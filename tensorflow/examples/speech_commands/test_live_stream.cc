/*
 *  File Name:  test_live_stream.cc
 *  Demo:       https://youtu.be/2CoRbuRRKbw
 *  Author:     Alireza Sameni
 *  Email:      alireza_sameni@live.com
 *  Date:       January, 2019
 *
 *  Edited code from:
 *  github.com/tensorflow/tensorflow/blob/master/tensorflow/
 *  examples/speech_commands/test_streaming_accuracy.cc
 */


/*
This program should be used in conjunction with jack_capture_tensorflow.c
jack_capture_tensorflow should be executed before running this program.

This file (test_live_stream.cc) should be coppied to the following path:
tensorflow/tensorflow/examples/speech_commands

Before building this file with bazel, the BUILD file should be editted.
Open tensorflow/tensorflow/examples/speech_commands/BUILD and put the following
code at the end of the file and save it:

tf_cc_binary(
    name = "test_live_stream",
    srcs = [
        "test_live_stream.cc",
    ],
    deps = [
        ":recognize_commands",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
    ],
)

Now this program can be built with Bazel.

bazel run tensorflow/examples/speech_commands:test_live_stream -- \
--graph=/tmp/conv_frozen.pb \
--labels=/tmp/speech_commands_train/conv_labels.txt
*/


#include <fstream>
#include <iomanip>
#include <unordered_set>
#include <vector>

#include <sys/mman.h> //for mmap()
#include <fcntl.h>   // for O_RDWR, S_IRUSR
#include <chrono>   //  for ms ns
#include <thread>  //   for sleep()

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/wav/wav_io.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/examples/speech_commands/recognize_commands.h"

// These are all common classes it's handy to reference with no namespace.
using tensorflow::Flag;
using tensorflow::Status;
using tensorflow::Tensor;
using tensorflow::int32;
using tensorflow::int64;
using tensorflow::string;
using tensorflow::uint16;
using tensorflow::uint32;

namespace {
unsigned char *mmap_audio_data;
unsigned char *mmap_toggle_var_data;

// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
Status LoadGraph(const string& graph_file_name,
                 std::unique_ptr<tensorflow::Session>* session) {
  tensorflow::GraphDef graph_def;
  Status load_graph_status =
      ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
  if (!load_graph_status.ok()) {
    return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                        graph_file_name, "'");
  }
  session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
  Status session_create_status = (*session)->Create(graph_def);
  if (!session_create_status.ok()) {
    return session_create_status;
  }
  return Status::OK();
}

// Takes a file name, and loads a list of labels from it, one per line, and
// returns a vector of the strings.
Status ReadLabelsFile(const string& file_name, std::vector<string>* result) {
  std::ifstream file(file_name);
  if (!file) {
    return tensorflow::errors::NotFound("Labels file '", file_name,
                                        "' not found.");
  }
  result->clear();
  string line;
  while (std::getline(file, line)) {
    result->push_back(line);
  }
  return Status::OK();
}

}  // namespace

int main(int argc, char* argv[]) {
  string graph = "";
  string labels = "";
  string input_data_name = "decoded_sample_data:0";
  string input_rate_name = "decoded_sample_data:1";
  string output_name = "labels_softmax";

  const int32 sample_rate = 16000;
  const int64 clip_duration_samples = 1*16000; // one second
  const int32 buff_duration_samples = 480; // (30ms)*16000Hz = 1440/3

  //const int64 clip_stride_samples = buff_duration_samples;
  const int64 clip_stride_samples = 5*buff_duration_samples; // yields less delay with same accuracy

  int32 average_window_ms = 400;
  int32 suppression_ms = 300;
  float detection_threshold = 0.6f;

  std::vector<Flag> flag_list = {
      Flag("graph", &graph, "model to be executed"),
      Flag("labels", &labels, "path to file containing labels"),
      Flag("input_data_name", &input_data_name,
           "name of input data node in model"),
      Flag("input_rate_name", &input_rate_name,
           "name of input sample rate node in model"),
      Flag("output_name", &output_name, "name of output node in model"),
      Flag("average_window_ms", &average_window_ms,
           "length of window to smooth results over"),
      Flag("suppression_ms", &suppression_ms,
           "how long to ignore others for after a recognition"),
      Flag("detection_threshold", &detection_threshold,
           "what score is required to trigger detection of a word"),
  };
  string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    LOG(ERROR) << usage;
    return -1;
  }

  // We need to call this to set up global state for TensorFlow.
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  if (argc > 1) {
    LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
    return -1;
  }

  // First we load and initialize the model.
  std::unique_ptr<tensorflow::Session> session;
  Status load_graph_status = LoadGraph(graph, &session);
  if (!load_graph_status.ok()) {
    LOG(ERROR) << load_graph_status;
    return -1;
  }

  std::vector<string> labels_list;
  Status read_labels_status = ReadLabelsFile(labels, &labels_list);
  if (!read_labels_status.ok()) {
    LOG(ERROR) << read_labels_status;
    return -1;
  }

  std::vector<float> Main_audio_data_vector (clip_duration_samples);


  Tensor audio_data_tensor(tensorflow::DT_FLOAT,
                           tensorflow::TensorShape({clip_duration_samples, 1}));

  Tensor sample_rate_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
  sample_rate_tensor.scalar<int32>()() = sample_rate;

  tensorflow::RecognizeCommands recognize_commands(
      labels_list, average_window_ms, detection_threshold, suppression_ms);


  //>>>>>>>
  const char mmap_audio_buf_file_name [] = "/tmp/mem_mapped_audio_buffer_file";
  const int size_of_audio_buf_file_in_byte = 4*buff_duration_samples; // sizeof(float)*(1440/3)=4*480
  const int audio_buf_fd = open(mmap_audio_buf_file_name, O_RDONLY, S_IRUSR);
  if(audio_buf_fd<0) {
      LOG(ERROR) << "mem_mapped_audio_buffer_file not found";
      return -1;
    }
  mmap_audio_data = (unsigned char*)
  		mmap((caddr_t)0, size_of_audio_buf_file_in_byte, PROT_READ, MAP_SHARED, audio_buf_fd, 0);
  close(audio_buf_fd);


  const char mmap_toggle_var_file_name [] = "/tmp/mem_mapped_toggle_var_file";
  const int size_of_toggle_var_file_in_byte = 1; // exactly one byte
  const int toggle_var_fd = open(mmap_toggle_var_file_name, O_RDONLY, S_IRUSR);
  if(audio_buf_fd<0) {
      LOG(ERROR) << "mem_mapped_toggle_var_file not found";
      return -1;
    }
  mmap_toggle_var_data = (unsigned char*)
   		mmap((caddr_t)0, size_of_toggle_var_file_in_byte, PROT_READ, MAP_SHARED, toggle_var_fd, 0);
  close(toggle_var_fd);
  //>>>>>>>

  const int size_of_audio_vector_in_float = size_of_audio_buf_file_in_byte/4; // 4 is sizeof(float)
  unsigned char bytes_array[4]; // array of 4 bytes to represent a single float value.
  int idx_float;  //index of audio data in float. range: 0 to 480-1
  int idx_byte;  // index of audio data in byte.  range: 0 to 1920-1
  unsigned char prev_mmap_toggle_var_data = 69;     //some random value other than 0 or 255
  unsigned char current_mmap_toggle_var_data = 85; // some random value other than 0 or 255
  int64 audio_data_offset = 0;
  LOG(INFO) << "Waiting for the Jack Audio Capture Module...";

  //to measure your_microphone_silence_bias, refer to "SECTION: Measuring Microphone Silence Bias"
  //you should edit this value to your own mic's silence bias:
  float your_microphone_silence_bias = 0.0f; // my low-end microphone_silence_average was +0.0079

  while (1) {
    while (1) {
      current_mmap_toggle_var_data = mmap_toggle_var_data[0];
      if(prev_mmap_toggle_var_data != current_mmap_toggle_var_data) {
          break;
        }
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
      }
    prev_mmap_toggle_var_data = current_mmap_toggle_var_data;

    // zero initialization
    std::vector<float> mmap_audio_data_vector (size_of_audio_vector_in_float);
    idx_float=0;
    for (idx_byte=0 ; idx_byte<size_of_audio_buf_file_in_byte ; idx_byte += 4) {
        memcpy(bytes_array, mmap_audio_data+idx_byte + 0, 4);
        mmap_audio_data_vector.at(idx_float) = *((float *)bytes_array);
        mmap_audio_data_vector.at(idx_float) -= your_microphone_silence_bias;
        idx_float++;
      }


//    //SECTION: Measuring Microphone Silence Bias
//    //uncomment this to see your microphone silence bias, then put this...
//    //...value to your_microphone_silence_bias variable.
//    //the mic should be in an approximately silent environment
//    float avg = accumulate(
//    		Main_audio_data_vector.begin(),
//			Main_audio_data_vector.end(), 0.0)/
//		    Main_audio_data_vector.size();
//    printf("\nsilence bias = %+1.4f\n", avg);
//    if(your_microphone_silence_bias==0) {
//        printf("Now you should change your_microphone_silence_bias to this value\n");
//      }



//    //uncomment this to see the audio PCM values:
//    for (idx_float=0 ; idx_float<size_of_audio_vector_in_float ; idx_float ++) {
//    	printf("%+1.4f    ", mmap_audio_data_vector.at(idx_float));
//    	if (idx_float%10==9) {
//    		printf("\n");
//    	  }
//      }


    // zero initialization
    std::vector<float> audio_data_vector_prev (clip_duration_samples-buff_duration_samples);

    std::copy(
    		&(Main_audio_data_vector[buff_duration_samples]),
			&(Main_audio_data_vector[clip_duration_samples-1]),
			audio_data_vector_prev.begin());

    audio_data_vector_prev.insert(
    		audio_data_vector_prev.end(),
    		mmap_audio_data_vector.begin(),
    		mmap_audio_data_vector.end() );

    std::copy(
    		audio_data_vector_prev.begin(),
            audio_data_vector_prev.end(),
    		Main_audio_data_vector.begin() );

    audio_data_vector_prev.clear();
    mmap_audio_data_vector.clear();

    std::copy(
    		Main_audio_data_vector.begin(),
    		Main_audio_data_vector.end(),
    		audio_data_tensor.flat<float>().data());

    // Actually run the audio through the model.
    std::vector<Tensor> outputs;
    Status run_status = session->Run({{input_data_name, audio_data_tensor},
                                      {input_rate_name, sample_rate_tensor}},
                                     {output_name}, {}, &outputs);

    if (!run_status.ok()) {
        LOG(ERROR) << "Running model failed: " << run_status;
        return -1;
      }

    audio_data_offset += clip_stride_samples;
    const int64 current_time_ms = (audio_data_offset * 1000) / sample_rate;
    string found_command;
    float score;
    bool is_new_command;
    Status recognize_status = recognize_commands.ProcessLatestResults(
          outputs[0], current_time_ms, &found_command, &score, &is_new_command);

    if (!recognize_status.ok()) {
        LOG(ERROR) << "Recognition processing failed: " << recognize_status;
        return -1;
      }

    if (is_new_command && (found_command != "_silence_")) {
    	string padding = ""; // white space padding to allign the printed scores.
    	for(int i=0 ; i < 15-found_command.size() ; i++) {
            padding += " ";
    	  }
     	LOG(INFO) << "The Recognized Command :" << found_command
     	<< padding << "Score: " << score ;
      }
    }

  munmap(mmap_audio_data, size_of_audio_buf_file_in_byte);
  munmap(mmap_toggle_var_data, size_of_toggle_var_file_in_byte);

  return 0;
}
