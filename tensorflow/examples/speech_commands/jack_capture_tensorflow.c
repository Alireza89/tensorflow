/*
 *  File Name:  jack_capture_tensorflow.c
 *  Demo:       https://youtu.be/2CoRbuRRKbw
 *  Author:     Alireza Sameni
 *  Email:      alireza_sameni@live.com
 *  Date:       January, 2019
 *
 *  Edited code from:
 *  github.com/jackaudio/jack2/blob/master/example-clients/capture_client.c
 */


/*
This program should be used in conjunction with test_live_stream.cc

JACK (jack audio connection kit audio server daemon) binaries need to be installed.
This program is tested on JACK2 version 1.9.12.

to configure JACK, run QjackCtl (an application to control the JACK) and set the
following parameters for the Alsa sound driver:
Sample Rate = 48000
Frames/Period = 1440

After setting the correct parameters, now the program can be executed:

# to compile and link:
gcc -g -o "jack_capture_tensorflow" \
"./jack_capture_tensorflow.c" -ljack -lpthread

# to run:
./jack_capture_tensorflow

please be informed that by default "system:capture_1" is chosen as the audio input source.
you can change this by editing the capture_source_name [] variable in the code.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <pthread.h>
#include <getopt.h>

#include <sys/mman.h> //for mmap()
#include <fcntl.h> //for O_RDWR, S_IRUSR|S_IWUSR

#include <jack/jack.h>
#include <jack/ringbuffer.h>

typedef struct _thread_info {
    pthread_t thread_id;
    jack_nframes_t duration;
    jack_nframes_t rb_size;
    jack_client_t *client;
    unsigned int channels;
    int bitdepth;
    char *path;
    volatile int can_capture;
    volatile int can_process;
    volatile int status;
} jack_thread_info_t;

char capture_source_name [] = "system:capture_1"; //Audio input source

unsigned char *mmap_audio_data;
unsigned char *audio_data_internal_buffer;
unsigned char *mmap_toggle_var_data;
const int size_of_audio_buf_file_in_byte = 4*480; // sizeof(float)*(1440/3)


/* JACK data */
unsigned int nports;
jack_port_t **ports;
jack_default_audio_sample_t **in;
jack_nframes_t nframes;
const size_t sample_size = sizeof(jack_default_audio_sample_t);

/* Synchronization between process thread and memory map thread. */
#define DEFAULT_RB_SIZE 16384		/* ringbuffer size in frames */
jack_ringbuffer_t *rb;
pthread_mutex_t memory_map_thread_lock = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t  data_ready = PTHREAD_COND_INITIALIZER;
long overruns = 0;


void *
memory_map_thread (void *arg)
{
	jack_thread_info_t *info = (jack_thread_info_t *) arg;
	jack_nframes_t samples_per_frame = info->channels;
	size_t bytes_per_frame = samples_per_frame * sample_size;
	void *framebuf = malloc (bytes_per_frame);

	pthread_setcanceltype (PTHREAD_CANCEL_ASYNCHRONOUS, NULL);
	pthread_mutex_lock (&memory_map_thread_lock);

	info->status = 0;

	unsigned char sequence_1_2_3 = 0;
	unsigned int idx_byte = 0;
	unsigned char bytes_sequence_of_single_32bit_float[4];

	while (1) {
		while ((info->can_capture)&&(jack_ringbuffer_read_space(rb) >= bytes_per_frame))
		{
			jack_ringbuffer_read (rb, framebuf, bytes_per_frame);
			if( sequence_1_2_3 % 3 == 0 )
			{
				sequence_1_2_3 = 0;
				*((float *)bytes_sequence_of_single_32bit_float) = *((float *)framebuf);
				memcpy(audio_data_internal_buffer+idx_byte, bytes_sequence_of_single_32bit_float, 4);
				idx_byte = idx_byte + 4;
			}
			sequence_1_2_3++;
		}

		memcpy(mmap_audio_data, audio_data_internal_buffer, size_of_audio_buf_file_in_byte);
		idx_byte = 0;
		sequence_1_2_3 = 0;
		mmap_toggle_var_data[0] *= (signed char) -1;
		pthread_cond_wait (&data_ready, &memory_map_thread_lock); /* wait until process() signals more data */

	}//while (1)

	return 0; //unreachable:
}

int
process (jack_nframes_t nframes, void *arg)
{
	int chn;
	size_t i;
	jack_thread_info_t *info = (jack_thread_info_t *) arg;

	/* Do nothing until we're ready to begin. */
	if ((!info->can_process) || (!info->can_capture))
		return 0;

	for (chn = 0; chn < nports; chn++)
		in[chn] = jack_port_get_buffer (ports[chn], nframes);

	/* Sndfile requires interleaved data.  It is simpler here to
	 * just queue interleaved samples to a single ringbuffer. */
	for (i = 0; i < nframes; i++) {
		for (chn = 0; chn < nports; chn++) {
			if (jack_ringbuffer_write (rb, (void *) (in[chn]+i),
					      sample_size)
			    < sample_size)
				overruns++;
		}
	}

	/* Tell the memory map thread there is work to do.  If it is already
	 * running, the lock will not be available.  We can't wait
	 * here in the process() thread, but we don't need to signal
	 * in that case, because the memory map thread will read all the
	 * data queued before waiting again. */
	if (pthread_mutex_trylock (&memory_map_thread_lock) == 0) {
	    pthread_cond_signal (&data_ready);
	    pthread_mutex_unlock (&memory_map_thread_lock);
	}

	return 0;
}

void
jack_shutdown (void *arg)
{
	fprintf (stderr, "JACK shutdown\n");
	// exit (0);
	abort();
}

void
setup_memory_map_thread (jack_thread_info_t *info)
{
	info->can_capture = 0;
	pthread_create (&info->thread_id, NULL, memory_map_thread, info);
}

void
run_memory_map_thread (jack_thread_info_t *info)
{
	info->can_capture = 1;
	pthread_join (info->thread_id, NULL);
	if (overruns > 0) {
		fprintf (stderr,
			 "jackrec failed with %ld overruns.\n", overruns);
		fprintf (stderr, " try a bigger buffer than -B %"
			 PRIu32 ".\n", info->rb_size);
		info->status = EPIPE;
	}
}

void
setup_ports (int sources, char *source_names[], jack_thread_info_t *info)
{
	unsigned int i;
	size_t in_size;

	/* Allocate data structures that depend on the number of ports. */
	nports = sources;
	ports = (jack_port_t **) malloc (sizeof (jack_port_t *) * nports);
	in_size =  nports * sizeof (jack_default_audio_sample_t *);
	in = (jack_default_audio_sample_t **) malloc (in_size);
	rb = jack_ringbuffer_create (nports * sample_size * info->rb_size);

	/* When JACK is running realtime, jack_activate() will have
	 * called mlockall() to lock our pages into memory map.  But, we
	 * still need to touch any newly allocated pages before
	 * process() starts using them.  Otherwise, a page fault could
	 * create a delay that would force JACK to shut us down. */
	memset(in, 0, in_size);
	memset(rb->buf, 0, rb->size);

	for (i = 0; i < nports; i++) {
		char name[64];

		sprintf (name, "input%d", i+1);

		if ((ports[i] = jack_port_register (info->client, name, JACK_DEFAULT_AUDIO_TYPE, JackPortIsInput, 0)) == 0) {
			fprintf (stderr, "cannot register input port \"%s\"!\n", name);
			jack_client_close (info->client);
			exit (1);
		}
	}

	for (i = 0; i < nports; i++) {
		if (jack_connect (info->client, source_names[i], jack_port_name (ports[i]))) {
			fprintf (stderr, "cannot connect input port %s to %s\n", jack_port_name (ports[i]), source_names[i]);
			jack_client_close (info->client);
			exit (1);
		}
	}

	info->can_process = 1;		/* process() can start, now */
}

int
main (int argc, char *argv[])
{
	//>>>>>>>
	const char mmap_audio_buf_file_name [] = "/tmp/mem_mapped_audio_buffer_file";
	const int audio_buf_fd = open(mmap_audio_buf_file_name, O_RDWR|O_CREAT, S_IRWXU);
	for(int i=0; i<size_of_audio_buf_file_in_byte; i++) {
		write(audio_buf_fd, "0", 1);
	  }

    mmap_audio_data = (unsigned char*)
    		mmap((caddr_t)0, size_of_audio_buf_file_in_byte, PROT_WRITE, MAP_SHARED, audio_buf_fd, 0);
    close(audio_buf_fd);

    audio_data_internal_buffer= malloc(size_of_audio_buf_file_in_byte);

	const char mmap_toggle_var_file_name [] = "/tmp/mem_mapped_toggle_var_file";
	const int size_of_toggle_var_file_in_byte = 1; //exactly one byte
    const int toggle_var_fd = open(mmap_toggle_var_file_name, O_RDWR|O_CREAT, S_IRWXU);
	for(int i=0; i<size_of_toggle_var_file_in_byte; i++) {
		write(toggle_var_fd, "0", 1);
	  }

    mmap_toggle_var_data = (unsigned char*)
    		mmap((caddr_t)0, size_of_toggle_var_file_in_byte, PROT_WRITE, MAP_SHARED, toggle_var_fd, 0);
    close(toggle_var_fd);
    //>>>>>>>


	jack_client_t *client;
	jack_thread_info_t thread_info;
	memset (&thread_info, 0, sizeof (thread_info));
	thread_info.rb_size = DEFAULT_RB_SIZE;


	if ((client = jack_client_open ("jackrec", JackNullOption, NULL)) == 0) {
		fprintf (stderr, "jack server not running?\n");
		exit (1);
	}

	thread_info.client = client;
	thread_info.channels = 1; //mono audio
	thread_info.bitdepth = 16; //for 16-bit wav
	thread_info.can_process = 0;
	thread_info.duration = JACK_MAX_FRAMES; //4294967295
	setup_memory_map_thread (&thread_info);

	jack_set_process_callback (client, process, &thread_info);
	jack_on_shutdown (client, jack_shutdown, &thread_info);

	if (jack_activate (client)) {
		fprintf (stderr, "cannot activate client");
	}


	char *capture_source[thread_info.channels]; //a mono microphone is sufficient
	capture_source[0] = capture_source_name;
	setup_ports (thread_info.channels, capture_source, &thread_info);

	run_memory_map_thread (&thread_info);

	jack_client_close (client);

	jack_ringbuffer_free (rb);

        munmap(mmap_audio_data, size_of_audio_buf_file_in_byte);
        munmap(mmap_toggle_var_data, size_of_toggle_var_file_in_byte);

	exit (0);
}
