#include <zlib.h>
#include <libgen.h>
#include "smithwaterman.h"
#include <ctype.h>
#include <strings.h>
#include <sys/time.h>
#include "kseq.h"
#include <stdlib.h>
#include <malloc.h>

/** Need to define this when you want to output performance information **/
//#define PROFILING

//#define ITERATIONS 5

void pfn_notify(const char *errinfo, const void *private_info, size_t cb, void *user_data)
{
    fprintf(stderr, "OpenCL Error (via pfn_notify): %s\n", errinfo);
}

int exists(const char *fname)
{
    FILE *file;
    if ((file = fopen(fname, "r")) != NULL)
    {
        fclose(file);
        return 1;
    }
    return 0;
}




/* argc:
#$1 - sequence file
#$2 - target file
#$3 - number of super blocks for sequence file (ceil((#seqs in file / #seqs to device))
#$4 - number of super blocks for target file (ceil((#seqs in file / #seqs to device))
#$5 - sequence_index_start
#$6 - sequence_index_end
#$7 - target_index_start
#$8 - target_index_end
#$9 - File location to store performance information
 */
KSEQ_INIT(gzFile, gzread)

int main(int argc, char **argv) {


	int iter;
	int total_number_sequences = 0;
	int total_number_targets = 0;
	int total_alignments = 0;

	int sequence_index_start = 0;
	int sequence_index_end = 0;
	int target_index_start = 0;
	int target_index_end = 0;


#ifdef PROFILING
	struct timeval startt, endt, startttotal, endttotal, starttiter, endtiter,
	timer_iter_total, timer_mm, timer_init, timer_H2D, timer_D2H, timer_kernel1, timer_kernel2, timer_plotAlignments, timer_kernel_builder,
	timer_total;

	struct timeval timer_iter_total_array[ITERATIONS];
	struct timeval timer_mm_array[ITERATIONS];
	struct timeval timer_init_array[ITERATIONS];
	struct timeval timer_H2D_array[ITERATIONS];
	struct timeval timer_D2H_array[ITERATIONS];
	struct timeval timer_kernel1_array[ITERATIONS];
	struct timeval timer_kernel2_array[ITERATIONS];
	struct timeval timer_plotAlignments_array[ITERATIONS];
	struct timeval timer_kernel_builder_array[ITERATIONS];

	timer_total.tv_usec = 0;
	timer_total.tv_sec = 0;
#endif

#ifdef PROFILING
	gettimeofday(&startttotal, NULL);
#endif
	for(iter=0; iter<ITERATIONS; iter++) {

		total_number_sequences = 0;
		total_number_targets = 0;

#ifdef PROFILING
		timer_iter_total.tv_usec = 0;
		timer_iter_total.tv_sec = 0;

		timer_mm.tv_usec = 0;
		timer_mm.tv_sec = 0;

		timer_init.tv_usec = 0;
		timer_init.tv_sec = 0;

		timer_H2D.tv_usec = 0;
		timer_H2D.tv_sec = 0;

		timer_D2H.tv_usec = 0;
		timer_D2H.tv_sec = 0;

		timer_kernel1.tv_usec = 0;
		timer_kernel1.tv_sec = 0;

		timer_kernel2.tv_usec = 0;
		timer_kernel2.tv_sec = 0;

		timer_plotAlignments.tv_usec = 0;
		timer_plotAlignments.tv_sec = 0;

		timer_kernel_builder.tv_usec = 0;
		timer_kernel_builder.tv_sec = 0;
#endif

#ifdef PROFILING
		gettimeofday(&starttiter, NULL);
#endif
		/** We do not have 11 arguments **/
		if (argc != 10) {
			fprintf(stderr,"Error: use: ./paswas_opencl <sequenceFile> <targetFile> <superBlocksX> <superBlocksY> <sequence_index_start> <sequence_index_end> <target_index_start> <target_index_end> <performanceFileLoc>!\n");
			exit(EXIT_FAILURE);
		}

		cl_platform_id platforms = NULL;
		cl_uint ret_num_platforms = 0;
		cl_uint ret_num_devices = 0;
		cl_device_id devices = NULL;
		cl_command_queue queue = NULL;
		cl_context context = NULL;
		cl_program program = NULL;
		cl_kernel kernel_calculateScore = NULL;
		cl_kernel kernel_traceback = NULL;
		cl_int error_check;

		error_check = clGetPlatformIDs(1, &platforms, &ret_num_platforms);
		if(error_check != CL_SUCCESS) {
			fprintf(stderr,"Could not find a valid OpenCL platform\n");
			exit(EXIT_FAILURE);
		}

		sequence_index_start = atoi(argv[5]);
		if(sequence_index_start<0){
			fprintf(stderr,"Please provide a valid sequence start index:%d\n",sequence_index_start);
			exit(EXIT_FAILURE);
		}
		fprintf(stderr,"sequence_index_start:%d\n",sequence_index_start);

		sequence_index_end = atoi(argv[6]);
		if(sequence_index_end<=sequence_index_start){
			fprintf(stderr,"Please provide a valid sequence end index:%d\n",sequence_index_end);
			exit(EXIT_FAILURE);
		}
		fprintf(stderr,"sequence_index_end:%d\n",sequence_index_end);

		target_index_start = atoi(argv[7]);
		if(target_index_start<0){
			fprintf(stderr,"Please provide a valid target start index:%d\n",target_index_start);
			exit(EXIT_FAILURE);
		}
		fprintf(stderr,"target_index_start:%d\n",target_index_start);

		target_index_end = atoi(argv[8]);
		if(target_index_end<=target_index_start){
			fprintf(stderr,"Please provide a valid target end index:%d\n",target_index_end);
			exit(EXIT_FAILURE);
		}
		fprintf(stderr,"target_index_end:%d\n",target_index_end);


	#ifdef NVIDIA
		error_check = clGetDeviceIDs(platforms, CL_DEVICE_TYPE_GPU, 1, &devices, &ret_num_devices);
	#endif
	#ifdef AMD_GPU
		error_check = clGetDeviceIDs(platforms, CL_DEVICE_TYPE_GPU, 1, &devices, &ret_num_devices);
	#endif
	#ifdef AMD_CPU
		error_check = clGetDeviceIDs(platforms, CL_DEVICE_TYPE_CPU, 1, &devices, &ret_num_devices);
	#endif
	#ifdef INTEL
		error_check = clGetDeviceIDs(platforms, CL_DEVICE_TYPE_CPU, 1, &devices, &ret_num_devices);
	#endif

		if(error_check != CL_SUCCESS) {
				fprintf(stderr,"No OpenCL devices found\n");
				exit(EXIT_FAILURE);
		}

		cl_context_properties properties[] = {
				CL_CONTEXT_PLATFORM, (cl_context_properties) platforms, 0
		};

		context = clCreateContext(properties, 1, &devices, &pfn_notify, NULL, &error_check);
		if(error_check != CL_SUCCESS) {
				fprintf(stderr,"Context could not be created\n");
				exit(EXIT_FAILURE);
		}

		queue = clCreateCommandQueue(context, devices, 0, &error_check);
		if(error_check != CL_SUCCESS) {
				fprintf(stderr,"Command queue could not be created");
				exit(EXIT_FAILURE);
		}

		dimensions superBlocks;
		superBlocks.x = atoi(argv[3]);
		superBlocks.y = atoi(argv[4]);
		if(!superBlocks.x || !superBlocks.y) {
			fprintf(stderr,"Please provide integer values for superblock_x or superblock_y\n");
		}
		superBlocks.z = 0;
		// variables needed for the application:

		fprintf(stderr,"Superblocksx: %d\tSuperblocksy:%d\n",superBlocks.x,superBlocks.y);
		fprintf(stderr,"#SEQUENCES: %d\t#TARGET:%d\n",NUMBER_SEQUENCES,NUMBER_TARGETS);

#ifdef PROFILING
		gettimeofday(&startt, NULL);
#endif

		char *descSequences = (char *) malloc(sizeof(char) * superBlocks.x*NUMBER_SEQUENCES*MAX_LINE_LENGTH);
		char *descTargets = (char *) malloc(sizeof(char) * superBlocks.y* NUMBER_TARGETS*MAX_LINE_LENGTH);
		descSequences[0] = '\0';
		descTargets[0] = '\0';

#ifdef PROFILING
		gettimeofday(&endt, NULL);
		timer_mm.tv_usec += (endt.tv_sec*1000000+endt.tv_usec) - (startt.tv_sec*1000000+startt.tv_usec);
#endif

#ifdef PROFILING
		gettimeofday(&startt, NULL);
#endif

		float h_scoringsMatrix[SCORINGS_MAT_SIZE*SCORINGS_MAT_SIZE] = {0};
		fillScoringsMatrix(h_scoringsMatrix);

#ifdef PROFILING
		gettimeofday(&endt, NULL);
		timer_init.tv_usec += (endt.tv_sec*1000000+endt.tv_usec) - (startt.tv_sec*1000000+startt.tv_usec);
#endif

		int min_align;
		error_check = clGetDeviceInfo(devices, CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(int), &min_align, NULL);
		if(error_check != CL_SUCCESS) {
			fprintf(stderr,"GetDeviceInfo failed");
			exit(EXIT_FAILURE);
		}
		fprintf(stderr, "ALIGN = %d\n", min_align);

#ifdef PROFILING
		gettimeofday(&startt, NULL);
#endif

#ifdef NO_ZERO_COPY
		char *h_sequences = 0, *h_targets = 0;
		cl_mem d_sequences, d_targets, d_matrix, d_globalMaxima, d_globalDirection;
		cl_mem d_indexIncrement, d_scoringsMatrix;
		GlobalMatrix *h_matrix = 0;
		GlobalMaxima *h_globalMaxima = 0;
		GlobalDirection *h_globalDirectionZeroCopy = 0;
		StartingPoints *h_startingPointsZeroCopy = 0;
		float *h_maxPossibleScoreZeroCopy = 0;
		cl_mem d_startingPointsZeroCopy, d_maxPossibleScoreZeroCopy;

		init(&h_sequences, &h_targets,
			 &d_sequences, &d_targets,
			 &h_matrix, &d_matrix,
				&h_globalMaxima, &d_globalMaxima,
				&d_globalDirection, &h_globalDirectionZeroCopy,
				&d_startingPointsZeroCopy, &h_startingPointsZeroCopy,
				&d_maxPossibleScoreZeroCopy, &h_maxPossibleScoreZeroCopy,
				&d_scoringsMatrix,
				&d_indexIncrement,
				superBlocks,
				context,
				queue,
				error_check);
#endif

		/**Data Structure Initialization with NVIDIA_zero_copy **/
#ifdef NVIDIA_ZERO_COPY
			char *h_sequences = 0, *h_targets = 0;
			cl_mem d_sequences, d_targets, d_matrix, d_globalMaxima, d_globalDirection;
			cl_mem d_indexIncrement, d_scoringsMatrix;
			GlobalMatrix *h_matrix = 0;
			GlobalMaxima *h_globalMaxima = 0;
			StartingPoints *h_startingPointsZeroCopy = 0;
			float *h_maxPossibleScoreZeroCopy = 0;
			GlobalDirection *h_globalDirectionZeroCopy = 0;
			cl_mem pinned_startingPointsZeroCopy, pinned_maxPossibleScoreZeroCopy, pinned_globalDirectionZeroCopy;
			cl_mem d_startingPointsZeroCopy, d_maxPossibleScoreZeroCopy;

			// allocate memory on host & device:
			init_zc(&h_sequences, &h_targets, 
					&d_sequences, &d_targets,
				&h_matrix, &d_matrix,
				&h_globalMaxima, &d_globalMaxima,
				&d_startingPointsZeroCopy, &h_startingPointsZeroCopy, &pinned_startingPointsZeroCopy,
				&d_maxPossibleScoreZeroCopy, &h_maxPossibleScoreZeroCopy, &pinned_maxPossibleScoreZeroCopy,
				&d_globalDirection, &h_globalDirectionZeroCopy, &pinned_globalDirectionZeroCopy,
				&d_scoringsMatrix,
				&d_indexIncrement,
				superBlocks,
				context, queue,
				error_check);
#endif

			/**Data Structure Initialization with INTEL_CPU_zero_copy **/
#ifdef INTEL_ZERO_COPY
			char *h_sequences = 0, *h_targets = 0;
			cl_mem d_sequences, d_targets, d_matrix, d_globalMaxima, d_globalDirection;
			cl_mem d_indexIncrement, d_scoringsMatrix;
			GlobalMatrix *h_matrix = 0;
			GlobalMaxima *h_globalMaxima = 0;
			GlobalDirection *h_globalDirectionZeroCopy = 0;
			StartingPoints *h_startingPointsZeroCopy = 0;
			float *h_maxPossibleScoreZeroCopy = 0;
			cl_mem d_startingPointsZeroCopy, d_maxPossibleScoreZeroCopy;

			/**Zero copy variables base address **/
			StartingPoints* startingPointsData = (StartingPoints*)memalign(min_align/8, sizeof(StartingPoints));
			float* maxPossibleScoreData = (float*)memalign(min_align/8, sizeof(float) * NUMBER_SEQUENCES * superBlocks.x);
			GlobalDirection* globalDirectionData = (GlobalDirection*)memalign(min_align/8, sizeof(GlobalDirection));
			// allocate memory on host & device:
			init_zc_CPU(&h_sequences, &h_targets,
					&d_sequences, &d_targets,
					&h_matrix, &d_matrix,
					&h_globalMaxima, &d_globalMaxima,
					&d_startingPointsZeroCopy, &startingPointsData,
					&d_maxPossibleScoreZeroCopy, &maxPossibleScoreData,
					&d_globalDirection, &globalDirectionData,
					&d_scoringsMatrix,
					&d_indexIncrement,
					superBlocks,
					context,
					queue,
					error_check);
#endif

#ifdef PROFILING
		gettimeofday(&endt, NULL);
		timer_mm.tv_usec += (endt.tv_sec*1000000+endt.tv_usec) - (startt.tv_sec*1000000+startt.tv_usec);
#endif

		/**Fill GlobalMatrix with 0's, feature is only available in OpenCL 1.2, however this is not needed for the shared memory case**/
#ifdef PROFILING
		gettimeofday(&startt, NULL);
#endif

		#if (defined(INTEL) || defined(AMD)) && (defined(GLOBAL_MEM4))
			float zero_pattern = 0.0;
			error_check = clEnqueueFillBuffer(queue, d_matrix, &zero_pattern, sizeof(float), 0, sizeof(GlobalMatrix), 0, NULL, NULL);
			if(error_check != CL_SUCCESS) {
					fprintf(stderr,"Failed to initialize GlobalMatrix to zero\n");
					exit(EXIT_FAILURE);
			}
			clFinish(queue);
		#endif

#ifdef PROFILING
		gettimeofday(&endt, NULL);
		timer_mm.tv_usec += (endt.tv_sec*1000000+endt.tv_usec) - (startt.tv_sec*1000000+startt.tv_usec);
#endif


#ifdef PROFILING
		gettimeofday(&startt, NULL);
#endif

#ifdef GLOBAL_MEM4
		GlobalSemaphores *h_semaphore = (GlobalSemaphores*) malloc(sizeof(GlobalSemaphores));
		cl_mem d_semaphore = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(GlobalSemaphores), NULL, &error_check);
		if(error_check != CL_SUCCESS) {
				fprintf(stderr,"Failed to create semaphore on device\n");
				exit(EXIT_FAILURE);
		}
		/** Initialize device buffer to all zeroes **/
		initSemaphor(&d_semaphore, context, queue, error_check);
#endif

#ifdef PROFILING
		gettimeofday(&endt, NULL);
		timer_mm.tv_usec += (endt.tv_sec*1000000+endt.tv_usec) - (startt.tv_sec*1000000+startt.tv_usec);
#endif

		/* Write scoringsMatrix to device buffer*/
#ifdef PROFILING
		gettimeofday(&startt, NULL);
#endif

		error_check = clEnqueueWriteBuffer(queue, d_scoringsMatrix, CL_TRUE, 0, sizeof(float) * 26 * 26, h_scoringsMatrix, 0, NULL, NULL);

#ifdef PROFILING
		gettimeofday(&endt, NULL);
		timer_H2D.tv_usec += (endt.tv_sec*1000000+endt.tv_usec) - (startt.tv_sec*1000000+startt.tv_usec);
#endif

		if(error_check != CL_SUCCESS) {
			fprintf(stderr,"Failed to write scoringsMatrix to device buffer");
			exit(EXIT_FAILURE);
		}

		gzFile targetFile, seqFile;
		kseq_t *target, *seq;

		targetFile = gzopen(argv[2], "r");
		seqFile = gzopen(argv[1], "r");

		if (!targetFile || !seqFile) {
			fprintf(stderr,"Error: could not open target/seq file!\n");
			return 1;
		}



#ifdef INTEL_ZERO_COPY

#ifdef PROFILING
		gettimeofday(&startt, NULL);
#endif
		h_maxPossibleScoreZeroCopy = (float *)clEnqueueMapBuffer(queue, d_maxPossibleScoreZeroCopy, CL_TRUE, CL_MAP_WRITE, 0, sizeof(float) * NUMBER_SEQUENCES * superBlocks.x, 0, NULL, NULL, &error_check);
		clFinish(queue);

#ifdef PROFILING
		gettimeofday(&endt, NULL);
		timer_H2D.tv_usec += (endt.tv_sec*1000000+endt.tv_usec) - (startt.tv_sec*1000000+startt.tv_usec);
#endif
		if(error_check != CL_SUCCESS) {
			fprintf(stderr, "clEnqueueMap CL_MAP_WRITE h_maxPossibleScoreZeroCopy => %d\n", error_check);
			exit(EXIT_FAILURE);
		}
#endif



		int t=target_index_start;
		int s=sequence_index_start;
		int target_offset = 0;
		int sequence_offset = 0;
		int l;

#ifdef PROFILING
		gettimeofday(&startt, NULL);
#endif
		target = kseq_init(targetFile);
		int skip=0;
		//Skip sequences which we do not need
		while(skip<target_index_start) {
			kseq_read(target);
			skip++;
		}

		while ((l = kseq_read(target)) >= 0 && t<target_index_end) {
			char * current = h_targets+(target_offset*(unsigned int)Y);
			char * currentDesc = descTargets + (target_offset*MAX_LINE_LENGTH);
			current[0] = '\0';
			currentDesc[0] = '\0';
			char * description = (char*) malloc( sizeof(char) * (target->name.l + target->comment.l + target->qual.l + 3));
			description[0] = '\0';

			strcpy(description, target->name.s);
			if(target->comment.l) {
				strcat(description, " ");
				strcat(description, target->comment.s);
			}

			if(target->qual.l) {
				strcat(description, " ");
				strcat(description, target->qual.s);
			}

			if(strlen(description)>=MAX_LINE_LENGTH) {
				description[MAX_LINE_LENGTH-1] = '\0';
			}
			strcpy(currentDesc,description);
			free(description);

			if(target->seq.l > Y) {
				fprintf(stderr, "Error: target read too long!\n");
				fprintf(stderr, "id: %s\n", currentDesc);
				fprintf(stderr, "Y:%d\n", Y);
				fprintf(stderr, "target_file:%s\n",argv[2]);
				return 1;
			}
			strncpy(current, target->seq.s, target->seq.l);

			int i=0;
			for (; i < target->seq.l; i++) {
				current[i] = toupper(current[i]);
				if (current[i] - 'A' < 0 || current[i] - 'A' > 25) {
					fprintf(stderr, "Error: wrong character in target file: '%c', desc: %s\n", current[i], current);
					return 1;
				}
			}
			for(; i < Y; i++) {
				current[i] = FILL_CHARACTER;
			}
			t++;
			target_offset++;
		}

		fprintf(stderr, "Read number of targets: %d from %s\n", target_offset,argv[2]);
		total_number_targets = target_offset;

		for (; target_offset < superBlocks.y * NUMBER_TARGETS; target_offset++) {
			for (int c=0; c < Y; c++)
				*(h_targets+target_offset*Y+c) = FILL_CHARACTER;
		}



		seq = kseq_init(seqFile);
		skip=0;

		//Skip sequences which we do not need
		while(skip<sequence_index_start) {
			kseq_read(seq);
			skip++;
		}

		while ((l = kseq_read(seq)) >= 0 && s<sequence_index_end) {
			h_maxPossibleScoreZeroCopy[sequence_offset] = 0;
			char * current = h_sequences +(sequence_offset*(unsigned int)X);
			char * currentDesc = descSequences + (sequence_offset*MAX_LINE_LENGTH);
			current[0] = '\0';
			currentDesc[0] = '\0';
			char * description = (char*) malloc( sizeof(char) * (seq->name.l + seq->comment.l + seq->qual.l + 3));
			description[0] = '\0';

			strcpy(description, seq->name.s);
			if(seq->comment.l) {
				strcat(description, " ");
				strcat(description, seq->comment.s);
			}
			if(seq->qual.l) {
				strcat(description, " ");
				strcat(description, seq->qual.s);
			}

			if(strlen(description)>=MAX_LINE_LENGTH) {
				description[MAX_LINE_LENGTH-1] = '\0';
			}

			strcpy(currentDesc,description);
			free(description);

			if(seq->seq.l > X) {
				fprintf(stderr, "Error: sequence read too long!\n");
				fprintf(stderr, "id: %s\n", currentDesc);
				fprintf(stderr, "X:%d\n", X);
				fprintf(stderr, "sequence_file:%s\n",argv[1]);
				return 1;
			}
			strncpy(current, seq->seq.s, seq->seq.l);

			int i=0;
			for (; i < seq->seq.l; i++) {
				current[i] = toupper(current[i]);
				h_maxPossibleScoreZeroCopy[sequence_offset] += HIGHEST_SCORE;
				if (current[i] - 'A' < 0 || current[i] - 'A' > 25) {
					fprintf(stderr, "Error: wrong character in seq file: '%c', desc: %s\n", current[i], current);
					return 1;
				}
			}
			for(; i < X; i++) {
				current[i] = FILL_CHARACTER;
			}
			h_maxPossibleScoreZeroCopy[sequence_offset] *= LOWER_LIMIT_MAX_SCORE;
			s++;
			sequence_offset++;
		}

		fprintf(stderr, "Read number of sequences: %d from %s\n", sequence_offset, argv[1]);
		total_number_sequences = sequence_offset;

		int count = 0;

		for (; sequence_offset < superBlocks.x * NUMBER_SEQUENCES; sequence_offset++) {
			count++;
			h_maxPossibleScoreZeroCopy[sequence_offset] = 0;
			for (int c=0; c < X; c++)
				*(h_sequences+sequence_offset*X+c) = FILL_CHARACTER;
		}



		kseq_destroy(target);
		gzclose(targetFile);
		kseq_destroy(seq);
		gzclose(seqFile);

#ifdef PROFILING
		gettimeofday(&endt, NULL);
		timer_init.tv_usec += (endt.tv_sec*1000000+endt.tv_usec) - (startt.tv_sec*1000000+startt.tv_usec);
#endif

		/* Write maxPossibleScoreZeroCopy to device buffer*/
#ifdef PROFILING
		gettimeofday(&startt, NULL);
#endif

	#if defined(NO_ZERO_COPY) || defined(NVIDIA_ZERO_COPY)
		error_check = clEnqueueWriteBuffer(queue, d_maxPossibleScoreZeroCopy, CL_TRUE, 0, sizeof(float) * NUMBER_SEQUENCES * superBlocks.x, h_maxPossibleScoreZeroCopy, 0, NULL, NULL);
	#endif
	#ifdef INTEL_ZERO_COPY
		error_check = clEnqueueUnmapMemObject(queue, d_maxPossibleScoreZeroCopy, h_maxPossibleScoreZeroCopy, 0, NULL, NULL);
	#endif
		clFinish(queue);

#ifdef PROFILING
		gettimeofday(&endt, NULL);
		timer_H2D.tv_usec += (endt.tv_sec*1000000+endt.tv_usec) - (startt.tv_sec*1000000+startt.tv_usec);
#endif

		if(error_check != CL_SUCCESS) {
			fprintf(stderr,"Failed to write maxPossibleScoreZeroCopy to buffer");
			exit(EXIT_FAILURE);
		}

#ifdef PROFILING
		gettimeofday(&startt, NULL);
#endif

		FILE *fp;
		size_t kernel_size;
		char *source;
		fp = fopen("kernel/smithwaterman_kern.cl", "rb");
		if(fp) {
			/* Set the file pointer to the end of the file */
			fseek(fp, 0, SEEK_END);

			kernel_size = ftell(fp);

			/* Set the file pointer back to the beginning of the file */
			fseek(fp, 0, SEEK_SET);

			source = (char*)malloc(kernel_size+1);
			fread(source, 1, kernel_size, fp);
			source[kernel_size] = '\0';

			fclose(fp);
		} else {
			fprintf(stderr,"Could not read from: kernel/smithwaterman_kern.cl\n");
			exit(EXIT_FAILURE);

		}

		program = clCreateProgramWithSource(context, 1, (const char **)&source, (const size_t *)&kernel_size, &error_check);
		if (error_check != CL_SUCCESS)
		{
			fprintf(stderr,"Could not create a program object with the provided source code");
			exit(EXIT_FAILURE);
		}

		if(source){
			free(source);
		}

		/**TODO Determine size of directives dynamically**/
		int size_directives = 2048;
		char *directives = (char *) malloc(sizeof(char) * size_directives);

#ifdef GLOBAL_MEM4
#ifdef NVIDIA
		sprintf(directives, "-DNVIDIA -DGLOBAL_MEM4 -DNUMBER_SEQUENCES=%d -DNUMBER_TARGETS=%d -DX=%d -DY=%d -DMINIMUM_SCORE=%f -DSHARED_X=%d -DSHARED_Y=%d -DWORKLOAD_X=%d -DWORKLOAD_Y=%d",NUMBER_SEQUENCES,NUMBER_TARGETS,X,Y,MINIMUM_SCORE,SHARED_X,SHARED_Y, WORKLOAD_X, WORKLOAD_Y);
#else
		sprintf(directives, "-DGLOBAL_MEM4 -DNUMBER_SEQUENCES=%d -DNUMBER_TARGETS=%d -DX=%d -DY=%d -DMINIMUM_SCORE=%f -DSHARED_X=%d -DSHARED_Y=%d -DWORKLOAD_X=%d -DWORKLOAD_Y=%d",NUMBER_SEQUENCES,NUMBER_TARGETS,X,Y,MINIMUM_SCORE,SHARED_X,SHARED_Y, WORKLOAD_X, WORKLOAD_Y);
#endif
#endif

#ifdef SHARED_MEM
#ifdef NVIDIA
		sprintf(directives, "-DNVIDIA -DSHARED_MEM -DNUMBER_SEQUENCES=%d -DNUMBER_TARGETS=%d -DX=%d -DY=%d -DMINIMUM_SCORE=%f -DSHARED_X=%d -DSHARED_Y=%d",NUMBER_SEQUENCES,NUMBER_TARGETS,X,Y,MINIMUM_SCORE,SHARED_X,SHARED_Y);
#else
		sprintf(directives, "-DSHARED_MEM -DNUMBER_SEQUENCES=%d -DNUMBER_TARGETS=%d -DX=%d -DY=%d -DMINIMUM_SCORE=%f -DSHARED_X=%d -DSHARED_Y=%d",NUMBER_SEQUENCES,NUMBER_TARGETS,X,Y,MINIMUM_SCORE,SHARED_X,SHARED_Y);
#endif
#endif

		error_check = clBuildProgram(program, 1, &devices,directives, NULL, NULL);
		free(directives);

		if (error_check != CL_SUCCESS)
		{
			size_t length;
			clGetProgramBuildInfo(program, devices, CL_PROGRAM_BUILD_LOG, 0, NULL, &length);
			char* buffer = (char*)malloc(length+1);
			clGetProgramBuildInfo(program, devices, CL_PROGRAM_BUILD_LOG, length, buffer, NULL);
			buffer[length] ='\0';
			fprintf(stderr,"Error: Failed to build program executable!\n");
			printf("%s\n", buffer);
			free(buffer);
			exit(EXIT_FAILURE);
		}

#ifdef PROFILING
		gettimeofday(&endt, NULL);
		timer_kernel_builder.tv_usec += (endt.tv_sec*1000000+endt.tv_usec) - (startt.tv_sec*1000000+startt.tv_usec);
#endif

		total_alignments = 0;



		for (int i=0; i < superBlocks.y; i++) {
#ifdef PROFILING
			gettimeofday(&startt, NULL);
#endif

			error_check = clEnqueueWriteBuffer(queue, d_targets, CL_TRUE, 0, sizeof(char) * Y*NUMBER_TARGETS, h_targets+(i*NUMBER_TARGETS*Y), 0, NULL, NULL);
			clFinish(queue);

#ifdef PROFILING
			gettimeofday(&endt, NULL);
			timer_H2D.tv_usec += (endt.tv_sec*1000000+endt.tv_usec) - (startt.tv_sec*1000000+startt.tv_usec);
#endif

			if(error_check != CL_SUCCESS) {
				fprintf(stderr,"Failed to write target data to buffer");
				exit(EXIT_FAILURE);
			}

			for (int j=0;j<superBlocks.x;j++) {
				// copy sequences to the device:
#ifdef PROFILING
				gettimeofday(&startt, NULL);
#endif

				error_check = clEnqueueWriteBuffer(queue, d_sequences, CL_TRUE, 0, sizeof(char)*X*NUMBER_SEQUENCES, h_sequences+(j*NUMBER_SEQUENCES*X), 0, NULL, NULL);
				clFinish(queue);

#ifdef PROFILING
				gettimeofday(&endt, NULL);
				timer_H2D.tv_usec += (endt.tv_sec*1000000+endt.tv_usec) - (startt.tv_sec*1000000+startt.tv_usec);
#endif
				if(error_check != CL_SUCCESS) {
					fprintf(stderr,"Failed to write sequence data to buffer");
					exit(EXIT_FAILURE);
				}


				// make sure database-type index is reset:
#ifdef PROFILING
				gettimeofday(&startt, NULL);
#endif

				initZeroCopy(&d_indexIncrement, context, queue, error_check);

#ifdef PROFILING
				gettimeofday(&endt, NULL);
				timer_H2D.tv_usec += (endt.tv_sec*1000000+endt.tv_usec) - (startt.tv_sec*1000000+startt.tv_usec);
#endif
				// fill the scorings matrix:
				kernel_calculateScore = clCreateKernel(program, "calculateScore", &error_check);
				if(error_check != CL_SUCCESS) {
					fprintf(stderr,"Failed to create calculate score kernel");
					exit(EXIT_FAILURE);
				}


#ifdef PROFILING
				gettimeofday(&startt, NULL);
#endif

				calculateScoreHost(d_matrix, d_sequences, d_targets, d_globalMaxima, d_globalDirection, d_scoringsMatrix, kernel_calculateScore, queue, error_check);

#ifdef PROFILING
				gettimeofday(&endt, NULL);
				timer_kernel1.tv_usec += (endt.tv_sec*1000000+endt.tv_usec) - (startt.tv_sec*1000000+startt.tv_usec);

#endif


				// create tracebacks and copy information through zero copy to the host:
				kernel_traceback = clCreateKernel(program, "traceback", &error_check);
				if(error_check != CL_SUCCESS) {
					fprintf(stderr,"Failed to create calculate traceback kernel");
					exit(EXIT_FAILURE);
				}

#ifdef PROFILING
				gettimeofday(&startt, NULL);
#endif

#ifdef GLOBAL_MEM4
				tracebackHost(d_matrix, d_globalMaxima, d_globalDirection, d_indexIncrement, d_startingPointsZeroCopy,
						d_maxPossibleScoreZeroCopy, j*NUMBER_SEQUENCES, kernel_traceback, queue, error_check, d_semaphore);
#else
				tracebackHost(d_matrix, d_globalMaxima, d_globalDirection, d_indexIncrement, d_startingPointsZeroCopy,
										d_maxPossibleScoreZeroCopy, j*NUMBER_SEQUENCES, kernel_traceback, queue, error_check);
#endif

#ifdef PROFILING
				gettimeofday(&endt, NULL);
				timer_kernel2.tv_usec += (endt.tv_sec*1000000+endt.tv_usec) - (startt.tv_sec*1000000+startt.tv_usec);

#endif				// get number of alignments:
				unsigned int index[1];

#ifdef PROFILING
				gettimeofday(&startt, NULL);
#endif
				error_check = clEnqueueReadBuffer(queue, d_indexIncrement, CL_TRUE, 0, sizeof(unsigned int), index, 0, NULL, NULL);
				clFinish(queue);
#ifdef PROFILING
				gettimeofday(&endt, NULL);
				timer_D2H.tv_usec += (endt.tv_sec*1000000+endt.tv_usec) - (startt.tv_sec*1000000+startt.tv_usec);
#endif
				if(error_check != CL_SUCCESS) {
					fprintf(stderr,"Failed to read from d_indexIncrement");
					exit(EXIT_FAILURE);
				}

				fprintf(stderr, "Number of alignments: %d @ %d in iteration: %d\n", *index, j, iter);				// plot the alignments:

				total_alignments+=index[0];

#ifdef PROFILING
				gettimeofday(&startt, NULL);
#endif

				/**Reading from zeroCopy Buffer**/
#if defined(NO_ZERO_COPY) || defined(NVIDIA_ZERO_COPY)
				error_check = clEnqueueReadBuffer(queue, d_globalDirection, CL_TRUE, 0, sizeof(GlobalDirection), h_globalDirectionZeroCopy, 0, NULL, NULL);
#endif

#ifdef INTEL_ZERO_COPY
				h_globalDirectionZeroCopy = (GlobalDirection *)clEnqueueMapBuffer(queue, d_globalDirection, CL_TRUE, CL_MAP_READ, 0, sizeof(GlobalDirection), 0, NULL, NULL, &error_check);
#endif
				clFinish(queue);

#ifdef PROFILING
				gettimeofday(&endt, NULL);
				timer_D2H.tv_usec += (endt.tv_sec*1000000+endt.tv_usec) - (startt.tv_sec*1000000+startt.tv_usec);
#endif
				if(error_check != CL_SUCCESS) {
					fprintf(stderr,"Failed to read from d_globalDirectionZeroCopy");
					exit(EXIT_FAILURE);
				}


#ifdef PROFILING
				gettimeofday(&startt, NULL);
#endif

				/**Reading from zeroCopy Buffer**/
	#if defined(NO_ZERO_COPY) || defined(NVIDIA_ZERO_COPY)
				error_check = clEnqueueReadBuffer(queue, d_startingPointsZeroCopy, CL_TRUE, 0, sizeof(StartingPoints), h_startingPointsZeroCopy, 0, NULL, NULL);
	#endif
	#ifdef INTEL_ZERO_COPY
				h_startingPointsZeroCopy = (StartingPoints *)clEnqueueMapBuffer(queue, d_startingPointsZeroCopy, CL_TRUE, CL_MAP_READ, 0, sizeof(StartingPoints), 0, NULL, NULL, &error_check);
	#endif
				clFinish(queue);

#ifdef PROFILING
				gettimeofday(&endt, NULL);
				timer_D2H.tv_usec += (endt.tv_sec*1000000+endt.tv_usec) - (startt.tv_sec*1000000+startt.tv_usec);
#endif

				if(error_check != CL_SUCCESS) {
					fprintf(stderr,"Failed to read from d_startingPointsZeroCopy");
					exit(EXIT_FAILURE);
				}

#ifdef PROFILING
				gettimeofday(&startt, NULL);
#endif

				plotAlignments(h_sequences, h_targets, h_globalDirectionZeroCopy, *index, h_startingPointsZeroCopy, j*NUMBER_SEQUENCES, i*NUMBER_TARGETS, descSequences, descTargets);

#ifdef PROFILING
				gettimeofday(&endt, NULL);
				timer_plotAlignments.tv_usec += (endt.tv_sec*1000000+endt.tv_usec) - (startt.tv_sec*1000000+startt.tv_usec);
#endif
			}


		}
#ifdef PROFILING
		gettimeofday(&startt, NULL);
#endif
		/** Unique to NVIDIA Zero-Copy **/
	#ifdef NVIDIA_ZERO_COPY
		clEnqueueUnmapMemObject(queue, pinned_maxPossibleScoreZeroCopy, (void*)h_maxPossibleScoreZeroCopy, 0, NULL, NULL);
		clEnqueueUnmapMemObject(queue, pinned_startingPointsZeroCopy, (void*)h_startingPointsZeroCopy, 0, NULL, NULL);
		clEnqueueUnmapMemObject(queue, pinned_globalDirectionZeroCopy, (void*)h_globalDirectionZeroCopy, 0, NULL, NULL);
		clFinish(queue);

		error_check = 0;
		error_check = clReleaseMemObject(d_sequences);
		error_check |= clReleaseMemObject(d_targets);
		error_check |= clReleaseMemObject(d_matrix);
		error_check |= clReleaseMemObject(d_globalMaxima);
		error_check |= clReleaseMemObject(d_globalDirection);
		error_check |= clReleaseMemObject(pinned_startingPointsZeroCopy);
		error_check |= clReleaseMemObject(d_startingPointsZeroCopy);
		error_check |= clReleaseMemObject(pinned_maxPossibleScoreZeroCopy);
		error_check |= clReleaseMemObject(d_maxPossibleScoreZeroCopy);
		error_check |= clReleaseMemObject(pinned_globalDirectionZeroCopy);
		error_check |= clReleaseMemObject(d_scoringsMatrix);
		error_check |= clReleaseMemObject(d_indexIncrement);

		if(error_check!= CL_SUCCESS) {
			fprintf(stderr, "Releasing memory objects has failed in NVIDIA zero copy\n");
			exit(EXIT_FAILURE);
		}

		free(h_sequences);
		free(h_targets);
		free(h_matrix);
		free(h_globalMaxima);
	#endif

		/** Unique to NO Zero-Copy **/
	#ifdef NO_ZERO_COPY
		error_check = 0;
		error_check |= clReleaseMemObject(d_sequences);
		error_check |= clReleaseMemObject(d_targets);
		error_check |= clReleaseMemObject(d_matrix);
		error_check |= clReleaseMemObject(d_globalMaxima);
		error_check |= clReleaseMemObject(d_globalDirection);
		error_check |= clReleaseMemObject(d_startingPointsZeroCopy);
		error_check |= clReleaseMemObject(d_maxPossibleScoreZeroCopy);
		error_check |= clReleaseMemObject(d_scoringsMatrix);
		error_check |= clReleaseMemObject(d_indexIncrement);

		if(error_check!= CL_SUCCESS) {
			fprintf(stderr, "Releasing memory objects has failed in no zero copy\n");
			exit(EXIT_FAILURE);
		}

		free(h_sequences);
		free(h_targets);
		free(h_matrix);
		free(h_globalMaxima);
		free(h_globalDirectionZeroCopy);
		free(h_startingPointsZeroCopy);
		free(h_maxPossibleScoreZeroCopy);
	#endif


		/** Unique to Intel Zero-Copy **/
	#ifdef INTEL_ZERO_COPY
		error_check = 0;
		error_check |= clReleaseMemObject(d_sequences);
		error_check |= clReleaseMemObject(d_targets);
		error_check |= clReleaseMemObject(d_matrix);
		error_check |= clReleaseMemObject(d_globalMaxima);
		error_check |= clReleaseMemObject(d_globalDirection);
		error_check |= clReleaseMemObject(d_startingPointsZeroCopy);
		error_check |= clReleaseMemObject(d_maxPossibleScoreZeroCopy);
		error_check |= clReleaseMemObject(d_scoringsMatrix);
		error_check |= clReleaseMemObject(d_indexIncrement);

		if(error_check!= CL_SUCCESS) {
			fprintf(stderr, "Releasing memory objects has failed in INTEL zero copy\n");
			exit(EXIT_FAILURE);
		}
		free(h_sequences);
		free(h_targets);
		free(h_matrix);
		free(h_globalMaxima);
		free(startingPointsData);
		free(maxPossibleScoreData);
		free(globalDirectionData);

	#endif

#ifdef GLOBAL_MEM4
		free(h_semaphore);
		clReleaseMemObject(d_semaphore);
#endif


		free(descTargets);
		free(descSequences);

		error_check = 0;
		error_check = clReleaseKernel(kernel_calculateScore);
		error_check |= clReleaseKernel(kernel_traceback);
		if(error_check!= CL_SUCCESS) {
			fprintf(stderr, "Could not release kernels\n");
			exit(EXIT_FAILURE);
		}
		error_check = clReleaseProgram(program);
		if(error_check!= CL_SUCCESS) {
			fprintf(stderr, "Could not release program\n");
			exit(EXIT_FAILURE);
		}
		error_check = clReleaseCommandQueue(queue);
		if(error_check!= CL_SUCCESS) {
			fprintf(stderr, "Could not release CommandQueue\n");
			exit(EXIT_FAILURE);
		}
		error_check = clReleaseContext(context);
		if(error_check!= CL_SUCCESS) {
			fprintf(stderr, "Could not release context\n");
			exit(EXIT_FAILURE);
		}




#ifdef PROFILING
		gettimeofday(&endt, NULL);
		timer_mm.tv_usec += (endt.tv_sec*1000000+endt.tv_usec) - (startt.tv_sec*1000000+startt.tv_usec);
#endif

#ifdef PROFILING
		gettimeofday(&endtiter, NULL);
		timer_iter_total.tv_usec += (endtiter.tv_sec*1000000+endtiter.tv_usec) - (starttiter.tv_sec*1000000+starttiter.tv_usec);
#endif

#ifdef PROFILING
		timer_iter_total_array[iter] = timer_iter_total;
		timer_mm_array[iter] = timer_mm;
		timer_init_array[iter] = timer_init;
		timer_H2D_array[iter] = timer_H2D;
		timer_D2H_array[iter] = timer_D2H;
		timer_kernel1_array[iter] = timer_kernel1;
		timer_kernel2_array[iter] = timer_kernel2;
		timer_plotAlignments_array[iter] = timer_plotAlignments;
		timer_kernel_builder_array[iter] = timer_kernel_builder;
#endif

	}

#ifdef PROFILING
	gettimeofday(&endttotal, NULL);
	timer_total.tv_usec += (endttotal.tv_sec*1000000+endttotal.tv_usec) - (startttotal.tv_sec*1000000+startttotal.tv_usec);
#endif

#ifdef PROFILING
	if(exists(argv[9])) {
		FILE *timeFile;
		timeFile = fopen(argv[9],"a+");
		if(timeFile) {
			for(iter=0; iter<ITERATIONS; iter++) {
				//fprintf(timeFile, "%d\t", iter);
				fprintf(timeFile, "%d\t",sequence_index_start);
				fprintf(timeFile, "%d\t",sequence_index_end);
				fprintf(timeFile, "%d\t",total_number_sequences);
				fprintf(timeFile, "%d\t", X);
				fprintf(timeFile, "%d\t", NUMBER_SEQUENCES);
				fprintf(timeFile, "%s\t", argv[3]);
				fprintf(timeFile, "%d\t",target_index_start);
				fprintf(timeFile, "%d\t",target_index_end);
				fprintf(timeFile, "%d\t",total_number_targets);
				fprintf(timeFile, "%d\t", Y);
				fprintf(timeFile, "%d\t", NUMBER_TARGETS);
				fprintf(timeFile, "%s\t", argv[4]);
				fprintf(timeFile, "%ld.%06ld\t",timer_mm_array[iter].tv_usec/1000000, timer_mm_array[iter].tv_usec%1000000);
				fprintf(timeFile, "%ld.%06ld\t",timer_init_array[iter].tv_usec/1000000, timer_init_array[iter].tv_usec%1000000);
				fprintf(timeFile, "%ld.%06ld\t",timer_H2D_array[iter].tv_usec/1000000, timer_H2D_array[iter].tv_usec%1000000);
				fprintf(timeFile, "%ld.%06ld\t",timer_D2H_array[iter].tv_usec/1000000, timer_D2H_array[iter].tv_usec%1000000);
				fprintf(timeFile, "%ld.%06ld\t",timer_kernel1_array[iter].tv_usec/1000000, timer_kernel1_array[iter].tv_usec%1000000);
				fprintf(timeFile, "%ld.%06ld\t",timer_kernel2_array[iter].tv_usec/1000000, timer_kernel2_array[iter].tv_usec%1000000);
				fprintf(timeFile, "%ld.%06ld\t",timer_plotAlignments_array[iter].tv_usec/1000000, timer_plotAlignments_array[iter].tv_usec%1000000);
				fprintf(timeFile, "%ld.%06ld\t",timer_iter_total_array[iter].tv_usec/1000000, timer_iter_total_array[iter].tv_usec%1000000);
				fprintf(timeFile, "%d\n",total_alignments);

			}
			fprintf(timeFile, "#Total execution time: %ld.%06ld\n",timer_total.tv_usec/1000000, timer_total.tv_usec%1000000);
			fprintf(timeFile, "\n");
			fclose(timeFile);

		} else{
			fprintf(stderr,"Cannot append to file: %s\n", argv[9]);
			exit(EXIT_FAILURE);
		}

	} else {
		FILE *timeFile;
		timeFile = fopen(argv[9],"a+");
		if(timeFile) {
			fprintf(timeFile, "#Start_s\tEnd_s\tTotal_s\tX\tDevice_s\tSuperblock_s\tStart_t\tEnd_t\tTotal_t\tY\tDevice_t\tSuperblock_t\tMem_Management\tInit\tH2D\tD2H\tkernel1\tkernel2\tPlotAlignments\titer_total\tAlignments\n");
			for(iter=0; iter<ITERATIONS; iter++) {
				fprintf(timeFile, "%d\t",sequence_index_start);
				fprintf(timeFile, "%d\t",sequence_index_end);
				fprintf(timeFile, "%d\t",total_number_sequences);
				fprintf(timeFile, "%d\t", X);
				fprintf(timeFile, "%d\t", NUMBER_SEQUENCES);
				fprintf(timeFile, "%s\t", argv[3]);
				fprintf(timeFile, "%d\t",target_index_start);
				fprintf(timeFile, "%d\t",target_index_end);
				fprintf(timeFile, "%d\t",total_number_targets);
				fprintf(timeFile, "%d\t", Y);
				fprintf(timeFile, "%d\t", NUMBER_TARGETS);
				fprintf(timeFile, "%s\t", argv[4]);
				fprintf(timeFile, "%ld.%06ld\t",timer_mm_array[iter].tv_usec/1000000, timer_mm_array[iter].tv_usec%1000000);
				fprintf(timeFile, "%ld.%06ld\t",timer_init_array[iter].tv_usec/1000000, timer_init_array[iter].tv_usec%1000000);
				fprintf(timeFile, "%ld.%06ld\t",timer_H2D_array[iter].tv_usec/1000000, timer_H2D_array[iter].tv_usec%1000000);
				fprintf(timeFile, "%ld.%06ld\t",timer_D2H_array[iter].tv_usec/1000000, timer_D2H_array[iter].tv_usec%1000000);
				fprintf(timeFile, "%ld.%06ld\t",timer_kernel1_array[iter].tv_usec/1000000, timer_kernel1_array[iter].tv_usec%1000000);
				fprintf(timeFile, "%ld.%06ld\t",timer_kernel2_array[iter].tv_usec/1000000, timer_kernel2_array[iter].tv_usec%1000000);
				fprintf(timeFile, "%ld.%06ld\t",timer_plotAlignments_array[iter].tv_usec/1000000, timer_plotAlignments_array[iter].tv_usec%1000000);
				fprintf(timeFile, "%ld.%06ld\t",timer_iter_total_array[iter].tv_usec/1000000, timer_iter_total_array[iter].tv_usec%1000000);
				fprintf(timeFile, "%d\n",total_alignments);

			}
			fprintf(timeFile, "#Total execution time: %ld.%06ld\n",timer_total.tv_usec/1000000, timer_total.tv_usec%1000000);
			fprintf(timeFile, "\n");
			fclose(timeFile);

		} else{
			fprintf(stderr,"Cannot create file: %s\n", argv[9]);
			exit(EXIT_FAILURE);
		}

	}
#endif

	return(0);
}



