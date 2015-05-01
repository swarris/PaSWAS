#include <zlib.h>
#include <stdio.h>
#include "kseq.h"
#include <math.h>

 #define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

typedef struct {
	unsigned int sequence;
	unsigned int target;
	unsigned int blockX;
	unsigned int blockY;
	unsigned int valueX;
	unsigned int valueY;
	float score;
  float maxScore;
  float posScore;
} StartingPoint;

//Structure which presents the data for input to the Smith-Waterman algorithm
typedef struct {
	int superblocks_x;
	int superblocks_y;
	int num_sequences;
	int num_targets;
	int X;
	int Y;
} Config;

//An intermediary structure for representing a single run
typedef struct {
	int i;
	int j;
	int alignments;
	int parallel;
} Candidate;


void lengths(char* filename,int *lengths, int size, int block);
float mem_usage(int conf_seq, int conf_target, int sequence_length, int target_length,
		int shared_x, int shared_y, int zero_copy, int global_mem, int superblock_x);
Config parameters(int sequence_length, int target_length, int block_x, int block_y, float mem, int nvidia_zero_copy, int global_mem, int number_sequences, int number_targets);
Candidate findMax(int *parallel, int *alignments, int n, int m, int k, int l, float threshold);
Candidate findSubMax(int *parallel, int *alignments, int start_index_i, int end_index_i, int start_index_j, int end_index_j, int m, float threshold);
void printConfig(Config c);
void update(int *alignments, int *parallel, int i, int j, int n, int m);



// STEP 1: declare the type of file handler and the read() function
KSEQ_INIT(gzFile, gzread)


/** Assumption: input file should be sorted from largest to smallest**/
int main(int argc, char *argv[])
{
	if (argc != 11) {
		fprintf(stderr, "Usage: <sequence.fasta> <target.fasta> <#Sequences in sequence.fasta> <#Targets in target.fasta> <Dimension of block x> <Dimension of block y> <Amount of RAM in (mb)> <nvidia_zero_copy> <global_mem> <threshold>\n");
		return 1;
	}

	char *sequences_file = argv[1];
	if(!sequences_file) {
		fprintf(stderr, "Please provide a sequence file\n");
		return 1;
	}

	char *targets_file = argv[2];
	if(!targets_file) {
		fprintf(stderr, "Please provide a target file\n");
		return 1;
	}

	int n = atoi(argv[3]);
	if(n <= 0) {
		fprintf(stderr, "Please provide an positive integer value for number_sequences\n");
		return 1;
	}
	int m = atoi(argv[4]);
	if(m <= 0) {
		fprintf(stderr, "Please provide an positive integer value for number_targets\n");
		return 1;
	}
	int block_x = atoi(argv[5]);
	if(block_x <= 0) {
		fprintf(stderr, "Please provide an positive integer value for blockx\n");
		return 1;
	}
	int block_y = atoi(argv[6]);
	if(block_y <= 0) {
		fprintf(stderr, "Please provide an positive integer value for blocky\n");
		return 1;
	}
	float memory =  atof(argv[7]);
	if(memory <= 0) {
		fprintf(stderr, "Please provide an positive float value for memory\n");
		return 1;
	}

	int zero_copy =  atoi(argv[8]);
	if(zero_copy < 0) {
		fprintf(stderr, "Please provide a valid mode for zero_copy:%d\n",zero_copy);
		return 1;
	}

	int global_mem =  atoi(argv[9]);
	if(global_mem < 0) {
		fprintf(stderr, "Please provide a valid mode for global_mem:%d\n",global_mem);
		return 1;
	}

	float threshold =  atof(argv[10]);
	if(threshold < 0) {
		fprintf(stderr, "Please provide a valid threshold value:%f\n",threshold);
		return 1;
	}

	int *sequence_lengths = (int*)malloc(sizeof(int)*n);
	if(!sequence_lengths) {
		fprintf(stderr, "Could not allocate memory for the sequence lengths\n");
		return 1;
	}

	int *target_lengths = (int*)malloc(sizeof(int)*m);
	if(!target_lengths) {
		fprintf(stderr, "Could not allocate memory for the target lengths\n");
		return 1;
	}

	//Matrix containing how much parallelism can be achieved for an entry (i,j)
	int *parallel = (int*)malloc(sizeof(int)*n*m);
	if(!parallel) {
		fprintf(stderr, "Could not allocate memory for the parallel matrix\n");
		return 1;
	}

	//Matrix containing how many alignments can be conducted for an entry (i,j)
	int *alignments = (int*)malloc(sizeof(int)*n*m);
	if(!alignments) {
		fprintf(stderr, "Could not allocate memory for the alignment matrix\n");
		return 1;
	}

	//Array which stores information about which alignments already have been conducted
	int *progress = (int*)malloc(sizeof(int)*n);
	if(!progress) {
		fprintf(stderr, "Could not allocate memory for the progress array\n");
		return 1;
	}

	//The size of this list indicates how many runs we need to evaluate a dataset
	Candidate *iterations = (Candidate*)malloc(sizeof(Candidate)*n*m);
	if(!iterations) {
		fprintf(stderr, "Could not allocate memory for the iterations matrix\n");
		return 1;
	}

	//Total amount of alignments to conduct
	int universe = n*m;

	lengths(sequences_file, sequence_lengths, n, block_x);
	lengths(targets_file, target_lengths, m, block_y);

	int i,j,sequences,targets, num_alignments, num_parallel, sequence_length, target_length;
	float mem_single_alignment;

	//Initialization of datastructures
	for(i=0; i<n; i++) {

		//Initially not a single alignment has been conducted
		progress[i] = m;

		sequences = n - i;
		sequence_length = sequence_lengths[i];
		for(j=0; j<m; j++) {
			targets = m - j;
			target_length = target_lengths[j];
			alignments[i*m+j] = sequences * targets;
			mem_single_alignment = mem_usage(1, 1, sequence_length, target_length, block_x, block_y, zero_copy, global_mem, 1);
			num_alignments = (int)floor(memory/mem_single_alignment);
			parallel[i*m+j] = num_alignments > alignments[i*m+j]?alignments[i*m+j]:num_alignments;
		}
	}

	int runs=0, k=0, l=0;
	while (universe > 0) {

		Candidate t = findMax(parallel,alignments,n,m,k,l,threshold);
		k = t.i;
		l = t.j;

		iterations[runs] = t;

		num_alignments = alignments[k*m+l];
		num_parallel = parallel[k*m+l];

		update(alignments, parallel, k, l, n, m);
		universe = universe - num_alignments;
		runs++;
	}

	int curr_i = n;
	int curr_j = m;
	for(i=0; i<runs; i++) {
		Candidate t = iterations[i];
		int start_sequence = t.i;
		int start_target = t.j;

		int end_sequence = start_sequence;

		int end_target = progress[t.i];

		//determine whether we already have done more alignments for a smaller sequence
		while(start_target<progress[end_sequence] && end_sequence<n) {
			progress[end_sequence] = start_target;
			end_sequence++;
		}

		//Load targets from start_target until (not including) end_target, the same for the sequences
		printf("%d\t%d\t%d\t%d\t",start_sequence,end_sequence,start_target,end_target);


		Config c = parameters(sequence_lengths[t.i], target_lengths[t.j], block_x, block_y, memory, zero_copy, global_mem, end_sequence-start_sequence, end_target-start_target);
		printConfig(c);

	}

	//Variables presented to the script are:
	//<Start_index_sequence>	<End_index_sequence>	<Start_index_target>	<End_index_target> <Sequences in parallel>	<Targets in parallel>	<Superblock_x>	<Superblock_y>	<X>	<Y>

	free(sequence_lengths);
	free(target_lengths);
	free(parallel);
	free(alignments);
	free(progress);
	free(iterations);
	return 0;
}

Candidate findMax(int *parallel, int *alignments, int n, int m, int k, int l, float threshold){
	int i,j,max=0, alignment=0;
	Candidate c, c1, c2;

	//k=0 and l=0 special case

	if (k==0 && l==0) {
		/**for(i=0; i<n; i++) {
			for(j=0; j<m; j++) {

			}
		}**/
		c = findSubMax(parallel, alignments, 0, n, 0, m, m, threshold);
	} else {
		/**for(i=0; i<k; i++) {
			for(j=l; j<m; j++) {

			}
		}**/
		c1 = findSubMax(parallel, alignments, 0, k, l, m, m, threshold);
		/**for(i=k; i<n; i++) {
			for(j=0; j<l; j++) {

			}
		}**/
		c2 = findSubMax(parallel, alignments, k, n, 0, l, m, threshold);
		if(c1.parallel>c2.parallel) {
			c = c1;
		}else if(c1.parallel==c2.parallel && c1.alignments > c2.alignments) { //tie breaker
			c = c1;
		} else {
			c = c2;
		}
	}

	return c;
}

Candidate findSubMax(int *parallel, int *alignments, int start_index_i, int end_index_i, int start_index_j, int end_index_j, int m, float threshold){
	int i,j,max=0, alignment=0, threshold_max=0;
	//IDEA BEHIND THIS ALGORITHM THE FARTHER YOU GO INTO THE MATRIX THE MORE PARALLELISM IT NEEDS TO HAVE
	float increase = threshold + 1.0;
	Candidate t;
	for(i=start_index_i; i<end_index_i; i++) {
		for(j=start_index_j; j<end_index_j; j++) {
			threshold_max = (int) ceil(increase*max);
			if(parallel[i*m+j]>=threshold_max) {
				if(parallel[i*m+j]==threshold_max && alignments[i*m+j] > alignment) { //Tie breaker
						t.i = i;
						t.j = j;
						max = parallel[i*m+j];
						alignment = alignments[i*m+j];
				} else if (parallel[i*m+j] > threshold_max) {
					t.i = i;
					t.j = j;
					max = parallel[i*m+j];
					alignment = alignments[i*m+j];
				}
			}

		}
	}
	t.alignments = alignment;
	t.parallel = max;
	return t;
}

void update(int *alignments, int *parallel, int i, int j, int n, int m) {
	int k,l, num_alignments;

	for(k=0; k<n; k++) {
		for(l=0; l<m; l++) {
			/** Determine intersection **/
			int index_i = max(k,i);
			int index_j = max(l,j);
			alignments[k*m+l] = alignments[k*m+l] - alignments[index_i*m+index_j];
			parallel[k*m+l] = parallel[k*m+l] > alignments[k*m+l]?alignments[k*m+l]:parallel[k*m+l];
		}
	}
}



float mem_usage(int conf_seq, int conf_target, int sequence_length, int target_length,
		int shared_x, int shared_y, int zero_copy, int global_mem, int superblock_x){
	float GlobalMatrix;
	float GlobalMaxima;
	float GlobalDirection;
	float StartingPoints;
	float maxPossibleScore;
	float total=0;

	if(global_mem) {
		GlobalMaxima = (1.0/1024)*(1.0/1024)*conf_seq*conf_target*sequence_length*target_length*sizeof(float);
		GlobalMatrix = (1.0/1024)*(1.0/1024)*conf_seq*conf_target*(sequence_length/shared_x)*(target_length/shared_y)*(shared_x+1)*(shared_y+1)*sizeof(float);
	} else {
		GlobalMaxima = (1.0/1024)*(1.0/1024)*conf_seq*conf_target*(sequence_length/shared_x)*(target_length/shared_y)*sizeof(float);
		GlobalMatrix = (1.0/1024)*(1.0/1024)*conf_seq*conf_target*sequence_length*target_length*sizeof(float);
	}

	GlobalDirection = (1.0/1024)*(1.0/1024)*conf_seq*conf_target*sequence_length*target_length*sizeof(char);
	StartingPoints = (1.0/1024)*(1.0/1024)*conf_seq*conf_target * 1000 * sizeof(StartingPoint);
	maxPossibleScore = (1.0/1024)*(1.0/1024)*conf_seq*superblock_x*sizeof(float);

	//total = GlobalMaxima + GlobalMatrix + 2*GlobalDirection + StartingPoints + maxPossibleScore;
	total = GlobalMaxima + GlobalMatrix + GlobalDirection + StartingPoints + maxPossibleScore;

	if(zero_copy) {
		total += GlobalDirection + StartingPoints + maxPossibleScore;
		//total += StartingPoints + maxPossibleScore;
	}
	//return floor(total);
	return total;
}

void lengths(char* filename,int *lengths, int size, int block) {
	gzFile fp;
	kseq_t *seq;
	int length, i=0;
	fp = gzopen(filename, "r"); // STEP 2: open the file handler
	seq = kseq_init(fp); // STEP 3: initialize seq

	while ((length = kseq_read(seq)) >= 0 && i<size) { // STEP 4: read sequence
		//int max_length = seq->seq.l;
		int remainder = length%block;
		if(remainder) {
			//Sequence needs to be padded
			int pad_length = block - remainder;
			length+=pad_length;
		}
		lengths[i]=length;
		i++;
	}
	kseq_destroy(seq); // STEP 5: destroy seq
	gzclose(fp); // STEP 6: close the file handler
}

Config parameters(int sequence_length, int target_length, int block_x, int block_y, float mem, int nvidia_zero_copy, int global_mem, int number_sequences, int number_targets) {

	int i,j;
	int s_x = 0;
	int s_y = 0;
	int sequence = 0;
	int target = 0;
	//int iterations = number_sequences * number_targets;
	int iterations = (number_sequences * number_targets) + 1;
	float mem_needed = 0;

	int superblock_x = 0;
	int superblock_y = 0;

	for(i=1; i<=number_sequences; i++) {
		superblock_x = (int) ceil((double)number_sequences/i);
		for(j=1; j<=number_targets; j++) {
			mem_needed = mem_usage(i,j,sequence_length,target_length,block_x,block_y,nvidia_zero_copy,global_mem,superblock_x);
			if(mem_needed >= 0 && mem >= mem_needed) {
				//valid configuration
				superblock_y = (int) ceil((double)number_targets/j);
				if(superblock_x*superblock_y<iterations) {
					sequence = i;
					target = j;
					s_x = superblock_x;
					s_y = superblock_y;
					iterations = s_x*s_y;
				}
			}
		}
	}
	Config conf;
	conf.superblocks_x = s_x;
	conf.superblocks_y = s_y;
	conf.num_sequences = sequence;
	conf.num_targets = target;
	conf.X=sequence_length;
	conf.Y=target_length;
	return conf;

}



void printConfig(Config c) {
	//Sequences	Targets	Superblock_x	Superblock_y	X	Y
	printf("%d\t%d\t%d\t%d\t%d\t%d\n",c.num_sequences,c.num_targets,c.superblocks_x,c.superblocks_y, c.X, c.Y);
}

