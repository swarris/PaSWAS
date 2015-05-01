#include <zlib.h>
#include <stdio.h>
#include "kseq.h"
#include <math.h>

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

typedef struct {
	int sequence_index;
	int target_index;
	int superblocks_x;
	int superblocks_y;
	int num_sequences;
	int num_targets;
	int X;
	int Y;
} Config;

void lengths(char* filename,int *lengths, int size, int block);
float mem_usage(int conf_seq, int conf_target, int sequence_length, int target_length,
		int shared_x, int shared_y, int zero_copy, int global_mem, int superblock_x);
Config parameters(int sequence_length, int target_length, int block_x, int block_y, float mem, int nvidia_zero_copy, int global_mem, int number_sequences, int number_targets);

// STEP 1: declare the type of file handler and the read() function
KSEQ_INIT(gzFile, gzread)


/** Assumption: input file should be sorted from largest to smallest**/
int main(int argc, char *argv[])
{
	if (argc != 10) {
		fprintf(stderr, "Usage: <sequence.fasta> <target.fasta> <#Sequences in sequence.fasta> <#Targets in target.fasta> <Dimension of block x> <Dimension of block y> <Amount of RAM in (mb)> <nvidia_zero_copy> <global_mem>\n");
		return 1;
	}

	char *sequences_file = argv[1];
	char *targets_file = argv[2];
	int number_sequences = atoi(argv[3]);
	if(number_sequences <= 0) {
		fprintf(stderr, "Please provide an positive integer value for number_sequences\n");
		return 1;
	}
	int number_targets = atoi(argv[4]);
	if(number_targets <= 0) {
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

	int *sequence_lengths = (int*)malloc(sizeof(int));
	int *target_lengths = (int*)malloc(sizeof(int));

	lengths(sequences_file, sequence_lengths, 1, block_x);
	lengths(targets_file, target_lengths, 1, block_y);

	Config c = parameters(sequence_lengths[0], target_lengths[0], block_x, block_y, memory, zero_copy, global_mem, number_sequences,number_targets);

	//<Start_index_sequence>	<End_index_sequence>	<Start_index_target>	<End_index_target> <Sequences in parallel>	<Targets in parallel>	<Superblock_x>	<Superblock_y>	<X>	<Y>
	printf("0\t%d\t0\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",number_sequences,number_targets,c.num_sequences,c.num_targets,c.superblocks_x,c.superblocks_y, c.X, c.Y);
	return 0;

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
	//Due to this datastructure, extrapolating from mem_usage(1,1) is difficult
	maxPossibleScore = (1.0/1024)*(1.0/1024)*conf_seq*superblock_x*sizeof(float);

	total = GlobalMaxima + GlobalMatrix + GlobalDirection + StartingPoints + maxPossibleScore;

	if(zero_copy) {
		total += GlobalDirection + StartingPoints + maxPossibleScore;
	}
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
	unsigned int iterations = (number_sequences * number_targets) + 1;
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
	conf.sequence_index = 0;
	conf.target_index = 0;
	conf.superblocks_x = s_x;
	conf.superblocks_y = s_y;
	conf.num_sequences = sequence;
	conf.num_targets = target;
	conf.X=sequence_length;
	conf.Y=target_length;

	return conf;

}

