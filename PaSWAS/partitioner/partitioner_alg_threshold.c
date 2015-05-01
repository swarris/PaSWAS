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

void greedyOnTarget(int total_targets, int *target_lengths, int block_x, int block_y, int zero_copy, int global_mem, float memory,
		Config *runs, int *size, float increase);
void lengths(char* filename,int *lengths, int size, int block);
float mem_usage(int conf_seq, int conf_target, int sequence_length, int target_length,
		int shared_x, int shared_y, int zero_copy, int global_mem, int superblock_x);
Config parameters(int sequence_length, int target_length, int block_x, int block_y, float mem, int nvidia_zero_copy, int global_mem, int number_sequences, int number_targets);
void printConfig(Config c);



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

	int *sequence_lengths = (int*)malloc(sizeof(int)*number_sequences);
	int *target_lengths = (int*)malloc(sizeof(int)*number_targets);

	lengths(sequences_file, sequence_lengths, number_sequences, block_x);
	lengths(targets_file, target_lengths, number_targets, block_y);

	Config *runs = (Config*) malloc(sizeof(Config)*number_sequences*number_targets);

	float threshold =  atof(argv[10]);
	if(threshold < 0) {
		fprintf(stderr, "Please provide a valid threshold value:%f\n",threshold);
		return 1;
	}

	int i=0, j=0, size=-1;
	Config conf;

	float increase = threshold + 1.0; //Each subsequent run should at least have an increase in parallel alignments of threshold+1.0
	int alignment=0;

	//Greedy on sequence
	while(i<number_sequences) {
		//0<=|Sequence_lengths|<number_sequences
		int sequence_length = sequence_lengths[i];
		//fprintf(stderr, "sequence_length:%d\n",sequence_length);
		int target_length = target_lengths[0];
		//fprintf(stderr, "target_length:%d\n",target_length);
		float mem_single_alignment = mem_usage(1, 1, sequence_length, target_length, block_x, block_y, 1, 1, 1);
		//fprintf(stderr, "mem_single_alignment:%f\n",mem_single_alignment);
		int sequences = number_sequences - i;
		//fprintf(stderr, "sequences:%d\n",sequences);
		//a_i: number of alignments we can do with ith sequence and largest target
		int a_i = (int)floor(memory/mem_single_alignment);
		//fprintf(stderr, "a_i:%d\n",a_i);
		//fprintf(stderr,"mem_single_alignment:%f\n",mem_single_alignment);
		if(a_i <= 0) {
			size++;
			alignment = 0;
			Config run;
			run.sequence_index=i;
			run.target_index=0;
			run.num_targets = 0;
			run.num_sequences = 0;
			run.superblocks_x = 0;
			run.superblocks_y = 0;
			run.X = sequence_length;
			run.Y = target_length;
			runs[size]=run;
			i++;
		}else {
			int lower_bound = (int) ceil(increase*alignment);
			if(a_i <= lower_bound){
				int update;
				runs[size].superblocks_x+=1;
				for(update = 0; update<size; update++){
					if(runs[update].sequence_index == runs[size].sequence_index) {
						runs[update].superblocks_x = runs[size].superblocks_x;
					}
				}

				i+=conf.num_sequences;
			} else { /**a_i >= lower_bound**/
				alignment = a_i;
				size++;
				if(alignment < sequences * number_targets) {
					Config run;
					conf = parameters(sequence_length, target_length, block_x, block_y, memory, zero_copy, global_mem,sequences,number_targets);
					run.sequence_index = i;
					run.target_index = 0;
					run.superblocks_x = 1;
					run.superblocks_y = 1;
					run.num_sequences = conf.num_sequences;
					run.num_targets = conf.num_targets;
					run.X = sequence_length;
					run.Y = target_length;

					runs[size] = run;

					//Greedy on target
					greedyOnTarget(number_targets, target_lengths, block_x, block_y, zero_copy, global_mem, memory, runs, &size, increase);
					i+=conf.num_sequences;
				} else {
					Config run;
					run.sequence_index = i;
					run.target_index = 0;
					run.num_targets = number_targets;
					run.num_sequences = sequences;
					run.superblocks_x = 1;
					run.superblocks_y = 1;
					run.X = sequence_length;
					run.Y = target_length;
					runs[size] = run;
					i+=sequences;
				}
			}
		}
	}

	int c;
	for(c=0; c<size+1; c++) {
		printConfig(runs[c]);
	}
	return 0;

}

void greedyOnTarget(int total_targets, int *target_lengths, int block_x, int block_y, int zero_copy, int global_mem, float memory,
		Config *runs, int *size, float increase){

	int j=runs[(*size)].num_targets;
	int superblock_x = runs[(*size)].superblocks_x;
	int start_sequence = runs[(*size)].sequence_index;
	int sequence_parallel = runs[(*size)].num_sequences;
	int sequence_length = runs[(*size)].X;


	int prev_t_alignment = runs[(*size)].num_targets;
	while(j<total_targets) {
		int remaining_targets = total_targets - j;
		int target_length = target_lengths[j];
		float mem_single_target = mem_usage(sequence_parallel, 1, sequence_length, target_length, block_x, block_y, zero_copy, global_mem, superblock_x);
		//Get an upper bound of how many alignments we can conduct on the device(should be at least one)
		int t_alignment = ((int)floor(memory/mem_single_target)) > remaining_targets?remaining_targets:(int)floor(memory/mem_single_target);
		if(t_alignment>0) {
			int threshold = (int) ceil(increase*prev_t_alignment);
			if(t_alignment <= threshold) { //moving to a smaller target did not change the number of alignments that can be conducted on the device
				//Update run
				runs[(*size)].superblocks_y+=1;
				j+=runs[(*size)].num_targets;

			} else {
				//Create new config
				Config c;
				//fprintf(stderr,"size:%d\n", (*size));
				(*size)++;
				c.sequence_index = start_sequence;
				c.superblocks_x = superblock_x;
				c.num_sequences = sequence_parallel;
				c.X = sequence_length;

				c.target_index = j;
				c.superblocks_y = 1;
				c.num_targets = t_alignment;
				c.Y = target_length;
				runs[(*size)]=c;
				prev_t_alignment = t_alignment;
				j+=t_alignment;
			}
		} else {
			//Create new config
			Config c;
			//fprintf(stderr,"size:%d\n", (*size));
			(*size)++;
			c.sequence_index = start_sequence;
			c.superblocks_x = superblock_x;
			c.num_sequences = sequence_parallel;
			c.X = sequence_length;

			c.target_index = j;
			c.superblocks_y = 0;
			c.num_targets = 0;
			c.Y = target_length;
			runs[(*size)]=c;
			j++;
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
	//Due to this datastructure extrapolating from mem_usage(1,1) is difficult
	maxPossibleScore = (1.0/1024)*(1.0/1024)*conf_seq*superblock_x*sizeof(float);

	//fprintf(stderr,"maxPossibleScore:%f\t",maxPossibleScore);

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
	unsigned int iterations = (number_sequences * number_targets) + 1;
	float mem_needed = 0;


	int superblock_x = 0;
	int superblock_y = 0;

	for(i=1; i<=number_sequences; i++) {
		superblock_x = (int) ceil((double)number_sequences/i);
		for(j=1; j<=number_targets; j++) {
			mem_needed = mem_usage(i,j,sequence_length,target_length,block_x,block_y,nvidia_zero_copy,global_mem,superblock_x);
			//fprintf(stderr,"mem_needed:%f\n",mem_needed);
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

void printConfig(Config c) {
	//<Start_index_sequence>	<End_index_sequence>	<Start_index_target>	<End_index_target> <Sequences in parallel>	<Targets in parallel>	<Superblock_x>	<Superblock_y>	<X>	<Y>
	printf("%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",c.sequence_index,(c.num_sequences * c.superblocks_x)+ c.sequence_index,c.target_index,(c.num_targets*c.superblocks_y)+c.target_index,c.num_sequences,c.num_targets,c.superblocks_x,c.superblocks_y, c.X, c.Y);
}

