/*
 * tagDetect.cu
 *
 *  Created on: 20-apr-2010
 *      Author: sven
 */

#include "smithwaterman.h"
#include <ctype.h>

/* argc:
 * 1 - device number >= 1
 * 2 - sequence file
 * 3 - target file
 * 4 - number of super blocks for sequence file (ceil((#seqs in file / #seqs to GPU))
 * 5 - number of super blocks for target file (ceil((#seqs in file / #seqs to GPU))
 */
int main(int argc, char **argv) {

	if (argc != 6) {
		printf("Error: use: paswas device sequenceFile targetFile superBlocksX superBlocksY!\n");
		return 1;
	}

	// select device:
	int device = 0;
	if (argc >= 2 && atoi(argv[1]))
		device = atoi(argv[1])-1;
	// make sure it has CC 1.3 and enable zero-copy:
	cudaError cudaError_t = cudaSetDeviceFlags(cudaDeviceMapHost);
	cutilSafeCall(cudaSetDevice(device));

	// variables needed for the application:
	char *h_sequences = 0, *h_targets = 0;
	char *d_sequences = 0, *d_targets = 0;
	GlobalMatrix *d_matrix = 0;
	GlobalMatrix *h_matrix = 0;
	GlobalMaxima *d_globalMaxima = 0;
	GlobalMaxima *d_internalMaxima = 0;
	GlobalMaxima *h_globalMaxima = 0;
	GlobalDirection *d_globalDirection = 0;
	GlobalDirection *h_globalDirectionZeroCopy = 0;
	GlobalDirection *d_globalDirectionZeroCopy = 0;
	unsigned int *d_indexIncrement = 0;
	StartingPoints *h_startingPointsZeroCopy = 0;
	StartingPoints *d_startingPointsZeroCopy = 0;
	float *h_maxPossibleScoreZeroCopy = 0;
	float *d_maxPossibleScoreZeroCopy = 0;


	dim3 superBlocks(atoi(argv[4]), atoi(argv[5]),0);

	char *descSequences = (char *) malloc(sizeof(char) * superBlocks.x*NUMBER_SEQUENCES*MAX_LINE_LENGTH);
	char *descTargets = (char *) malloc(sizeof(char) * superBlocks.y* NUMBER_TARGETS*MAX_LINE_LENGTH);


	// allocate memory on host & device:
	init(&h_sequences, &h_targets, &d_sequences, &d_targets,
		&h_matrix, &d_matrix,
		&h_globalMaxima, &d_globalMaxima,
		&d_internalMaxima,
		&d_globalDirection,
		&h_globalDirectionZeroCopy,
		&d_globalDirectionZeroCopy,
		&h_startingPointsZeroCopy,
		&d_startingPointsZeroCopy,
		&h_maxPossibleScoreZeroCopy,
		&d_maxPossibleScoreZeroCopy,
		superBlocks,
		device);


	FILE *tagsFile = NULL, *seqFile = NULL;

	tagsFile = fopen(argv[3], "r");
	seqFile = fopen(argv[2], "r");

	if (!tagsFile || !seqFile) {
		printf("Error: could not open tag/seq file!\n");
		return 1;
	}

	int t=0;
	for (; t < superBlocks.y * NUMBER_TARGETS && !feof(tagsFile); t++){
		char * current = h_targets+(t*Y);
		char * currentDesc = descTargets + (t*MAX_LINE_LENGTH);
		fgets(currentDesc, MAX_LINE_LENGTH, tagsFile);
		char *endLine = (index(currentDesc, '\n'));
		if (endLine) *endLine = '\0';
		fgets(current, Y, tagsFile);
		bool inString = true;
		for (int i=0; i < Y; i++) {
			if (inString && (current[i] == '\n' || !current[i]))
				inString = false;
			if (inString) {
				current[i] = toupper(current[i]);
				if (current[i] - 'A' < 0 || current[i] - 'A' > 25) {
				  fprintf(stderr, "Error: wrong character in target file: '%c', desc: %s\n", current[i], current);
				  return 1;
				}
			}
			else
				current[i] = FILL_CHARACTER;
		}
		//fprintf(stderr, "|%s|\n|%s|\n", currentDesc, current);
	}

	fprintf(stderr, "Read number of targets: %d\n", t-1);

	for (; t < superBlocks.y * NUMBER_TARGETS; t++) {
		for (int c=0; c < Y; c++)
			*(h_targets+t*Y+c) = FILL_CHARACTER;
	}


	int s=0;
	for (; (s < superBlocks.x * NUMBER_SEQUENCES) && !feof(seqFile); s++) {
		h_maxPossibleScoreZeroCopy[s] = 0;
		char * current = h_sequences +(s*X);
		char * currentDesc = descSequences + (s*MAX_LINE_LENGTH);
		fgets(currentDesc, MAX_LINE_LENGTH, seqFile);
		char * endOfLine = (index(currentDesc, '\n'));
		if (endOfLine) *endOfLine = '\0';
		fgets(current, X, seqFile);
		//endIndex = strlen(current);
		bool inString = true;

		for (int i=0; i < X; i++) {

			if (inString && (current[i] == '\n' || !current[i]))
				inString = false;
			if (inString) {
				current[i] = toupper(current[i]);
				h_maxPossibleScoreZeroCopy[s] += HIGHEST_SCORE;
				if (current[i] - 'A' < 0 || current[i] - 'A' > 25) {
				  fprintf(stderr, "Error: wrong character in file: '%c', file ref: %i\n", current[i], s);
				  return 1;
				}
			}
			else
			  current[i] = FILL_CHARACTER;
		}
		if (inString && !feof(seqFile)) {
			fprintf(stderr, "Error: read too long!\n");
			fprintf(stderr, "id: %s\n", currentDesc);
			return 1;
		}
		h_maxPossibleScoreZeroCopy[s] *= LOWER_LIMIT_MAX_SCORE;
		//		fprintf(stderr, "|%s|\n|%s|\n", currentDesc, current);
	}

	fprintf(stderr, "Read number of sequences: %d\n", s-1);

	for (; s < superBlocks.x * NUMBER_SEQUENCES; s++) {
	  h_maxPossibleScoreZeroCopy[s] = 0;
		for (int c=0; c < X; c++)
			*(h_sequences+s*X+c) = FILL_CHARACTER;
	}

	fclose(tagsFile);
	fclose(seqFile);

	for (int i=0; i < superBlocks.y; i++) {
		cutilSafeCall(cudaMemcpy(d_targets, h_targets+(i*NUMBER_TARGETS*Y), sizeof(char) * Y*NUMBER_TARGETS, cudaMemcpyHostToDevice));
		fprintf(stderr, "Starting alignments on target block %i\n", i);
		for (int j=0;j<superBlocks.x;j++) {
			// copy sequences to the device:
			cutilSafeCall(cudaMemcpy(d_sequences, h_sequences+(j*NUMBER_SEQUENCES*X), sizeof(char) * X*NUMBER_SEQUENCES, cudaMemcpyHostToDevice));

			// make sure database-type index is reset:
			initZeroCopy(&d_indexIncrement);
			// fill the scorings matrix:
			calculateScoreHost(d_matrix, d_sequences, d_targets, d_globalMaxima, d_internalMaxima, d_globalDirection);
			// create tracebacks and copy information through zero copy to the host:
			tracebackHost(d_matrix, d_globalMaxima, d_internalMaxima, d_globalDirection, d_globalDirectionZeroCopy, d_indexIncrement, d_startingPointsZeroCopy, d_maxPossibleScoreZeroCopy, j*NUMBER_SEQUENCES);
			// get number of alignments:
			unsigned int index[1];
			cudaMemcpy(index, d_indexIncrement, sizeof(int), cudaMemcpyDeviceToHost);
			fprintf(stderr, "Number of alignments: %d @ %d\n", *index, j);
			// plot the alignments:
			plotAlignments(h_sequences, h_targets, h_globalDirectionZeroCopy, *index, h_startingPointsZeroCopy, j*NUMBER_SEQUENCES, i*NUMBER_TARGETS, descSequences, descTargets);
			//release memory:
			cudaFree(d_indexIncrement);
		}
	}


	cudaFree(d_sequences);
	cudaFree(d_targets);
	cudaFree(d_matrix);
	cudaFree(d_globalMaxima);
	cudaFree(d_internalMaxima);
	cudaFree(d_indexIncrement);
	cudaFreeHost(h_globalDirectionZeroCopy);
	cudaFreeHost(h_startingPointsZeroCopy);
	free(h_sequences);
	free(h_targets);
	free(h_matrix);
	free(h_globalMaxima);
	free(descTargets);
	free(descSequences);
	return(0);
}
