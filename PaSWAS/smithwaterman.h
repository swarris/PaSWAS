#ifndef SMITHWATERMAN_H_
#define SMITHWATERMAN_H_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
//#include <cutil.h>
#include <math.h>
#include <builtin_types.h>
#include <time.h>
//#include <cutil_inline.h>
#include <helper_cuda.h>
#include "definitions.h"
#include "typedefs.h"

#define cutilSafeCall(x) checkCudaErrors(x)

/**
 * init initializes host and device memory blocks
 * @ h_sequence1 Target sequence(s) on the host
 * @ h_sequence2 Query sequence(s) on the host
 * @ d_sequence1 Target sequence(s) on the device
 * @ d_sequence2 Query sequence(s) on the device
 * @ h_matrix    scorings matrix on the host
 * @ d_matrix    scorings matrix on the device
 * @ h_globalMaxima Maximum values on the host
 * @ d_globalMaxima Maximum values on the device
 * @ device Number of the device to allocate memory on
 */
void init(char **h_sequences, char **h_targets, char **d_sequences, char **d_targets,
	GlobalMatrix **h_matrix, GlobalMatrix **d_matrix,
	GlobalMaxima **h_globalMaxima, GlobalMaxima **d_globalMaxima,
	GlobalMaxima **d_internalMaxima,
	GlobalDirection **d_globalDirection,
	GlobalDirection **h_globalDirectionZeroCopy, GlobalDirection **d_globalDirectionZeroCopy,
	StartingPoints **h_startingPointsZeroCopy,
	StartingPoints **d_startingPointsZeroCopy,
	  float **h_maxPossibleScoreZeroCopy,
	  float **d_maxPossibleScoreZeroCopy,
	dim3 superBlocks,
	int device);

void initZeroCopy(unsigned int **d_IndexIncrement);


void calculateScoreHost(GlobalMatrix *d_matrix, char *d_sequences, char *d_targets, GlobalMaxima *d_globalMaxima, GlobalMaxima *d_internalMaxima, GlobalDirection *d_globalDirection);

/**
 * The calculateScore function checks the alignment per block. It calculates the score for each cell in
 * shared memory.
 * @matrix   The scorings matrix
 * @x        The start x block position in the alignment to be calculated
 * @y        The start y block position in the alignment to be calculated
 * @numberOfBlocks The amount of blocks within an alignment which can be calculated
 * @seq1     The upper sequence in the alignment
 * @seq2     The left sequence in the alignment
 */
extern "C"
__global__ void calculateScore(GlobalMatrix *matrix, unsigned int x, unsigned int y, unsigned int numberOfBlocks, char *sequences, char *targets, GlobalMaxima *globalMaxima, GlobalMaxima *internalMaxima, GlobalDirection *globalDirection);


void tracebackHost(GlobalMatrix *d_matrix, GlobalMaxima *d_globalMaxima, GlobalMaxima *d_internalMaxima, GlobalDirection *d_globalDirection, GlobalDirection *d_globalDirectionZeroCopy, unsigned int *d_indexIncrement, StartingPoints *d_startingPoints, float *d_maxPossibleScore, int inBlock);
extern "C"
__global__ void traceback(GlobalMatrix *matrix, unsigned int x, unsigned int y, unsigned int numberOfBlocks, GlobalMaxima *globalMaxima, GlobalMaxima *internalMaxima, GlobalDirection *globalDirection, GlobalDirection *globalDirectionZeroCopy, unsigned int *indexIncrement, StartingPoints *startingPoints, float *maxPossibleScore, int inBlock);


void plotAlignments(char *sequences, char *targets, GlobalDirection *globalDirectionZeroCopy, unsigned int index, StartingPoints *startingPoints, int offset, int offsetTarget, char *descSequences, char *descTargets);

#endif /*SMITHWATERMAN_H_*/
