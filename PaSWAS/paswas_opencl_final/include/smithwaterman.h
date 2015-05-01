#ifndef SMITHWATERMAN_H_
#define SMITHWATERMAN_H_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifdef APPLE
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "definitions.h"
#include "typedefs.h"

void init(char **h_sequences, char **h_targets,
		cl_mem *d_sequences, cl_mem *d_targets,
		GlobalMatrix **h_matrix, cl_mem *d_matrix,
		GlobalMaxima **h_globalMaxima, cl_mem *d_globalMaxima,
		cl_mem *d_globalDirection, GlobalDirection **h_globalDirectionZeroCopy,
		cl_mem *d_startingPointsZeroCopy, StartingPoints **h_startingPointsZeroCopy,
	    cl_mem *d_maxPossibleScoreZeroCopy, float **h_maxPossibleScoreZeroCopy,
	    cl_mem *d_scoringsMatrix,
	    cl_mem *d_indexIncrement,
		dimensions superBlocks,
		cl_context context,
		cl_command_queue queue,
		cl_int error_check);

void init_zc(char **h_sequences, char **h_targets,
		cl_mem *d_sequences, cl_mem *d_targets,
		GlobalMatrix **h_matrix, cl_mem *d_matrix,
		GlobalMaxima **h_globalMaxima, cl_mem *d_globalMaxima,
		cl_mem *d_startingPointsZeroCopy, StartingPoints **h_startingPointsZeroCopy, cl_mem *pinned_startingPointsZeroCopy,
		cl_mem *d_maxPossibleScoreZeroCopy, float **h_maxPossibleScoreZeroCopy, cl_mem *pinned_maxPossibleScoreZeroCopy,
		cl_mem *d_globalDirection, GlobalDirection **h_globalDirectionZeroCopy, cl_mem *pinned_globalDirectionZeroCopy,
		cl_mem *d_scoringsMatrix,
	    cl_mem *d_indexIncrement,
		dimensions superBlocks,
		cl_context context,
		cl_command_queue queue,
		cl_int error_check);

void init_zc_CPU(char **h_sequences, char **h_targets,
		cl_mem *d_sequences, cl_mem *d_targets,
		GlobalMatrix **h_matrix, cl_mem *d_matrix,
		GlobalMaxima **h_globalMaxima, cl_mem *d_globalMaxima,
		cl_mem *d_startingPointsZeroCopy, StartingPoints **startingPointsData,
		cl_mem *d_maxPossibleScoreZeroCopy, float **maxPossibleScoreData,
		cl_mem *d_globalDirection, GlobalDirection **globalDirectionData,
		cl_mem *d_scoringsMatrix,
	    cl_mem *d_indexIncrement,
		dimensions superBlocks,
		cl_context context,
		cl_command_queue queue,
		cl_int error_check);

void initZeroCopy(cl_mem *d_indexIncrement, cl_context context, cl_command_queue queue, cl_int error_check);
void calculateScoreHost(cl_mem d_matrix, cl_mem d_sequences, cl_mem d_targets, cl_mem d_globalMaxima, cl_mem d_globalDirection, cl_mem d_scoringsMatrix, cl_kernel kernel, cl_command_queue queue, cl_int error_check);

#ifdef GLOBAL_MEM4
void tracebackHost(cl_mem d_matrix, cl_mem d_globalMaxima, cl_mem d_globalDirection, cl_mem d_indexIncrement, cl_mem d_startingPoints, cl_mem d_maxPossibleScore, int inBlock, cl_kernel kernel, cl_command_queue queue, cl_int error_check, cl_mem d_semaphor);
void initSemaphor(cl_mem *d_semaphor, cl_context context, cl_command_queue queue, cl_int error_check);
#endif

#ifdef SHARED_MEM
void tracebackHost(cl_mem d_matrix, cl_mem d_globalMaxima, cl_mem d_globalDirection, cl_mem d_indexIncrement, cl_mem d_startingPoints, cl_mem d_maxPossibleScore, int inBlock, cl_kernel kernel, cl_command_queue queue, cl_int error_check);
#endif

void plotAlignments(char *sequences, char *targets, GlobalDirection *globalDirection, unsigned int index, StartingPoints *startingPoints, int offset, int offsetTarget, char *descSequences, char *descTargets);
void fillScoringsMatrix(float h_scoringsMatrix[]);
void readContentsScoringMatrix(float *h_scoringsMatrix, int size, char *file_loc);



#endif /*SMITHWATERMAN_H_*/
