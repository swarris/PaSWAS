/*
 * typedefs.h
 *
 *  Created on: 13-nov-2009
 *      Author: sven
 */

#ifndef TYPEDEFS_H_
#define TYPEDEFS_H_

#include "definitions.h"


#if defined(SHARED_MEM)
/* Scorings matrix for each thread block */
typedef struct {
	float value[SHARED_X][SHARED_Y];
}  LocalMatrix;

/* Scorings matrix for each sequence alignment */
typedef struct {
	LocalMatrix matrix[XdivSHARED_X][YdivSHARED_Y];
} ScoringsMatrix;

/* Scorings matrix for entire application */
typedef struct {
	ScoringsMatrix metaMatrix[NUMBER_SEQUENCES][NUMBER_TARGETS];
} GlobalMatrix;



typedef struct {
    float value[XdivSHARED_X][YdivSHARED_Y];
} BlockMaxima;

typedef struct {
    BlockMaxima blockMaxima[NUMBER_SEQUENCES][NUMBER_TARGETS];
} GlobalMaxima;



typedef struct {
	unsigned char value[SHARED_X][SHARED_Y];
} LocalDirection;

typedef struct {
	LocalDirection localDirection[XdivSHARED_X][YdivSHARED_Y];
} Direction;

typedef struct {
	Direction direction[NUMBER_SEQUENCES][NUMBER_TARGETS];
} GlobalDirection;


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
#endif

#if defined(GLOBAL_MEM4)
/* Scorings matrix for each sequence alignment */
typedef struct {
	float value[X+1][Y+1];
} ScoringsMatrix;

/* Scorings matrix for entire application */
typedef struct {
	ScoringsMatrix metaMatrix[NUMBER_SEQUENCES][NUMBER_TARGETS];
} GlobalMatrix;



typedef struct {
	float value[WORKGROUP_X][WORKGROUP_Y];
}  BlockMaxima;

typedef struct {
	BlockMaxima blockMaxima[XdivSHARED_X][YdivSHARED_Y];
} AlignMaxima;

/* Maximum matrix for entire application */
typedef struct {
	AlignMaxima alignMaxima[NUMBER_SEQUENCES][NUMBER_TARGETS];
} GlobalMaxima;


typedef struct {
	unsigned char value[X][Y];
} Direction;

typedef struct {
	Direction direction[NUMBER_SEQUENCES][NUMBER_TARGETS];
} GlobalDirection;


typedef struct {
	unsigned int sequence;
	unsigned int target;
	unsigned int valueX;
	unsigned int valueY;
	float score;
	float maxScore;
	float posScore;
} StartingPoint;


typedef struct {
	int s[1];
} Semaphore;

typedef struct {
	Semaphore semaphore[X][Y];
} Semaphores;

typedef struct {
	Semaphores semaphores[NUMBER_SEQUENCES][NUMBER_TARGETS];
} GlobalSemaphores;
#endif

typedef struct {
	StartingPoint startingPoint[MAXIMUM_NUMBER_STARTING_POINTS];
} StartingPoints;

typedef struct _dimensions {
	unsigned int x;
	unsigned int y;
	unsigned int z;
} dimensions;

typedef enum { false, true } bool;

#endif /* TYPEDEFS_H_ */
