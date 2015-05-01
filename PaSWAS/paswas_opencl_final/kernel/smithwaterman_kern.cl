//#include "smithwaterman.h"

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
//#pragma OPENCL EXTENSION cl_intel_printf : enable

/** kernel contains a for-loop in which the score is calculated. */
#define DIAGONAL SHARED_X + SHARED_Y

#define SCORINGS_MAT_SIZE 26
/** amount of blocks across the X axis */
#define XdivSHARED_X (X/SHARED_X)
/** amount of blocks across the Y axis */
#define YdivSHARED_Y (Y/SHARED_Y)

/**
 * Ensure that SHARED_X and SHARED_Y are divisible by WORKLOAD_X and WORKLOAD_Y
 * SHARED_X: Size of single block x
 * SHARED_Y: Size of single block y
 * WORKLOAD_X: number of elements in x that a single thread has to work on
 * WORKLOAD_Y: number of elements in y that a single thread has to work on
 * SHARED_X=>WORKLOAD_X && SHARED_X%WORKLOAD_X = 0
 * SHARED_Y=>WORKLOAD_Y && SHARED_Y%WORKLOAD_Y = 0
 * WORKGROUP_X=SHARED_X/WORKLOAD_X: Workgroup size x
 * WORKGROUP_Y=SHARED_Y/WORKLOAD_Y: Workgroup size y
 * **/

#define WORKGROUP_X (SHARED_X/WORKLOAD_X)
#define WORKGROUP_Y (SHARED_Y/WORKLOAD_Y)

/** amount of workgroups across the X axis */
#define XdivWORKGROUP_X (X/WORKGROUP_X)
/** amount of workgroups across the Y axis */
#define YdivWORKGROUP_Y (Y/WORKGROUP_Y)

/** start of the alphabet, so scoringsmatrix index can be calculated */
#define characterOffset 'A'
/** character used to fill the sequence if length < X */
#define FILL_CHARACTER 'x'
#define FILL_SCORE -1E10

/** Direction definitions for the direction matrix. These are needed for the traceback */
#define NO_DIRECTION 0
#define UPPER_LEFT_DIRECTION 1
#define UPPER_DIRECTION 2
#define LEFT_DIRECTION 3
#define STOP_DIRECTION 4

/** Specifies the value for which a traceback can be started. If the
 * value in the alignment matrix is larger than or equal to
 * LOWER_LIMIT_SCORE * maxValue the traceback is started at this point.
 * A lower value for LOWER_LIMIT_SCORE will give more aligments.
 */
#define LOWER_LIMIT_SCORE 1.0

/** this value is used to allocate enough memory to store the starting points */
#define MAXIMUM_NUMBER_STARTING_POINTS (NUMBER_SEQUENCES*NUMBER_TARGETS*1000)

/**** Scorings Section ****/
/** score used for a gap */
#define gapScore  -5.0

/**** Other definitions ****/

/** bit mask to get the negative value of a float, or to keep it negative */
#define SIGN_BIT_MASK 0x80000000
//#define max(a, b) (((a) > (b)) ? (a) : (b))

#ifdef SHARED_MEM
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

#ifdef GLOBAL_MEM4
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


void getSemaphore(__global int * semaphore) {
   int occupied = atom_xchg(semaphore, 1);
   while(occupied > 0)
   {
     occupied = atom_xchg(semaphore, 1);
   }
}

void releaseSemaphore(__global int * semaphore)
{
   int prevVal = atom_xchg(semaphore, 0);
}


#ifdef SHARED_MEM
__kernel void calculateScore(
		__global GlobalMatrix *matrix, unsigned int x, unsigned int y, unsigned int numberOfBlocks,
		__global char *sequences, __global char *targets, __global GlobalMaxima *globalMaxima, __global GlobalDirection *globalDirection,
		__global float *scoringsMatrix) {
	
	/**
     * shared memory block for calculations. It requires
     * extra (+1 in both directions) space to hold
     * Neighboring cells
     */
	__local float s_matrix[SHARED_X+1][SHARED_Y+1];
    /**
     * shared memory block for storing the maximum value of each neighboring cell.
     * Careful: the s_maxima[SHARED_X][SHARED_Y] does not contain the maximum value
     * after the calculation loop! This value is determined at the end of this
     * function.
     */
	__local float s_maxima[SHARED_X][SHARED_Y];

    // calculate indices:
    //unsigned int yDIVnumSeq = (blockIdx.y/NUMBER_SEQUENCES);
    // 1 is in y-direction and 0 is in x-direction
    unsigned int blockx = x - get_group_id(1)/NUMBER_TARGETS;//yDIVnumSeq;
    unsigned int blocky = y + get_group_id(1)/NUMBER_TARGETS;//yDIVnumSeq;
    unsigned int tIDx = get_local_id(0);
    unsigned int tIDy = get_local_id(1);
    unsigned int bIDx = get_group_id(0);
    unsigned int bIDy = get_group_id(1)%NUMBER_TARGETS;///numberOfBlocks;
    unsigned char direction = NO_DIRECTION;

    // indices of the current characters in both sequences.
    int seqIndex1 = tIDx + bIDx * X + blockx * SHARED_X;
    int seqIndex2 = tIDy + bIDy * Y + blocky * SHARED_Y;


    /* the next block is to get the maximum value from surrounding blocks. This maximum values is compared to the
     * first element in the shared score matrix s_matrix.
     */
    float maxPrev = 0.0f;
    if (!tIDx && !tIDy) {
        if (blockx && blocky) {
            maxPrev = fmax(fmax(globalMaxima->blockMaxima[bIDx][bIDy].value[blockx-1][blocky-1], globalMaxima->blockMaxima[bIDx][bIDy].value[blockx-1][blocky]), globalMaxima->blockMaxima[bIDx][bIDy].value[blockx][blocky-1]);
        }
        else if (blockx) {
            maxPrev = globalMaxima->blockMaxima[bIDx][bIDy].value[blockx-1][blocky];
        }
        else if (blocky) {
            maxPrev = globalMaxima->blockMaxima[bIDx][bIDy].value[blockx][blocky-1];
        }
    }
    // local scorings variables:
    float currentScore, ulS, lS, uS;
    float innerScore = 0.0f;
    /**
     * tXM1 and tYM1 are to store the current value of the thread Index. tIDx and tIDy are
     * both increased with 1 later on.
     */
    unsigned int tXM1 = tIDx;
    unsigned int tYM1 = tIDy;

    // shared location for the parts of the 2 sequences, for faster retrieval later on:
    __local char s_seq1[SHARED_X];
    __local char s_seq2[SHARED_Y];

    // copy sequence data to shared memory (shared is much faster than global)
    if (!tIDy)
        s_seq1[tIDx] = sequences[seqIndex1];
    if (!tIDx)
        s_seq2[tIDy] = targets[seqIndex2];

    // set both matrices to zero
    s_matrix[tIDx][tIDy] = 0.0f;
    s_maxima[tIDx][tIDy] = 0.0f;

    if (tIDx == SHARED_X-1  && ! tIDy)
        s_matrix[SHARED_X][0] = 0.0f;
    if (tIDy == SHARED_Y-1  && ! tIDx)
        s_matrix[0][SHARED_Y] = 0.0f;

    /**** sync barrier ****/
    s_matrix[tIDx][tIDy] = 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);

    // initialize outer parts of the matrix:
    if (!tIDx || !tIDy) {
        if (tIDx == SHARED_X-1)
            s_matrix[tIDx+1][tIDy] = 0.0f;
        if (tIDy == SHARED_Y-1)
            s_matrix[tIDx][tIDy+1] = 0.0f;
        if (blockx && !tIDx) {
            s_matrix[0][tIDy+1] = (*matrix).metaMatrix[bIDx][bIDy].matrix[blockx-1][blocky].value[SHARED_X-1][tIDy];
        }
        if (blocky && !tIDy) {
            s_matrix[tIDx+1][0] = (*matrix).metaMatrix[bIDx][bIDy].matrix[blockx][blocky-1].value[tIDx][SHARED_Y-1];
        }
        if (blockx && blocky && !tIDx && !tIDy){
            s_matrix[0][0] = (*matrix).metaMatrix[bIDx][bIDy].matrix[blockx-1][blocky-1].value[SHARED_X-1][SHARED_Y-1];
        }
    }
    // set inner score (aka sequence match/mismatch score):
    char charS1 = s_seq1[tIDx];
    char charS2 = s_seq2[tIDy];
    
    int offset_x = charS1-characterOffset;
    int offset_y = charS2-characterOffset;

    innerScore = charS1 == FILL_CHARACTER || charS2 == FILL_CHARACTER ? FILL_SCORE : scoringsMatrix[(offset_y*SCORINGS_MAT_SIZE) + offset_x];

    // transpose the index
    ++tIDx;
    ++tIDy;
    // set shared matrix to zero (starting point!)
    s_matrix[tIDx][tIDy] = 0.0f;


    // wait until all elements have been copied to the shared memory block
        /**** sync barrier ****/
    barrier(CLK_LOCAL_MEM_FENCE);

    currentScore = 0.0f;

    for (int i=0; i < DIAGONAL; ++i) {
        if (i == tXM1+ tYM1) {
            // calculate only when there are two valid characters
            // this is necessary when the two sequences are not of equal length
            // this is the SW-scoring of the cell:

          ulS = s_matrix[tXM1][tYM1] + innerScore;
          lS = s_matrix[tXM1][tIDy] + gapScore;
          uS = s_matrix[tIDx][tYM1] + gapScore;

            if (currentScore < lS) { // score comes from left
                currentScore = lS;
                direction = LEFT_DIRECTION;
            }
            if (currentScore < uS) { // score comes from above
                currentScore = uS;
                direction = UPPER_DIRECTION;
            }
            if (currentScore < ulS) { // score comes from upper left
                currentScore = ulS;
                direction = UPPER_LEFT_DIRECTION;
            }
            s_matrix[tIDx][tIDy] = innerScore == FILL_SCORE ? 0.0 : currentScore; // copy score to matrix
        }

        else if (i-1 == tXM1 + tYM1 ){
                // use this to find fmax
            if (i==1) {
                s_maxima[0][0] = fmax(maxPrev, currentScore);
            }
            else if (!tXM1 && tYM1) {
                s_maxima[0][tYM1] = fmax(s_maxima[0][tYM1-1], currentScore);
            }
            else if (!tYM1 && tXM1) {
                s_maxima[tXM1][0] = fmax(s_maxima[tXM1-1][0], currentScore);
            }
            else if (tXM1 && tYM1 ){
                s_maxima[tXM1][tYM1] = fmax(s_maxima[tXM1-1][tYM1], fmax(s_maxima[tXM1][tYM1-1], currentScore));
            }
        }
        // wait until all threads have calculated their new score
            /**** sync barrier ****/
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // copy end score to the scorings matrix:
    (*matrix).metaMatrix[bIDx][bIDy].matrix[blockx][blocky].value[tXM1][tYM1] = s_matrix[tIDx][tIDy];
    (*globalDirection).direction[bIDx][bIDy].localDirection[blockx][blocky].value[tXM1][tYM1] = direction;

    if (tIDx==SHARED_X && tIDy==SHARED_Y)
        globalMaxima->blockMaxima[bIDx][bIDy].value[blockx][blocky] = fmax(currentScore, fmax(s_maxima[SHARED_X-2][SHARED_Y-1], s_maxima[SHARED_X-1][SHARED_Y-2]));

    // wait until all threads have copied their score:
        /**** sync barrier ****/
    barrier(CLK_LOCAL_MEM_FENCE);
}


__kernel void traceback(
		__global GlobalMatrix *matrix, unsigned int x, unsigned int y, unsigned int numberOfBlocks, __global GlobalMaxima *globalMaxima,
		__global GlobalDirection *globalDirection, volatile __global unsigned int *indexIncrement,
		__global StartingPoints *startingPoints, __global float *maxPossibleScore, int inBlock) {

	/**
     * shared memory block for calculations. It requires
     * extra (+1 in both directions) space to hold
     * Neighboring cells
     */
	__local float s_matrix[SHARED_X+1][SHARED_Y+1];
    /**
     * shared memory for storing the maximum value of this alignment.
     */
	__local float s_maxima[1];
	__local float s_maxPossibleScore[1];

    // calculate indices:
    unsigned int yDIVnumSeq = (get_group_id(1)/NUMBER_TARGETS);
    unsigned int blockx = x - yDIVnumSeq;
    unsigned int blocky = y + yDIVnumSeq;
    unsigned int tIDx = get_local_id(0);
    unsigned int tIDy = get_local_id(1);
    unsigned int bIDx = get_group_id(0);
    unsigned int bIDy = get_group_id(1)%NUMBER_TARGETS;
    
    //unsigned int index = atom_inc(&indexIncrement[0]);


    float value = 0.0;

    if (!tIDx && !tIDy) {
        s_maxima[0] = globalMaxima->blockMaxima[bIDx][bIDy].value[XdivSHARED_X-1][YdivSHARED_Y-1];
        s_maxPossibleScore[0] = maxPossibleScore[bIDx+inBlock];//maxPossibleScore[bIDx];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if (s_maxima[0]>= MINIMUM_SCORE) { // if the maximum score is below threshold, there is nothing to do

        s_matrix[tIDx][tIDy] = (*matrix).metaMatrix[bIDx][bIDy].matrix[blockx][blocky].value[tIDx][tIDy];

        unsigned char direction = globalDirection->direction[bIDx][bIDy].localDirection[blockx][blocky].value[tIDx][tIDy];


        // wait until all elements have been copied to the shared memory block
        /**** sync barrier ****/
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int i=DIAGONAL-1; i >= 0; --i) {

            if ((i == tIDx + tIDy) && direction == UPPER_LEFT_DIRECTION && s_matrix[tIDx][tIDy] >= LOWER_LIMIT_SCORE * s_maxima[0] && s_matrix[tIDx][tIDy] >= s_maxPossibleScore[0]) {
                // found starting point!
                // reserve index:
                unsigned int index = atom_inc(&indexIncrement[0]);
                StartingPoint start;
                //__global StartingPoint *start = &(startingPoints->startingPoint[index]);
                start.sequence = bIDx;
                start.target = bIDy;
                start.blockX = blockx;
                start.blockY = blocky;
                start.valueX = tIDx;
                start.valueY = tIDy;
                start.score = s_matrix[tIDx][tIDy];
                start.maxScore = s_maxima[0];
                start.posScore = s_maxPossibleScore[0];
                startingPoints->startingPoint[index] = start;
                // mark this value:             
                //s_matrix[tIDx][tIDy] = __int_as_float(SIGN_BIT_MASK | __float_as_int(s_matrix[tIDx][tIDy]));
#ifdef NVIDIA
                s_matrix[tIDx][tIDy] = __int_as_float(SIGN_BIT_MASK | __float_as_int(s_matrix[tIDx][tIDy]));
#else
				s_matrix[tIDx][tIDy] = as_float(SIGN_BIT_MASK | as_int(s_matrix[tIDx][tIDy]));
#endif
               
            }
                
            barrier(CLK_LOCAL_MEM_FENCE);

            if ((i == tIDx + tIDy) && s_matrix[tIDx][tIDy] < 0 && direction == UPPER_LEFT_DIRECTION) {
                if (tIDx && tIDy){
                    value = s_matrix[tIDx-1][tIDy-1];
                    if (value == 0.0f) {
                    	direction = STOP_DIRECTION;
                    }     
                    else {
                    	//s_matrix[tIDx-1][tIDy-1] = __int_as_float(SIGN_BIT_MASK | __float_as_int(value));
#ifdef NVIDIA
						s_matrix[tIDx-1][tIDy-1] = __int_as_float(SIGN_BIT_MASK | __float_as_int(value));
#else
						s_matrix[tIDx-1][tIDy-1] = as_float(SIGN_BIT_MASK | as_int(value));
#endif
                    }
                        
                    	
                }
                else if (!tIDx && tIDy && blockx) {
                    value = (*matrix).metaMatrix[bIDx][bIDy].matrix[blockx-1][blocky].value[SHARED_X-1][tIDy-1];
                    if (value == 0.0f) {
                    	direction = STOP_DIRECTION;
                    }   
                    else {
                    	//(*matrix).metaMatrix[bIDx][bIDy].matrix[blockx-1][blocky].value[SHARED_X-1][tIDy-1] = __int_as_float(SIGN_BIT_MASK | __float_as_int(value));
#ifdef NVIDIA
                    	(*matrix).metaMatrix[bIDx][bIDy].matrix[blockx-1][blocky].value[SHARED_X-1][tIDy-1] = __int_as_float(SIGN_BIT_MASK | __float_as_int(value));
#else
                    	(*matrix).metaMatrix[bIDx][bIDy].matrix[blockx-1][blocky].value[SHARED_X-1][tIDy-1] = as_float(SIGN_BIT_MASK | as_int(value));
#endif
                    	
                    }
                        
                }
                else if (!tIDx && !tIDy && blockx && blocky) {
                    value = (*matrix).metaMatrix[bIDx][bIDy].matrix[blockx-1][blocky-1].value[SHARED_X-1][SHARED_Y-1];
                    if (value == 0.0f) {
                    	direction = STOP_DIRECTION;
                    }
                    else {
                    	//(*matrix).metaMatrix[bIDx][bIDy].matrix[blockx-1][blocky-1].value[SHARED_X-1][SHARED_Y-1] = __int_as_float(SIGN_BIT_MASK | __float_as_int(value));
#ifdef NVIDIA
                    	(*matrix).metaMatrix[bIDx][bIDy].matrix[blockx-1][blocky-1].value[SHARED_X-1][SHARED_Y-1] = __int_as_float(SIGN_BIT_MASK | __float_as_int(value));
#else
	              		(*matrix).metaMatrix[bIDx][bIDy].matrix[blockx-1][blocky-1].value[SHARED_X-1][SHARED_Y-1] = as_float(SIGN_BIT_MASK | as_int(value));
#endif
                    }
                        
                }
                else if (tIDx && !tIDy && blocky) {
                    value = (*matrix).metaMatrix[bIDx][bIDy].matrix[blockx][blocky-1].value[tIDx-1][SHARED_Y-1];
                    if (value == 0.0f) {
                    	direction = STOP_DIRECTION;
                    }    
                    else {
                    	//(*matrix).metaMatrix[bIDx][bIDy].matrix[blockx][blocky-1].value[tIDx-1][SHARED_Y-1] = __int_as_float(SIGN_BIT_MASK | __float_as_int(value));
#ifdef NVIDIA
						(*matrix).metaMatrix[bIDx][bIDy].matrix[blockx][blocky-1].value[tIDx-1][SHARED_Y-1] = __int_as_float(SIGN_BIT_MASK | __float_as_int(value));
#else
						(*matrix).metaMatrix[bIDx][bIDy].matrix[blockx][blocky-1].value[tIDx-1][SHARED_Y-1] = as_float(SIGN_BIT_MASK | as_int(value));
#endif
                    }
                        
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            if ((i == tIDx + tIDy) && s_matrix[tIDx][tIDy] < 0 && direction == UPPER_DIRECTION) {
                if (!tIDy) {
                    if (blocky) {
                        value = (*matrix).metaMatrix[bIDx][bIDy].matrix[blockx][blocky-1].value[tIDx][SHARED_Y-1];
                        if (value == 0.0f) {
                        	direction = STOP_DIRECTION;
                        }
                        else {
                        	//(*matrix).metaMatrix[bIDx][bIDy].matrix[blockx][blocky-1].value[tIDx][SHARED_Y-1] = __int_as_float(SIGN_BIT_MASK | __float_as_int(value));
#ifdef NVIDIA
                        	(*matrix).metaMatrix[bIDx][bIDy].matrix[blockx][blocky-1].value[tIDx][SHARED_Y-1] = __int_as_float(SIGN_BIT_MASK | __float_as_int(value));
#else
                        	(*matrix).metaMatrix[bIDx][bIDy].matrix[blockx][blocky-1].value[tIDx][SHARED_Y-1] = as_float(SIGN_BIT_MASK | as_int(value));
#endif
                        }
                            
                    }
                }
                else {
                    value = s_matrix[tIDx][tIDy-1];
                    if (value == 0.0f) {
                    	direction = STOP_DIRECTION;
                    }
                    else {
                    	//s_matrix[tIDx][tIDy-1] = __int_as_float(SIGN_BIT_MASK | __float_as_int(value));
#ifdef NVIDIA
                    	s_matrix[tIDx][tIDy-1] = __int_as_float(SIGN_BIT_MASK | __float_as_int(value));
#else
						s_matrix[tIDx][tIDy-1] = as_float(SIGN_BIT_MASK | as_int(value));
#endif
                    }
                        
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);
            if ((i == tIDx + tIDy) && s_matrix[tIDx][tIDy] < 0 && direction == LEFT_DIRECTION) {
                if (!tIDx){
                    if (blockx) {
                        value = (*matrix).metaMatrix[bIDx][bIDy].matrix[blockx-1][blocky].value[SHARED_X-1][tIDy];
                        if (value == 0.0f) {
                            direction = STOP_DIRECTION;
                        }
                        else {
                        	//(*matrix).metaMatrix[bIDx][bIDy].matrix[blockx-1][blocky].value[SHARED_X-1][tIDy] = __int_as_float(SIGN_BIT_MASK | __float_as_int(value));
#ifdef NVIDIA
							(*matrix).metaMatrix[bIDx][bIDy].matrix[blockx-1][blocky].value[SHARED_X-1][tIDy] = __int_as_float(SIGN_BIT_MASK | __float_as_int(value));
#else
							(*matrix).metaMatrix[bIDx][bIDy].matrix[blockx-1][blocky].value[SHARED_X-1][tIDy] = as_float(SIGN_BIT_MASK | as_int(value));
#endif
                        }
                            
                    }
                }
                else {
                    value = s_matrix[tIDx-1][tIDy];
                    if (value == 0.0f) {
                    	direction = STOP_DIRECTION;
                    }
                    else {
                    	//s_matrix[tIDx-1][tIDy] = __int_as_float(SIGN_BIT_MASK | __float_as_int(value));
#ifdef NVIDIA
                    	s_matrix[tIDx-1][tIDy] = __int_as_float(SIGN_BIT_MASK | __float_as_int(value));
#else
                    	s_matrix[tIDx-1][tIDy] = as_float(SIGN_BIT_MASK | as_int(value));
#endif

                    }
                        
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);

        }

        // copy end score to the scorings matrix:
        if (s_matrix[tIDx][tIDy] < 0) {
            (*matrix).metaMatrix[bIDx][bIDy].matrix[blockx][blocky].value[tIDx][tIDy] = s_matrix[tIDx][tIDy];
            globalDirection->direction[bIDx][bIDy].localDirection[blockx][blocky].value[tIDx][tIDy] = direction;
        }
        /**** sync barrier ****/
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
#endif

#ifdef GLOBAL_MEM4
__kernel void calculateScore(
		__global GlobalMatrix *matrix, unsigned int x, unsigned int y, unsigned int numberOfBlocks,
		__global char *sequences, __global char *targets, __global GlobalMaxima *globalMaxima, __global GlobalDirection *globalDirection,
		__global float *scoringsMatrix) {
	// calculate indices:
	//unsigned int yDIVnumSeq = (blockIdx.y/NUMBER_SEQUENCES);
	unsigned int blockx = x - get_group_id(1)/NUMBER_TARGETS;//0<=(get_group_id(1)/NUMBER_TARGETS)<numberOfBlocks
	unsigned int blocky = y + get_group_id(1)/NUMBER_TARGETS;//0<=(get_group_id(1)/NUMBER_TARGETS)<numberOfBlocks
	unsigned int tIDx = get_local_id(0);
	unsigned int tIDy = get_local_id(1);
	unsigned int bIDx = get_group_id(0);
	unsigned int bIDy = get_group_id(1)%NUMBER_TARGETS;///numberOfBlocks;
	
	float thread_max = 0.0;

		for (int i=0; i < WORKGROUP_X + WORKGROUP_Y; ++i) {
		if(i==tIDx+tIDy) {
			for(int j=0; j<WORKLOAD_X; j++) {
				
				unsigned int aIDx = tIDx*WORKLOAD_X + j + blockx * SHARED_X; //0<=alignmentIDx<X
				unsigned int aXM1 = aIDx;
				
				++aIDx; //1<=alignmentIDx<=X
								
				int seqIndex1 = tIDx * WORKLOAD_X + j + bIDx * X + blockx * SHARED_X;
				char s1 = sequences[seqIndex1];
				
				/** Number of target characters a single work-item is responsible for **/
				for(int k=0; k<WORKLOAD_Y; k++) {
					
					unsigned char direction = NO_DIRECTION;
					int seqIndex2 = tIDy*WORKLOAD_Y + k + bIDy * Y + blocky * SHARED_Y;
					char s2 = targets[seqIndex2];

					unsigned int aIDy = tIDy*WORKLOAD_Y + k + blocky * SHARED_Y; //0<=alignmentIDy<Y
					unsigned int aYM1 = aIDy;
					
					++aIDy; //1<=alignmentIDy<=Y
		
					float currentScore = 0.0;
					float ulS = 0.0;
					float lS = 0.0; 
					float uS = 0.0;
					float innerScore = 0.0;
					
					
					int offset_x = s1-characterOffset;
					int offset_y = s2-characterOffset;
					
					innerScore = s1 == FILL_CHARACTER || s2 == FILL_CHARACTER ? FILL_SCORE : scoringsMatrix[(offset_y*SCORINGS_MAT_SIZE) + offset_x];					
					ulS = (*matrix).metaMatrix[bIDx][bIDy].value[aXM1][aYM1] + innerScore;
					lS = (*matrix).metaMatrix[bIDx][bIDy].value[aXM1][aIDy] + gapScore;
					uS = (*matrix).metaMatrix[bIDx][bIDy].value[aIDx][aYM1] + gapScore;
					
					if (currentScore < lS) { // score comes from left
						currentScore = lS;
						direction = LEFT_DIRECTION;
					}
					if (currentScore < uS) { // score comes from above
						currentScore = uS;
						direction = UPPER_DIRECTION;
					}
					if (currentScore < ulS) { // score comes from upper left
						currentScore = ulS;
						direction = UPPER_LEFT_DIRECTION;
					}
					(*matrix).metaMatrix[bIDx][bIDy].value[aIDx][aIDy] = currentScore;
					(*globalDirection).direction[bIDx][bIDy].value[aXM1][aYM1] = direction;
					thread_max = fmax(currentScore, thread_max);
					
				}
			}

		}
		if(i-1==tIDx+tIDy) { //got a thread_maximum
			if(i==1) {
				//get the maximum value of surrounding blocks
				float maxPrev = 0.0;
				if (blockx && blocky) {
					maxPrev = fmax(fmax(globalMaxima->alignMaxima[bIDx][bIDy].blockMaxima[blockx-1][blocky-1].value[WORKGROUP_X-1][WORKGROUP_Y-1], globalMaxima->alignMaxima[bIDx][bIDy].blockMaxima[blockx-1][blocky].value[WORKGROUP_X-1][WORKGROUP_Y-1]), globalMaxima->alignMaxima[bIDx][bIDy].blockMaxima[blockx][blocky-1].value[WORKGROUP_X-1][WORKGROUP_Y-1]);
				}
				else if (blockx) {
					maxPrev = globalMaxima->alignMaxima[bIDx][bIDy].blockMaxima[blockx-1][blocky].value[WORKGROUP_X-1][WORKGROUP_Y-1];
				}
				else if (blocky) {
					maxPrev = globalMaxima->alignMaxima[bIDx][bIDy].blockMaxima[blockx][blocky-1].value[WORKGROUP_X-1][WORKGROUP_Y-1];
				}
				
				(*globalMaxima).alignMaxima[bIDx][bIDy].blockMaxima[blockx][blocky].value[0][0] = fmax(maxPrev, thread_max);
			} else if(!tIDx && tIDy) {
				(*globalMaxima).alignMaxima[bIDx][bIDy].blockMaxima[blockx][blocky].value[0][tIDy] = fmax((*globalMaxima).alignMaxima[bIDx][bIDy].blockMaxima[blockx][blocky].value[0][tIDy-1],thread_max);
			} else if(tIDx && !tIDy) {
				(*globalMaxima).alignMaxima[bIDx][bIDy].blockMaxima[blockx][blocky].value[tIDx][0] = fmax((*globalMaxima).alignMaxima[bIDx][bIDy].blockMaxima[blockx][blocky].value[tIDx-1][0],thread_max);
			} else if(tIDx && tIDy) {
				(*globalMaxima).alignMaxima[bIDx][bIDy].blockMaxima[blockx][blocky].value[tIDx][tIDy] = fmax((*globalMaxima).alignMaxima[bIDx][bIDy].blockMaxima[blockx][blocky].value[tIDx-1][tIDy], fmax(thread_max,(*globalMaxima).alignMaxima[bIDx][bIDy].blockMaxima[blockx][blocky].value[tIDx][tIDy-1]));
			}
			
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		
	}
	
	if (tIDx==WORKGROUP_X-1 && tIDy==WORKGROUP_Y-1) {
		(*globalMaxima).alignMaxima[bIDx][bIDy].blockMaxima[blockx][blocky].value[tIDx][tIDy] = fmax(thread_max, fmax((*globalMaxima).alignMaxima[bIDx][bIDy].blockMaxima[blockx][blocky].value[tIDx-1][tIDy], (*globalMaxima).alignMaxima[bIDx][bIDy].blockMaxima[blockx][blocky].value[tIDx][tIDy-1]));
	}
}

__kernel void traceback(
		__global GlobalMatrix *matrix, unsigned int x, unsigned int y, unsigned int numberOfBlocks, __global GlobalMaxima *globalMaxima,
		__global GlobalDirection *globalDirection, volatile __global unsigned int *indexIncrement,
		__global StartingPoints *startingPoints, __global float *maxPossibleScore, int inBlock, __global GlobalSemaphores *globalSemaphores) {

		unsigned int blockx = x - get_group_id(1)/NUMBER_TARGETS;//0<=(get_group_id(1)/NUMBER_TARGETS)<numberOfBlocks
		unsigned int blocky = y + get_group_id(1)/NUMBER_TARGETS;//0<=(get_group_id(1)/NUMBER_TARGETS)<numberOfBlocks
		unsigned int tIDx = get_local_id(0);
		unsigned int tIDy = get_local_id(1);
		unsigned int bIDx = get_group_id(0);
		unsigned int bIDy = get_group_id(1)%NUMBER_TARGETS;///numberOfBlocks;
	    
		float maximum = (*globalMaxima).alignMaxima[bIDx][bIDy].blockMaxima[XdivSHARED_X-1][YdivSHARED_Y-1].value[WORKGROUP_X-1][WORKGROUP_Y-1];
	
		
		if(maximum >= MINIMUM_SCORE) {
			float mpScore = maxPossibleScore[bIDx+inBlock];
			for(int i=WORKGROUP_X+WORKGROUP_Y-1; i>=0; --i) {
				if(i==tIDx+tIDy) {
					for(int j=WORKLOAD_X-1; j>=0; j--){
						unsigned int aIDx = tIDx*WORKLOAD_X + j + blockx * SHARED_X; //0<=alignmentIDx<X
						unsigned int aXM1 = aIDx;
						++aIDx; //1<=alignmentIDx<=X
						for(int k=WORKLOAD_Y-1; k>=0; k--) {
							unsigned int aIDy = tIDy*WORKLOAD_Y + k + blocky * SHARED_Y; //0<=alignmentIDy<Y
							unsigned int aYM1 = aIDy;
							++aIDy; //1<=alignmentIDy<=Y
							float score = (*matrix).metaMatrix[bIDx][bIDy].value[aIDx][aIDy];
							unsigned char direction = (*globalDirection).direction[bIDx][bIDy].value[aXM1][aYM1];
							if (direction == UPPER_LEFT_DIRECTION && score >= LOWER_LIMIT_SCORE * maximum && score >= mpScore) {
								// found starting point!
								unsigned int index = atom_inc(&indexIncrement[0]);
								// now copy this to host:
								StartingPoint start;
								start.sequence = bIDx;
								start.target = bIDy;
								start.valueX = aXM1;
								start.valueY = aYM1;
								start.score = (*matrix).metaMatrix[bIDx][bIDy].value[aIDx][aIDy];
								start.maxScore = maximum;
								start.posScore = mpScore;
								startingPoints->startingPoint[index] = start;
								//Mark this value
#ifdef NVIDIA
								(*matrix).metaMatrix[bIDx][bIDy].value[aIDx][aIDy] = __int_as_float(SIGN_BIT_MASK | __float_as_int(score);
#else
								(*matrix).metaMatrix[bIDx][bIDy].value[aIDx][aIDy] = as_float(SIGN_BIT_MASK | as_int(score));
#endif
							}
							score = (*matrix).metaMatrix[bIDx][bIDy].value[aIDx][aIDy];
							if (score < 0 && direction == UPPER_LEFT_DIRECTION) {
								score = (*matrix).metaMatrix[bIDx][bIDy].value[aIDx-1][aIDy-1];
								if (score == 0.0) {
									direction = STOP_DIRECTION;
									globalDirection->direction[bIDx][bIDy].value[aXM1][aYM1] = direction;
								}
								else {
									getSemaphore(&((*globalSemaphores).semaphores[bIDx][bIDy].semaphore[aXM1-1][aYM1-1].s[0]));
#ifdef NVIDIA
									(*matrix).metaMatrix[bIDx][bIDy].value[aIDx-1][aIDy-1] = __int_as_float(SIGN_BIT_MASK | __float_as_int(score));
#else
									(*matrix).metaMatrix[bIDx][bIDy].value[aIDx-1][aIDy-1] = as_float(SIGN_BIT_MASK | as_int(score));
#endif
									releaseSemaphore(&((*globalSemaphores).semaphores[bIDx][bIDy].semaphore[aXM1-1][aYM1-1].s[0]));
								}
							}
							if (score < 0 && direction == LEFT_DIRECTION) {
								score = (*matrix).metaMatrix[bIDx][bIDy].value[aIDx-1][aIDy];
								if (score == 0.0) {
									direction = STOP_DIRECTION;
									globalDirection->direction[bIDx][bIDy].value[aXM1][aYM1] = direction;
								}
								else {
									getSemaphore(&((*globalSemaphores).semaphores[bIDx][bIDy].semaphore[aXM1-1][aYM1].s[0]));
#ifdef NVIDIA
									(*matrix).metaMatrix[bIDx][bIDy].value[aIDx-1][aIDy] = __int_as_float(SIGN_BIT_MASK | __float_as_int(score));
#else
									(*matrix).metaMatrix[bIDx][bIDy].value[aIDx-1][aIDy] = as_float(SIGN_BIT_MASK | as_int(score));
#endif
									releaseSemaphore(&((*globalSemaphores).semaphores[bIDx][bIDy].semaphore[aXM1-1][aYM1].s[0]));
								}
							}
							if (score < 0 && direction == UPPER_DIRECTION) {
								score = (*matrix).metaMatrix[bIDx][bIDy].value[aIDx][aIDy-1];
								if (score == 0.0) {
									direction = STOP_DIRECTION;
									globalDirection->direction[bIDx][bIDy].value[aXM1][aYM1] = direction;
								}
								else {
									getSemaphore(&((*globalSemaphores).semaphores[bIDx][bIDy].semaphore[aXM1][aYM1-1].s[0]));
#ifdef NVIDIA
									(*matrix).metaMatrix[bIDx][bIDy].value[aIDx][aIDy-1] = __int_as_float(SIGN_BIT_MASK | __float_as_int(score));
#else
									(*matrix).metaMatrix[bIDx][bIDy].value[aIDx][aIDy-1] = as_float(SIGN_BIT_MASK | as_int(score));
#endif
									releaseSemaphore(&((*globalSemaphores).semaphores[bIDx][bIDy].semaphore[aXM1][aYM1-1].s[0]));
								}
							}


						}
					}
					
				}
				barrier(CLK_LOCAL_MEM_FENCE);
			}
		}
}
#endif
