#ifndef DEFINITIONS_H_
#define DEFINITIONS_H_

/**
 * Warris's adaptation of the Smith-Waterman algorithm (WASWA).
 *
 * Requires a NVidia Geforce CUDA 2.1 with at least 1.3 compute capability.
 *
 * @author Sven Warris
 * @version 1.1
 */

/** maximum X per block (used in dimensions for blocks and amount of shared memory */
//#define SHARED_X 8
/** maximum Y per block (used in dimensions for blocks and amount of shared memory */
//#define SHARED_Y 8

/** kernel contains a for-loop in which the score is calculated. */
#define DIAGONAL SHARED_X + SHARED_Y

/** amount of score elements in a single block */
#define blockSize (SHARED_X * SHARED_Y)

#define SCORINGS_MAT_SIZE 26
#define DNA_RNA_LOC "scoringsmatrix/DNA_RNA.txt"
#define BLOSUM62_LOC "scoringsmatrix/BLOSUM62.txt"
#define BASIC_LOC "scoringsmatrix/BASIC.txt"

/**
 * Todo ensure that SHARED_X and SHARED_Y are divisible by WORKLOAD_X and WORKLOAD_Y
 * SHARED_X: Size of single block x
 * SHARED_Y: Size of single block y
 * WORKLOAD_X: number of elements in x that a single thread has to work on
 * WORKLOAD_Y: number of elements in y that a single thread has to work on
 * SHARED_X=>WORKLOAD_X && SHARED_X%WORKLOAD_X = 0
 * SHARED_Y=>WORKLOAD_Y && SHARED_Y%WORKLOAD_Y = 0
 * WORKGROUP_X=SHARED_X/WORKLOAD_X: Workgroup size x
 * WORKGROUP_Y=SHARED_Y/WORKLOAD_Y: Workgroup size y
 * **/

/** amount of blocks across the X axis */
#define XdivSHARED_X (X/SHARED_X)
/** amount of blocks across the Y axis */
#define YdivSHARED_Y (Y/SHARED_Y)

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

#define LOWER_LIMIT_MAX_SCORE 0.0


/** Specifies the value for which a traceback can be started. If the
 * value in the alignment matrix is larger than or equal to
 * LOWER_LIMIT_SCORE * maxValue the traceback is started at this point.
 * A lower value for LOWER_LIMIT_SCORE will give more aligments.
 */
#define LOWER_LIMIT_SCORE 1.0

/** Only report alignments with a given minimum score. A good setting is:
 * (length shortest seq)*(lowest positive score) - (number of allowed gaps/mismatches)*(lowest negative score)
 * For testing: keep it low to get many alignments back.
 * @todo: make this a config at runtime.
 */

#define ALLOWED_ERRORS 1.0

/** this value is used to allocate enough memory to store the starting points */
#define MAXIMUM_NUMBER_STARTING_POINTS (NUMBER_SEQUENCES*NUMBER_TARGETS*1000)

/** these characters are used for the alignment output plot */
#define GAP_CHAR_SEQ '-'
#define GAP_CHAR_ALIGN ' '
#define MISMATCH_CHAR '.'
#define MATCH_CHAR '|'

/**** Scorings Section ****/

/** score used for a gap */
#define gapScore  -5.0
#define HIGHEST_SCORE 5.0


/**** Other definitions ****/

/** bit mask to get the negative value of a float, or to keep it negative */
#define SIGN_BIT_MASK 0x80000000
#define MAX_LINE_LENGTH 100
#define min(a, b) (((a) < (b)) ? (a) : (b))
#endif /*DEFINITIONS_H_*/
