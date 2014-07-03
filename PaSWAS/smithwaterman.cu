#include "smithwaterman.h"


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
void init(char **h_sequences, char **h_targets,
		char **d_sequences, char **d_targets,
		GlobalMatrix **h_matrix, GlobalMatrix **d_matrix,
		GlobalMaxima **h_globalMaxima, GlobalMaxima **d_globalMaxima,
		GlobalDirection **d_globalDirection,
		GlobalDirection **h_globalDirectionZeroCopy, GlobalDirection **d_globalDirectionZeroCopy,
		StartingPoints **h_startingPointsZeroCopy, StartingPoints **d_startingPointsZeroCopy,
	  float **h_maxPossibleScoreZeroCopy, float **d_maxPossibleScoreZeroCopy,
		dim3 superBlocks,
		int device){

	if (!*h_sequences)
		*h_sequences = (char*)malloc(X * sizeof(char)*NUMBER_SEQUENCES*superBlocks.x);

	if (!*h_targets)
		*h_targets = (char*)malloc(Y * sizeof(char)*NUMBER_TARGETS*superBlocks.y);

	cutilSafeCall(cudaMalloc((void**)d_sequences, sizeof(char) *  (X*NUMBER_SEQUENCES)));
	cutilSafeCall(cudaMalloc((void**)d_targets, sizeof(char) *  (Y*NUMBER_TARGETS)));

	// create matrix:
	cudaMalloc((void**)d_matrix, sizeof(GlobalMatrix));
	if (!*h_matrix)
		*h_matrix = (GlobalMatrix *) malloc(sizeof(GlobalMatrix));

	cudaMalloc((void**)d_globalMaxima, sizeof(GlobalMaxima));
	if (!*h_globalMaxima)
		*h_globalMaxima = (GlobalMaxima *) malloc(sizeof(GlobalMaxima));

	cudaMalloc((void**)d_globalDirection, sizeof(GlobalDirection));

	cudaHostAlloc((void**)h_startingPointsZeroCopy, sizeof(StartingPoints), cudaHostAllocMapped);
	cudaHostGetDevicePointer((void **)d_startingPointsZeroCopy, (void *)*h_startingPointsZeroCopy, 0);

	cudaHostAlloc((void**)h_maxPossibleScoreZeroCopy, sizeof(float) * NUMBER_SEQUENCES * superBlocks.x, cudaHostAllocMapped);
	cudaHostGetDevicePointer((void **)d_maxPossibleScoreZeroCopy, (void *)*h_maxPossibleScoreZeroCopy, 0);

	// init host direction matrix for zero copy:
	cudaHostAlloc((void**)h_globalDirectionZeroCopy, sizeof(GlobalDirection), cudaHostAllocMapped);
	cudaHostGetDevicePointer((void **)d_globalDirectionZeroCopy, (void *)*h_globalDirectionZeroCopy, 0);

	fprintf(stderr, "%d mb of memory allocated on device number: %d\n",
			(int)((sizeof(char) *  (Y*NUMBER_TARGETS) +
			sizeof(char) *  (X*NUMBER_SEQUENCES) +
			sizeof(GlobalMatrix) + sizeof(GlobalMaxima) + sizeof(GlobalDirection)) / 1024/1024), device);
}

void initZeroCopy(unsigned int **d_indexIncrement){
	// create index:
	cudaMalloc((void **)d_indexIncrement, sizeof(int));
	unsigned int index[1];
	*index = 0;
	cudaMemcpy(*d_indexIncrement, index,sizeof(int), cudaMemcpyHostToDevice);

}



void calculateScoreHost(GlobalMatrix *d_matrix, char *d_sequences, char *d_targets, GlobalMaxima *d_globalMaxima, GlobalDirection *d_globalDirection) {
	unsigned int maxNumberOfBlocks = min(XdivSHARED_X,YdivSHARED_Y);
	unsigned int startDecreaseAt = XdivSHARED_X+YdivSHARED_Y - maxNumberOfBlocks;
	unsigned int numberOfBlocks = 0;
	unsigned int x = 0;
	unsigned int y = 0;
	dim3 dimBlock(SHARED_X,SHARED_Y,1);

//		int i= maxNumberOfBlocks;

	for (unsigned int i=1; i < XdivSHARED_X+YdivSHARED_Y; ++i) {
		if (i <= maxNumberOfBlocks)
			numberOfBlocks = i;
		else if( i >= startDecreaseAt)
			numberOfBlocks = XdivSHARED_X+YdivSHARED_Y - i;
		else
			numberOfBlocks = maxNumberOfBlocks;
		dim3 dimGridSW(NUMBER_SEQUENCES,NUMBER_TARGETS*numberOfBlocks , 1);//numberOfBlocks);
		//printf("%d, %d, %d\n", x,y,numberOfBlocks);

		calculateScore<<<dimGridSW, dimBlock>>>(d_matrix, x, y, numberOfBlocks,  d_sequences, d_targets, d_globalMaxima, d_globalDirection);
		cudaThreadSynchronize();
		if (x == XdivSHARED_X - 1)
			++y;
		if (x < XdivSHARED_X - 1)
			++x;
	}
}

/**
 * The calculateScore function checks the alignment per block. It calculates the score for each cell in
 * shared memory
 * @matrix   The scorings matrix
 * @x        The start x block position in the alignment to be calculated
 * @y        The start y block position in the alignment to be calculated
 * @numberOfBlocks The amount of blocks within an alignment which can be calculated
 * @seq1     The upper sequence in the alignment
 * @seq2     The left sequence in the alignment
 */
__global__ void calculateScore(
		GlobalMatrix *matrix, unsigned int x, unsigned int y, unsigned int numberOfBlocks,
		char *sequences, char *targets,
		GlobalMaxima *globalMaxima,
		GlobalDirection *globalDirection
		) {
	/**
	 * shared memory block for calculations. It requires
	 * extra (+1 in both directions) space to hold
	 * Neighboring cells
	 */
	__shared__ float s_matrix[SHARED_X+1][SHARED_Y+1];
	/**
	 * shared memory block for storing the maximum value of each neighboring cell.
	 * Careful: the s_maxima[SHARED_X][SHARED_Y] does not contain the maximum value
	 * after the calculation loop! This value is determined at the end of this
	 * function.
	 */
	__shared__ float s_maxima[SHARED_X][SHARED_Y];

	// calculate indices:
	//unsigned int yDIVnumSeq = (blockIdx.y/NUMBER_SEQUENCES);
	unsigned int blockx = x - blockIdx.y/NUMBER_TARGETS;//yDIVnumSeq;
	unsigned int blocky = y + blockIdx.y/NUMBER_TARGETS;//yDIVnumSeq;
	unsigned int tIDx = threadIdx.x;
	unsigned int tIDy = threadIdx.y;
	unsigned int bIDx = blockIdx.x;
	unsigned int bIDy = blockIdx.y%NUMBER_TARGETS;///numberOfBlocks;
	unsigned char direction = NO_DIRECTION;


	// indices of the current characters in both sequences.
	int seqIndex1 = tIDx + bIDx * X + blockx * SHARED_X;
	int seqIndex2 = tIDy + bIDy * Y + blocky * SHARED_Y;


	/* the next block is to get the maximum value from surrounding blocks. This maximum values is compared to the
	 * first element in the shared score matrix s_matrix.
	 */
	float maxPrev = 0.0;
	if (!tIDx && !tIDy) {
		if (blockx && blocky) {
			maxPrev = max(max(globalMaxima->blockMaxima[bIDx][bIDy].value[blockx-1][blocky-1], globalMaxima->blockMaxima[bIDx][bIDy].value[blockx-1][blocky]), globalMaxima->blockMaxima[bIDx][bIDy].value[blockx][blocky-1]);
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
	float innerScore = 0.0;
	/**
	 * tXM1 and tYM1 are to store the current value of the thread Index. tIDx and tIDy are
	 * both increased with 1 later on.
	 */
	unsigned int tXM1 = tIDx;
	unsigned int tYM1 = tIDy;

	// shared location for the parts of the 2 sequences, for faster retrieval later on:
	__shared__ char s_seq1[SHARED_X];
	__shared__ char s_seq2[SHARED_Y];

	// copy sequence data to shared memory (shared is much faster than global)
	if (!tIDy)
		s_seq1[tIDx] = sequences[seqIndex1];
	if (!tIDx)
		s_seq2[tIDy] = targets[seqIndex2];

	// set both matrices to zero
	s_matrix[tIDx][tIDy] = 0.0;
	s_maxima[tIDx][tIDy] = 0.0;

	if (tIDx == SHARED_X-1  && ! tIDy)
		s_matrix[SHARED_X][0] = 0.0;
	if (tIDy == SHARED_Y-1  && ! tIDx)
		s_matrix[0][SHARED_Y] = 0.0;

	/**** sync barrier ****/
	s_matrix[tIDx][tIDy] = 0.0;
	__syncthreads();

	// initialize outer parts of the matrix:
	if (!tIDx || !tIDy) {
		if (tIDx == SHARED_X-1)
			s_matrix[tIDx+1][tIDy] = 0.0;
		if (tIDy == SHARED_Y-1)
			s_matrix[tIDx][tIDy+1] = 0.0;
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

	innerScore = charS1 == FILL_CHARACTER || charS2 == FILL_CHARACTER ? FILL_SCORE : scoringsMatrix[charS1-characterOffset][charS2-characterOffset];

	// transpose the index
	++tIDx;
	++tIDy;
	// set shared matrix to zero (starting point!)
	s_matrix[tIDx][tIDy] = 0.0;


	// wait until all elements have been copied to the shared memory block
		/**** sync barrier ****/
	__syncthreads();

	currentScore = 0.0;
	//printf("begin!\n");

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
			s_matrix[tIDx][tIDy] = currentScore; // copy score to matrix
		}

		else if (i-1 == tXM1 + tYM1 ){
				// use this to find max
			if (i==1) {
				s_maxima[0][0] = max(maxPrev, currentScore);
			}
			else if (!tXM1 && tYM1) {
				s_maxima[0][tYM1] = max(s_maxima[0][tYM1-1], currentScore);
			}
			else if (!tYM1 && tXM1) {
				s_maxima[tXM1][0] = max(s_maxima[tXM1-1][0], currentScore);
			}
			else if (tXM1 && tYM1 ){
				s_maxima[tXM1][tYM1] = max(s_maxima[tXM1-1][tYM1], max(s_maxima[tXM1][tYM1-1], currentScore));
			}
		}
		// wait until all threads have calculated their new score
			/**** sync barrier ****/
		__syncthreads();
	}
	// copy end score to the scorings matrix:
	(*matrix).metaMatrix[bIDx][bIDy].matrix[blockx][blocky].value[tXM1][tYM1] = s_matrix[tIDx][tIDy];
	(*globalDirection).direction[bIDx][bIDy].localDirection[blockx][blocky].value[tXM1][tYM1] = direction;

	if (tIDx==SHARED_X && tIDy==SHARED_Y)
		globalMaxima->blockMaxima[bIDx][bIDy].value[blockx][blocky] = max(currentScore, max(s_maxima[SHARED_X-2][SHARED_Y-1], s_maxima[SHARED_X-1][SHARED_Y-2]));

	// wait until all threads have copied their score:
		/**** sync barrier ****/
	__syncthreads();
}


void tracebackHost(GlobalMatrix *d_matrix, GlobalMaxima *d_globalMaxima, GlobalDirection *d_globalDirection, GlobalDirection *d_globalDirectionZeroCopy, unsigned int *d_indexIncrement, StartingPoints *d_startingPoints, float *d_maxPossibleScore, int inBlock) {
	unsigned int maxNumberOfBlocks = min(XdivSHARED_X,YdivSHARED_Y);
	unsigned int startDecreaseAt = XdivSHARED_X+YdivSHARED_Y - maxNumberOfBlocks;
	unsigned int numberOfBlocks = 0;
	unsigned int x = XdivSHARED_X-1;
	unsigned int y = YdivSHARED_Y-1;
	dim3 dimBlock(SHARED_X,SHARED_Y,1);

//		int i= maxNumberOfBlocks;

	for (unsigned int i=1; i < XdivSHARED_X+YdivSHARED_Y; ++i) {
		if (i <= maxNumberOfBlocks)
			numberOfBlocks = i;
		else if( i >= startDecreaseAt)
			numberOfBlocks = XdivSHARED_X+YdivSHARED_Y - i;
		else
			numberOfBlocks = maxNumberOfBlocks;
		dim3 dimGridSW(NUMBER_SEQUENCES,NUMBER_TARGETS*numberOfBlocks , 1);
//		printf("%d, %d, %d\n", x,y,numberOfBlocks);
		traceback<<<dimGridSW, dimBlock>>>(d_matrix, x, y, numberOfBlocks, d_globalMaxima, d_globalDirection, d_globalDirectionZeroCopy, d_indexIncrement, d_startingPoints, d_maxPossibleScore, inBlock);
		cudaThreadSynchronize();
		if (y == 0)
			--x;
		if (y > 0)
			--y;
	}
}

__global__ void traceback(GlobalMatrix *matrix, unsigned int x, unsigned int y, unsigned int numberOfBlocks, GlobalMaxima *globalMaxima, GlobalDirection *globalDirection, GlobalDirection *globalDirectionZeroCopy, unsigned int *indexIncrement, StartingPoints *startingPoints, float *maxPossibleScore, int inBlock) {
	/**
	 * shared memory block for calculations. It requires
	 * extra (+1 in both directions) space to hold
	 * Neighboring cells
	 */
	__shared__ float s_matrix[SHARED_X+1][SHARED_Y+1];
	/**
	 * shared memory for storing the maximum value of this alignment.
	 */
	__shared__ float s_maxima[1];
	__shared__ float s_maxPossibleScore[1];

	// calculate indices:
	unsigned int yDIVnumSeq = (blockIdx.y/NUMBER_TARGETS);
	unsigned int blockx = x - yDIVnumSeq;
	unsigned int blocky = y + yDIVnumSeq;
	unsigned int tIDx = threadIdx.x;
	unsigned int tIDy = threadIdx.y;
	unsigned int bIDx = blockIdx.x;
	unsigned int bIDy = blockIdx.y%NUMBER_TARGETS;

	float value;

	if (!tIDx && !tIDy) {
//		printf("%d, %d (%d, %d)\n", bIDx, bIDy, blockx, blocky);
		s_maxima[0] = globalMaxima->blockMaxima[bIDx][bIDy].value[XdivSHARED_X-1][YdivSHARED_Y-1];
		s_maxPossibleScore[0] = maxPossibleScore[bIDx+inBlock];
	}

	__syncthreads();
	if (s_maxima[0]>= MINIMUM_SCORE) { // if the maximum score is below threshold, there is nothing to do

		s_matrix[tIDx][tIDy] = (*matrix).metaMatrix[bIDx][bIDy].matrix[blockx][blocky].value[tIDx][tIDy];

		unsigned char direction = globalDirection->direction[bIDx][bIDy].localDirection[blockx][blocky].value[tIDx][tIDy];


		// wait until all elements have been copied to the shared memory block
		/**** sync barrier ****/
		__syncthreads();

		for (int i=DIAGONAL-1; i >= 0; --i) {

			if ((i == tIDx + tIDy) && s_matrix[tIDx][tIDy] >= LOWER_LIMIT_SCORE * s_maxima[0] && s_matrix[tIDx][tIDy] >= s_maxPossibleScore[0]) {
				// found starting point!
				// reserve index:
				unsigned int index = atomicAdd(indexIncrement, 1);
				// now copy this to host:
				StartingPoint *start = &(startingPoints->startingPoint[index]);
				start->sequence = bIDx;
				start->target = bIDy;
				start->blockX = blockx;
				start->blockY = blocky;
				start->valueX = tIDx;
				start->valueY = tIDy;
				start->score = s_matrix[tIDx][tIDy];
				start->maxScore = s_maxima[0];
				start->posScore = s_maxPossibleScore[0];
				//				startingPoints->startingPoint[index] = start;
				// mark this value:
				s_matrix[tIDx][tIDy] = __int_as_float(SIGN_BIT_MASK | __float_as_int(s_matrix[tIDx][tIDy]));
			}
				//printf("Dir: %d\n", direction);
			__syncthreads();

			if ((i == tIDx + tIDy) && s_matrix[tIDx][tIDy] < 0 && direction == UPPER_LEFT_DIRECTION) {
				if (tIDx && tIDy){
					value = s_matrix[tIDx-1][tIDy-1];
					if (value == 0.0)
						direction = STOP_DIRECTION;
					else
						s_matrix[tIDx-1][tIDy-1] = __int_as_float(SIGN_BIT_MASK | __float_as_int(value));
				}
				else if (!tIDx && tIDy && blockx) {
					value = (*matrix).metaMatrix[bIDx][bIDy].matrix[blockx-1][blocky].value[SHARED_X-1][tIDy-1];
					if (value == 0.0)
						direction = STOP_DIRECTION;
					else
						(*matrix).metaMatrix[bIDx][bIDy].matrix[blockx-1][blocky].value[SHARED_X-1][tIDy-1] = __int_as_float(SIGN_BIT_MASK | __float_as_int(value));
				}
				else if (!tIDx && !tIDy && blockx && blocky) {
					value = (*matrix).metaMatrix[bIDx][bIDy].matrix[blockx-1][blocky-1].value[SHARED_X-1][SHARED_Y-1];
					if (value == 0.0)
						direction = STOP_DIRECTION;
					else
						(*matrix).metaMatrix[bIDx][bIDy].matrix[blockx-1][blocky-1].value[SHARED_X-1][SHARED_Y-1] = __int_as_float(SIGN_BIT_MASK | __float_as_int(value));
				}
				else if (tIDx && !tIDy && blocky) {
					value = (*matrix).metaMatrix[bIDx][bIDy].matrix[blockx][blocky-1].value[tIDx-1][SHARED_Y-1];
					if (value == 0.0)
						direction = STOP_DIRECTION;
					else
						(*matrix).metaMatrix[bIDx][bIDy].matrix[blockx][blocky-1].value[tIDx-1][SHARED_Y-1] = __int_as_float(SIGN_BIT_MASK | __float_as_int(value));
				}
			}
			__syncthreads();

			if ((i == tIDx + tIDy) && s_matrix[tIDx][tIDy] < 0 && direction == UPPER_DIRECTION) {
				if (!tIDy) {
					if (blocky) {
						value = (*matrix).metaMatrix[bIDx][bIDy].matrix[blockx][blocky-1].value[tIDx][SHARED_Y-1];
						if (value == 0.0)
							direction = STOP_DIRECTION;
						else
							(*matrix).metaMatrix[bIDx][bIDy].matrix[blockx][blocky-1].value[tIDx][SHARED_Y-1] = __int_as_float(SIGN_BIT_MASK | __float_as_int(value));
					}
				}
				else {
					value = s_matrix[tIDx][tIDy-1];
					if (value == 0.0)
						direction = STOP_DIRECTION;
					else
						s_matrix[tIDx][tIDy-1] = __int_as_float(SIGN_BIT_MASK | __float_as_int(value));
				}
			}

			__syncthreads();
			if ((i == tIDx + tIDy) && s_matrix[tIDx][tIDy] < 0 && direction == LEFT_DIRECTION) {
				if (!tIDx){
					if (blockx) {
						value = (*matrix).metaMatrix[bIDx][bIDy].matrix[blockx-1][blocky].value[SHARED_X-1][tIDy];
						if (value == 0.0)
							direction = STOP_DIRECTION;
						else
							(*matrix).metaMatrix[bIDx][bIDy].matrix[blockx-1][blocky].value[SHARED_X-1][tIDy] = __int_as_float(SIGN_BIT_MASK | __float_as_int(value));
					}
				}
				else {
					value = s_matrix[tIDx-1][tIDy];
					if (value == 0.0)
						direction = STOP_DIRECTION;
					else
						s_matrix[tIDx-1][tIDy] = __int_as_float(SIGN_BIT_MASK | __float_as_int(value));
				}
			}

			__syncthreads();

		}

		// copy end score to the scorings matrix:
		if (s_matrix[tIDx][tIDy] < 0) {
			(*matrix).metaMatrix[bIDx][bIDy].matrix[blockx][blocky].value[tIDx][tIDy] = s_matrix[tIDx][tIDy];
			globalDirectionZeroCopy->direction[bIDx][bIDy].localDirection[blockx][blocky].value[tIDx][tIDy] = direction;
		}
		/**** sync barrier ****/
		__syncthreads();
	}
}



void plotAlignments(char *sequences, char *targets, GlobalDirection *globalDirectionZeroCopy, unsigned int index, StartingPoints *startingPoints, int offset, int offsetTarget, char *descSequences, char *descTargets) {

	char *target = 0;
	char *sequence = 0;
	char *alignment = 0;

	target = (char*)malloc(sizeof(char)*2*(X+Y));
	sequence = (char*)malloc(sizeof(char)*2*(X+Y));
	alignment = (char*)malloc(sizeof(char)*2*(X+Y));

	if (!offset && !offsetTarget)
	  //		printf("QUERYLOCUS\tTARGETFRAME\tTARGETLOCUS\tPERCENTALIGNMENT\tALIGNMENTLENGTH\tGAPS\tMATCHES\tQUERYSTART\tQUERYEND\tTARGETSTART\tTARGETEND\tSIGNIFICANCE\tSCORE\tTARGETNT\tALIGN\tQUERYNT\n");
		printf("QUERYLOCUS\tTARGETLOCUS\tRELATIVESCORE\tALIGNMENTLENGTH\tGAPS\tMATCHES\tQUERYSTART\tQUERYEND\tTARGETSTART\tTARGETEND\tSCORE\tTARGETNT\tALIGN\tQUERYNT\n");

	float maxScore = 0;
	for (int i=0; i<index; i++){
	  int
	    alignmentLength = 0,
	    gapsSeq = 0,
	    gapsTarget = 0,
	    matches = 0,
	    mismatches = 0,
	    qEnd = startingPoints->startingPoint[i].blockX * SHARED_X + startingPoints->startingPoint[i].valueX,
	    tEnd = startingPoints->startingPoint[i].blockY * SHARED_Y + startingPoints->startingPoint[i].valueY;


	  /*
	  printf("[%d][%d][%d] vs [%d][%d][%d] with score %1.0f. \n", (startingPoints->startingPoint[i].sequence+offset),
				startingPoints->startingPoint[i].blockX,
				startingPoints->startingPoint[i].valueX,
		 startingPoints->startingPoint[i].target+offsetTarget,
				startingPoints->startingPoint[i].blockY,
				startingPoints->startingPoint[i].valueY,
				startingPoints->startingPoint[i].score
		);
	  */

		maxScore = startingPoints->startingPoint[i].score > maxScore ? startingPoints->startingPoint[i].score : maxScore;

		long targetStartingPoint = startingPoints->startingPoint[i].target;
		long sequenceStartingPoint = startingPoints->startingPoint[i].sequence;

		long blockX = startingPoints->startingPoint[i].blockX;
		long blockY = startingPoints->startingPoint[i].blockY;
		long valueX = startingPoints->startingPoint[i].valueX;
		long valueY = startingPoints->startingPoint[i].valueY;

		long localIndex = 0;

		unsigned char direction = globalDirectionZeroCopy->direction[sequenceStartingPoint][targetStartingPoint].localDirection[blockX][blockY].value[valueX][valueY];

		while ( blockX >= 0 && blockY >=0 && valueX >= 0 && valueY >= 0 &&
				direction != STOP_DIRECTION && direction != NO_DIRECTION) {
			direction = globalDirectionZeroCopy->direction[sequenceStartingPoint][targetStartingPoint].localDirection[blockX][blockY].value[valueX][valueY];

			//printf("%d ", targetStartingPoint*Y + blockY * SHARED_Y + valueY);
			//printf("d: %d\n", direction);
			alignmentLength++;
			switch (direction) {
				case UPPER_LEFT_DIRECTION:
					target[localIndex] = targets[targetStartingPoint*Y + blockY * SHARED_Y + valueY + offsetTarget*Y];
					sequence[localIndex] = sequences[sequenceStartingPoint*X + blockX * SHARED_X + valueX + offset*X];
					alignment[localIndex] = target[localIndex] == sequence[localIndex] ? MATCH_CHAR : MISMATCH_CHAR;
					matches += target[localIndex] == sequence[localIndex] ? 1 : 0;
					mismatches += target[localIndex] != sequence[localIndex] ? 1 : 0;
					if (valueX == 0){
						blockX--;
						valueX = SHARED_X-1;
					}
					else
						valueX--;
					if (valueY == 0){
						blockY--;
						valueY = SHARED_Y-1;
					}
					else
						valueY--;
					break;
				case LEFT_DIRECTION:
				  gapsTarget++;
					target[localIndex] = GAP_CHAR_SEQ;
					sequence[localIndex] = sequences[sequenceStartingPoint*X + blockX * SHARED_X + valueX + offset*X];
					alignment[localIndex] = GAP_CHAR_ALIGN;
					if (valueX == 0){
						blockX--;
						valueX = SHARED_X-1;
					}
					else
						valueX--;
					break;
				case UPPER_DIRECTION:
				  gapsSeq++;
					target[localIndex] = targets[targetStartingPoint*Y + blockY * SHARED_Y + valueY + offsetTarget*Y];
					sequence[localIndex] = GAP_CHAR_SEQ;
					alignment[localIndex] = GAP_CHAR_ALIGN;
					if (valueY == 0){
						blockY--;
						valueY = SHARED_Y-1;
					}
					else
						valueY--;
					break;
				case STOP_DIRECTION: // end of alignment
					matches += target[localIndex] == sequence[localIndex] ? 1 : 0;
					target[localIndex] = targets[targetStartingPoint*Y + blockY * SHARED_Y + valueY + offsetTarget*Y];
					sequence[localIndex] = sequences[sequenceStartingPoint*X + blockX * SHARED_X + valueX + offset*X];
					alignment[localIndex] = target[localIndex] == sequence[localIndex] ? MATCH_CHAR : MISMATCH_CHAR;
				break;
				case NO_DIRECTION: // end of alignment
				default:// huh? should not be the case!
				  fprintf(stderr, "Warning: wrong value in direction matrix: %d!\n", direction);
					blockX = -1;
					break;
			}
			localIndex++;
		}
		if ((double)(mismatches+gapsSeq+gapsTarget) / (double)alignmentLength  < ALLOWED_ERRORS) {
		  target[localIndex] = '\0';
		  sequence[localIndex] = '\0';
		  alignment[localIndex] = '\0';

		// now reverse:
		  
		  char swap;
		  for (long iSwap=0; iSwap < localIndex/2; iSwap++) {
			swap = target[iSwap];
			target[iSwap] = target[localIndex-1-iSwap];
			target[localIndex-1-iSwap] = swap;
			swap = sequence[iSwap];
			sequence[iSwap] = sequence[localIndex-1-iSwap];
			sequence[localIndex-1-iSwap] = swap;
			swap = alignment[iSwap];
			alignment[iSwap] = alignment[localIndex-1-iSwap];
			alignment[localIndex-1-iSwap] = swap;
		  }
		  //		  printf("%s\t%c\t%s\t", descSequences + ((startingPoints->startingPoint[i].sequence+offset) * MAX_LINE_LENGTH)+1, startingPoints->startingPoint[i].sequence%2?'D':'C',
		  printf("%s\t%s\t", descSequences + ((startingPoints->startingPoint[i].sequence+offset) * MAX_LINE_LENGTH)+1,
			 descTargets + ((startingPoints->startingPoint[i].target+offsetTarget) * MAX_LINE_LENGTH)+1);
		  // "0\t%i\t%i\t%i\t%i\t%i\t%i\t%i\t0\t%1.0f\t%s\t%s\t%s\n"
		  printf("%1.3f\t%i\t%i\t%i\t%i\t%i\t%i\t%i\t%1.0f\t%s\t%s\t%s\n",startingPoints->startingPoint[i].score/(float)alignmentLength,
			 alignmentLength, (gapsSeq + gapsTarget), matches, qEnd - alignmentLength+gapsSeq+1, qEnd+gapsSeq, tEnd - alignmentLength+gapsTarget+1, tEnd-gapsTarget, startingPoints->startingPoint[i].score, target, alignment, sequence
			 );
		}
	}

	free(target);
	free(sequence);
	free(alignment);
}



