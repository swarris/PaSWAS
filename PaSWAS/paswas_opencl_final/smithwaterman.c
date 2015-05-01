#include "smithwaterman.h"

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
		cl_int error_check){


	if (!*h_sequences)
		*h_sequences = (char*)malloc(X * sizeof(char)*NUMBER_SEQUENCES*superBlocks.x);

	*d_sequences = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(char) * (X*NUMBER_SEQUENCES), NULL, &error_check);
	if(error_check != CL_SUCCESS) {
			fprintf(stderr,"Device sequence buffer could not be created");
			exit(EXIT_FAILURE);
	}

	if (!*h_targets)
		*h_targets = (char*)malloc(Y * sizeof(char)*NUMBER_TARGETS*superBlocks.y);

	*d_targets = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(char) *  (Y*NUMBER_TARGETS), NULL, &error_check);
	if(error_check != CL_SUCCESS) {
			fprintf(stderr,"Device target buffer could not be created");
			exit(EXIT_FAILURE);
	}

	// create matrix:
	*d_matrix = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(GlobalMatrix), NULL, &error_check);
	if(error_check != CL_SUCCESS) {
			fprintf(stderr,"Device matrix buffer could not be created");
			exit(EXIT_FAILURE);
	}

	if (!*h_matrix)
		*h_matrix = (GlobalMatrix *) malloc(sizeof(GlobalMatrix));

	*d_globalMaxima = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(GlobalMaxima), NULL, &error_check);
	if(error_check != CL_SUCCESS) {
			fprintf(stderr,"Device globalMaxima buffer could not be created");
			exit(EXIT_FAILURE);
	}

	if (!*h_globalMaxima)
		*h_globalMaxima = (GlobalMaxima *) malloc(sizeof(GlobalMaxima));

	if (!*h_globalDirectionZeroCopy)
		*h_globalDirectionZeroCopy = (GlobalDirection *) malloc(sizeof(GlobalDirection));

	*d_globalDirection = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(GlobalDirection), NULL, &error_check);
	if(error_check != CL_SUCCESS) {
			fprintf(stderr,"Device globalDirection buffer could not be created");
			exit(EXIT_FAILURE);
	}

	if (!*h_startingPointsZeroCopy)
		*h_startingPointsZeroCopy = (StartingPoints *) malloc(sizeof(StartingPoints));

	*d_startingPointsZeroCopy = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(StartingPoints), NULL, &error_check);
	if(error_check != CL_SUCCESS) {
			fprintf(stderr,"Device startingPointsZeroCopy buffer could not be created");
			exit(EXIT_FAILURE);
	}

	if (!*h_maxPossibleScoreZeroCopy)
		*h_maxPossibleScoreZeroCopy = (float *) malloc(sizeof(float) * NUMBER_SEQUENCES * superBlocks.x);

	*d_maxPossibleScoreZeroCopy = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * NUMBER_SEQUENCES * superBlocks.x, NULL, &error_check);
	if(error_check != CL_SUCCESS) {
			fprintf(stderr,"Device maxPossibleScoreZeroCopy buffer could not be created");
			exit(EXIT_FAILURE);
	}

    *d_scoringsMatrix = clCreateBuffer(context, CL_MEM_READ_ONLY, SCORINGS_MAT_SIZE * SCORINGS_MAT_SIZE * sizeof(float), NULL, &error_check);
	if(error_check != CL_SUCCESS) {
			fprintf(stderr,"Device scorings buffer could not be created");
			exit(EXIT_FAILURE);
	}

    *d_indexIncrement = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(unsigned int), NULL, &error_check);
	if(error_check != CL_SUCCESS) {
			fprintf(stderr,"Device index could not be created");
			exit(EXIT_FAILURE);
	}

   fprintf(stderr, "%d mb of memory allocated for CalculateScore kernel\n",
			(int)((sizeof(GlobalMatrix) +
				  sizeof(unsigned int) * 3 +
				  X * sizeof(char)*NUMBER_SEQUENCES +
				  Y * sizeof(char)*NUMBER_TARGETS +
				  sizeof(GlobalMaxima) +
				  sizeof(GlobalDirection) +
				  26 * 26 * sizeof(float)) / 1024/1024));

	fprintf(stderr, "%d mb of memory allocated for Traceback kernel\n",
			(int)((sizeof(GlobalMatrix) +
				  sizeof(unsigned int) * 3 +
				  sizeof(GlobalMaxima) +
				  sizeof(GlobalDirection) * 2 +
				  sizeof(int) * 2 +
				  sizeof(StartingPoints) +
				  sizeof(float) * NUMBER_SEQUENCES * superBlocks.x) / 1024/1024));
}

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
		cl_int error_check){


	if (!*h_sequences)
		*h_sequences = (char*)malloc(X * sizeof(char)*NUMBER_SEQUENCES*superBlocks.x);

	*d_sequences = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(char) * (X*NUMBER_SEQUENCES), NULL, &error_check);
	if(error_check != CL_SUCCESS) {
			fprintf(stderr,"Device sequence buffer could not be created\n");
			exit(EXIT_FAILURE);
	}

	if (!*h_targets)
		*h_targets = (char*)malloc(Y * sizeof(char)*NUMBER_TARGETS*superBlocks.y);

	*d_targets = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(char) *  (Y*NUMBER_TARGETS), NULL, &error_check);
	if(error_check != CL_SUCCESS) {
			fprintf(stderr,"Device target buffer could not be created\n");
			exit(EXIT_FAILURE);
	}

	// create matrix:
	if (!*h_matrix)
		*h_matrix = (GlobalMatrix*) malloc(sizeof(GlobalMatrix));

	*d_matrix = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(GlobalMatrix), NULL, &error_check);
	if(error_check != CL_SUCCESS) {
			fprintf(stderr,"Device matrix buffer could not be created\n");
			exit(EXIT_FAILURE);
	}


	if (!*h_globalMaxima)
			*h_globalMaxima = (GlobalMaxima*) malloc(sizeof(GlobalMaxima));

	*d_globalMaxima = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(GlobalMaxima), NULL, &error_check);
	if(error_check != CL_SUCCESS) {
			fprintf(stderr,"Device globalMaxima buffer could not be created\n");
			exit(EXIT_FAILURE);
	}


	/** Begin: Initializing Zero Copy variables **/

	/** Begin: Initializing StartingPoint Zero Copy variable **/
	*pinned_startingPointsZeroCopy = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(StartingPoints), NULL, &error_check);
	if(error_check != CL_SUCCESS) {
			fprintf(stderr,"Could not initialize pinned StartingPoint buffer\n");
			exit(EXIT_FAILURE);
	}

	*d_startingPointsZeroCopy = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(StartingPoints), NULL, &error_check);
	if(error_check != CL_SUCCESS) {
			fprintf(stderr,"Device startingPointsZeroCopy buffer could not be created\n");
			exit(EXIT_FAILURE);
	}

	if (!*h_startingPointsZeroCopy) {
		*h_startingPointsZeroCopy = (StartingPoints*) clEnqueueMapBuffer(queue, *pinned_startingPointsZeroCopy, CL_TRUE, CL_MAP_READ, 0,
						sizeof(StartingPoints), 0, NULL, NULL, &error_check);
		if(error_check != CL_SUCCESS) {
				fprintf(stderr,"Could not pin StartingPoint region\n");
				exit(EXIT_FAILURE);
		}

	}
	/** End: Initializing StartingPoint Zero Copy variables **/


	/** Begin: Initializing MaxPossibleScore Zero Copy variable **/
	*pinned_maxPossibleScoreZeroCopy = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(float) * NUMBER_SEQUENCES * superBlocks.x, NULL, &error_check);
	if(error_check != CL_SUCCESS) {
			fprintf(stderr,"Could not initialize pinned StartingPoint buffer\n");
			exit(EXIT_FAILURE);
	}

	*d_maxPossibleScoreZeroCopy = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * NUMBER_SEQUENCES * superBlocks.x, NULL, &error_check);
	if(error_check != CL_SUCCESS) {
			fprintf(stderr,"Device maxPossibleScoreZeroCopy buffer could not be created\n");
			exit(EXIT_FAILURE);
	}

	if (!*h_maxPossibleScoreZeroCopy) {
		*h_maxPossibleScoreZeroCopy = (float*) clEnqueueMapBuffer(queue, *pinned_maxPossibleScoreZeroCopy, CL_TRUE, CL_MAP_WRITE, 0,
				sizeof(float) * NUMBER_SEQUENCES * superBlocks.x, 0, NULL, NULL, &error_check);
		if(error_check != CL_SUCCESS) {
				fprintf(stderr,"Could not pin MaxPossibleScore region\n");
				exit(EXIT_FAILURE);
		}

	}
	/** End: Initializing MaxPossibleScore Zero Copy variable **/

	/** Begin: Initializing GlobalDirection Zero Copy variable **/
	*pinned_globalDirectionZeroCopy = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(GlobalDirection), NULL, &error_check);
	if(error_check != CL_SUCCESS) {
			fprintf(stderr,"Could not initialize pinned GlobalDirection buffer\n");
			exit(EXIT_FAILURE);
	}

	*d_globalDirection = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(GlobalDirection), NULL, &error_check);
		if(error_check != CL_SUCCESS) {
				fprintf(stderr,"Device globalDirectionZeroCopy buffer could not be created\n");
				exit(EXIT_FAILURE);
	}

	if (!*h_globalDirectionZeroCopy) {
		*h_globalDirectionZeroCopy = (GlobalDirection*) clEnqueueMapBuffer(queue, *pinned_globalDirectionZeroCopy, CL_TRUE, CL_MAP_READ, 0,
				sizeof(GlobalDirection), 0, NULL, NULL, &error_check);
		if(error_check != CL_SUCCESS) {
				fprintf(stderr,"Could not pin GlobalDirection region\n");
				exit(EXIT_FAILURE);
		}

	}
	/** End: Initializing GlobalDirection Zero Copy variable **/
	/** End: Initializing Zero Copy variables **/

    *d_scoringsMatrix = clCreateBuffer(context, CL_MEM_READ_ONLY, SCORINGS_MAT_SIZE * SCORINGS_MAT_SIZE * sizeof(float), NULL, &error_check);
	if(error_check != CL_SUCCESS) {
			fprintf(stderr,"Device scorings buffer could not be created\n");
			exit(EXIT_FAILURE);
	}

    *d_indexIncrement = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(unsigned int), NULL, &error_check);
	if(error_check != CL_SUCCESS) {
			fprintf(stderr,"Device index could not be created");
			exit(EXIT_FAILURE);
	}

   fprintf(stderr, "%d mb of memory allocated for CalculateScore kernel\n",
			(int)((sizeof(GlobalMatrix) +
				  sizeof(unsigned int) * 3 +
				  X * sizeof(char)*NUMBER_SEQUENCES +
				  Y * sizeof(char)*NUMBER_TARGETS +
				  sizeof(GlobalMaxima) +
				  sizeof(GlobalDirection) +
				  26 * 26 * sizeof(float)) / 1024/1024));

	fprintf(stderr, "%d mb of memory allocated for Traceback kernel\n",
			(int)((sizeof(GlobalMatrix) +
				  sizeof(unsigned int) * 3 +
				  sizeof(GlobalMaxima) +
				  sizeof(GlobalDirection) * 2 +
				  sizeof(int) * 2 +
				  sizeof(StartingPoints) +
				  sizeof(float) * NUMBER_SEQUENCES * superBlocks.x) / 1024/1024));

	fprintf(stderr, "%d mb of memory allocated for zero-copy variables\n",
			(int)((sizeof(GlobalDirection) +
				  sizeof(StartingPoints) +
				  sizeof(float) * NUMBER_SEQUENCES * superBlocks.x) / 1024/1024));
}

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
		cl_int error_check){


	if (!*h_sequences)
		*h_sequences = (char*)malloc(X * sizeof(char)*NUMBER_SEQUENCES*superBlocks.x);

	*d_sequences = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(char) * (X*NUMBER_SEQUENCES), NULL, &error_check);
	if(error_check != CL_SUCCESS) {
			fprintf(stderr,"Device sequence buffer could not be created\n");
			exit(EXIT_FAILURE);
	}

	if (!*h_targets)
		*h_targets = (char*)malloc(Y * sizeof(char)*NUMBER_TARGETS*superBlocks.y);

	*d_targets = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(char) *  (Y*NUMBER_TARGETS), NULL, &error_check);
	if(error_check != CL_SUCCESS) {
			fprintf(stderr,"Device target buffer could not be created\n");
			exit(EXIT_FAILURE);
	}


	if (!*h_matrix)
		*h_matrix = (GlobalMatrix*) malloc(sizeof(GlobalMatrix));

	*d_matrix = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(GlobalMatrix), NULL, &error_check);
	if(error_check != CL_SUCCESS) {
			fprintf(stderr,"Device matrix buffer could not be created\n");
			exit(EXIT_FAILURE);
	}

	if (!*h_globalMaxima)
			*h_globalMaxima = (GlobalMaxima*) malloc(sizeof(GlobalMaxima));

	*d_globalMaxima = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(GlobalMaxima), NULL, &error_check);
	if(error_check != CL_SUCCESS) {
			fprintf(stderr,"Device globalMaxima buffer could not be created\n");
			exit(EXIT_FAILURE);
	}

	/** Begin: Initializing Zero Copy variables **/

	/** Begin: Initializing StartingPoint Zero Copy variable **/
	*d_startingPointsZeroCopy = clCreateBuffer(context, CL_MEM_USE_HOST_PTR, sizeof(StartingPoints), *startingPointsData, &error_check);
	if(error_check != CL_SUCCESS) {
			fprintf(stderr,"startingPointsZeroCopy buffer could not be created\n");
			exit(EXIT_FAILURE);
	}
	/** End: Initializing StartingPoint Zero Copy variables **/


	/** Begin: Initializing MaxPossibleScore Zero Copy variable **/
	*d_maxPossibleScoreZeroCopy = clCreateBuffer(context, CL_MEM_USE_HOST_PTR, sizeof(float) * NUMBER_SEQUENCES * superBlocks.x, *maxPossibleScoreData, &error_check);
	if(error_check != CL_SUCCESS) {
			fprintf(stderr,"maxPossibleScoreZeroCopy buffer could not be created\n");
			exit(EXIT_FAILURE);
	}
	/** End: Initializing MaxPossibleScore Zero Copy variable **/

	/** Begin: Initializing GlobalDirection Zero Copy variable **/
	*d_globalDirection = clCreateBuffer(context, CL_MEM_USE_HOST_PTR, sizeof(GlobalDirection), *globalDirectionData, &error_check);
	if(error_check != CL_SUCCESS) {
			fprintf(stderr,"globalDirectionZeroCopy buffer could not be created\n");
			exit(EXIT_FAILURE);
	}
	/** End: Initializing GlobalDirection Zero Copy variable **/
	/** End: Initializing Zero Copy variables **/

    *d_scoringsMatrix = clCreateBuffer(context, CL_MEM_READ_ONLY, SCORINGS_MAT_SIZE * SCORINGS_MAT_SIZE * sizeof(float), NULL, &error_check);
	if(error_check != CL_SUCCESS) {
			fprintf(stderr,"Device scorings buffer could not be created\n");
			exit(EXIT_FAILURE);
	}

    *d_indexIncrement = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(unsigned int), NULL, &error_check);
	if(error_check != CL_SUCCESS) {
			fprintf(stderr,"Device index could not be created");
			exit(EXIT_FAILURE);
	}

   fprintf(stderr, "%d mb of memory allocated for CalculateScore kernel\n",
			(int)((sizeof(GlobalMatrix) +
				  sizeof(unsigned int) * 3 +
				  X * sizeof(char)*NUMBER_SEQUENCES +
				  Y * sizeof(char)*NUMBER_TARGETS +
				  sizeof(GlobalMaxima) +
				  sizeof(GlobalDirection) +
				  26 * 26 * sizeof(float)) / 1024/1024));

	fprintf(stderr, "%d mb of memory allocated for Traceback kernel\n",
			(int)((sizeof(GlobalMatrix) +
				  sizeof(unsigned int) * 3 +
				  sizeof(GlobalMaxima) +
				  sizeof(GlobalDirection) * 2 +
				  sizeof(int) * 2 +
				  sizeof(StartingPoints) +
				  sizeof(float) * NUMBER_SEQUENCES * superBlocks.x) / 1024/1024));

	fprintf(stderr, "%d mb of memory allocated for zero-copy variables\n",
			(int)((sizeof(GlobalDirection) +
				  sizeof(StartingPoints) +
				  sizeof(float) * NUMBER_SEQUENCES * superBlocks.x) / 1024/1024));
}

void initZeroCopy(cl_mem *d_indexIncrement, cl_context context, cl_command_queue queue, cl_int error_check){
	// create index:

		unsigned int index[1];
		*index = 0;
		error_check = clEnqueueWriteBuffer(queue, *d_indexIncrement, CL_TRUE, 0, sizeof(unsigned int), index, 0, NULL, NULL);
		clFinish(queue);
		if(error_check != CL_SUCCESS) {
				fprintf(stderr,"Data could not be written to indexIncrement buffer\n");
				exit(EXIT_FAILURE);
		}

}

#ifdef SHARED_MEM
void calculateScoreHost(cl_mem d_matrix, cl_mem d_sequences, cl_mem d_targets, cl_mem d_globalMaxima, cl_mem d_globalDirection, cl_mem d_scoringsMatrix, cl_kernel kernel, cl_command_queue queue, cl_int error_check) {
	unsigned int maxNumberOfBlocks = min(XdivSHARED_X,YdivSHARED_Y);
	unsigned int startDecreaseAt = XdivSHARED_X+YdivSHARED_Y - maxNumberOfBlocks;
	unsigned int numberOfBlocks = 0;
	unsigned int x = 0;
	unsigned int y = 0;
	size_t local[2];
	local[0] = SHARED_X;
	local[1] = SHARED_Y;


	for (unsigned int i=1; i < XdivSHARED_X+YdivSHARED_Y; ++i) {
		if (i <= maxNumberOfBlocks)
			numberOfBlocks = i;
		else if( i >= startDecreaseAt)
			numberOfBlocks = XdivSHARED_X+YdivSHARED_Y - i;
		else
			numberOfBlocks = maxNumberOfBlocks;

		size_t global[2];
		global[0] = NUMBER_SEQUENCES * SHARED_X;
		global[1] = numberOfBlocks * NUMBER_TARGETS * SHARED_Y;

		error_check = 0;
		error_check = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_matrix);
		error_check |= clSetKernelArg(kernel, 1, sizeof(unsigned int), &x);
		error_check |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &y);
		error_check |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &numberOfBlocks);
		error_check |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_sequences);
		error_check |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &d_targets);
		error_check |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &d_globalMaxima);
		error_check |= clSetKernelArg(kernel, 7, sizeof(cl_mem), &d_globalDirection);
		error_check |= clSetKernelArg(kernel, 8, sizeof(cl_mem), &d_scoringsMatrix);


		if(error_check != CL_SUCCESS) {
			fprintf(stderr,"Failed to set calculate score kernel arguments");
			exit(EXIT_FAILURE);
		}


		error_check = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, NULL);

		if(error_check != CL_SUCCESS) {
			fprintf(stderr,"Distributing calculateScoreKernel kernel has failed\n");
			exit(EXIT_FAILURE);
		}


		// Wait for the commands to complete
		clFinish(queue);
		if (x == XdivSHARED_X - 1)
			++y;
		if (x < XdivSHARED_X - 1)
			++x;
	}
}

void tracebackHost(cl_mem d_matrix, cl_mem d_globalMaxima, cl_mem d_globalDirection, cl_mem d_indexIncrement, cl_mem d_startingPoints, cl_mem d_maxPossibleScore, int inBlock, cl_kernel kernel, cl_command_queue queue, cl_int error_check) {
	unsigned int maxNumberOfBlocks = min(XdivSHARED_X,YdivSHARED_Y);
	unsigned int startDecreaseAt = XdivSHARED_X+YdivSHARED_Y - maxNumberOfBlocks;
	unsigned int numberOfBlocks = 0;
	unsigned int x = XdivSHARED_X-1;
	unsigned int y = YdivSHARED_Y-1;
	size_t local[2];
	local[0] = SHARED_X;
	local[1] = SHARED_Y;


	for (unsigned int i=1; i < XdivSHARED_X+YdivSHARED_Y; ++i) {
		if (i <= maxNumberOfBlocks)
			numberOfBlocks = i;
		else if( i >= startDecreaseAt)
			numberOfBlocks = XdivSHARED_X+YdivSHARED_Y - i;
		else
			numberOfBlocks = maxNumberOfBlocks;

		size_t global[2];
		global[0] = NUMBER_SEQUENCES * SHARED_X;
		global[1] = numberOfBlocks * NUMBER_TARGETS * SHARED_Y;

		error_check = 0;
		error_check = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_matrix);
		error_check |= clSetKernelArg(kernel, 1, sizeof(unsigned int), &x);
		error_check |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &y);
		error_check |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &numberOfBlocks);
		error_check |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_globalMaxima);
		error_check |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &d_globalDirection);
		error_check |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &d_indexIncrement);
		error_check |= clSetKernelArg(kernel, 7, sizeof(cl_mem), &d_startingPoints);
		error_check |= clSetKernelArg(kernel, 8, sizeof(cl_mem), &d_maxPossibleScore);
		error_check |= clSetKernelArg(kernel, 9, sizeof(int), &inBlock);


		if(error_check != CL_SUCCESS) {
			fprintf(stderr,"Failed to set traceback kernel arguments");
			exit(EXIT_FAILURE);
		}

		error_check = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, NULL);
		if(error_check != CL_SUCCESS) {
			fprintf(stderr,"Distributing traceback kernel kernel has failed\n");
			exit(EXIT_FAILURE);
		}

		clFinish(queue);


		if (y == 0)
			--x;
		if (y > 0)
			--y;
	}
}


void plotAlignments(char *sequences, char *targets, GlobalDirection *globalDirectionZeroCopy, unsigned int index, StartingPoints *startingPoints, int offset, int offsetTarget, char *descSequences, char *descTargets) {

	char *target = 0;
	char *sequence = 0;
	char *alignment = 0;

	target = (char*)malloc(sizeof(char)*2*(X+Y));
	sequence = (char*)malloc(sizeof(char)*2*(X+Y));
	alignment = (char*)malloc(sizeof(char)*2*(X+Y));

	if (!offset && !offsetTarget) {
		printf("QUERYLOCUS\tTARGETLOCUS\tRELATIVESCORE\tALIGNMENTLENGTH\tGAPS\tMATCHES\tQUERYSTART\tQUERYEND\tTARGETSTART\tTARGETEND\tSCORE\tTARGETNT\tALIGN\tQUERYNT\n");
	}


	float maxScore = 0;
	for (int i=0; i<index; i++){
			int alignmentLength = 0,
			gapsSeq = 0,
			gapsTarget = 0,
			matches = 0,
			mismatches = 0,
			qEnd = startingPoints->startingPoint[i].blockX * SHARED_X + startingPoints->startingPoint[i].valueX,
			tEnd = startingPoints->startingPoint[i].blockY * SHARED_Y + startingPoints->startingPoint[i].valueY;

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
						target[localIndex] = targets[targetStartingPoint*Y + blockY * SHARED_Y + valueY + offsetTarget*Y];
						sequence[localIndex] = sequences[sequenceStartingPoint*X + blockX * SHARED_X + valueX + offset*X];
						matches += target[localIndex] == sequence[localIndex] ? 1 : 0;
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

				  printf("%s\t%s\t", descSequences + ((startingPoints->startingPoint[i].sequence+offset) * MAX_LINE_LENGTH),
							 descTargets + ((startingPoints->startingPoint[i].target+offsetTarget) * MAX_LINE_LENGTH));
				  printf("%1.3f\t%i\t%i\t%i\t%i\t%i\t%i\t%i\t%1.0f\t%s\t%s\t%s\n",startingPoints->startingPoint[i].score/(float)alignmentLength,
					 alignmentLength, (gapsSeq + gapsTarget), matches, qEnd - alignmentLength+gapsSeq+1, qEnd+gapsSeq, tEnd - alignmentLength+gapsTarget+1, tEnd-gapsTarget, startingPoints->startingPoint[i].score, target, alignment, sequence
					 );


			}
	}


	free(target);
	free(sequence);
	free(alignment);
}
#endif


#ifdef GLOBAL_MEM4
void initSemaphor(cl_mem *d_semaphore, cl_context context, cl_command_queue queue, cl_int error_check) {
	//int semaphor[1];
	//*semaphor = 0;
	int zero_pattern = 0;
	error_check = clEnqueueFillBuffer(queue, *d_semaphore, &zero_pattern, sizeof(int), 0, sizeof(GlobalSemaphores), 0, NULL, NULL);
	clFinish(queue);
	if(error_check != CL_SUCCESS) {
			fprintf(stderr,"Semaphore matrix could not be initialized\n");
			exit(EXIT_FAILURE);
	}
}


void calculateScoreHost(cl_mem d_matrix, cl_mem d_sequences, cl_mem d_targets, cl_mem d_globalMaxima, cl_mem d_globalDirection, cl_mem d_scoringsMatrix, cl_kernel kernel, cl_command_queue queue, cl_int error_check) {
	unsigned int maxNumberOfBlocks = min(XdivSHARED_X,YdivSHARED_Y);
	unsigned int startDecreaseAt = XdivSHARED_X+YdivSHARED_Y - maxNumberOfBlocks;
	unsigned int numberOfBlocks = 0;
	unsigned int x = 0;
	unsigned int y = 0;
	size_t local[2];
	local[0] = WORKGROUP_X; //SHARED_X/WORKLOAD_X
	local[1] = WORKGROUP_Y; //SHARED_Y/WORKLOAD_Y

	for (unsigned int i=1; i < XdivSHARED_X+YdivSHARED_Y; ++i) {
			if (i <= maxNumberOfBlocks)
				numberOfBlocks = i;
			else if( i >= startDecreaseAt)
				numberOfBlocks = XdivSHARED_X+YdivSHARED_Y - i;
			else
				numberOfBlocks = maxNumberOfBlocks;

			size_t global[2];
			global[0] = NUMBER_SEQUENCES * WORKGROUP_X;
			global[1] = numberOfBlocks * NUMBER_TARGETS * WORKGROUP_Y;

			error_check = 0;
			error_check = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_matrix);
			error_check |= clSetKernelArg(kernel, 1, sizeof(unsigned int), &x);
			error_check |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &y);
			error_check |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &numberOfBlocks);
			error_check |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_sequences);
			error_check |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &d_targets);
			error_check |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &d_globalMaxima);
			error_check |= clSetKernelArg(kernel, 7, sizeof(cl_mem), &d_globalDirection);
			error_check |= clSetKernelArg(kernel, 8, sizeof(cl_mem), &d_scoringsMatrix);


			if(error_check != CL_SUCCESS) {
				fprintf(stderr,"Failed to set calculate score kernel arguments\n");
				exit(EXIT_FAILURE);
			}


			error_check = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, NULL);

			if(error_check != CL_SUCCESS) {
				fprintf(stderr,"Distributing calculateScoreKernel kernel has failed\n");
				exit(EXIT_FAILURE);
			}


			// Wait for the commands to complete
			clFinish(queue);

			if (x == XdivSHARED_X - 1)
				++y;
			if (x < XdivSHARED_X - 1)
				++x;
	}

}



void tracebackHost(cl_mem d_matrix, cl_mem d_globalMaxima, cl_mem d_globalDirection, cl_mem d_indexIncrement, cl_mem d_startingPoints, cl_mem d_maxPossibleScore, int inBlock, cl_kernel kernel, cl_command_queue queue, cl_int error_check, cl_mem d_semaphor) {

		unsigned int maxNumberOfBlocks = min(XdivSHARED_X,YdivSHARED_Y);
		unsigned int startDecreaseAt = XdivSHARED_X+YdivSHARED_Y - maxNumberOfBlocks;
		unsigned int numberOfBlocks = 0;
		unsigned int x = XdivSHARED_X-1;
		unsigned int y = YdivSHARED_Y-1;
		size_t local[2];
		local[0] = WORKGROUP_X; //SHARED_X/WORKLOAD_X
		local[1] = WORKGROUP_Y; //SHARED_Y/WORKLOAD_Y

		for (unsigned int i=1; i < XdivSHARED_X+YdivSHARED_Y; ++i) {
				if (i <= maxNumberOfBlocks)
					numberOfBlocks = i;
				else if( i >= startDecreaseAt)
					numberOfBlocks = XdivSHARED_X+YdivSHARED_Y - i;
				else
					numberOfBlocks = maxNumberOfBlocks;

				size_t global[2];
				global[0] = NUMBER_SEQUENCES * WORKGROUP_X;
				global[1] = numberOfBlocks * NUMBER_TARGETS * WORKGROUP_Y;


				error_check = 0;
				error_check = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_matrix);
				error_check |= clSetKernelArg(kernel, 1, sizeof(unsigned int), &x);
				error_check |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &y);
				error_check |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &numberOfBlocks);
				error_check |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_globalMaxima);
				error_check |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &d_globalDirection);
				error_check |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &d_indexIncrement);
				error_check |= clSetKernelArg(kernel, 7, sizeof(cl_mem), &d_startingPoints);
				error_check |= clSetKernelArg(kernel, 8, sizeof(cl_mem), &d_maxPossibleScore);
				error_check |= clSetKernelArg(kernel, 9, sizeof(int), &inBlock);
				error_check |= clSetKernelArg(kernel, 10, sizeof(int), &d_semaphor);


				if(error_check != CL_SUCCESS) {
					fprintf(stderr,"Failed to set traceback kernel arguments");
					exit(EXIT_FAILURE);
				}

				error_check = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, NULL);
				if(error_check != CL_SUCCESS) {
					fprintf(stderr,"Distributing traceback kernel kernel has failed\n");
					exit(EXIT_FAILURE);
				}
				// Wait for the commands to complete
				clFinish(queue);


				if (y == 0)
					--x;
				if (y > 0)
					--y;
		}

}


void plotAlignments(char *sequences, char *targets, GlobalDirection *globalDirectionZeroCopy, unsigned int index, StartingPoints *startingPoints, int offset, int offsetTarget, char *descSequences, char *descTargets) {

	char *target = 0;
	char *sequence = 0;
	char *alignment = 0;

	target = (char*)malloc(sizeof(char)*2*(X+Y));
	sequence = (char*)malloc(sizeof(char)*2*(X+Y));
	alignment = (char*)malloc(sizeof(char)*2*(X+Y));

	if (!offset && !offsetTarget) {
		printf("QUERYLOCUS\tTARGETLOCUS\tRELATIVESCORE\tALIGNMENTLENGTH\tGAPS\tMATCHES\tQUERYSTART\tQUERYEND\tTARGETSTART\tTARGETEND\tSCORE\tTARGETNT\tALIGN\tQUERYNT\n");
	}


	float maxScore = 0;
	for (int i=0; i<index; i++){
			int alignmentLength = 0,
			gapsSeq = 0,
			gapsTarget = 0,
			matches = 0,
			mismatches = 0,
			qEnd = startingPoints->startingPoint[i].valueX,
			tEnd = startingPoints->startingPoint[i].valueY;

			maxScore = startingPoints->startingPoint[i].score > maxScore ? startingPoints->startingPoint[i].score : maxScore;

			long targetStartingPoint = startingPoints->startingPoint[i].target;
			long sequenceStartingPoint = startingPoints->startingPoint[i].sequence;

			long valueX = startingPoints->startingPoint[i].valueX;
			long valueY = startingPoints->startingPoint[i].valueY;
			long localIndex = 0;

			unsigned char direction = globalDirectionZeroCopy->direction[sequenceStartingPoint][targetStartingPoint].value[valueX][valueY];


			while (valueX >= 0 && valueY >= 0 && direction != STOP_DIRECTION && direction != NO_DIRECTION) {
				direction = globalDirectionZeroCopy->direction[sequenceStartingPoint][targetStartingPoint].value[valueX][valueY];
				alignmentLength++;
				switch (direction) {
					case UPPER_LEFT_DIRECTION:
						target[localIndex] = targets[targetStartingPoint*Y + valueY + offsetTarget*Y];
						sequence[localIndex] = sequences[sequenceStartingPoint*X + valueX + offset*X];
						alignment[localIndex] = target[localIndex] == sequence[localIndex] ? MATCH_CHAR : MISMATCH_CHAR;
						matches += target[localIndex] == sequence[localIndex] ? 1 : 0;
						mismatches += target[localIndex] != sequence[localIndex] ? 1 : 0;
						valueX--;
						valueY--;
						break;
					case LEFT_DIRECTION:
						gapsTarget++;
						target[localIndex] = GAP_CHAR_SEQ;
						sequence[localIndex] = sequences[sequenceStartingPoint*X + valueX + offset*X];
						alignment[localIndex] = GAP_CHAR_ALIGN;
						valueX--;
						break;
					case UPPER_DIRECTION:
						gapsSeq++;
						target[localIndex] = targets[targetStartingPoint*Y + valueY + offsetTarget*Y];
						sequence[localIndex] = GAP_CHAR_SEQ;
						alignment[localIndex] = GAP_CHAR_ALIGN;
						valueY--;
						break;
					case STOP_DIRECTION: // end of alignment
						target[localIndex] = targets[targetStartingPoint*Y + valueY + offsetTarget*Y];
						sequence[localIndex] = sequences[sequenceStartingPoint*X + valueX + offset*X];
						matches += target[localIndex] == sequence[localIndex] ? 1 : 0;
						alignment[localIndex] = target[localIndex] == sequence[localIndex] ? MATCH_CHAR : MISMATCH_CHAR;
						break;
					case NO_DIRECTION: // end of alignment
					default:// huh? should not be the case!
						fprintf(stderr, "Warning: wrong value in direction matrix: %d!\n", direction);
						valueX = -1;
						valueY = -1;
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
				  printf("%s\t%s\t", descSequences + ((startingPoints->startingPoint[i].sequence+offset) * MAX_LINE_LENGTH),
				  							 descTargets + ((startingPoints->startingPoint[i].target+offsetTarget) * MAX_LINE_LENGTH));
				  printf("%1.3f\t%i\t%i\t%i\t%i\t%i\t%i\t%i\t%1.0f\t%s\t%s\t%s\n",startingPoints->startingPoint[i].score/(float)alignmentLength,
					 alignmentLength, (gapsSeq + gapsTarget), matches, qEnd - alignmentLength+gapsSeq+1, qEnd+gapsSeq, tEnd - alignmentLength+gapsTarget+1, tEnd-gapsTarget, startingPoints->startingPoint[i].score, target, alignment, sequence
					 );


			}
	}


	free(target);
	free(sequence);
	free(alignment);
}
#endif



void fillScoringsMatrix(float *h_scoringsMatrix) {
#ifdef DNA_RNA
	readContentsScoringMatrix(h_scoringsMatrix, SCORINGS_MAT_SIZE, DNA_RNA_LOC);
#endif
#ifdef BLOSUM62
	readContentsScoringMatrix(h_scoringsMatrix, SCORINGS_MAT_SIZE, BLOSUM62_LOC);
#endif
#ifdef BASIC
	readContentsScoringMatrix(h_scoringsMatrix, SCORINGS_MAT_SIZE, BASIC_LOC);
#endif
}

void readContentsScoringMatrix(float *h_scoringsMatrix, int size, char *file_loc) {
	FILE *fp = 0;
	int i,j;
	fp = fopen(file_loc, "r");
	if(fp) {
		 for(i = 0; i < size; i++)
		 {
			  for(j = 0; j < size; j++)
			  {
				   if (!fscanf(fp, "%f", &h_scoringsMatrix[i*size+j]))
				   {
					   fprintf(stderr,"Cannot read float element\n");
					   exit(EXIT_FAILURE);
				   }
			  }
		 }
		 fclose(fp);
	}
	else {
		 printf("File Not Found: %s\n",file_loc);
		 exit(EXIT_FAILURE);
	}
}





