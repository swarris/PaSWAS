all: no_partitioner partitioner_alg2_threshold partitioner_alg_threshold
		
no_partitioner: no_partitioner.c kseq.h
	$(CC) -g -O2 no_partitioner.c -o no_partitioner -lm -lz

partitioner_alg_threshold: partitioner_alg_threshold.c kseq.h
	$(CC) -g -O2 partitioner_alg_threshold.c -o partitioner_alg_threshold -lm -lz
	
partitioner_alg2_threshold: partitioner_alg2_threshold.c kseq.h
	$(CC) -g -O2 partitioner_alg2_threshold.c -o partitioner_alg2_threshold -lm -lz

clean:
		rm -f *.o no_partitioner partitioner_alg_threshold partitioner_alg2_threshold 