export
CC=gcc
INCLUDE_DIR = ./include  
CFLAGS= -std=c99 -Wall $(DIRECTIVES) -I$(INCLUDE_DIR) -I${OPENCL_INCLUDE} -L${OPENCL_LIB} 
DIRECTIVES=$(MEM) $(SCORE) $(NUMBER_SEQUENCES) $(NUMBER_TARGETS) $(X) $(Y) $(MINIMUM_SCORE) $(SHARED_X) $(SHARED_Y) $(ITERATIONS) $(VENDOR) $(ZERO_COPY) ${WORKLOAD_X} ${WORKLOAD_Y}
DEPS=smithwaterman.h typedefs.h definitions.h kseq.h
#OBJ=gpuAlign.o smithwaterman.o
OBJ=gpuAlign.c smithwaterman.o
LIBS=-lOpenCL -lz

all: paswas_opencl

%.o: %.c $(DEPS)
	$(CC) $(CFLAGS) -c -o $@ $<

paswas_opencl: $(OBJ)
	$(CC) $(CFLAGS) -o  $@ $^ $(LIBS)

clean: 
	rm -f *.o paswas_opencl


	