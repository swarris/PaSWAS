Parallel Smith-Waterman Alignment Software
==========================================

This project contains the CUDA code and some python tools for PaSWAS.
PaSWAS can be used to align DNA, RNA and protein sequences. It runs on any NVIDIA-based
GPU. The (optional) Python tools require Python 2.7+ and BioPython.

A full python-based application, which is easier to use, can be downloaded from:
http://code.google.com/p/pypaswas

Prerequisites
-------------

To compile and run PaSWAS you need:
- NVIDIA GPU, compute capability 1.2 (almost all GPUs based on GT200 (released 2008) and higher)
- CUDA development kit: https://developer.nvidia.com (CUDA 5+)
- C++ compiler
- Make

To use the Python tools, which are optional, you need:
- Python 2.7+
- BioPython

The application is known to run on different versions of Ubuntu and Debian running on either servers, desktops or laptops. 
The above packages are readily available for these distribution, so we recommend using such a system.

Installation
------------

Get the source from GIT:
git clone https://github.com/swarris/PaSWAS.git

Go to PaSWAS/onGPU and edit:
- makefile: 'gcc -L/usr/local/cuda-6.0/lib64' should point to your cuda libs
- subdir.mk: 'nvcc --ptxas-options=-v -keep -arch=compute_30 -I/usr/local/cuda-6.0/samples/common/inc/' should 
point to your cuda includes. Also, replace compute_30 with the architecture of your GPU, in this case 3.0 (see http://en.wikipedia.org/wiki/CUDA).
 
The example data are located in the ./data folder. Please unzip the files before using them. 

Prepare your data
-----------------

PaSWAS is capable of reading FASTA files, other formats are not possible. See pyPaSWAS for more options. 
The sequences need to be ordered by length as well. Sequences longer than max_length will be removed.

Use this to convert your data:
python ./tools/prepare.py input type max_length output.fasta 

For example:
python ./tools/prepare.py ./data/primers.fna fasta 300 /tmp/primers.fa
python ./tools/prepare.py ./data/454_primers.fna fasta 300 /tmp/454_primers.fa

You need to set-up the length of the longest sequence in the code and get the super block values. Get these numbers by running:
python ./tools/count.py [sequences.fasta] [targets.fasta] [mem on gpu]
where [mem on gpu] gives the memory on the GPU in megabytes, for example: 512.

For example:
python ../tools/count.py /tmp/454_primers.fa /tmp/primers.fa 512

 
Compiling the code
------------------

Open in your favorite text editor the file PaSWAS/definition.h
Edit these values, outputed by count.py:
#define NUMBER_SEQUENCES 25
#define NUMBER_TARGETS 25
#define X (640) 
#define Y (1280) 

You can also edit these based on your own data. Make sure that X and Y are multiples of 8.

Set this to filter out low scoring alignments:
#define MINIMUM_SCORE 200.0

Go to PaSWAS/onGPU and run:
make clean; make

This should end with: 
'Finished building target: paswas'

Running the program
-------------------

The PaSWAS application is located in the onGPU folder. The command line options are:
./paswas device sequenceFile targetFile superBlocksX superBlocksY

- device: select the GPU in the system. Starts at 1. Useful when there are more GPUs present.
- sequenceFile: name of the fasta file 
- targetFile: name of the fasta file
- superBlocksX, superBlocksY: you need to split up the analysis of the fasta files in 'super blocks'. Memory on the GPU is limited, therefore sending all sequences to the GPU is usually not possible. Use the tools/count.py tool to get numbers on these. See: Preparing your data

Example command line:
./paswas 1 /tmp/454_primers.fa /tmp/primers.fa 1480 1 > /tmp/hits


 


