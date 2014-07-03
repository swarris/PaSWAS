import sys
import math

from Bio import  SeqIO
from Bio.Seq import Seq

mem_size_matrix = 32*8
mem_fill_factor = 0.9
mem_size_gpu = int(sys.argv[3])*1024*1024


seqList1 = list(SeqIO.parse(open(sys.argv[1],"r"), "fasta"))
seqList2 = list(SeqIO.parse(open(sys.argv[2],"r"), "fasta"))

length_sequences = int(math.ceil((len(seqList1[0])/8.0)*8.0))
length_targets = int(math.ceil((len(seqList2[0])/8.0)*8.0))
                               
number_targets = int(math.floor(math.sqrt((mem_size_gpu * mem_fill_factor) / ((length_targets * length_targets * (mem_size_matrix) +
                            (length_targets * length_targets * 32) /
                            (64))))))

number_sequences = int(math.floor(mem_size_gpu * mem_fill_factor) / ((length_sequences * length_targets * (mem_size_matrix) +
                            (length_sequences * length_targets * 32) /
                            (64) * number_targets)))


print("#define NUMBER_SEQUENCES {}\n#define X ({}) ".format(number_sequences, int(math.ceil(len(seqList1[0]) / 8.0)*8.0)))
print("#define NUMBER_TARGETS {}\n#define Y ({})\n".format(number_targets, int(math.ceil(len(seqList2[0]) / 8.0)*8.0)))

print("superBlockX -> {}\nsuperBlockY -> {}\n".format(int(math.ceil(len(seqList1)/number_sequences))+1, int(math.ceil(len(seqList2)/number_targets))+1))