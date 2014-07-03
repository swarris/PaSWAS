import sys

from Bio import  SeqIO
from Bio.Seq import Seq
from Bio.SeqIO import FastaIO

max_length = int(sys.argv[3])
seqList = list(SeqIO.parse(open(sys.argv[1],"r"), sys.argv[2]))
fOut = FastaIO.FastaWriter(open(sys.argv[4], "w"), wrap=None)

seqList.sort(key=lambda seqIO : len(seqIO.seq), reverse=True) 
fOut.write_file(filter(lambda seq : len(seq) <= max_length, seqList))
