import pdb
import os
import sys
from collections import OrderedDict

def read_labels(labels_file):
    with open(labels_file) as fin:
        count = 1
        for term in fin:
            term = term.strip("\n")
            term = term.split()
            if (term[0] == "_empty_" or term[0] == "_singletons_"):
                continue
            full_arr = []
            for i in range(len(term)):
                if (i <= 1):
                    full_arr.append("OTHER")
                else:
                    full_arr.append(term[i])
            ret_str = ' '.join(full_arr)
            print(ret_str)

def main():
    read_labels(sys.argv[1])

if __name__ == '__main__':
    main()

