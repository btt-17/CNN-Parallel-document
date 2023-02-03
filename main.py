import re
import argparse 
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import Counter
from train import *
from preprocess import *
from URLNet import *

def main(args):
    
    word_vocab_embed, char_vocab_embed = preprocess(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_len_chars",dest="max_len_chars", help="", default=None)
    parser.add_argument("--max_len_words",dest="max_len_words", help="", default=None)
    parser.add_argument("--vocabulary_directory", "--vc_dir", dest="vc_dir", default=None)
    parser.add_argument("--file_path",dest="file_path", help="",default=None)
    parser.add_argument("--embedding_dim", dest="embedding_dim", help="",default=32)
    parser.add_argument("--mode", dest="mode", help="",default=1)
    

    args = parser.parse_args()
    main(args)