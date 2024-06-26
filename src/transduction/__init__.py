
#---- ^^
import os
from os.path import exists
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
import math
import copy
import time
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
import altair as alt
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
import torchtext.datasets as datasets
import spacy
import GPUtil
import warnings
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


# Set to False to skip notebook execution (e.g. for debugging)
warnings.filterwarnings("ignore")
RUN_EXAMPLES = True
#---- $$

print("@@ torch.__version__:", torch.__version__)

#---- ^^
# Some convenience helper functions used throughout the notebook


# def is_interactive_notebook():
#     return __name__ == "__main__"


def show_example(fn, args=[]):
    #print("@@ __name__:", __name__)  # transduction

    # if __name__ == "__main__" and RUN_EXAMPLES:
    #     return fn(*args)
    #====
    if RUN_EXAMPLES:
        return fn(*args)


def execute_example(fn, args=[]):
    # if __name__ == "__main__" and RUN_EXAMPLES:
    #     fn(*args)
    #====
    if RUN_EXAMPLES:
        fn(*args)


class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None


class DummyScheduler:
    def step(self):
        None
#---- $$


#----
# Load spacy tokenizer models, download them if they haven't been
# downloaded already
def load_tokenizers():

    try:
        spacy_de = spacy.load("de_core_news_sm")
    except IOError:
        os.system("python -m spacy download de_core_news_sm")
        spacy_de = spacy.load("de_core_news_sm")

    try:
        spacy_en = spacy.load("en_core_web_sm")
    except IOError:
        os.system("python -m spacy download en_core_web_sm")
        spacy_en = spacy.load("en_core_web_sm")

    return spacy_de, spacy_en

#----
def tokenize(text, tokenizer):
    return [tok.text for tok in tokenizer.tokenizer(text)]


def yield_tokens(data_iter, tokenizer, index):
    for from_to_tuple in data_iter:
        yield tokenizer(from_to_tuple[index])

#----
def build_vocabulary(spacy_de, spacy_en):
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    if 1:#================ @@ https://github.com/pytorch/text/issues/1756#issuecomment-1889888918
        ##from torchtext.datasets import multi30k, Multi30k

        # Update URLs to point to data stored by user
        datasets.multi30k.URL["train"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
        datasets.multi30k.URL["valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"
        datasets.multi30k.URL["test"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/mmt_task1_test2016.tar.gz"

        # Update hash since there is a discrepancy between user hosted test split and that of the test split in the original dataset
        datasets.multi30k.MD5["test"] = "876a95a689a2a20b243666951149fd42d9bfd57cbbf8cd2c79d3465451564dd2"
    #================

    print("Building German Vocabulary ...")
    print("@@ 1111--aaaa")
    train, val, test = datasets.Multi30k(language_pair=("de", "en"))
    print("@@ 1111--bbbb")
    vocab_src = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_de, index=0),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    print("Building English Vocabulary ...")
    train, val, test = datasets.Multi30k(language_pair=("de", "en"))
    vocab_tgt = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_en, index=1),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    vocab_src.set_default_index(vocab_src["<unk>"])
    vocab_tgt.set_default_index(vocab_tgt["<unk>"])

    return vocab_src, vocab_tgt


def load_vocab(spacy_de, spacy_en):
    print("@@ 0000")
    if not exists("vocab.pt"):
        print("@@ 1111")
        vocab_src, vocab_tgt = build_vocabulary(spacy_de, spacy_en)
        print("@@ 2222")
        torch.save((vocab_src, vocab_tgt), "vocab.pt")
    else:
        vocab_src, vocab_tgt = torch.load("vocab.pt")
    print("Finished.\nVocabulary sizes:")
    print(len(vocab_src))
    print(len(vocab_tgt))
    return vocab_src, vocab_tgt
