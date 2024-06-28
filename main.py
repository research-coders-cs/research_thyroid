import transduction  # 'research-ai-transduction' pkg (i.e. 'src/transduction/*')

from transduction import example_mask  # !!!!

from transduction.helper import show_example
from transduction.vocab import load_tokenizers, load_vocab  # ok with torch 1.11 (enforced by `torchdata==0.3.0`); err with torch 2.3.0

import logging
logger = logging.getLogger('@@')
logger.setLevel(level=logging.DEBUG if 1 else logging.INFO)

# `mr` when 'src/transduction' is intact
# `mt` when 'src/transduction' is modified

if __name__ == '__main__':

    print("__main__: ^^")

    if 1:  # Encoder/Decoder part
        print("!!!! showme", example_mask())


    if 0:
        # global variables used later in the script
        spacy_de, spacy_en = show_example(load_tokenizers)
        vocab_src, vocab_tgt = show_example(load_vocab, args=[spacy_de, spacy_en])
