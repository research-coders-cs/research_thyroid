
#----
#import wsdan  # 'research-thyroid-wsdan' pkg (i.e. 'src/wsdan/*')
#from wsdan.demo import test as demo_test
#from wsdan.demo import train as demo_train
#from wsdan.demo import train_with_doppler as demo_train_with_doppler
#from wsdan.digitake.preprocess import build_dataset
#----
import transduction  # 'research-ai-transduction' pkg (i.e. 'src/transduction/*')
from transduction import show_example
from transduction import load_tokenizers, load_vocab
#----

import logging
logger = logging.getLogger('@@')
logger.setLevel(level=logging.DEBUG if 1 else logging.INFO)

if __name__ == '__main__':

    print("hello")

    if 1:
        # global variables used later in the script
        spacy_de, spacy_en = show_example(load_tokenizers)
        vocab_src, vocab_tgt = show_example(load_vocab, args=[spacy_de, spacy_en])

