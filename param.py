'''
Config Parameters for NER model
'''
import os
import logging
from preprocess import get_trimmed_glove_vectors, load_vocab, get_processing_word

class Config():
    # dimensions of word and char embeddings
    dim_word = 300
    dim_char = 100
    hidden_size_char = 100
    hidden_size_lstm = 300

    ######## Manual Config Setup ########

    # GloVe Path setup
    output_glove = "glove.6B/glove.6B.{}d.txt".format(dim_word)
    output_trimmed = "glove.6B.{}d.trimmed.npz".format(dim_word)

    # Decoding layer options
    use_multiclass_svm = True
    use_softmax = False
    use_svm = False

    # Training hyperparameters
    train_embeddings = False
    nepochs = 20                # Number of epoch : 10, 15, 20 for experiment. 20 gives the best performance.
    dropout = 0.5
    batch_size = 30
    lr = 0.001                  # Learning rate
    lr_decay = 0.9
    clip = -1
    nepoch_no_imprv = 3

    #####################################

    # general config
    dir_output = "results/test/"
    dir_model = dir_output + "model.weights/"
    path_log = dir_output + "log.txt"

    use_pretrained = True
    use_chars = True
    max_iter = None

    # dataset
    conll_dev = "CoNLL2003/valid.txt"
    conll_test = "CoNLL2003/test.txt"
    conll_train = "CoNLL2003/train.txt"

    # Output files
    output_words = "context/output_words.txt"
    output_tags = "context/output_tags.txt"
    output_chars = "context/output_chars.txt"

    def __init__(self, load=True):
        # directory for training outputs
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

        self.logger = getLogger(self.path_log)

        if load:
            self.load()

    def load(self):
        # Load vocabulary
        self.vocab_words = load_vocab(self.output_words)
        self.vocab_tags = load_vocab(self.output_tags)
        self.vocab_chars = load_vocab(self.output_chars)

        self.nwords = len(self.vocab_words)
        self.nchars = len(self.vocab_chars)
        self.ntags = len(self.vocab_tags)

        # Get processing words
        self.processing_word = get_processing_word(self.vocab_words,self.vocab_chars, lowercase=True, chars=self.use_chars)
        self.processing_tag = get_processing_word(self.vocab_tags, lowercase=False, allow_unk=False)

        # Get pre-trained embeddings
        self.embeddings = (get_trimmed_glove_vectors(self.output_trimmed)
                           if self.use_pretrained else None)



def getLogger(filename):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(
            '%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)

    return logger