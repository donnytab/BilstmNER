'''
Load dataset and train BSLTM-SVM model
'''
from preprocess import get_vocabs, UNK, NUM, get_glove_vocab, write_vocab, load_vocab, get_char_vocab, export_trimmed_glove_vectors, get_processing_word
from preprocess import DatasetHandler
from bilstm_model import NERModel
from param import Config

if __name__ == "__main__":
    # Load configs
    config = Config(load=False)
    target_word = get_processing_word(lowercase=True)

    # Target words from datasets
    dataset_dev   = DatasetHandler(config.conll_dev, target_word)
    dataset_test  = DatasetHandler(config.conll_test, target_word)
    dataset_train = DatasetHandler(config.conll_train, target_word)

    # Build Word and Tag vocab
    vocab_words, vocab_tags = get_vocabs([dataset_train, dataset_dev, dataset_test])
    vocab_glove = get_glove_vocab(config.output_glove)
    vocab = vocab_words & vocab_glove
    vocab.add(UNK)
    vocab.add(NUM)

    # Process vocab
    write_vocab(vocab, config.output_words)
    write_vocab(vocab_tags, config.output_tags)
    vocab = load_vocab(config.output_words)
    export_trimmed_glove_vectors(vocab, config.output_glove,config.output_trimmed, config.dim_word)

    # Build and save char vocab
    train = DatasetHandler(config.conll_train)
    vocab_chars = get_char_vocab(train)
    write_vocab(vocab_chars, config.output_chars)


    # build model
    train_config = Config()
    model = NERModel(train_config)
    model.build()
    # model.restore_session("results/crf/model.weights/") # optional, restore weights
    # model.reinitialize_weights("proj")

    # create datasets
    dev = DatasetHandler(train_config.conll_dev, train_config.processing_word,train_config.processing_tag, train_config.max_iter)
    train = DatasetHandler(train_config.conll_train, train_config.processing_word,train_config.processing_tag, train_config.max_iter)

    # train model
    model.train(train, train)
