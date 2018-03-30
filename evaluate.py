from param import Config
from bilstm_model import NERModel
from preprocess import CoNLLDataset

'''
tensorflow visualization
input command: tensorboard --logdir="results/test/"
tensorboard at: http://localhost:6006
'''


if __name__ == "__main__":
    config = Config()
    model = NERModel(config)
    model.build()

    # create datasets
    dev   = CoNLLDataset(config.filename_dev, config.processing_word,
                         config.processing_tag, config.max_iter)
    train = CoNLLDataset(config.filename_train, config.processing_word,
                         config.processing_tag, config.max_iter)

    # train model
    model.train(train, dev)

    test = CoNLLDataset(config.filename_test, config.processing_word,config.processing_tag, config.max_iter)
    model.evaluate(test)