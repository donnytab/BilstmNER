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
    ner_model = NERModel(config)
    ner_model.build()

    test = CoNLLDataset(config.filename_test, config.processing_word,config.processing_tag, config.max_iter)
    ner_model.evaluate(test)