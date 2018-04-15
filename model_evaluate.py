'''
Evaluate performance of BLSTM-SVM model
'''
from param import Config
from bilstm_model import BilstmModel
from preprocess import DatasetHandler

'''
tensorflow visualization
input command: tensorboard --logdir="results/test/"
tensorboard at: http://localhost:6006
'''

if __name__ == "__main__":
    # Load configs
    config = Config()

    # build model
    model = BilstmModel(config)
    model.build()
    model.restore_session(config.dir_model)

    # build testing dataset
    test = DatasetHandler(config.conll_test, config.processing_word,config.processing_tag, config.max_iter)

    # Evaluate testing set
    model.evaluate(test)