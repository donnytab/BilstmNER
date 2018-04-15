from param import Config
from bilstm_model import NERModel
from preprocess import DatasetHandler

'''
tensorflow visualization
input command: tensorboard --logdir="results/test/"
tensorboard at: http://localhost:6006
'''

if __name__ == "__main__":
    config = Config()
    model = NERModel(config)
    model.build()
    model.restore_session(config.dir_model)

    test = DatasetHandler(config.output_test, config.processing_word,
                        config.processing_tag, config.max_iter)
    model.evaluate(test)