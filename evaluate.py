from param import Config
from bilstm_model import NERModel
from preprocess import CoNLLDataset

if __name__ == "__main__":
    config = Config()
    ner_model = NERModel(config)
    ner_model.build()
    ner_model.restore_session()

    test = CoNLLDataset(config.filename_test, config.processing_word,config.processing_tag, config.max_iter)
    ner_model.evaluate()
    #interactive_shell(model)