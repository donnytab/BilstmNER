BLSTM-SVM Named Entity Recognition System

1. Download & unzip GloVe pre-trained word vector file (glove.6B) http://nlp.stanford.edu/data/glove.6B.zip

2. Change path of glove.6B in param.py

3. Build and train model using command:

	python3 model_train.py
    
  Training process takes a lot of time since each training epoch takes 500 to 600 seconds on average to finish.

4. Evaluate model using command:

	python3 model_evaluate.py