BLSTM-SVM Named Entity Recognition System


1. Download & unzip GloVe pre-trained word vector files under glove.6B directory
   (http://nlp.stanford.edu/data/glove.6B.zip)


2. Change parameters (eg. training epoch, embedding dimension) in param.py


3. Build and train model using command:

	   python3 model_train.py
    
   Training process takes a lot of time since each training epoch takes 500 to 600 seconds
   on average to finish. Change nepochs in param.py to reduce total training time.


4. Evaluate model using command:

	   python3 model_evaluate.py