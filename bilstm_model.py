'''
Bidirectional LSTM model with multiclass SVM option for NER system
'''
import numpy as np
import os
import time
import tensorflow as tf
from multiclass_svm import multiclass_svm
from preprocess import minibatches, pad_sequences, get_chunks

class BilstmModel():

    def __init__(self, config):
        self.config = config
        self.logger = config.logger
        self.sess   = None
        self.saver  = None
        self.idx_to_tag = {idx: tag for tag, idx in
                           self.config.vocab_tags.items()}

    # Initialization
    def initialize_session(self):
        self.logger.info("INITIALIZE NER SYSTEM...")
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    # Save session state
    def save_session(self):
        if not os.path.exists(self.config.dir_model):
            os.makedirs(self.config.dir_model)
        self.saver.save(self.sess, self.config.dir_model)

    # Shutdown session
    def close_session(self):
        self.sess.close()

    # Tensorflow Placeholders
    def add_placeholders(self):
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None],name="word_ids")
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None],name="sequence_lengths")
        self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None],name="char_ids")
        self.word_lengths = tf.placeholder(tf.int32, shape=[None, None],name="word_lengths")
        # tag_indices
        self.labels = tf.placeholder(tf.int32, shape=[None, None],name="labels")
        # hyper parameters
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[],name="lr")

    # Retrieve fed data
    def get_feed_dict(self, words, labels=None, lr=None, dropout=None):
        # perform padding of the given data
        if self.config.use_chars:
            char_ids, word_ids = zip(*words)
            word_ids, sequence_lengths = pad_sequences(word_ids, 0)
            char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0,
                nlevels=2)
        else:
            word_ids, sequence_lengths = pad_sequences(words, 0)

        # build feed dictionary
        feed = {
            self.word_ids: word_ids,
            self.sequence_lengths: sequence_lengths
        }

        if self.config.use_chars:
            feed[self.char_ids] = char_ids
            feed[self.word_lengths] = word_lengths

        if labels is not None:
            labels, _ = pad_sequences(labels, 0)
            feed[self.labels] = labels

        # lr : learning rate
        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout

        return feed, sequence_lengths


    # Add pretrained word vector
    def add_word_embeddings_op(self):
        with tf.variable_scope("words"):
            if self.config.embeddings is None:
                _word_embeddings = tf.get_variable(
                        name="_word_embeddings",
                        dtype=tf.float32,
                        shape=[self.config.nwords, self.config.dim_word])
            else:
                _word_embeddings = tf.Variable(
                        self.config.embeddings,
                        name="_word_embeddings",
                        dtype=tf.float32,
                        trainable=self.config.train_embeddings)

            word_embeddings = tf.nn.embedding_lookup(_word_embeddings,
                    self.word_ids, name="word_embeddings")

        with tf.variable_scope("chars"):
            if self.config.use_chars:
                # get char embeddings matrix
                _char_embeddings = tf.get_variable(
                        name="_char_embeddings",
                        dtype=tf.float32,
                        shape=[self.config.nchars, self.config.dim_char])
                char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
                        self.char_ids, name="char_embeddings")

                # put the time dimension on axis=1
                s = tf.shape(char_embeddings)
                char_embeddings = tf.reshape(char_embeddings,
                        shape=[s[0]*s[1], s[-2], self.config.dim_char])
                word_lengths = tf.reshape(self.word_lengths, shape=[s[0]*s[1]])

                # bilstm on chars
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char, state_is_tuple=True)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char, state_is_tuple=True)
                _output = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, char_embeddings, sequence_length=word_lengths, dtype=tf.float32)

                # read and concat output
                _, ((_, output_fw), (_, output_bw)) = _output
                output = tf.concat([output_fw, output_bw], axis=-1)

                # shape = (batch size, max sentence length, char hidden size)
                output = tf.reshape(output, shape=[s[0], s[1], 2*self.config.hidden_size_char])
                word_embeddings = tf.concat([word_embeddings, output], axis=-1)

        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout)


    # Label for prediction
    # def add_pred_op(self):
    #     if self.config.use_svm:
    #         self.labels_pred = self.logits
    #
    #     self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)


    def build(self):
        # NER specific functions
        self.add_placeholders()
        self.add_word_embeddings_op()

        # Add vector scores
        # dimension : number of tags
        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            (output_fw, output_bw), output_state = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, self.word_embeddings,
                    sequence_length=self.sequence_lengths, dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, self.dropout)

        with tf.variable_scope("proj"):
            W = tf.get_variable("W", dtype=tf.float32,shape=[2*self.config.hidden_size_lstm, self.config.ntags])

            b = tf.get_variable("b", shape=[self.config.ntags],dtype=tf.float32, initializer=tf.zeros_initializer())

            nsteps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2*self.config.hidden_size_lstm])
            pred = tf.matmul(output, W) + b

            # Classification inputs : [batch_size, max_seq_len, num_tags]
            self.logits = tf.reshape(pred, [-1, nsteps, self.config.ntags])

        if self.config.use_svm:
            self.labels_pred = self.logits

        self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)

        # Option for binary SVM
        if self.config.use_svm:
            svm_labels = tf.one_hot(self.labels, self.config.ntags)
            hinge_loss = tf.contrib.losses.hinge_loss(labels=svm_labels, logits=self.logits)
            mask = tf.sequence_mask(self.sequence_lengths)
            hinge_loss = tf.boolean_mask(hinge_loss, mask)
            self.loss = tf.reduce_mean(hinge_loss)

        # Option for multiclass SVM
        if self.config.use_multiclass_svm:
            multi_hinge_logits = tf.reshape(self.logits, [-1, self.config.ntags])
            multiclass_hinge_loss = multiclass_svm(labels=self.labels, logits=multi_hinge_logits)
            self.loss = tf.reduce_sum(multiclass_hinge_loss)

        # Option for softmax
        if self.config.use_softmax:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)

        tf.summary.scalar("loss", self.loss)

        # Generic functions that add training op and initialize session
        self.add_train_op(self.config.lr_method, self.lr, self.loss,self.config.clip)
        self.initialize_session()


    # words: list of sentences
    # Generate prediction batches
    def predict_batch(self, words):
        fd, sequence_lengths = self.get_feed_dict(words, dropout=1.0)
        labels_pred = self.sess.run(self.labels_pred, feed_dict=fd)
        return labels_pred, sequence_lengths


    # Run training and evaluation on dataset
    # train, dev: dataset
    def run_epoch(self, train, dev, epoch):
        batch_size = self.config.batch_size
        nbatches = (len(train) + batch_size - 1) // batch_size

        # iterate over dataset
        for i, (words, labels) in enumerate(minibatches(train, batch_size)):
            fd, _ = self.get_feed_dict(words, labels, self.config.lr,
                    self.config.dropout)

            _, train_loss, summary = self.sess.run(
                    [self.train_op, self.loss, self.merged], feed_dict=fd)

            # tensorboard
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch*nbatches + i)

        metrics = self.run_evaluate(dev)
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                for k, v in metrics.items()])
        self.logger.info(msg)


        return metrics["f1"]    # return F1 score

    # Evaluation
    # test: test dataset
    def run_evaluate(self, test):
        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.
        p, r, f1 = 0., 0., 0.
        for words, labels in minibatches(test, self.config.batch_size):
            labels_pred, sequence_lengths = self.predict_batch(words)

            for lab, lab_pred, length in zip(labels, labels_pred, sequence_lengths):

                lab = lab[:length]
                lab_pred = lab_pred[:length]
                accs += [a==b for (a, b) in zip(lab, lab_pred)]

                lab_chunks = set(get_chunks(lab, self.config.vocab_tags))
                lab_pred_chunks = set(get_chunks(lab_pred, self.config.vocab_tags))

                correct_preds += len(lab_chunks & lab_pred_chunks)
                total_preds += len(lab_pred_chunks)
                total_correct += len(lab_chunks)

        print("total_preds : ", total_preds)
        print("correct_preds : ", correct_preds)
        print("total_correct : ", total_correct)

        if correct_preds > 0:
            # Precision
            p = correct_preds / total_preds

            # Recall
            r = correct_preds / total_correct

            # F1 score
            f1 = 2 * p * r / (p + r)
            acc = np.mean(accs)

            # Tensorboard visualization
            tf.summary.scalar("accuracy", acc)
            tf.summary.scalar("f1", f1)

        else:
            acc = f1 = 0.

        print("Precision : ", p)
        print("Recall : ", r)

        return {"acc": 100*acc, "f1": 100*f1}


    # Tags for words in sentences
    def predict(self, words_raw):
        words = [self.config.processing_word(w) for w in words_raw]
        if type(words[0]) == tuple:
            words = zip(*words)
        pred_ids, _ = self.predict_batch([words])
        preds = [self.idx_to_tag[idx] for idx in list(pred_ids[0])]

        return preds


    # performs an update on a batch
    # lr: learning rate
    # lr_method: sgd method name
    # clip: clipping of gradient. If < 0, no clipping
    def add_train_op(self, lr_method, lr, loss, clip=-1):
        _lr_m = lr_method.lower() # lower to make sure

        with tf.variable_scope("train_step"):
            # Use AdamOptimizer
            optimizer = tf.train.AdamOptimizer(lr)

            if clip > 0: # gradient clipping if clip is positive
                grads, vs     = zip(*optimizer.compute_gradients(loss))
                grads, gnorm  = tf.clip_by_global_norm(grads, clip)
                self.train_op = optimizer.apply_gradients(zip(grads, vs))
            else:
                self.train_op = optimizer.minimize(loss)

    # Summary for tensorboard
    # Merge all the summaries and write them out
    def add_summary(self):
        self.merged      = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.config.dir_output,
                self.sess.graph)


    # Training based on train (sentences, tags) and dev dataset
    def train(self, train, dev):

        best_score = 0
        nepoch_no_imprv = 0
        self.add_summary()

        for epoch in range(self.config.nepochs):
            print("TRAINING EPOCH ", epoch+1, "...")
            startTime = time.time()
            score = self.run_epoch(train, dev, epoch)
            self.config.lr *= self.config.lr_decay

            # early stopping and saving best parameters
            if score >= best_score:
                nepoch_no_imprv = 0
                self.save_session()
                best_score = score
            else:
                nepoch_no_imprv += 1
                if nepoch_no_imprv >= self.config.nepoch_no_imprv:
                    break

            endTime = time.time()
            elapsedTime = endTime - startTime
            print("Time : ", elapsedTime)
            print("TRAINING EPOCH ", epoch, " FINISHED\n")

    # Evaluation based on test dataset
    def evaluate(self, test):
        self.logger.info("TESTING MODEL...")
        metrics = self.run_evaluate(test)
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                for k, v in metrics.items()])
        self.logger.info(msg)

    # Load trained weight data
    def restore_session(self, dir_model):
        self.logger.info("RESTORE MODEL...")
        self.saver.restore(self.sess, dir_model)