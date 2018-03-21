import tensorflow as tf
import numpy as np
import re
from scipy import sparse

# corpus_raw = 'He is the king . The king is royal . She is the royal  queen '
corpus_raw = ""
file = open("CoNLL2003/train.txt", "r")
if file:
    for row in file:
        word = row.split(" ", 1)[0]
        if word != "-DOCSTART-":
            corpus_raw = corpus_raw + word + " "

# print(corpus_raw)
print("Embedding Finished...")

# convert to lower case
corpus_raw = corpus_raw.lower()

words = []
regex = r'\b\w+\b'
processed_corpus = re.findall(regex, corpus_raw);
for word in processed_corpus:
    if word != '.': # because we don't want to treat . as a word
        words.append(word)

words = set(words) # so that all duplicate words are removed
word2int = {}
int2word = {}
vocab_size = len(words) # gives the total number of unique words
print("vocab: ")
print(vocab_size)

with open('WordID.txt', 'w') as out:
    for i,word in enumerate(words):
            word2int[word] = str(i)
            int2word[i] = str(word)
            out.writelines(str(i) + " " + str(word) + "\n")

# print("625")
# print(word2int["625"])
# print(int2word["3451"])

# raw sentences is a list of sentences.
# raw_sentences = re.split('. | ! | ? ', corpus_raw);
raw_sentences = corpus_raw.split('. ')
sentences = []
for sentence in raw_sentences:
    sentences.append(re.findall(regex, sentence))

WINDOW_SIZE = 2

data = []
for sentence in sentences:
    for word_index, word in enumerate(sentence):
        for nb_word in sentence[max(word_index - WINDOW_SIZE, 0) : min(word_index + WINDOW_SIZE, len(sentence)) + 1] :
            if nb_word != word:
                data.append([word, nb_word])

# function to convert numbers to one hot vectors
def to_one_hot(data_point_index, vocab_size):
    temp = np.zeros(vocab_size)
    temp[data_point_index] = 1
    return temp

x_train = [] # input word
y_train = [] # output word

print("word2int")
print(word2int)

for data_word in data:
    # print("dataword: ")
    # print(data_word)
    # print(data_word[0])
    # print(data_word[1])
    # print("")

    index_x = word2int[data_word[0]]
    index_y = word2int[data_word[1]]
    x_train.append(to_one_hot(int(index_x), vocab_size))
    y_train.append(to_one_hot(int(index_y), vocab_size))

print("x_train")
print("y_train")

# convert them to numpy arrays
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

print("Finish asarray")

# making placeholders for x_train and y_train
x = tf.placeholder(tf.float32, shape=(None, vocab_size))
y_label = tf.placeholder(tf.float32, shape=(None, vocab_size))

print("Finish placeholder")

EMBEDDING_DIM = 5 # you can choose your own number
W1 = tf.Variable(tf.random_normal([vocab_size, EMBEDDING_DIM]))
b1 = tf.Variable(tf.random_normal([EMBEDDING_DIM])) #bias
hidden_representation = tf.add(tf.matmul(x,W1), b1)

W2 = tf.Variable(tf.random_normal([EMBEDDING_DIM, vocab_size]))
b2 = tf.Variable(tf.random_normal([vocab_size]))
prediction = tf.nn.softmax(tf.add( tf.matmul(hidden_representation, W2), b2))

print("Finish prediction")

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init) #make sure you do this!

print("Start reduce_mean")
# define the loss function:
cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction), reduction_indices=[1]))

# define the training step:
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy_loss)

n_iters = 10000
# train for n_iter iterations

print("Start n_iters")
for _ in range(n_iters):
    sess.run(train_step, feed_dict={x: x_train, y_label: y_train})
    #print('loss is : ', sess.run(cross_entropy_loss, feed_dict={x: x_train, y_label: y_train}))

vectors = sess.run(W1 + b1)

print("Finish session")

def euclidean_dist(vec1, vec2):
    return np.sqrt(np.sum((vec1-vec2)**2))

def find_closest(word_index, vectors):
    min_dist = 10000 # to act like positive infinity
    min_index = -1
    query_vector = vectors[word_index]
    for index, vector in enumerate(vectors):
        if euclidean_dist(vector, query_vector) < min_dist and not np.array_equal(vector, query_vector):
            min_dist = euclidean_dist(vector, query_vector)
            min_index = index
    return min_index


from sklearn.manifold import TSNE

model = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
vectors = model.fit_transform(vectors)

from sklearn import preprocessing

normalizer = preprocessing.Normalizer()
vectors =  normalizer.fit_transform(vectors, 'l2')

with open("vector.txt", "w") as vec:
    print("Writing vectors...")
    vectorList = str(vectors).split(',')
    for wordVec in vectorList:
        vec.write(wordVec)
    print("Finish vectors output")
# print(vectors)

import matplotlib.pyplot as plt


fig, ax = plt.subplots()
# print(words)
for word in words:
    # print(word, vectors[word2int[word]][1])
    ax.annotate(word, (vectors[word2int[word]][0],vectors[word2int[word]][1] ))
plt.show()