# import tensorflow as tf
#
#
# if __name__ == "__main__":
#     sess = tf.Session()
#
#     tf_ones = tf.ones([2,2])
#     tf_zeros = tf.zeros([2,2])
#     tf_sample1 = tf.constant([0,3,4,5], shape=[2,2])
#     tf_sample2 = tf.constant([1, 1, 1, 1], shape=[2, 2])
#     print(sess.run(tf.matmul(tf_sample1, tf_sample2)))

import tensorflow as tf
#batch_size = 2
labels = tf.constant([[0, 0, 0, 1],[0, 1, 0, 0]])
logits = tf.constant([[-3.4, 2.5, -1.2, 5.5],[-3.4, 2.5, -1.2, 5.5]])

loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
loss_s = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(labels,1), logits=logits)

with tf.Session() as sess:
    print("softmax loss:", sess.run(loss))
    print("sparse softmax loss:", sess.run(loss_s))