import tensorflow as tf


if __name__ == "__main__":
    sess = tf.Session()

    tf_ones = tf.ones([2,2])
    tf_zeros = tf.zeros([2,2])
    tf_sample1 = tf.constant([0,3,4,5], shape=[2,2])
    tf_sample2 = tf.constant([1, 1, 1, 1], shape=[2, 2])
    print(sess.run(tf.matmul(tf_sample1, tf_sample2)))