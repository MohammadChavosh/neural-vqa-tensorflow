import tensorflow as tf
import numpy as np

val = 3
m = tf.placeholder(tf.int32)
m_feed = np.array([[val, 0, val, 0, val],
          [0, 0, 0, 0, 0],
          [val, 0, val, 0, 0]])

# tmp_indices = tf.where(tf.equal(tf.less(2, m), True))
result = m + tf.expand_dims(tf.maximum(2 - tf.reduce_max(m, axis=1), 0), 1)
tmp_indices = tf.where(tf.equal(tf.less(1, result), True))
number_prediction = tf.segment_max(tmp_indices[:, 1], tmp_indices[:, 0])

with tf.Session() as sess:
    print(sess.run([result, number_prediction], feed_dict={m: m_feed}))  # [2, 0, 1]
