import tensorflow as tf
import numpy as np

val = 3
m = tf.placeholder(tf.int32)
m_feed = np.array([[val, 0, val, 0, val],
          [val, 0, 0, 0, val],
          [val, 0, val, 0, 0]])

print np.shape(m_feed)
print np.shape(m_feed)[1] - np.argmax(m_feed[:, ::-1], axis=1) - 1
# tmp_indices = tf.where(tf.equal(tf.less(2, m), True))
# result = tf.reduce_max(tmp_indices)
#
# with tf.Session() as sess:
#     print(sess.run([result, tmp_indices], feed_dict={m: m_feed}))  # [2, 0, 1]
