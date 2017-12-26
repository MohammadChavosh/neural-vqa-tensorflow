import tensorflow as tf

val = 3
m = tf.placeholder(tf.int32)
m_feed = [val, 0, val, 0, val]

tmp_indices = tf.where(tf.equal(tf.less(2, m), True))
result = tf.reduce_min(tmp_indices)

with tf.Session() as sess:
    print(sess.run([result, tmp_indices], feed_dict={m: m_feed}))  # [2, 0, 1]
