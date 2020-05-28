import tensorflow as tf

x = tf.Variable(tf.random.uniform([2, 10], -1, 1))
print(x)


y = tf.split(x, num_or_size_splits=2, axis=1)
print(y)