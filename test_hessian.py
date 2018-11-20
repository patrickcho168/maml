import tensorflow as tf

x = tf.placeholder(tf.float32, shape=1)
y = x*x*x
grad = tf.gradients(y,x)
alpha = 1
x2 = x - alpha*grad
y2 = x2*x2*x2
grad2 = tf.gradients(y2, x)
with tf.Session() as sess:
    print(sess.run([y, grad, y2, grad2], feed_dict={x:[4]}))