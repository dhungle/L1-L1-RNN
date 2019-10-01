import torch 
import numpy as np 
import tensorflow as tf 

import tensorflow as tf
class Soft_TF:
    def soft_pos(z, ld1, ld2, alpha):
        cond1 = tf.greater(z, alpha + ld1 + ld2)
        cond2 = tf.math.logical_and(tf.greater_equal(z, alpha + ld1 - ld2), tf.greater_equal(ld1 + alpha + ld2, z))
        cond3 = tf.math.logical_and(tf.math.greater(z, ld1 - ld2), tf.math.greater(alpha + ld1 - ld2, z))
        cond4 = tf.math.logical_and(tf.math.greater_equal(ld1 - ld2, z), tf.math.greater_equal(z, 0 - ld1 - ld2))
        cond5 = tf.math.greater(0 - ld1 - ld2, z)
        f1 = lambda: z - ld1 - ld2
        f2 = lambda: z - z + alpha
        f3 = lambda: z - ld1 + ld2
        f4 = lambda: tf.zeros(shape=tf.shape(z))
        f5 = lambda: z + ld1 + ld2
        return tf.where(cond1, f1(), tf.where(cond2, f2(), tf.where(cond3, f3(), tf.where(cond4, f4(), f5()))))


    def soft_neg(z, ld1, ld2, alpha):
        cond1 = tf.greater(z, ld1 + ld2)
        cond2 = tf.math.logical_and(tf.greater_equal(ld1 + ld2, z), tf.greater_equal(z, 0 - ld1 + ld2))
        cond3 = tf.math.logical_and(tf.greater(ld2 - ld1, z), tf.greater(z, alpha - ld1 + ld2))
        cond4 = tf.math.logical_and(tf.greater_equal(alpha - ld1 + ld2, z), tf.greater_equal(z, alpha - ld1 - ld2))
        f1 = lambda: z - ld1 - ld2
        f2 = lambda: tf.zeros(shape=tf.shape(z))
        f3 = lambda: z + ld1 - ld2
        f4 = lambda: z - z + alpha
        f5 = lambda: z + ld1 + ld2
        return tf.where(cond1, f1(), tf.where(cond2, f2(), tf.where(cond3, f3(), tf.where(cond4, f4(), f5()))))

    def soft_l1(z, ld1, ld2, alpha):
        return tf.where(tf.greater_equal(alpha, 0), soft_pos(z, ld1, ld2, alpha), soft_neg(z, ld1, ld2, alpha))

    def test():
        x = np.linspace(-20, 20, 1001, dtype=np.float32)
        z = tf.placeholder(tf.float32)

        y_tf = soft_l1(z, tf.constant(3.0), tf.constant(5.0), tf.constant(0.0))

        with tf.Session() as sess:
            y_np = sess.run(y_tf, feed_dict={z:x})

        import matplotlib.pyplot as plt
        plt.plot(x, y_np)
        plt.show()

if __name__ == "__main__":
    Soft_TF().test()

