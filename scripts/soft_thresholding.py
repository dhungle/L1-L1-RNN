# import tensorflow as tf
import numpy as np 
import torch 
import torch.nn.functional as functional
import matplotlib.pyplot as plt

class Soft_TF:
    def soft_pos(self, z, ld1, ld2, alpha):
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


    def soft_neg(self, z, ld1, ld2, alpha):
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

    def soft_l1_l1(self, z, ld1, ld2, alpha):
        return tf.where(tf.greater_equal(alpha, 0), self.soft_pos(z, ld1, ld2, alpha), self.soft_neg(z, ld1, ld2, alpha))

    def test(self):
        x = np.linspace(-20, 20, 1001, dtype=np.float32)
        z = tf.placeholder(tf.float32)

        y_tf = self.soft_l1_l1(z, tf.constant(3.0), tf.constant(-1.0), tf.constant(1.0))

        with tf.Session() as sess:
            y_np = sess.run(y_tf, feed_dict={z:x})

        plt.plot(x, y_np)
        plt.show()
    
class Soft_Torch:
    def soft_l1(self, z, w0):
        out = torch.sign(z) * functional.relu(torch.abs(z) - w0)
        return out

    def soft_l1_l1_reweighted(self, z, ld0, ld1, alpha0, alpha1, alpha2):
        epsilon = 1e-6
        w0 = 1./(torch.abs(alpha1) + epsilon)
        w1 = 1./(torch.abs(alpha2 - alpha1) + epsilon)
        w0_reweighted = ld0 * z.numel() * w0 / torch.sum(w0)
        w1_reweighted = ld1 * z.numel() * w1 / torch.sum(w1)

        condition = alpha0 <= alpha1
        alpha0_sorted = torch.where(condition, alpha0, alpha1)
        alpha1_sorted = torch.where(condition, alpha1, alpha0)
        w0_sorted = torch.where(condition, w0_reweighted, w1_reweighted)
        w1_sorted = torch.where(condition, w1_reweighted, w0_reweighted)
        
        cond1 = z >= alpha1_sorted + w0_sorted + w1_sorted
        cond2 = z >= alpha1_sorted + w0_sorted - w1_sorted
        cond3 = z >= alpha0_sorted + w0_sorted - w1_sorted
        cond4 = z >= alpha0_sorted - w0_sorted - w1_sorted

        res1 = z - w0_sorted - w1_sorted 
        res2 = alpha1_sorted
        res3 = z - w0_sorted + w1_sorted 
        res4 = alpha0_sorted
        res5 = z + w0_sorted + w1_sorted 
        
        return torch.where(cond1, res1, torch.where(cond2, res2, torch.where(cond3, res3, torch.where(cond4, res4, res5))))

    def soft_l1_l1(self, z, w0, w1, alpha0, alpha1):
        condition = alpha0 <= alpha1
        alpha0_sorted = torch.where(condition, alpha0, alpha1)
        alpha1_sorted = torch.where(condition, alpha1, alpha0)
        w0_sorted = torch.where(condition, w0, w1)
        w1_sorted = torch.where(condition, w0, w1)

        cond1 = z >= alpha1_sorted + w0_sorted + w1_sorted
        cond2 = z >= alpha1_sorted + w0_sorted - w1_sorted
        cond3 = z >= alpha0_sorted + w0_sorted - w1_sorted
        cond4 = z >= alpha0_sorted - w0_sorted - w1_sorted

        res1 = z - w0_sorted - w1_sorted
        res2 = alpha1_sorted
        res3 = z - w0_sorted + w1_sorted
        res4 = alpha0_sorted
        res5 = z + w0_sorted + w1_sorted
        return torch.where(cond1, res1,
                           torch.where(cond2, res2, torch.where(cond3, res3, torch.where(cond4, res4, res5))))


    def soft_l1_l1_slow(self, z, w0, w1, alpha0, alpha1):
        z_flat = z.view(z.numel())
        ans = torch.Tensor(z_flat.size())
        alpha0 = alpha0.view(alpha0.numel())
        alpha1 = alpha1.view(alpha1.numel())
        for i in range(ans.size()[0]):
            if alpha0[i] <= alpha1[i]:
                if z_flat[i] > alpha1[i] + w0[i] + w1[i]:
                    ans[i] = z_flat[i] - w0[i] - w1[i]
                elif z_flat[i] >= alpha1[i] + w0[i] - w1[i]:
                    ans[i] = alpha1[i]
                elif z_flat[i] >= alpha0[i] + w0[i] - w1[i]:
                    ans[i] = z_flat[i] - w0[i] + w1[i]
                elif z_flat[i] >= alpha0[i] - w0[i] - w1[i]:
                    ans[i] = alpha0[i]
                else:
                    ans[i] = z_flat[i] + w0[i] + w1[i]
            else:
                if z_flat[i] > alpha0[i] + w0[i] + w1[i]:
                    ans[i] = z_flat[i] - w0[i] - w1[i]
                elif z_flat[i] >= alpha0[i] - w0[i] + w1[i]:
                    ans[i] = alpha0[i]
                elif z_flat[i] >= alpha1[i] - w0[i] + w1[i]:
                    ans[i] = z_flat[i] + w0[i] - w1[i]
                elif z_flat[i] >= alpha1[i] - w0[i] - w1[i]:
                    ans[i] = alpha1[i]
                else:
                    ans[i] = z_flat[i] + w0[i] + w1[i]
        return ans.view(z.size())
        
    def test_soft_l1_l1(self):
        z = torch.linspace(-20, 20, 1001)
        
        for _ in range(10):
            w0 = torch.tensor(np.random.randint(0, 10, size=1)[0], dtype=torch.float32).repeat(z.size())
            w1 = torch.tensor(np.random.randint(0, 10, size=1)[0], dtype=torch.float32).repeat(z.size())
            alpha0 = torch.ones(z.size(), dtype=torch.float32) * np.random.randint(-10, 10, size=1)[0].astype('float') * 0.0
            alpha1 = torch.ones(z.size(), dtype=torch.float32) * np.random.randint(-10, 10, size=1)[0].astype('float')

            print('w0: {}, w1: {}, alpha0: {}, alpha1: {}'.format(w0, w1, alpha0, alpha1))
            y1 = self.soft_l1_l1(z, w0, w1, alpha0, alpha1)
            y2 = self.soft_l1_l1_slow(z, w0, w1, alpha0, alpha1)
            plt.subplot(2, 1, 1)
            plt.plot(z.data.numpy(), y1.data.numpy())
            plt.title('after l1_l1')

            plt.subplot(2, 1, 2)
            plt.plot(z.data.numpy(), y2.data.numpy())
            plt.title('after l1')
            plt.show()


if __name__ == "__main__":
    soft_tf = Soft_Torch()
    soft_tf.test_soft_l1_l1()