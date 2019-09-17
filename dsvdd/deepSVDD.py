import tensorflow as tf
import numpy as np
from math import ceil
from sklearn.metrics import roc_auc_score

from .utils import task


class DeepSVDD:
    def __init__(self, keras_model, input_shape=(28, 28, 1), objective='one-class',
                 nu=0.1, representation_dim=32, batch_size=128):
        self.represetation_dim = representation_dim
        self.objective = objective
        self.keras_model = keras_model
        self.nu = nu
        self.R = tf.get_variable('R', [], dtype=tf.float32, trainable=False)
        self.c = tf.get_variable('c', [self.represetation_dim], dtype=tf.float32, trainable=False)
        self.warm_up_n_epochs = 10
        self.batch_size = batch_size

        with task('Build graph'):
            self.x = tf.placeholder(tf.float32, [None] + list(input_shape))
            self.latent_op = self.keras_model(self.x)
            self.dist_op = tf.reduce_sum(tf.square(self.latent_op - self.c), axis=-1)

            if self.objective == 'soft-boundary':
                self.score_op = self.dist_op - self.R ** 2
                penalty = tf.maximum(self.score_op, tf.zeros_like(self.score_op))
                self.loss_op = self.R ** 2 + (1 / self.nu) * penalty

            else:  # one-class
                self.score_op = self.dist_op
                self.loss_op = self.score_op

            opt = tf.train.AdamOptimizer()
            self.train_op = opt.minimize(self.loss_op)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

    def __del__(self):
        self.sess.close()

    def fit(self, X, X_test, y_test, epochs=10):
        N = X.shape[0]
        BS = self.batch_size
        BN = int(ceil(N / BS))

        self.sess.run(tf.global_variables_initializer())
        self._init_c(X)

        for i_epoch in range(epochs):
            ind = np.random.permutation(N)
            x_train = X[ind]
            for i_batch in range(BN):
                x_batch = x_train[i_batch * BS: (i_batch + 1) * BS]
                ops = {
                    'train': self.train_op,
                    'loss': tf.reduce_mean(self.loss_op),
                    'dist': self.dist_op
                }
                results = self.sess.run(ops, feed_dict={self.x: x_batch})

                if self.objective == 'soft-boundary' and i_epoch >= self.warm_up_n_epochs:
                    self.sess.run(tf.assign(self.R, self._get_R(results['dist'], self.nu)))

            else:
                pred = self.predict(X_test)  # pred: large->fail small->pass
                auc = roc_auc_score(y_test, -pred)  # Y: 1->pass 0->fail
                print('Epoch: %3d AUC: %.3f' % (i_epoch, auc))

    def predict(self, X):
        N = X.shape[0]
        BS = self.batch_size
        BN = int(ceil(N / BS))
        scores = list()
        for i_batch in range(BN):
            x_batch = X[i_batch * BS: (i_batch + 1) * BS]
            s_batch = self.sess.run(self.score_op, feed_dict={self.x: x_batch})
            scores.append(s_batch)

        return np.concatenate(scores)

    def _init_c(self, X, eps=1e-1):
        N = X.shape[0]
        BS = self.batch_size
        BN = int(ceil(N / BS))

        with task('1. Get output'):
            latent_sum = np.zeros(self.latent_op.shape[-1])
            for i_batch in range(BN):
                x_batch = X[i_batch * BS: (i_batch + 1) * BS]
                latent_v = self.sess.run(self.latent_op, feed_dict={self.x: x_batch})
                latent_sum += latent_v.sum(axis=0)

            c = latent_sum / N

        with task('2. Modify eps'):
            c[(abs(c) < eps) & (c < 0)] = -eps
            c[(abs(c) < eps) & (c > 0)] = eps

        self.sess.run(tf.assign(self.c, c))

    def _get_R(self, dist, nu):
        return np.quantile(np.sqrt(dist), 1 - nu)
