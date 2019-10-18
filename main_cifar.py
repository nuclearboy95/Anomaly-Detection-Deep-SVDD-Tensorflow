from dsvdd import *
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from sklearn.metrics import roc_auc_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'


def main(cls=1):
    tf.reset_default_graph()
    from dsvdd.utils import plot_most_normal_and_abnormal_images
    # build model and DeepSVDD
    keras_model = cifar_lenet(128)
    keras_model.summary()
    svdd = DeepSVDD(keras_model, input_shape=(32, 32, 3), representation_dim=128,
                    objective='one-class')

    # get dataset
    X_train, X_test, y_test = get_cifar10(cls)

    # train DeepSVDD
    svdd.fit(X_train, X_test, y_test, epochs=10, verbose=True)

    # test DeepSVDD
    score = svdd.predict(X_test)
    auc = roc_auc_score(y_test, -score)
    print('AUROC: %.3f' % auc)

    plot_most_normal_and_abnormal_images(X_test, score)
    plt.show()


if __name__ == '__main__':
    main()
