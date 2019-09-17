from codes.deepSVDD import DeepSVDD
from codes import *
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'


def main():
    from codes.utils import plot_most_normal_and_abnormal_images
    keras_model = cifar_lenet(128)
    svdd = DeepSVDD(keras_model, input_shape=(28, 28, 1), representation_dim=128)

    for cls in range(10):
        X_train, X_test, y_test = get_mnist(cls)

        svdd.fit(X_train, X_test, y_test, epochs=10)
        score = svdd.predict(X_test)

        plot_most_normal_and_abnormal_images(X_test, score)
        plt.savefig('results/mnist_%d.png' % cls)
        plt.close()


if __name__ == '__main__':
    main()
