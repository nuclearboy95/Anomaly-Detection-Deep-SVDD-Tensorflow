from dsvdd import *
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'


def main():
    from dsvdd.utils import plot_most_normal_and_abnormal_images
    keras_model = mnist_lenet(32)
    svdd = DeepSVDD(keras_model, input_shape=(28, 28, 1), representation_dim=32)

    X_train, X_test, y_test = get_mnist(1)

    svdd.fit(X_train, X_test, y_test, epochs=10)
    score = svdd.predict(X_test)

    plot_most_normal_and_abnormal_images(X_test, score)
    plt.show()


if __name__ == '__main__':
    main()
