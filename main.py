from codes.deepSVDD import DeepSVDD
from codes import *


def main():
    X_train, X_test, y_test = get_cifar10(9)
    keras_model = cifar_lenet(128)
    svdd = DeepSVDD(keras_model, input_shape=X_train.shape[1:], representation_dim=128)
    svdd.fit(X_train, X_test, y_test, epochs=10)
    score = svdd.predict(X_test)

    from codes.utils import plot_most_normal_and_abnormal_images
    plot_most_normal_and_abnormal_images(X_test, score)


if __name__ == '__main__':
    main()
