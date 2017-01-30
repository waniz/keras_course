from keras.datasets import cifar10
import matplotlib.pyplot as plt
from scipy.misc import toimage  # install pillow in case of mistake 'pip install pillow'


(X_train, y_train), (X_test, y_test) = cifar10.load_data()

for i in range(1, 10):
    plt.subplot(3, 3, 0 + i)
    plt.imshow(toimage(X_train[i]))

plt.show()

