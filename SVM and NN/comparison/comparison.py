import tensorflow as tf
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

def plot_accuracies(accuracies, labels):
    x = range(len(accuracies))
    plt.bar(x, accuracies, align='center', alpha=0.7)
    plt.xticks(x, labels)
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f'comparison.png')

print("Loading Fashion-MNIST dataset...")

# Loading Fashion-MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

print("Filtering data points from classes 5 and 7...")

mask_train = (y_train == 5) | (y_train == 7)
mask_test = (y_test == 5) | (y_test == 7)

x_train, y_train = x_train[mask_train], y_train[mask_train]
x_test, y_test = x_test[mask_test], y_test[mask_test]

# Updating class labels for binary classification: class 5 -> class 0, class 7 -> class 1
y_train = np.where(y_train == 5, 0, 1)
y_test = np.where(y_test == 5, 0, 1)

# Set the probability to flip the labels
p = 0.2 

# Iterating through each label in y_train
for i in range(len(y_train)):
    # Generating a random value for each label
    random_value = np.random.rand()
    
    # If random value is less than p, flip the label
    if random_value < p:
        y_train[i] = 1 - y_train[i]  # Since labels are 0 and 1, (1 - y) label will flip the label

# Normalizing the pixel values
print("Normalizing the pixel values...")

x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshaping the data to be 1-D arrays
print("Reshaping the data to be 1-D arrays...")

x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

# Standardizing data
print("Standardizing data...")

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

sgd_svm_scores = -1
gaussian_svm_score = -1
nn_score = -1

# Gaussian SVM Training / Testing

print("Fitting Gaussian SVM on Data...")
gaussian_svm = SVC(C=0.7880462815669912, gamma=0.0016102620275609393)

gaussian_svm.fit(x_train, y_train)
pred_y = gaussian_svm.predict(x_test)
gaussian_svm_score = accuracy_score(y_test, pred_y)
print(f"Accuracy Score: {gaussian_svm_score}")

# SGD SVM Training / Testing

print("Fitting SGD SVM on Data...")
alpha = 1/(len(x_train) * 0.001)
sgd_svm = SGDClassifier(alpha=alpha)

sgd_svm.fit(x_train, y_train)
pred_y = sgd_svm.predict(x_test)
sgd_svm_score = accuracy_score(y_test, pred_y)
print(f"Accuracy Score: {sgd_svm_score}")

# Neural Network Training / Testing

print("Fitting NN on Data...")
hidden_layer_sizes = tuple([40] * 5)
nn = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation='logistic')

nn.fit(x_train, y_train)
pred_y = nn.predict(x_test)
nn_score = accuracy_score(y_test, pred_y)
print(f"Accuracy Score: {nn_score}")

accuracies = [gaussian_svm_score, sgd_svm_score, nn_score]
labels = ['Gaussian SVM', 'SGD SVM', 'NN']
plot_accuracies(accuracies, labels)