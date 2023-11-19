from sklearn.linear_model import SGDClassifier
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

def plot(filename):
    plt.figure()  # Creating a figure without specifying the size

    # Plotting the data points and lines
    plt.plot(c_values, train_errors, marker='o', linestyle='--', linewidth=2, label='Train Error')
    plt.plot(c_values, test_errors, marker='o', linestyle='-', linewidth=2, label='Test Error')
    plt.xscale('log')
    plt.xlabel('Regularization Parameter C')
    plt.ylabel('Error')
    plt.title('Error vs Regularization Parameter C')
    
    # Positioning the legend outside of the plot on the right
    legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Automatically adjusting the layout to accommodate the legend
    plt.tight_layout()
    plt.savefig(filename, bbox_extra_artists=(legend,), bbox_inches='tight')

print("Loading Fashion-MNIST dataset...")

# Loading Fashion-MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Cut the sets for faster computation
reduced_data_n = 2
mid_idx = len(x_train) // reduced_data_n
x_train, y_train = x_train[:mid_idx], y_train[:mid_idx]
x_test, y_test = x_test[:mid_idx], y_test[:mid_idx]

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

best_c_score = -1
best_c = -1
c_values = []
average_scores = []

num_values = 30
c_values = np.logspace(-3, 3, num=num_values) # Produces values between 10^-3 and 10^3

k = 5 # > 1

print("Variables Initialized...")

# Create subsets for k-fold validation
subset_size = len(x_train) // k
train_subsets_X = [x_train[i*subset_size:(i+1)*subset_size] for i in range(k)]
train_subsets_y = [y_train[i*subset_size:(i+1)*subset_size] for i in range(k)]

print(f"C Values to Consider: \n{c_values}")

print("Starting k-fold cross-validation...")

# K Fold Cross-validation
for i, c in enumerate(c_values):
    print(f"{i+1}/{len(c_values)}")
    print(f"C = {c}")
    validation_scores = []
    for i in range(k):
        print(f"Testing on fold {i+1}...")
        # Concatenate k-1 subsets for training, use one for validation
        train_X = np.concatenate([train_subsets_X[j] for j in range(k) if j != i], axis=0)
        train_y = np.concatenate([train_subsets_y[j] for j in range(k) if j != i], axis=0)
        valid_X = train_subsets_X[i]
        valid_y = train_subsets_y[i]

        # Convert C values to alpha values for SGD
        n = len(train_X)
        alpha = 1/(n * c)
        
        classifier = SGDClassifier(alpha=alpha)
        classifier.fit(train_X, train_y)
        pred_y = classifier.predict(valid_X)
        score = accuracy_score(valid_y, pred_y)
        validation_scores.append(score)

    # Average the validation scores for this ensemble size
    average_score = np.mean(validation_scores)
    average_scores.append(average_score)
    print(f"Average validation: {average_score}")

    if average_score > best_c_score:
        best_c_score = average_score
        best_c = c

print(f"Best C found: {best_c}")

# Plotting
alpha_values = [1/(n * c) for c in c_values]
train_errors = [1 - accuracy_score(y_train, SGDClassifier(alpha=alpha).fit(x_train, y_train).predict(x_train)) for alpha in alpha_values]
test_errors = [1 - accuracy_score(y_test, SGDClassifier(alpha=alpha).fit(x_train, y_train).predict(x_test)) for alpha in alpha_values]

plot('sgd_error_plot.png')
