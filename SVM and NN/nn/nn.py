from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Function to train and get errors
def get_errors(num_layers, nodes, activation_function):
    train_errors, test_errors = [], []
    
    for node in nodes:
        hidden_layer_sizes = tuple([node] * num_layers)
        
        # Training the model
        classifier = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation_function)
        classifier.fit(x_train, y_train)
        
        # Getting the training error
        train_pred = classifier.predict(x_train)
        train_errors.append(1 - accuracy_score(y_train, train_pred))
        
        # Getting the test error
        test_pred = classifier.predict(x_test)
        test_errors.append(1 - accuracy_score(y_test, test_pred))
        
    return train_errors, test_errors

def plot_errors(num_hidden_layers, nodes_per_layer, activation_function):
    plt.figure()  # Letting the figure size adjust automatically
    
    colors = ['b', 'r', 'g', 'm', 'y']
    for idx, num_layer in enumerate(num_hidden_layers):
        train_errors, test_errors = get_errors(num_layer, nodes_per_layer, activation_function)
        
        plt.plot(nodes_per_layer, train_errors, f'{colors[idx]}--', label=f'Train Error ({num_layer} layers)')
        plt.plot(nodes_per_layer, test_errors, f'{colors[idx]}-', label=f'Test Error ({num_layer} layers)')
    
    plt.xlabel('Number of Nodes')
    plt.ylabel('Error')
    plt.title(f'Error vs Number of Nodes with {activation_function} Activation')
    
    # Positioning the legend outside of the plot to the right
    legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.xticks(nodes_per_layer)
    
    # Saving the figure with automatic size adjustment to include the legend
    plt.savefig(f'nn_error_plot_{activation_function}.png', bbox_extra_artists=(legend,), bbox_inches='tight')

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

# Define hyperparameters to grid search
num_hidden_layers = list(range(1, 6))  # Number of hidden layers to consider
nodes_per_layer = list(range(10, 101, 10))  # Number of nodes per layer to consider
activation_functions = ['logistic', 'relu', 'tanh']
best_score = -1  # Initialize the best score
validation_scores = []

num_folds = 5

print("Variables Initialized...")

# Create subsets for k-fold validation
subset_size = len(x_train) // num_folds
train_subsets_X = [x_train[i*subset_size:(i+1)*subset_size] for i in range(num_folds)]
train_subsets_y = [y_train[i*subset_size:(i+1)*subset_size] for i in range(num_folds)]

print(f"Number of Hidden Layers to Consider: \n{num_hidden_layers}")
print(f"Number of Nodes per Layer to Consider: \n{nodes_per_layer}")
print(f"Activation Functions to Consider: \n{activation_functions}")

print("Starting k-fold cross-validation...")

# Iterating over hyperparameters
for k, num_layer in enumerate(num_hidden_layers):
    for j, nodes in enumerate(nodes_per_layer):
        for i, activation_function in enumerate(activation_functions):
            hidden_layer_size = tuple([nodes] * num_layer)  # Creates a tuple based on the number of layers and nodes
            validation_scores = []
            
            print(f"{k+1}/{len(num_hidden_layers)} Hidden Layer Parameters")
            print(f"{j+1}/{len(nodes_per_layer)} Node Parameters")
            print(f"{i+1}/{len(activation_functions)} Activation Functions")

            print(f"{num_layer} hidden layers, {nodes} nodes per layer, activation function = {activation_function}")

            for fold in range(num_folds):
                # Print the current hyperparameters being considered
                print(f"Testing on fold {fold+1}...")
            
                # Concatenate subsets for training, use one for validation
                train_X = np.concatenate([train_subsets_X[j] for j in range(num_folds) if j != fold], axis=0)
                train_y = np.concatenate([train_subsets_y[j] for j in range(num_folds) if j != fold], axis=0)
                valid_X = train_subsets_X[fold]
                valid_y = train_subsets_y[fold]

                # Modified the classifier definition to use hyperparameters
                classifier = MLPClassifier(hidden_layer_sizes=hidden_layer_size, activation=activation_function)
                classifier.fit(train_X, train_y)
                pred_y = classifier.predict(valid_X)
                
                score = accuracy_score(valid_y, pred_y)  # Calculate accuracy as the score
                validation_scores.append(score)  # Save scores

            average_score = np.mean(validation_scores)
            print(f"Average validation score: {average_score}")
           
            # If the score is the best found so far, save the hyperparameters
            if average_score > best_score:
                best_score = average_score
                best_hyperparameters = {
                    'num_hidden_layers': num_layer,
                    'nodes_per_layer': nodes,
                    'activation_function': activation_function
                }

print(f"Best hyperparameters: {best_hyperparameters} with validation score: {best_score}")

# Iterating over activation functions for plotting
for activation_function in activation_functions:
    plot_errors(num_hidden_layers, nodes_per_layer, activation_function)