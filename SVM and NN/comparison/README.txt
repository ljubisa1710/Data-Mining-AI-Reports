This script is designed to perform model comparison on a subset of the Fashion-MNIST dataset using three different classifiers: Gaussian SVM, SGD SVM, and a Neural Network (NN).

Dependencies:

    TensorFlow
    scikit-learn
    NumPy
    Matplotlib

Functionality:

    The script starts by loading the Fashion-MNIST dataset.
    Filters and preprocesses the data, focusing on classes 5 and 7 for binary classification.
    Trains and evaluates the Gaussian SVM, SGD SVM, and Neural Network (NN) classifiers.
    Outputs the accuracy scores of each model.
    Generates a bar chart that visualizes the comparison of model accuracies.

Execution:
Run the script in a Python environment by using the command python comparison.py.

Output:

    Text outputs of accuracy scores in the console.
    A bar chart saved as 'comparison.png', comparing the accuracy of the three models.