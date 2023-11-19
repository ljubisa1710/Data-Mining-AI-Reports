Dependencies:

    scikit-learn
    TensorFlow
    NumPy
    Matplotlib

Functionality:

    The script processes a subset of the Fashion-MNIST dataset, focusing on binary classification between classes 5 and 7.
    Introduces label noise by flipping labels with a certain probability.
    Implements k-fold cross-validation for hyperparameter tuning of the SGDClassifier's regularization parameter C, which is inverse of alpha (C = (1 / n * alpha)​).
    Evaluates performance by computing the accuracy of predictions on validation sets.
    Generates error plots that visualize the effect of different regularization parameters on training and test errors.

Execution:
Execute the script in a Python environment using the command: python sgdclassifier_tuning.py.

Output:

    Console outputs detailing progress through the hyperparameter tuning process, including performance metrics for each fold and each configuration.
    An error plot saved as 'sgd_error_plot.png', visualizing the relationship between the regularization parameter and error rates on both the training and validation sets.