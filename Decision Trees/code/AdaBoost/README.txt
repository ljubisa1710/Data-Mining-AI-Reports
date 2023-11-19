Basic Function

This Python script performs AdaBoost classification on the SpamBase dataset using Decision Trees as the base estimator. 
The script evaluates the performance of the model using various test sizes, maximum depths of the trees, and the number of trees in the ensemble. 
It generates and saves line graphs depicting model accuracy and a 3D plot for a combination of parameters.

How It Runs

    The script reads the SpamBase data from a CSV file.
    The data is encoded using Label Encoding.
    The data is split into training and testing sets based on different test sizes.
    An AdaBoost Classifier is trained using Decision Trees with varying maximum depths and number of trees.
    The script calculates and stores the accuracy of each model configuration.
    Line graphs and a 3D plot are generated and saved as PNG files.

How to Run

    Make sure you have Python 3.x and the required libraries (pandas, scikit-learn, matplotlib) installed.
    Place your 'spambase.data' file in the ../Data/ folder.
    Run the Python script.

Files Created

    Line graphs depicting model accuracy based on different maximum depths and number of trees are saved as PNG files in the ./graphs/adaboost/ folder.
    A 3D plot depicting model accuracy based on a combination of maximum depths and number of trees is also saved in the same folder.