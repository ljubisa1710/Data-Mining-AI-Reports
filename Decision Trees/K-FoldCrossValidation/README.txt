Ensemble Classifiers for SpamBase Data
Basic Function

This Python script performs ensemble classification on the SpamBase dataset using RandomForest and AdaBoost classifiers. 
It utilizes k-fold cross-validation to tune the ensemble sizes for both classifiers and evaluates their performance on a test set. 
The script also generates and saves visualizations of the average validation scores for different ensemble sizes.

How It Runs

    The script reads the SpamBase data from a CSV file.
    The data is preprocessed and encoded using LabelEncoder.
    The data is split into training and testing sets.
    k-fold cross-validation is performed to find the optimal ensemble size for RandomForest and AdaBoost classifiers.
    The script calculates and stores the average validation score for each ensemble size.
    Visualizations (line graphs and bar graphs) are generated and saved as PNG files.

How to Run

    Make sure you have Python 3.x and the required libraries (pandas, scikit-learn, matplotlib, os) installed.
    Place your 'spambase.data' file in the ../Data/ folder.
    Run the Python script.

Files Created

    Line graphs depicting average validation scores for different ensemble sizes are saved as PNG files in the ../graphs/k_fold/ folder.
    Bar graphs comparing the test scores of the best ensemble sizes for RandomForest and AdaBoost are also saved in the same folder.

Additional Notes

    The script uses a fixed random state for reproducibility.
    The ensemble sizes range from 1 to 100, with increments of 1.
    The script prints out progress updates during the k-fold cross-validation process.