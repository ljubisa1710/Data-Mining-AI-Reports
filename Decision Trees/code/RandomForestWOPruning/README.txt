Random Forest Classifier for SpamBase Data

Basic Function

This Python script performs Random Forest classification on the SpamBase dataset using a variety of hyperparameters: Number of Trees, Max Features, and Max Samples. It evaluates the performance of the model using various test sizes and generates line graphs depicting model accuracy for different configurations.

How It Runs

1. The script reads the SpamBase data from a CSV file.
2. The data is split into training and testing sets based on different test sizes.
3. A Random Forest Classifier is trained using different combinations of Number of Trees, Max Features, and Max Samples.
4. The script calculates the accuracy for each model configuration.
5. Line graphs visualizing the accuracy metrics are generated and saved as PNG files in two different formats:
    - One with "Test Size" on the X-axis and various parameter values as different lines.
    - Another with parameter values on the X-axis and "Test Size" as different lines.

How to Run

1. Make sure you have Python 3.x and the required libraries (pandas, scikit-learn, matplotlib) installed.
2. Place your 'spambase.data' file in the `../Data/` folder.
3. Run the Python script.

Files Created

- Line graphs depicting model accuracy for varying Number of Trees, Max Features, and Max Samples are saved as PNG files in the `./graphs` folder. Two types of graphs are generated for each parameter:
    - One with "Test Size" as the X-axis (`<parameter_name>_Variable_X.png`)
    - Another with the parameter values as the X-axis (`<parameter_name>_By_TestSize.png`)