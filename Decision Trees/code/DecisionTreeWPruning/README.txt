Decision Tree Classifier for SpamBase Data

Basic Function

This Python script performs Decision Tree classification on the SpamBase dataset using both 'entropy' and 'gini' as splitting criteria. 
The script evaluates the performance of the model using various test sizes and Complexity Control Parameter (CCP) alpha values. 
It generates and saves visualizations of the Decision Trees and line graphs depicting model accuracy.

How It Runs

1. The script reads the SpamBase data from a CSV file.
2. The data is split into training and testing sets based on different test sizes.
3. A Decision Tree Classifier is trained using different criteria ('gini' and 'entropy') and various CCP alpha values.
4. The script calculates and stores the accuracy of each model configuration in separate folders based on the criterion used.
5. Visualizations (Decision Trees and line graphs) are generated and saved as PNG files.

How to Run

1. Make sure you have Python 3.x and the required libraries (pandas, scikit-learn, matplotlib) installed.
2. Place your 'spambase.data' file in the ../Data/ folder.
3. Run the Python script.

Files Created

- Line graphs depicting model accuracy based on different criteria are saved as PNG files in the ./graphs/line_graph/ folder.