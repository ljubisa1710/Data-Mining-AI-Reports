import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import os

print("Starting the program...")

# Reading data and preparing folder
data = pd.read_csv('../Data/spambase.data', header=None)
print("Data read successfully.")

graph_folder = '../graphs/random_forest'
if not os.path.exists(graph_folder):
    os.makedirs(graph_folder)

# Encoding categorical variables
for column in data.columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])

print("Data encoding complete.")

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

print("Variable initialization complete.")

test_sizes = [0.2, 0.3, 0.4, 0.5]
num_trees = [5, 10, 20, 50, 100]
max_features_list = ['sqrt', 'log2', 0.25, 0.5, 0.75, 0.99]
max_samples = [1/5, 1/4, 1/3, 1/2, 1/1]
color_list = ['b', 'g', 'r', 'c', 'm', 'y']

# First scenario: varying parameters
for scenario, parameter_list, param_name in [
    ("n_estimators", num_trees, "Number of Trees"),
    ("max_features", max_features_list, "Max Features"),
    ("max_samples", max_samples, "Max Samples")
]:
    print(f"Running scenario with varying {param_name}...")
    plt.figure(figsize=(12, 8))
    color_iterator = iter(color_list)

    # Loop through each parameter value in the scenario
    for param_value in parameter_list:
        current_color = next(color_iterator)
        test_accuracies = []
        train_accuracies = []
        print(f"  Evaluating with {param_name} = {param_value}...")

        # Loop through each test size
        for test_size in test_sizes:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            if scenario == "max_samples":
                samples = int((X_test.shape[0] - 1) * param_value)
                clf = RandomForestClassifier(max_samples=samples, random_state=42)
            else:
                clf = RandomForestClassifier(**{scenario: param_value}, random_state=42)

            clf.fit(X_train, y_train)
            test_accuracies.append(accuracy_score(y_test, clf.predict(X_test)))
            train_accuracies.append(accuracy_score(y_train, clf.predict(X_train)))

        plt.plot(test_sizes, test_accuracies, linestyle='-', color=current_color, label=f"Test Accuracy, {param_name} = {param_value}")
        plt.plot(test_sizes, train_accuracies, linestyle='--', color=current_color, label=f"Train Accuracy, {param_name} = {param_value}")

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.title(f"Random Forest Accuracy for Different {param_name}")
    plt.xlabel("Test Size")
    plt.ylabel("Accuracy")
    plt.savefig(os.path.join(graph_folder, f"{param_name}_Variable_X.png"), bbox_inches='tight')
    plt.close()

# Second scenario: varying test sizes
for scenario, parameter_list, param_name in [
    ("n_estimators", num_trees, "Number of Trees"),
    ("max_features", max_features_list, "Max Features"),
    ("max_samples", max_samples, "Max Samples")
]:
    print(f"Running scenario with varying Test Sizes for {param_name}...")
    plt.figure(figsize=(12, 8))
    color_iterator = iter(color_list)

    # Loop through each test size
    for test_size in test_sizes:
        print(f"  Evaluating with Test Size = {test_size}...")
        current_color = next(color_iterator)
        test_accuracies = []
        train_accuracies = []

        # Loop through each parameter value in the scenario
        for param_value in parameter_list:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            if scenario == "max_samples":
                samples = int((X_test.shape[0] - 1) * param_value)
                clf = RandomForestClassifier(max_samples=samples, random_state=42)
            else:
                clf = RandomForestClassifier(**{scenario: param_value}, random_state=42)

            clf.fit(X_train, y_train)
            test_accuracies.append(accuracy_score(y_test, clf.predict(X_test)))
            train_accuracies.append(accuracy_score(y_train, clf.predict(X_train)))

        plt.plot(parameter_list, test_accuracies, linestyle='-', color=current_color, label=f"Test Accuracy, Test Size = {test_size}")
        plt.plot(parameter_list, train_accuracies, linestyle='--', color=current_color, label=f"Train Accuracy, Test Size = {test_size}")

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.title(f"Random Forest Accuracy for Different {param_name}")
    plt.xlabel(f"{param_name}")
    plt.ylabel("Accuracy")
    plt.savefig(os.path.join(graph_folder, f"{param_name}_By_TestSize.png"), bbox_inches='tight')
    plt.close()

print("All scenarios completed.")
