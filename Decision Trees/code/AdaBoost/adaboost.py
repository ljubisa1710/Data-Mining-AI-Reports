from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import os

print("Starting the program...")

# Reading data and preparing folder
data = pd.read_csv('../Data/spambase.data', header=None)
print("Data read successfully.")

graph_folder = '../graphs/adaboost'
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

max_depth = [1, 5, 10, 25, 50, 100]
num_trees = [5, 10, 20, 50, 100]
test_sizes = [0.2, 0.3, 0.4, 0.5]
colour_list = ['b', 'g', 'r', 'c', 'm', 'y']

print("Entering the first loop...")
for scenario, parameter_list, param_name in [
    ("max_depth", max_depth, "Max Depth of The Tree"),
    ("n_estimators", num_trees, "Number of Trees")
]:
    print(f"Running scenario with varying {param_name}...")
    
    plt.figure(figsize=(12, 8))
    color_iterator = iter(colour_list)

    for param_value in parameter_list:
        current_color = next(color_iterator)
        test_accuracies = []
        train_accuracies = []
        print(f"  Evaluating with {param_name} = {param_value}...")
        
        for test_size in test_sizes:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            if (scenario == "max_depth"):
                clf = AdaBoostClassifier(DecisionTreeClassifier(**{scenario: param_value}))
            elif (scenario == "n_estimators"):
                clf = AdaBoostClassifier(DecisionTreeClassifier(), **{scenario: param_value})

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
    
    print(f"Plot saved for varying {param_name}.")

print("First loop complete.")

print("Entering the second loop...")
for scenario, parameter_list, param_name in [
    ("max_depth", max_depth, "Max Depth of The Tree"),
    ("n_estimators", num_trees, "Number of Trees")
]:
    print(f"Running scenario with varying Test Sizes for {param_name}...")
    
    plt.figure(figsize=(12, 8))
    color_iterator = iter(colour_list)

    for test_size in test_sizes:
        current_color = next(color_iterator)
        test_accuracies = []
        train_accuracies = []
        print(f"  Evaluating with Test Size = {test_size}...")

        for param_value in parameter_list:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            if (scenario == "max_depth"):
                clf = AdaBoostClassifier(DecisionTreeClassifier(**{scenario: param_value}))
            elif (scenario == "n_estimators"):
                clf = AdaBoostClassifier(DecisionTreeClassifier(), **{scenario: param_value})

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

    print(f"Plot saved for varying Test Sizes for {param_name}.")

print("Second loop complete.")

print("Generating 3D plot...")
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
z_values = []
x_values = []
y_values = []

for depth in max_depth:
    for trees in num_trees:
        clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=depth), n_estimators=trees)
        clf.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, clf.predict(X_test))

        x_values.append(depth)
        y_values.append(trees)
        z_values.append(accuracy)

ax.scatter(x_values, y_values, z_values, c='r', marker='o')

ax.set_xlabel('Max Depth')
ax.set_ylabel('Number of Trees')
ax.set_zlabel('Accuracy')

plt.savefig(os.path.join(graph_folder, '3D_Plot_Accuracy_vs_MaxDepth_and_NumTrees.png'))
print("3D plot saved.")

print("Program complete.")
