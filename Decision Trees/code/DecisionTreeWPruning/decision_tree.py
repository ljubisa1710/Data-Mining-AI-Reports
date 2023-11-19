import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import os

print("Starting the program...")

# Reading data and preparing folder
data = pd.read_csv('../Data/spambase.data', header=None)
print("Data read successfully.")

graph_folder = '../graphs/decision_tree'
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

# Initialize dictionary to store results
results = {}
# Define a color map for CCP Alpha values
color_map = {0.0: 'b', 0.001: 'g', 0.01: 'r', 0.1: 'c'}

if not os.path.exists(graph_folder):
    os.makedirs(graph_folder)

test_sizes = [0.2, 0.3, 0.4, 0.5]
ccp_alphas = [0.0, 0.001, 0.01, 0.1]
line_graph_results = {'Test Size': [], 'CCP Alpha': [], 'Criteria': [], 'Test Accuracy': [], 'Train Accuracy': []}
split_criteria = ['entropy', 'gini']

for criterion in split_criteria:
    print(f'Criterion: {criterion}') 
    for test_size in test_sizes:
        for ccp_alpha in ccp_alphas:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            clf = DecisionTreeClassifier(criterion=criterion, ccp_alpha=ccp_alpha, random_state=42)
            clf.fit(X_train, y_train)

            # Test accuracy
            y_test_pred = clf.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_test_pred)

            # Training accuracy
            y_train_pred = clf.predict(X_train)
            train_accuracy = accuracy_score(y_train, y_train_pred)

            print(f"    Test size: {test_size}, CCP Alpha: {ccp_alpha}, Test Accuracy: {test_accuracy:.2f}")

            line_graph_results['Test Size'].append(test_size)
            line_graph_results['CCP Alpha'].append(ccp_alpha)
            line_graph_results['Criteria'].append(criterion)
            line_graph_results['Test Accuracy'].append(test_accuracy)
            line_graph_results['Train Accuracy'].append(train_accuracy)

# Line graphs for 'gini' and 'entropy'
df = pd.DataFrame(line_graph_results)

for criterion in split_criteria:
    plt.figure(figsize=(10, 6))
    df_criterion = df[df['Criteria'] == criterion]
    
    for test_size in test_sizes:
        subset = df_criterion[df_criterion['Test Size'] == test_size]

        plt.plot(subset['CCP Alpha'], subset['Test Accuracy'], '-', label=f"Test, Test Size={test_size}")
        plt.plot(subset['CCP Alpha'], subset['Train Accuracy'], '--', label=f"Train, Test Size={test_size}")

    plt.title(f"Decision Tree Errors by CCP Alpha ({criterion})")
    plt.xlabel("CCP Alpha")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.tight_layout()  # Adjust layout to avoid overlap
    
    # Position legend outside of plot area, make it semi-transparent
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Line Type, Test Size")
    
    plt.savefig(os.path.join(graph_folder, f"variable_X_{criterion}.png"), bbox_inches='tight')
    plt.close()


for criterion in split_criteria:
    plt.figure(figsize=(10, 6))
    df_criterion = df[df['Criteria'] == criterion]
    
    for ccp_alpha in ccp_alphas:
        subset = df_criterion[df_criterion['CCP Alpha'] == ccp_alpha]

        color = color_map[ccp_alpha]  # Get color from the color map

        plt.plot(subset['Test Size'], subset['Test Accuracy'], '-', color=color, label=f"Test, α={ccp_alpha}")
        plt.plot(subset['Test Size'], subset['Train Accuracy'], '--', color=color, label=f"Train, α={ccp_alpha}")

    plt.title(f"Decision Tree Errors by Test Size ({criterion})")
    plt.xlabel("Test Size")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.tight_layout()  # Adjust layout to avoid overlap
    
    # Position legend outside of plot area, make it semi-transparent
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Line Type, Pruning Constant (α)")
    
    plt.savefig(os.path.join(graph_folder, f"TestSize_X_{criterion}.png"), bbox_inches='tight')
    plt.close()