import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import os

print("Starting the program...")

# Reading data and preparing folder
data = pd.read_csv('../Data/spambase.data', header=None)
print("Data read successfully.")

graph_folder = '../graphs/k_fold'

if not os.path.exists(graph_folder):
    os.makedirs(graph_folder)

# Encoding categorical variables
for column in data.columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])

print("Data encoding complete.")

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Initialize variables
ensemble_sizes = list(range(0, 500, 10))
ensemble_sizes[0] = 1

best_rf_score = 0
best_ab_score = 0
best_rf_size = 0
best_ab_size = 0

k = 5
training_set_percent = 0.80

print("Variable initialization complete.")

# Create training and test sets
train_end_idx = int(training_set_percent * len(data))
X_train = X.iloc[:train_end_idx, :]
y_train = y.iloc[:train_end_idx]
X_test = X.iloc[train_end_idx:, :]
y_test = y.iloc[train_end_idx:]

# Create subsets for k-fold validation
subset_size = len(X_train) // k
train_subsets_X = [X_train.iloc[i*subset_size:(i+1)*subset_size, :] for i in range(k)]
train_subsets_y = [y_train.iloc[i*subset_size:(i+1)*subset_size] for i in range(k)]

rf_average_scores = []
ab_average_scores = []

print("Starting RandomForest k-fold cross-validation...")

# RandomForest
for size in ensemble_sizes:
    validation_scores = []
    for i in range(k):
        print(f"Performing {i+1}-th fold with ensemble size {size}...")
        # Concatenate k-1 subsets for training, use one for validation
        train_X = pd.concat([train_subsets_X[j] for j in range(k) if j != i], axis=0)
        train_y = pd.concat([train_subsets_y[j] for j in range(k) if j != i], axis=0)
        valid_X = train_subsets_X[i]
        valid_y = train_subsets_y[i]
        
        # Initialize and fit the classifier
        clf = RandomForestClassifier(n_estimators=size, random_state=42)
        clf.fit(train_X, train_y)
        pred_y = clf.predict(valid_X)
        score = accuracy_score(valid_y, pred_y)
        validation_scores.append(score)

    # Average the validation scores for this ensemble size
    average_score = sum(validation_scores) / len(validation_scores)
    rf_average_scores.append(average_score)
    print(f"Average validation score with ensemble size {size}: {average_score}")

    if average_score > best_rf_score:
        best_rf_score = average_score
        best_rf_size = size

print(f"Best RandomForest ensemble size found: {best_rf_size}")

print("Starting AdaBoost k-fold cross-validation...")

# AdaBoost
for size in ensemble_sizes:
    validation_scores = []
    for i in range(k):
        print(f"Performing {i+1}-th fold with ensemble size {size}...")
        # Same logic as above
        train_X = pd.concat([train_subsets_X[j] for j in range(k) if j != i], axis=0)
        train_y = pd.concat([train_subsets_y[j] for j in range(k) if j != i], axis=0)
        valid_X = train_subsets_X[i]
        valid_y = train_subsets_y[i]

        # Initialize and fit the classifier
        clf = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=size)
        clf.fit(train_X, train_y)
        pred_y = clf.predict(valid_X)
        score = accuracy_score(valid_y, pred_y)
        validation_scores.append(score)

    # Average the validation scores for this ensemble size
    average_score = sum(validation_scores) / len(validation_scores)
    ab_average_scores.append(average_score)
    print(f"Average validation score with ensemble size {size}: {average_score}")

    if average_score > best_ab_score:
        best_ab_score = average_score
        best_ab_size = size

print(f"Best AdaBoost ensemble size found: {best_ab_size}")

print("Plotting graphs...")

# Plotting
plt.figure()
plt.plot(ensemble_sizes, rf_average_scores, label='RandomForest')
plt.plot(ensemble_sizes, ab_average_scores, label='AdaBoost')
plt.legend()
plt.xlabel('Ensemble Size')
plt.ylabel('Average Validation Score')
plt.title('Ensemble Size vs Average Validation Score')
plt.savefig(os.path.join(graph_folder, f"Ensemble Size vs Average Validation Score.png"), bbox_inches='tight')
plt.close()

# Final evaluation on the test set
clf_rf = RandomForestClassifier(n_estimators=best_rf_size, random_state=42)
clf_rf.fit(X_train, y_train)
rf_test_score = accuracy_score(y_test, clf_rf.predict(X_test))

clf_ab = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=best_ab_size, random_state=42)
clf_ab.fit(X_train, y_train)
ab_test_score = accuracy_score(y_test, clf_ab.predict(X_test))

# Bar graph for test scores
labels = ['RandomForest', 'AdaBoost']
scores = [rf_test_score, ab_test_score]

plt.figure()
plt.barh(labels, scores)
plt.xlabel('Test Score')
plt.title('Test Score Comparison')
plt.savefig(os.path.join(graph_folder, f"Best Scores.png"), bbox_inches='tight')
plt.close()

print(f"Test Score with Best RandomForest Ensemble Size ({best_rf_size}): {rf_test_score}")
print(f"Training Score with Best RandomForest Ensemble Size ({best_rf_size}): {best_rf_score}")
print(f"Test Score with Best AdaBoost Ensemble Size ({best_ab_size}): {ab_test_score}")
print(f"Training Score with Best AdaBoost Ensemble Size ({best_ab_size}): {best_ab_score}")
