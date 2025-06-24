import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, balanced_accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# 1. Data Loading and Understanding
# ---------------------------------
# The wine dataset from UCI typically does not have a header.
# Column names are usually obtained from the wine.names file.
# For this example, we'll define them. There are 13 features and 1 target class.
# Make sure 'wine.data' is in the same directory or provide the full path.
try:
    # Attempt to load the data
    # Replace 'wine.data' with the actual path to your data file if it's located elsewhere.
    # The dataset is typically comma-separated.
    df = pd.read_csv('wine.data', header=None)

    # Assign column names (based on wine.names description)
    # The first column in wine.data is the class label (1, 2, or 3)
    # The subsequent 13 columns are the features.
    # We will reorder them later if needed, or select X and y appropriately.
    # Let's assume the raw file has class first, then features.
    # If your wine.data has features first and then class, adjust accordingly.
    
    # Standard column names for the wine dataset (features + class)
    # Source: https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.names
    feature_names = [
        'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium',
        'Total_phenols', 'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins',
        'Color_intensity', 'Hue', 'OD280_OD315_of_diluted_wines', 'Proline'
    ]
    
    # Check if the first column is the class or if it's at the end.
    # Typically, in wine.data, the class is the FIRST column.
    if df.shape[1] == 14: # 1 class + 13 features
        df.columns = ['Class'] + feature_names
        X = df[feature_names]
        y = df['Class']
    elif df.shape[1] == 13 and 'target' not in df.columns: # Only features, need separate target (less common for raw .data)
        print("Error: Data file seems to contain only features. Please ensure your wine.data file includes the target class or load it separately.")
        exit()
    else: # Potentially other formats, user might need to adjust.
          # This is a generic catch; specific handling might be needed.
        print(f"Warning: Unexpected number of columns ({df.shape[1]}). Please verify data format.")
        print("Assuming the first column is the target and the rest are features for now.")
        X = df.iloc[:, 1:]
        y = df.iloc[:, 0]
        X.columns = feature_names[:X.shape[1]]


    print("Dataset loaded successfully.")
    print("Shape of dataset:", df.shape)
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    print("\nBasic statistics of features:")
    print(X.describe())
    print("\nClass distribution:")
    print(y.value_counts())

except FileNotFoundError:
    print("Error: 'wine.data' not found. Please make sure the file is in the correct directory or provide the full path.")
    exit()
except Exception as e:
    print(f"An error occurred during data loading: {e}")
    exit()


# 2. Data Preprocessing
# ---------------------
# Splitting data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nShape of X_train: {X_train.shape}, Shape of X_test: {X_test.shape}")
print(f"Shape of y_train: {y_train.shape}, Shape of y_test: {y_test.shape}")
print(f"Shape of X_train_scaled: {X_train_scaled.shape}, Shape of X_test_scaled: {X_test_scaled.shape}")


# 3. Model Training and Evaluation
# --------------------------------
# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=10000, solver='lbfgs', multi_class='auto'), # Increased max_iter for convergence
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(kernel='linear', probability=True, random_state=42), # probability=True for predict_proba if needed, can be slower
    "Gaussian Naive Bayes": GaussianNB(),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5) # Example k=5
}

# Store results
results = {}
confusion_matrices = {}

print("\n--- Model Training and Evaluation ---")
for name, model in models.items():
    print(f"\nTraining {name}...")
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    # For multi-class, specify average for precision, recall, f1
    # 'macro': Calculate metrics for each label, and find their unweighted mean.
    # 'weighted': Calculate metrics for each label, and find their average weighted by support.
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    results[name] = {
        "Accuracy": accuracy,
        "Balanced Accuracy": balanced_acc,
        "Precision (Macro)": precision,
        "Recall (Macro)": recall,
        "F1-Score (Macro)": f1
    }
    confusion_matrices[name] = cm
    
    print(f"{name} Evaluation:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Balanced Accuracy: {balanced_acc:.4f}")
    print(f"  Precision (Macro): {precision:.4f}")
    print(f"  Recall (Macro): {recall:.4f}")
    print(f"  F1-Score (Macro): {f1:.4f}")
    print(f"  Confusion Matrix:\n{cm}")

# 4. Display Results
# ------------------
print("\n\n--- Overall Model Comparison ---")
results_df = pd.DataFrame(results).T # Transpose for better readability
print(results_df)

# Plotting the results (optional, but good for reports)
results_df.plot(kind='bar', figsize=(14, 8))
plt.title('Comparison of Classification Models')
plt.ylabel('Score')
plt.xticks(rotation=45, ha="right")
plt.legend(loc='lower right')
plt.tight_layout()
plt.show() # This will display the plot. Save it for your report.

# Plotting confusion matrices
num_models = len(models)
# Determine grid size for subplots
cols = 3 
rows = int(np.ceil(num_models / cols))

fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
axes = axes.flatten() # Flatten in case of single row/column

for i, (name, cm) in enumerate(confusion_matrices.items()):
    if i < len(axes): # Ensure we don't try to plot more than available subplots
        ax = axes[i]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                    xticklabels=np.unique(y), yticklabels=np.unique(y))
        ax.set_title(f'Confusion Matrix: {name}')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
    else:
        break # No more subplots available

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show() # This will display the confusion matrices. Save them for your report.

print("\n--- End of Script ---")
