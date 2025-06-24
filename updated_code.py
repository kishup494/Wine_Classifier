import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,f1_score, confusion_matrix, balanced_accuracy_score)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings("ignore")

# Load data
def load_data(path='wine.data'):
    feature_names = [
        'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium',
        'Total_phenols', 'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins',
        'Color_intensity', 'Hue', 'OD280_OD315_of_diluted_wines', 'Proline'
    ]
    df = pd.read_csv(path, header=None)
    df.columns = ['Class'] + feature_names
    return df, feature_names

# Preprocess data
def preprocess_data(df, feature_names):
    X = df[feature_names]
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=42,
                                                        stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# Train and evaluate models
def train_models(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=10000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVM": SVC(kernel='linear', probability=True, random_state=42),
        "Gaussian NB": GaussianNB(),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }

    results = {}
    confusion_matrices = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average='macro', zero_division=0),
            "Recall": recall_score(y_test, y_pred, average='macro', zero_division=0),
            "F1": f1_score(y_test, y_pred, average='macro', zero_division=0)
        }
        confusion_matrices[name] = confusion_matrix(y_test, y_pred)

    return models, results, confusion_matrices

# Plot model comparison
def plot_results(results):
    df = pd.DataFrame(results).T
    df.plot(kind='bar', figsize=(14, 8))
    plt.title('Model Comparison')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Feature importance using SHAP
def explain_model(model, X_train, feature_names):
    explainer = shap.Explainer(model.predict, X_train)
    shap_values = explainer(X_train)
    shap.summary_plot(shap_values, X_train, feature_names=feature_names)

# Confusion matrices
def plot_confusion_matrices(conf_mats, class_labels):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, (model_name, cm) in enumerate(conf_mats.items()):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=class_labels, yticklabels=class_labels)
        axes[i].set_title(model_name)
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")
    plt.tight_layout()
    plt.show()

# === MAIN ===
if __name__ == "__main__":
    df, features = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df, features)
    models, results, conf_mats = train_models(X_train, X_test, y_train, y_test)

    print("\n--- Model Performance ---")
    for model, metrics in results.items():
        print(f"\n{model}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

    plot_results(results)
    plot_confusion_matrices(conf_mats, class_labels=np.unique(df['Class']))

    # Feature importance for Random Forest
    print("\nExplaining Random Forest with SHAP:")
    explain_model(models['Random Forest'], X_train, features)
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 5, 10],
}

grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=3, scoring='accuracy')
grid.fit(X_train, y_train)

print("\nBest parameters for Random Forest:")
print(grid.best_params_)
