"""
==================================================================
           WEATHER PREDICTION ML MODEL (Australia)
   Predicts whether it will RAIN TOMORROW using weatherAUS.csv
==================================================================

Dataset : weatherAUS.csv (Kaggle - Rain in Australia)
Target  : RainTomorrow (Yes / No -> Binary Classification)
Models  : Logistic Regression, Random Forest, XGBoost
"""

# --------------------------- IMPORTS ----------------------------
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
)
from xgboost import XGBClassifier
import joblib

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# ----------------------- CONFIGURATION --------------------------
DATASET_PATH   = r"D:\weatherAUS.csv"
OUTPUT_DIR     = r"D:\New folder\output"
RANDOM_STATE   = 42
TEST_SIZE      = 0.20

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================================================================
# 1.  LOAD DATA
# ================================================================
def load_data(path: str) -> pd.DataFrame:
    """Load dataset and print basic info."""
    print("=" * 65)
    print("  [LOAD] Loading Dataset")
    print("=" * 65)
    df = pd.read_csv(path)
    print(f"  Shape           : {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"  Target column   : RainTomorrow")
    print(f"  Target balance  :\n{df['RainTomorrow'].value_counts().to_string()}")
    print(f"  Missing values  : {df.isnull().sum().sum():,} total")
    print()
    return df


# ================================================================
# 2.  EXPLORATORY DATA ANALYSIS  (EDA)
# ================================================================
def run_eda(df: pd.DataFrame):
    """Generate and save EDA visualizations."""
    print("=" * 65)
    print("  [EDA] Exploratory Data Analysis")
    print("=" * 65)

    # --- 2a. Target distribution ---
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#4CAF50", "#F44336"]
    df["RainTomorrow"].value_counts().plot.bar(color=colors, edgecolor="black", ax=ax)
    ax.set_title("RainTomorrow Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Rain Tomorrow")
    ax.set_ylabel("Count")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "01_target_distribution.png"), dpi=150)
    plt.close(fig)
    print("  [DONE] Saved 01_target_distribution.png")

    # --- 2b. Correlation heatmap (numeric cols) ---
    numeric_df = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, ax=ax, annot_kws={"size": 7})
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "02_correlation_heatmap.png"), dpi=150)
    plt.close(fig)
    print("  [DONE] Saved 02_correlation_heatmap.png")

    # --- 2c. Temperature distribution ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col, color in zip(axes, ["MinTemp", "MaxTemp"], ["#2196F3", "#FF9800"]):
        df[col].dropna().hist(bins=40, color=color, edgecolor="black", alpha=0.8, ax=ax)
        ax.set_title(f"{col} Distribution", fontsize=12, fontweight="bold")
        ax.set_xlabel(col)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "03_temperature_distribution.png"), dpi=150)
    plt.close(fig)
    print("  [DONE] Saved 03_temperature_distribution.png")

    # --- 2d. Rainfall by RainTomorrow ---
    fig, ax = plt.subplots(figsize=(8, 4))
    df.boxplot(column="Rainfall", by="RainTomorrow", ax=ax,
               patch_artist=True,
               boxprops=dict(facecolor="#81D4FA"),
               medianprops=dict(color="red", linewidth=2))
    ax.set_title("Rainfall by RainTomorrow", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 50)
    plt.suptitle("")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "04_rainfall_boxplot.png"), dpi=150)
    plt.close(fig)
    print("  [DONE] Saved 04_rainfall_boxplot.png")
    print()


# ================================================================
# 3.  DATA PREPROCESSING
# ================================================================
def preprocess(df: pd.DataFrame):
    """Clean, encode, and split the data."""
    print("=" * 65)
    print("  [PRE] Data Preprocessing")
    print("=" * 65)

    df = df.copy()

    # Drop date column (not useful for model)
    df.drop(columns=["Date"], inplace=True, errors="ignore")

    # --- Separate numeric and categorical columns ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c != "RainTomorrow"]

    # --- Fill missing values ---
    # Numeric: fill with median
    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)

    # Categorical: fill with mode
    for col in cat_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Target: drop rows where target is missing
    df.dropna(subset=["RainTomorrow"], inplace=True)

    print(f"  Numeric columns  : {len(numeric_cols)}")
    print(f"  Categorical cols : {len(cat_cols)}")
    print(f"  Rows after clean : {len(df):,}")

    # --- Encode categorical features ---
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # Encode target
    df["RainTomorrow"] = df["RainTomorrow"].map({"No": 0, "Yes": 1})

    # --- Feature / Target split ---
    X = df.drop(columns=["RainTomorrow"])
    y = df["RainTomorrow"]

    # --- Train / Test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # --- Feature Scaling ---
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train),
                                  columns=X_train.columns, index=X_train.index)
    X_test_scaled  = pd.DataFrame(scaler.transform(X_test),
                                  columns=X_test.columns, index=X_test.index)

    print(f"  Training set     : {X_train_scaled.shape[0]:,} samples")
    print(f"  Test set         : {X_test_scaled.shape[0]:,} samples")
    print(f"  Features         : {X_train_scaled.shape[1]}")
    print()

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoders


# ================================================================
# 4.  MODEL TRAINING & EVALUATION
# ================================================================
def evaluate_model(name, model, X_test, y_test):
    """Return a dict of evaluation metrics."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "Model"     : name,
        "Accuracy"  : accuracy_score(y_test, y_pred),
        "Precision" : precision_score(y_test, y_pred),
        "Recall"    : recall_score(y_test, y_pred),
        "F1 Score"  : f1_score(y_test, y_pred),
        "ROC AUC"   : roc_auc_score(y_test, y_proba) if y_proba is not None else None,
    }
    return metrics, y_pred, y_proba


def train_models(X_train, X_test, y_train, y_test):
    """Train multiple models and compare performance."""
    print("=" * 65)
    print("  [TRN] Model Training")
    print("=" * 65)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=15, random_state=RANDOM_STATE,
            class_weight="balanced", n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=RANDOM_STATE, use_label_encoder=False,
            eval_metric="logloss", scale_pos_weight=3, n_jobs=-1
        ),
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        print(f"  -> Training {name} ...")
        model.fit(X_train, y_train)
        trained_models[name] = model

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5,
                                     scoring="accuracy", n_jobs=-1)

        metrics, y_pred, y_proba = evaluate_model(name, model, X_test, y_test)
        metrics["CV Mean Acc"] = cv_scores.mean()
        results.append(metrics)

        print(f"    - Accuracy  : {metrics['Accuracy']:.4f}")
        print(f"    - Precision : {metrics['Precision']:.4f}")
        print(f"    - Recall    : {metrics['Recall']:.4f}")
        print(f"    - F1 Score  : {metrics['F1 Score']:.4f}")
        print(f"    - ROC AUC   : {metrics['ROC AUC']:.4f}")
        print(f"    - CV Acc    : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    print()

    # --- Results comparison table ---
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("ROC AUC", ascending=False).reset_index(drop=True)
    print("=" * 65)
    print("  [TAB] Model Comparison")
    print("=" * 65)
    print(results_df.to_string(index=False, float_format="%.4f"))
    print()

    # Save results
    results_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)
    print("  [DONE] Saved model_comparison.csv")

    return trained_models, results_df


# ================================================================
# 5.  VISUALIZATION OF RESULTS
# ================================================================
def plot_results(trained_models, X_test, y_test, results_df):
    """Generate evaluation charts."""
    print()
    print("=" * 65)
    print("  [PLT] Generating Result Plots")
    print("=" * 65)

    colors_map = {
        "Logistic Regression": "#2196F3",
        "Random Forest"      : "#4CAF50",
        "XGBoost"            : "#FF9800",
    }

    # --- 5a. Model comparison bar chart ---
    fig, ax = plt.subplots(figsize=(10, 5))
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]
    x = np.arange(len(metrics_to_plot))
    width = 0.25

    for i, (_, row) in enumerate(results_df.iterrows()):
        name = row["Model"]
        vals = [row[m] for m in metrics_to_plot]
        ax.bar(x + i * width, vals, width, label=name,
               color=colors_map.get(name, "#999"), edgecolor="black", alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics_to_plot, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "05_model_comparison.png"), dpi=150)
    plt.close(fig)
    print("  [DONE] Saved 05_model_comparison.png")

    # --- 5b. ROC Curves ---
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, model in trained_models.items():
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc = roc_auc_score(y_test, y_proba)
            ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})",
                    color=colors_map.get(name, "#999"), linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Baseline")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "06_roc_curves.png"), dpi=150)
    plt.close(fig)
    print("  [DONE] Saved 06_roc_curves.png")

    # --- 5c. Confusion matrices ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, (name, model) in zip(axes, trained_models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["No Rain", "Rain"],
                    yticklabels=["No Rain", "Rain"], ax=ax,
                    annot_kws={"size": 13})
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "07_confusion_matrices.png"), dpi=150)
    plt.close(fig)
    print("  [DONE] Saved 07_confusion_matrices.png")

    # --- 5d. Feature Importance (best tree model) ---
    best_tree = "XGBoost" if "XGBoost" in trained_models else "Random Forest"
    model = trained_models[best_tree]
    importances = model.feature_importances_
    feature_names = X_test.columns
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    feat_imp.tail(15).plot.barh(color="#26A69A", edgecolor="black", ax=ax)
    ax.set_title(f"Top 15 Feature Importances ({best_tree})",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "08_feature_importance.png"), dpi=150)
    plt.close(fig)
    print("  [DONE] Saved 08_feature_importance.png")
    print()


# ================================================================
# 6.  SAVE BEST MODEL
# ================================================================
def save_best_model(trained_models, results_df, scaler):
    """Save the best performing model to disk."""
    print("=" * 65)
    print("  [SAV] Saving Best Model")
    print("=" * 65)

    best_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_name]

    model_path  = os.path.join(OUTPUT_DIR, "best_model.pkl")
    scaler_path = os.path.join(OUTPUT_DIR, "scaler.pkl")

    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"  Best model   : {best_name}")
    print(f"  Model saved  : {model_path}")
    print(f"  Scaler saved : {scaler_path}")
    print()

    return best_name


# ================================================================
# 7.  CLASSIFICATION REPORT
# ================================================================
def print_classification_report(trained_models, X_test, y_test, best_name):
    """Print detailed classification report for the best model."""
    print("=" * 65)
    print(f"  [REP] Classification Report - {best_name}")
    print("=" * 65)
    y_pred = trained_models[best_name].predict(X_test)
    print(classification_report(y_test, y_pred,
                                target_names=["No Rain", "Rain"]))


# ================================================================
# 8.  MAIN PIPELINE
# ================================================================
def main():
    print()
    print("================================================================")
    print("       WEATHER PREDICTION - RAIN TOMORROW (AUS)       ")
    print("================================================================")
    print()

    # Step 1 - Load
    df = load_data(DATASET_PATH)

    # Step 2 - EDA
    run_eda(df)

    # Step 3 - Preprocess
    X_train, X_test, y_train, y_test, scaler, label_encoders = preprocess(df)

    # Step 4 - Train
    trained_models, results_df = train_models(X_train, X_test, y_train, y_test)

    # Step 5 - Visualize
    plot_results(trained_models, X_test, y_test, results_df)

    # Step 6 - Save
    best_name = save_best_model(trained_models, results_df, scaler)

    # Step 7 - Report
    print_classification_report(trained_models, X_test, y_test, best_name)

    print("=" * 65)
    print("  [FIN] PIPELINE COMPLETE!")
    print(f"  [OUT] All outputs saved to: {OUTPUT_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
