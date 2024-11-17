import os
import matplotlib

matplotlib.use("Agg")

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import boxcox
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix


def visualize_results_pre_modeling():
    df = pd.read_parquet("data/processed_data.pq")

    # Class imbalance
    if not os.path.exists("models/class_imbalance.png"):
        plt.figure(figsize=(10, 6))
        sns.countplot(x="label", data=df)
        plt.title("Class Imbalance")
        plt.savefig("models/class_imbalance.png")
        plt.close()

    # Feature distributions
    for feature in ["length", "income", "weight", "count", "neighbors"]:
        if not os.path.exists(f"models/{feature}_distribution.png"):
            plt.figure(figsize=(10, 6))
            sns.histplot(df[feature], kde=True)
            plt.title(f"Distribution of {feature}")
            plt.savefig(f"models/{feature}_distribution.png")
            plt.close()

        # Boxcox transformation
        if not os.path.exists(f"models/{feature}_boxcox_distribution.png"):
            transformed, _ = boxcox(df[feature] + 1)  # Adding 1 to avoid log(0)
            plt.figure(figsize=(10, 6))
            sns.histplot(transformed, kde=True)
            plt.title(f"Distribution of {feature} after Boxcox Transformation")
            plt.savefig(f"models/{feature}_boxcox_distribution.png")
            plt.close()

    # Pairplot
    if not os.path.exists("models/pairplot.png"):
        sns.pairplot(df, hue="label")
        plt.savefig("models/pairplot.png")
        plt.close()

    # Bivariate plots
    for feature1 in ["length", "income", "weight"]:
        for feature2 in ["count", "neighbors"]:
            if not os.path.exists(f"models/{feature1}_vs_{feature2}.png"):
                plt.figure(figsize=(10, 6))
                sns.scatterplot(x=df[feature1], y=df[feature2], hue=df["label"])
                plt.title(f"{feature1} vs {feature2}")
                plt.savefig(f"models/{feature1}_vs_{feature2}.png")
                plt.close()

    # Correlation heatmap
    if not os.path.exists("models/correlation_heatmap.png"):
        plt.figure(figsize=(12, 10))
        sns.heatmap(df.select_dtypes(("int", "float")).corr(), annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.savefig("models/correlation_heatmap.png")
        plt.close()


def visualize_results_post_modeling():
    df = pd.read_parquet("data/model_predictions.pq")

    df["probability"] = df["probability"].apply(lambda x: x["values"][1])

    # ROC Curve
    plt.figure(figsize=(10, 6))
    fpr, tpr, _ = roc_curve(df["prediction"], df["probability"])
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig("models/roc_curve.png")
    plt.close()

    # Precision-Recall Curve
    plt.figure(figsize=(10, 6))
    precision, recall, _ = precision_recall_curve(df["prediction"], df["probability"])
    plt.plot(recall, precision, label="Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.savefig("models/precision_recall_curve.png")
    plt.close()

    # Confusion Matrix
    plt.figure(figsize=(10, 6))
    cm = confusion_matrix(df["label"], df["prediction"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("models/confusion_matrix.png")
    plt.close()
