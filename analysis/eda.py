import seaborn as sns
import matplotlib.pyplot as plt
import os
import mlflow

def visualize_eda(df, base_dir="eda_artifacts"):
    """
    Generates and logs EDA visualizations for both univariate and bivariate analysis.

    Args:
        df (pd.DataFrame): The input DataFrame to analyze.
        base_dir (str): Directory where plots will be saved. Defaults to 'eda_artifacts'.
    """

    # ──────────────────────────────────────────────
    # 🎨 Setup Seaborn styles and output directories
    # ──────────────────────────────────────────────
    sns.set(style='whitegrid', palette='pastel')
    uni_dir = os.path.join(base_dir, "univariate")
    bi_dir = os.path.join(base_dir, "bivariate")
    os.makedirs(uni_dir, exist_ok=True)
    os.makedirs(bi_dir, exist_ok=True)

    # ──────────────────────────────────────────────
    # 🔍 Identify numeric and categorical columns
    # ──────────────────────────────────────────────
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # ──────────────────────────────────────────────
    # 📊 Univariate plots: Histograms and Count plots
    # ──────────────────────────────────────────────
    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], kde=True, bins=30, color='skyblue')
        plt.title(f'Distribution of {col}')
        plt.tight_layout()
        plt.savefig(os.path.join(uni_dir, f"{col}_hist.png"))
        plt.close()

    for col in categorical_cols:
        plt.figure(figsize=(6, 4))
        sns.countplot(y=col, data=df, order=df[col].value_counts().index)
        plt.title(f'Count of {col}')
        plt.tight_layout()
        plt.savefig(os.path.join(uni_dir, f"{col}_count.png"))
        plt.close()

    # ──────────────────────────────────────────────
    # 🎯 Determine target variable for bivariate analysis
    # ──────────────────────────────────────────────
    target = 'adjusted_total_usd' if 'adjusted_total_usd' in df.columns \
             else df.select_dtypes(include='float64').columns[-1]

    # ──────────────────────────────────────────────
    # 📦 Bivariate plots: Boxplots for categorical vs target
    # ──────────────────────────────────────────────
    for col in categorical_cols:
        if target in df.columns:
            plt.figure(figsize=(6, 4))
            sns.boxplot(x=col, y=target, data=df)
            plt.title(f'{target} by {col}')
            plt.tight_layout()
            plt.savefig(os.path.join(bi_dir, f"{target}_by_{col}.png"))
            plt.close()

    # Exclude target from correlation pairs
    if target in numeric_cols:
        numeric_cols.remove(target)

    # 📈 Bivariate plots: Scatter plots for numeric vs target
    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.scatterplot(x=col, y=target, data=df)
        plt.title(f'{target} vs {col}')
        plt.tight_layout()
        plt.savefig(os.path.join(bi_dir, f"{target}_vs_{col}.png"))
        plt.close()

    # 🔥 Correlation heatmap if enough numeric variables
    if len(numeric_cols) > 1:
        plt.figure(figsize=(10, 8))
        corr = df.select_dtypes(include=['int64', 'float64']).corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(os.path.join(bi_dir, "correlation_heatmap.png"))
        plt.close()

    # ──────────────────────────────────────────────
    # 📁 Log all generated artifacts to MLflow
    # ──────────────────────────────────────────────
    mlflow.log_artifacts(uni_dir, artifact_path="eda/univariate")
    mlflow.log_artifacts(bi_dir, artifact_path="eda/bivariate")
