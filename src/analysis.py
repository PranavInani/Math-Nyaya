import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import os
from src.config import FINAL_RESULTS_DIR, model_name_to_filename

def analyze_results(model_name, prompt_style='cot'):
    """
    Generates and saves confusion matrices and classification reports.
    """
    filename = os.path.join(FINAL_RESULTS_DIR, f"{model_name_to_filename(model_name)}_{prompt_style}.csv")
    if not os.path.exists(filename):
        print(f"Results file not found: {filename}")
        return

    df = pd.read_csv(filename)

    output_with_col = f'Output_with_Panchvakya_{prompt_style}'
    output_without_col = f'Output_without_Panchvakya_{prompt_style}'

    # Normalize answers
    for col in ['correct_letter', output_with_col, output_without_col]:
        df[col] = df[col].astype(str).str.strip().str.upper()

    df_clean = df.dropna(subset=['correct_letter', output_with_col, output_without_col])
    labels = sorted(df_clean['correct_letter'].dropna().unique().tolist())

    # With Panchvakya
    cm_with = confusion_matrix(df_clean['correct_letter'], df_clean[output_with_col], labels=labels)
    acc_with = accuracy_score(df_clean['correct_letter'], df_clean[output_with_col])

    # Without Panchvakya
    cm_without = confusion_matrix(df_clean['correct_letter'], df_clean[output_without_col], labels=labels)
    acc_without = accuracy_score(df_clean['correct_letter'], df_clean[output_without_col])

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.heatmap(cm_with, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'With Panchvakya ({prompt_style})\n{model_name}\nAccuracy: {acc_with:.2%}')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.subplot(1, 2, 2)
    sns.heatmap(cm_without, annot=True, fmt='d', cmap='Oranges', xticklabels=labels, yticklabels=labels)
    plt.title(f'Without Panchvakya ({prompt_style})\n{model_name}\nAccuracy: {acc_without:.2%}')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()
    
    plot_filename = os.path.join(FINAL_RESULTS_DIR, f"{model_name_to_filename(model_name)}_{prompt_style}_analysis.png")
    plt.savefig(plot_filename)
    print(f"Analysis plot saved to {plot_filename}")
    plt.close()

    print(f"\n--- Classification Report (With Panchvakya - {prompt_style} - {model_name}) ---")
    print(classification_report(df_clean['correct_letter'], df_clean[output_with_col]))

    print(f"\n--- Classification Report (Without Panchvakya - {prompt_style} - {model_name}) ---")
    print(classification_report(df_clean['correct_letter'], df_clean[output_without_col]))

if __name__ == '__main__':
    from src.config import MODELS_TO_TEST
    for model in MODELS_TO_TEST:
        analyze_results(model, 'cot')
        analyze_results(model, 'zero_shot')
