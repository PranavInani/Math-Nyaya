import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.config import MODELS_TO_TEST, FINAL_RESULTS_DIR, model_name_to_filename
import numpy as np

def load_all_results():
    """
    Load results from all models and prompt styles.
    Returns a comprehensive DataFrame with all results.
    """
    all_results = []
    
    for model in MODELS_TO_TEST:
        safe_model_name = model_name_to_filename(model)
        
        # Load COT results
        cot_file = os.path.join(FINAL_RESULTS_DIR, f"{safe_model_name}_cot.csv")
        if os.path.exists(cot_file):
            cot_df = pd.read_csv(cot_file)
            cot_results = calculate_accuracy(cot_df, 'cot')
            cot_results['model'] = model
            cot_results['model_short'] = model.split('/')[-1]
            all_results.append(cot_results)
        
        # Load Zero-Shot results
        zero_shot_file = os.path.join(FINAL_RESULTS_DIR, f"{safe_model_name}_zero_shot.csv")
        if os.path.exists(zero_shot_file):
            zero_shot_df = pd.read_csv(zero_shot_file)
            zero_shot_results = calculate_accuracy(zero_shot_df, 'zero_shot')
            zero_shot_results['model'] = model
            zero_shot_results['model_short'] = model.split('/')[-1]
            all_results.append(zero_shot_results)
    
    if not all_results:
        print("No results found! Make sure you've run the evaluations first.")
        return None
    
    return pd.DataFrame(all_results)

def calculate_accuracy(df, prompt_style):
    """
    Calculate accuracy for with and without Panchavakya.
    """
    with_col = f'Output_with_Panchvakya_{prompt_style}'
    without_col = f'Output_without_Panchvakya_{prompt_style}'
    
    # Calculate accuracy with Panchavakya
    with_panchavakya_correct = (df[with_col] == df['correct_letter']).sum()
    with_panchavakya_total = df[with_col].notna().sum()
    with_panchavakya_accuracy = (with_panchavakya_correct / with_panchavakya_total * 100) if with_panchavakya_total > 0 else 0
    
    # Calculate accuracy without Panchavakya
    without_panchavakya_correct = (df[without_col] == df['correct_letter']).sum()
    without_panchavakya_total = df[without_col].notna().sum()
    without_panchavakya_accuracy = (without_panchavakya_correct / without_panchavakya_total * 100) if without_panchavakya_total > 0 else 0
    
    return {
        'prompt_style': prompt_style.upper(),
        'with_panchavakya_accuracy': with_panchavakya_accuracy,
        'without_panchavakya_accuracy': without_panchavakya_accuracy,
        'with_panchavakya_total': with_panchavakya_total,
        'without_panchavakya_total': without_panchavakya_total
    }

def create_comparison_charts(results_df):
    """
    Create comprehensive comparison charts with enhanced annotations.
    """
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots - 1 row, 2 columns for the two main charts
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle('Math-Nyaya Model Performance Comparison', fontsize=24, fontweight='bold', y=0.95)
    
    # Chart 1: Overall Accuracy Comparison (grouped bar chart)
    ax1 = axes[0]
    create_grouped_bar_chart(results_df, ax1)
    
    # Chart 2: COT vs Zero-Shot Performance
    ax2 = axes[1]
    create_method_comparison_chart(results_df, ax2)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])  # Adjust layout to accommodate suptitle
    
    # Save the comprehensive chart
    output_path = os.path.join(FINAL_RESULTS_DIR, 'model_comparision.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Comprehensive comparison chart saved to: {output_path}")
    
    plt.show()

def create_grouped_bar_chart(results_df, ax):
    """
    Create a grouped bar chart showing all accuracies with percentage annotations.
    """
    # Prepare data for grouped bar chart
    models = results_df['model_short'].unique()
    x = np.arange(len(models))
    width = 0.15
    
    # Get data for each combination
    cot_with = []
    cot_without = []
    zero_shot_with = []
    zero_shot_without = []
    
    for model in models:
        model_data = results_df[results_df['model_short'] == model]
        
        cot_row = model_data[model_data['prompt_style'] == 'COT']
        zero_shot_row = model_data[model_data['prompt_style'] == 'ZERO_SHOT']
        
        cot_with.append(cot_row['with_panchavakya_accuracy'].iloc[0] if len(cot_row) > 0 else 0)
        cot_without.append(cot_row['without_panchavakya_accuracy'].iloc[0] if len(cot_row) > 0 else 0)
        zero_shot_with.append(zero_shot_row['with_panchavakya_accuracy'].iloc[0] if len(zero_shot_row) > 0 else 0)
        zero_shot_without.append(zero_shot_row['without_panchavakya_accuracy'].iloc[0] if len(zero_shot_row) > 0 else 0)
    
    # Create bars
    bars1 = ax.bar(x - 1.5*width, cot_with, width, label='COT with Panchavakya', alpha=0.8, color='#1f77b4')
    bars2 = ax.bar(x - 0.5*width, cot_without, width, label='COT without Panchavakya', alpha=0.8, color='#aec7e8')
    bars3 = ax.bar(x + 0.5*width, zero_shot_with, width, label='Zero-Shot with Panchavakya', alpha=0.8, color='#ff7f0e')
    bars4 = ax.bar(x + 1.5*width, zero_shot_without, width, label='Zero-Shot without Panchavakya', alpha=0.8, color='#ffbb78')
    
    # Add percentage annotations on bars
    def add_percentage_labels(bars, values):
        for bar, value in zip(bars, values):
            height = bar.get_height()
            if height > 0:  # Only add label if bar has height
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{value:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    add_percentage_labels(bars1, cot_with)
    add_percentage_labels(bars2, cot_without)
    add_percentage_labels(bars3, zero_shot_with)
    add_percentage_labels(bars4, zero_shot_without)
    
    ax.set_xlabel('Models', fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Overall Model Performance Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)  # Slightly higher to accommodate labels

def create_method_comparison_chart(results_df, ax):
    """
    Create a comparison chart between COT and Zero-Shot methods with percentage annotations.
    """
    models = results_df['model_short'].unique()
    x = np.arange(len(models))
    width = 0.35
    
    cot_avg = []
    zero_shot_avg = []
    
    for model in models:
        model_data = results_df[results_df['model_short'] == model]
        
        cot_row = model_data[model_data['prompt_style'] == 'COT']
        zero_shot_row = model_data[model_data['prompt_style'] == 'ZERO_SHOT']
        
        # Average of with and without Panchavakya
        cot_avg_acc = ((cot_row['with_panchavakya_accuracy'].iloc[0] + cot_row['without_panchavakya_accuracy'].iloc[0]) / 2) if len(cot_row) > 0 else 0
        zero_shot_avg_acc = ((zero_shot_row['with_panchavakya_accuracy'].iloc[0] + zero_shot_row['without_panchavakya_accuracy'].iloc[0]) / 2) if len(zero_shot_row) > 0 else 0
        
        cot_avg.append(cot_avg_acc)
        zero_shot_avg.append(zero_shot_avg_acc)
    
    bars1 = ax.bar(x - width/2, cot_avg, width, label='Chain-of-Thought (COT)', alpha=0.8, color='#2ca02c')
    bars2 = ax.bar(x + width/2, zero_shot_avg, width, label='Zero-Shot', alpha=0.8, color='#d62728')
    
    # Add percentage annotations
    for bar, value in zip(bars1, cot_avg):
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{value:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    for bar, value in zip(bars2, zero_shot_avg):
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{value:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Models', fontweight='bold')
    ax.set_ylabel('Average Accuracy (%)', fontweight='bold')
    ax.set_title('COT vs Zero-Shot Performance', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)  # Slightly higher to accommodate labels

def print_summary_table(results_df):
    """
    Print a summary table of all results.
    """
    print("\n" + "="*100)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("="*100)
    
    for model in results_df['model_short'].unique():
        model_data = results_df[results_df['model_short'] == model]
        
        print(f"\n{model.upper()}:")
        print("-" * 50)
        
        for _, row in model_data.iterrows():
            print(f"{row['prompt_style']:>12} | With Panchavakya: {row['with_panchavakya_accuracy']:>6.1f}% | Without Panchavakya: {row['without_panchavakya_accuracy']:>6.1f}% | Impact: {row['with_panchavakya_accuracy'] - row['without_panchavakya_accuracy']:>+6.1f}%")
    
    print("\n" + "="*100)

def main():
    """
    Main function to create all comparison charts.
    """
    print("Loading results from all models...")
    results_df = load_all_results()
    
    if results_df is None:
        return
    
    print(f"Loaded results for {len(results_df['model'].unique())} models")
    
    # Print summary table
    print_summary_table(results_df)
    
    # Create comparison charts
    print("\nGenerating comprehensive comparison charts...")
    create_comparison_charts(results_df)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()
