import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.config import MODELS_TO_TEST, FINAL_RESULTS_DIR, model_name_to_filename
import numpy as np

# Constants for model and strategy mappings
STRATEGY_MAPPING = {
    'COT_with_Panchavakya': 'C+',
    'COT_without_Panchavakya': 'C-', 
    'ZeroShot_with_Panchavakya': 'Z+',
    'ZeroShot_without_Panchavakya': 'Z-'
}

MODEL_MAPPING = {
    'qwen3-32b': 'Qwen',
    'deepseek-r1-distill-llama-70b': 'DeepSeek',
    'gemma2-9b-it': 'Gemma',
    'llama-3.1-8b-instant': 'Llama31',
    'llama-3.3-70b-versatile': 'Llama33',
    'mistral-saba-24b': 'Mistral'
}

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

def add_percentage_labels_to_bars(ax, bars, values, fontsize=8):
    """
    Helper function to add percentage labels on top of bars.
    """
    for bar, value in zip(bars, values):
        height = bar.get_height()
        if height > 0:  # Only add label if bar has height
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{value:.1f}%', ha='center', va='bottom', 
                   fontsize=fontsize, fontweight='bold')

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
    
    # Add percentage annotations on bars using helper function
    add_percentage_labels_to_bars(ax, bars1, cot_with)
    add_percentage_labels_to_bars(ax, bars2, cot_without)
    add_percentage_labels_to_bars(ax, bars3, zero_shot_with)
    add_percentage_labels_to_bars(ax, bars4, zero_shot_without)
    
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
    
    # Add percentage annotations using helper function
    add_percentage_labels_to_bars(ax, bars1, cot_avg, fontsize=9)
    add_percentage_labels_to_bars(ax, bars2, zero_shot_avg, fontsize=9)
    
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

def load_detailed_results():
    """
    Load detailed question-by-question results from all models and strategies.
    Returns a comprehensive DataFrame with question-level results.
    """
    all_detailed_results = []
    
    for model in MODELS_TO_TEST:
        safe_model_name = model_name_to_filename(model)
        model_short = model.split('/')[-1]
        
        # Define file paths and strategy mappings
        files_and_strategies = [
            (os.path.join(FINAL_RESULTS_DIR, f"{safe_model_name}_cot.csv"), 
             [('COT_with_Panchavakya', 'Output_with_Panchvakya_cot'),
              ('COT_without_Panchavakya', 'Output_without_Panchvakya_cot')]),
            (os.path.join(FINAL_RESULTS_DIR, f"{safe_model_name}_zero_shot.csv"),
             [('ZeroShot_with_Panchavakya', 'Output_with_Panchvakya_zero_shot'),
              ('ZeroShot_without_Panchavakya', 'Output_without_Panchvakya_zero_shot')])
        ]
        
        # Process each file and strategy combination
        for file_path, strategies in files_and_strategies:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                for idx, row in df.iterrows():
                    for strategy_name, output_column in strategies:
                        question_result = {
                            'question_id': idx + 1,
                            'correct_answer': row['correct_letter'],
                            'model': model,
                            'model_short': model_short,
                            'strategy': strategy_name,
                            'predicted_answer': row.get(output_column, ''),
                            'is_correct': row.get(output_column, '') == row['correct_letter']
                        }
                        all_detailed_results.append(question_result)
    
    return pd.DataFrame(all_detailed_results)

def create_question_results_table(detailed_df):
    """
    Create a pivot table showing question-by-question results for all models and strategies.
    """
    if detailed_df.empty:
        print("No detailed results found!")
        return None
    
    # Create a column name that combines model and strategy
    detailed_df['model_strategy'] = detailed_df['model_short'] + '_' + detailed_df['strategy']
    
    # Create pivot table with questions as rows and model_strategy as columns
    pivot_df = detailed_df.pivot_table(
        index='question_id', 
        columns='model_strategy', 
        values='is_correct',
        aggfunc='first'  # Use first since there should be only one value per combination
    )
    
    # Convert boolean to int for better display (1 = correct, 0 = incorrect)
    pivot_df = pivot_df.astype(int)
    
    return pivot_df

def save_detailed_results_csv(detailed_df):
    """
    Save the detailed question-by-question results to CSV files.
    """
    if detailed_df.empty:
        print("No detailed results to save!")
        return
    
    # Save the raw detailed results
    detailed_output_path = os.path.join(FINAL_RESULTS_DIR, 'detailed_question_results.csv')
    detailed_df.to_csv(detailed_output_path, index=False)
    print(f"Detailed results saved to: {detailed_output_path}")
    
    # Create and save the pivot table
    pivot_df = create_question_results_table(detailed_df)
    if pivot_df is not None:
        pivot_output_path = os.path.join(FINAL_RESULTS_DIR, 'question_results_table.csv')
        pivot_df.to_csv(pivot_output_path)
        print(f"Question results table saved to: {pivot_output_path}")
        
        return pivot_df
    
    return None

def print_detailed_summary(detailed_df):
    """
    Print a detailed summary of question-by-question performance.
    """
    if detailed_df.empty:
        print("No detailed results to summarize!")
        return
    
    print("\n" + "="*120)
    print("DETAILED QUESTION-BY-QUESTION PERFORMANCE SUMMARY")
    print("="*120)
    
    # Group by model and strategy to show performance
    summary_stats = detailed_df.groupby(['model_short', 'strategy']).agg({
        'is_correct': ['count', 'sum', 'mean']
    }).round(3)
    
    summary_stats.columns = ['Total_Questions', 'Correct_Answers', 'Accuracy']
    summary_stats['Accuracy_Percent'] = (summary_stats['Accuracy'] * 100).round(1)
    
    print(summary_stats)
    
    # Find questions that are most difficult (lowest success rate)
    question_difficulty = detailed_df.groupby('question_id').agg({
        'is_correct': ['count', 'sum', 'mean']
    }).round(3)
    
    question_difficulty.columns = ['Total_Attempts', 'Correct_Attempts', 'Success_Rate']
    question_difficulty['Success_Rate_Percent'] = (question_difficulty['Success_Rate'] * 100).round(1)
    question_difficulty = question_difficulty.sort_values('Success_Rate')
    
    print(f"\n\nMOST DIFFICULT QUESTIONS (Top 10):")
    print("-" * 120)
    print(question_difficulty.head(10))
    
    print(f"\n\nEASIEST QUESTIONS (Top 10):")
    print("-" * 120)
    print(question_difficulty.tail(10))
    
    print("\n" + "="*120)

def create_csv_style_table(detailed_df):
    """
    Create separate CSV-style table visualizations for COT and Zero-Shot strategies.
    """
    if detailed_df.empty:
        print("No detailed results found!")
        return
    
    # Create separate tables for COT and Zero-Shot
    create_strategy_table(detailed_df, 'COT', 'Chain-of-Thought (COT) Results')
    create_strategy_table(detailed_df, 'ZeroShot', 'Zero-Shot Results')

def create_strategy_table(detailed_df, strategy_prefix, title):
    """
    Create a table for a specific strategy (COT or ZeroShot) with full model names.
    """
    # Filter data for the specific strategy
    strategy_df = detailed_df[detailed_df['strategy'].str.startswith(strategy_prefix)].copy()
    
    if strategy_df.empty:
        print(f"No data found for {strategy_prefix} strategy!")
        return
    
    # Create column names with full model names and compact strategy labels
    strategy_df['strategy_clean'] = strategy_df['strategy'].str.replace(f'{strategy_prefix}_', '')
    strategy_df['strategy_clean'] = strategy_df['strategy_clean'].str.replace('with_Panchavakya', '(with)')
    strategy_df['strategy_clean'] = strategy_df['strategy_clean'].str.replace('without_Panchavakya', '(without)')
    
    # Shorten the DeepSeek model name for better display
    strategy_df['model_display'] = strategy_df['model_short'].str.replace('deepseek-r1-distill-llama-70b', 'deepseek-distil')
    strategy_df['model_strategy_full'] = strategy_df['model_display'] + '\n' + strategy_df['strategy_clean']
    
    # Create pivot table with full names
    pivot_df = strategy_df.pivot_table(
        index='question_id', 
        columns='model_strategy_full', 
        values='is_correct',
        aggfunc='first'
    )
    pivot_df = pivot_df.astype(int)
    
    # Create figure with matplotlib table
    num_columns = len(pivot_df.columns)
    num_rows = len(pivot_df.index)
    
    # Calculate figure size based on content - compact cells
    cell_width = 1.5  # Reduced for compact display
    cell_height = 0.6  # Increased height for text wrapping
    fig_width = max(16, (num_columns + 1) * cell_width)
    fig_height = max(12, (num_rows + 2) * cell_height)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')  # Hide axes
    
    # Prepare data for table
    table_data = []
    
    # Add header row with full model names
    headers = ['Question'] + list(pivot_df.columns)
    table_data.append(headers)
    
    # Add data rows with checkmarks and cross marks (using more compatible symbols)
    for idx, row in pivot_df.iterrows():
        row_data = [f'Q{idx}'] + ['✓' if val == 1 else '✗' for val in row.values]
        table_data.append(row_data)
    
    # Create the table
    table = ax.table(
        cellText=table_data[1:],  # Data rows
        colLabels=table_data[0],  # Header row
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    # Style the table with smaller fonts and better spacing
    table.auto_set_font_size(False)
    table.set_fontsize(8)  # Reduced font size
    table.scale(1.0, 1.8)  # Adjusted scaling
    
    # Color cells based on values
    for i in range(len(table_data[1:])):  # Skip header
        for j in range(1, len(table_data[0])):  # Skip question number column
            cell = table[(i+1, j)]
            value = table_data[i+1][j]
            
            if value == '✓':
                cell.set_facecolor('#51cf66')  # Green for correct
                cell.set_text_props(weight='bold', color='white', size=14)
            elif value == '✗':
                cell.set_facecolor('#ff6b6b')  # Red for incorrect
                cell.set_text_props(weight='bold', color='white', size=14)
            
            cell.set_edgecolor('white')
            cell.set_linewidth(1)
    
    # Style header row with better text handling for multiline
    for j in range(len(table_data[0])):
        cell = table[(0, j)]
        cell.set_facecolor('#2196F3')  # Blue header
        cell.set_text_props(weight='bold', color='white', size=7)  # Smaller font for headers
        cell.set_edgecolor('white')
        cell.set_linewidth(1)
        # Set more proportional cell height for headers
        cell.set_height(0.08)  # Reduced height for more compact headers
    
    # Style question number column
    for i in range(1, len(table_data)):
        cell = table[(i, 0)]
        cell.set_facecolor('#E0E0E0')  # Light gray for question numbers
        cell.set_text_props(weight='bold', color='black', size=8)
        cell.set_edgecolor('white')
        cell.set_linewidth(1)
    
    # Add title above the table
    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    
    # Add simple legend
    legend_text = (
        "✓ = Correct | ✗ = Incorrect\n"
        "(with) = Using Sanskrit logical framework\n"
        "(without) = Standard approach"
    )
    
    plt.figtext(0.5, 0.02, legend_text, fontsize=10, ha='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    # Adjust layout
    plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.12)
    
    # Save the table with high quality
    strategy_clean = strategy_prefix.lower()
    output_path = os.path.join(FINAL_RESULTS_DIR, f'question_results_{strategy_clean}_table.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.5)
    print(f"{title} table saved to: {output_path}")
    
    plt.show()
    
    return pivot_df

def main():
    """
    Main function to create all comparison charts and detailed analysis.
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
    
    # Load detailed question-by-question results
    print("\nLoading detailed question-by-question results...")
    detailed_df = load_detailed_results()
    
    if not detailed_df.empty:
        print(f"Loaded detailed results for {len(detailed_df)} question-model-strategy combinations")
        
        # Print detailed summary
        print_detailed_summary(detailed_df)
        
        # Save detailed results to CSV
        print("\nSaving detailed results...")
        pivot_df = save_detailed_results_csv(detailed_df)
        
        # Create separate CSV-style table visualizations for COT and Zero-Shot
        print("\nGenerating separate results tables for COT and Zero-Shot...")
        create_csv_style_table(detailed_df)
        
        # Print sample of the pivot table
        if pivot_df is not None:
            print(f"\nSample of Question Results Table (first 10 questions):")
            print("-" * 120)
            print(pivot_df.head(10))
    else:
        print("No detailed results found!")
    
    print("\nComplete analysis finished!")

if __name__ == "__main__":
    main()
