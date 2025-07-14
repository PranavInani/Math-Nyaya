import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from .config import FINAL_RESULTS_DIR, MCQ_RESULTS_DIR, NUMERICAL_RESULTS_DIR, model_name_to_filename, MODELS_TO_TEST

def create_comprehensive_analysis():
    """
    Creates a comprehensive analysis comparing MCQ and numerical question performance
    across all models and prompt styles.
    """
    print("Creating comprehensive analysis of MCQ and numerical results...")
    
    mcq_results = []
    numerical_results = []
    
    # Collect MCQ results
    for model in MODELS_TO_TEST:
        safe_model_name = model_name_to_filename(model)
        
        for prompt_style in ['cot', 'zero_shot']:
            mcq_file = os.path.join(MCQ_RESULTS_DIR, f"{safe_model_name}_{prompt_style}.csv")
            if os.path.exists(mcq_file):
                df = pd.read_csv(mcq_file)
                
                # Calculate MCQ accuracies
                output_with_col = f'Output_with_Panchvakya_{prompt_style}'
                output_without_col = f'Output_without_Panchvakya_{prompt_style}'
                
                if output_with_col in df.columns and output_without_col in df.columns:
                    # Normalize answers for comparison
                    df['correct_letter'] = df['correct_letter'].astype(str).str.strip().str.upper()
                    df[output_with_col] = df[output_with_col].astype(str).str.strip().str.upper()
                    df[output_without_col] = df[output_without_col].astype(str).str.strip().str.upper()
                    
                    df_clean = df.dropna(subset=['correct_letter', output_with_col, output_without_col])
                    
                    acc_with = (df_clean['correct_letter'] == df_clean[output_with_col]).mean()
                    acc_without = (df_clean['correct_letter'] == df_clean[output_without_col]).mean()
                    
                    mcq_results.append({
                        'Model': model.split('/')[-1],
                        'Prompt_Style': prompt_style.upper(),
                        'With_Panchvakya': acc_with,
                        'Without_Panchvakya': acc_without,
                        'Improvement': acc_with - acc_without,
                        'Question_Type': 'MCQ',
                        'Total_Questions': len(df_clean)
                    })
    
    # Collect numerical results
    for model in MODELS_TO_TEST:
        safe_model_name = model_name_to_filename(model)
        
        for prompt_style in ['cot', 'zero_shot']:
            num_file = os.path.join(NUMERICAL_RESULTS_DIR, f"{safe_model_name}_{prompt_style}.csv")
            if os.path.exists(num_file):
                df = pd.read_csv(num_file)
                
                # Calculate numerical accuracies
                acc_with_col = f'Accuracy_with_Panchvakya_{prompt_style}'
                acc_without_col = f'Accuracy_without_Panchvakya_{prompt_style}'
                
                if acc_with_col in df.columns and acc_without_col in df.columns:
                    acc_with = df[acc_with_col].mean()
                    acc_without = df[acc_without_col].mean()
                    
                    numerical_results.append({
                        'Model': model.split('/')[-1],
                        'Prompt_Style': prompt_style.upper(),
                        'With_Panchvakya': acc_with,
                        'Without_Panchvakya': acc_without,
                        'Improvement': acc_with - acc_without,
                        'Question_Type': 'Numerical',
                        'Total_Questions': len(df)
                    })
    
    # Combine results
    all_results = mcq_results + numerical_results
    
    if not all_results:
        print("No results found to analyze.")
        return
    
    results_df = pd.DataFrame(all_results)
    
    # Create visualizations
    create_comprehensive_plots(results_df)
    
    # Save summary table
    summary_file = os.path.join(FINAL_RESULTS_DIR, "comprehensive_summary.csv")
    results_df.to_csv(summary_file, index=False)
    print(f"Comprehensive summary saved to {summary_file}")
    
    # Print summary statistics
    print_summary_statistics(results_df)

def create_comprehensive_plots(results_df):
    """Create comprehensive visualization plots"""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Accuracy comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Comprehensive Model Performance Analysis: MCQ vs Numerical Questions', fontsize=16, fontweight='bold')
    
    # Plot 1: Accuracy with Panchvakya by question type
    ax1 = axes[0, 0]
    accuracy_data = results_df.pivot_table(values='With_Panchvakya', 
                                         index=['Model', 'Prompt_Style'], 
                                         columns='Question_Type', 
                                         aggfunc='mean').reset_index()
    
    if 'MCQ' in accuracy_data.columns and 'Numerical' in accuracy_data.columns:
        scatter = ax1.scatter(accuracy_data['MCQ'], accuracy_data['Numerical'], 
                            c=accuracy_data['Prompt_Style'].map({'COT': 'blue', 'ZERO_SHOT': 'red'}),
                            s=100, alpha=0.7)
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax1.set_xlabel('MCQ Accuracy (With Panchvakya)')
        ax1.set_ylabel('Numerical Accuracy (With Panchvakya)')
        ax1.set_title('MCQ vs Numerical Performance\n(With Panchvakya)')
        ax1.grid(True, alpha=0.3)
        
        # Add model labels
        for idx, row in accuracy_data.iterrows():
            ax1.annotate(f"{row['Model']}\n({row['Prompt_Style']})", 
                        (row['MCQ'], row['Numerical']), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Plot 2: Improvement comparison
    ax2 = axes[0, 1]
    improvement_data = results_df.pivot_table(values='Improvement', 
                                            index=['Model', 'Prompt_Style'], 
                                            columns='Question_Type', 
                                            aggfunc='mean').reset_index()
    
    if 'MCQ' in improvement_data.columns and 'Numerical' in improvement_data.columns:
        scatter = ax2.scatter(improvement_data['MCQ'], improvement_data['Numerical'], 
                            c=improvement_data['Prompt_Style'].map({'COT': 'blue', 'ZERO_SHOT': 'red'}),
                            s=100, alpha=0.7)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax2.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        ax2.set_xlabel('MCQ Improvement (Panchvakya Effect)')
        ax2.set_ylabel('Numerical Improvement (Panchvakya Effect)')
        ax2.set_title('Panchvakya Improvement:\nMCQ vs Numerical')
        ax2.grid(True, alpha=0.3)
        
        # Add model labels
        for idx, row in improvement_data.iterrows():
            ax2.annotate(f"{row['Model']}\n({row['Prompt_Style']})", 
                        (row['MCQ'], row['Numerical']), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Plot 3: Model comparison bar chart
    ax3 = axes[1, 0]
    avg_by_model = results_df.groupby(['Model', 'Question_Type']).agg({
        'With_Panchvakya': 'mean',
        'Without_Panchvakya': 'mean',
        'Improvement': 'mean'
    }).reset_index()
    
    models = avg_by_model['Model'].unique()
    x = np.arange(len(models))
    width = 0.35
    
    mcq_data = avg_by_model[avg_by_model['Question_Type'] == 'MCQ']
    num_data = avg_by_model[avg_by_model['Question_Type'] == 'Numerical']
    
    if not mcq_data.empty:
        bars1 = ax3.bar(x - width/2, mcq_data['With_Panchvakya'], width, 
                       label='MCQ (With Panchvakya)', alpha=0.8)
    if not num_data.empty:
        bars2 = ax3.bar(x + width/2, num_data['With_Panchvakya'], width, 
                       label='Numerical (With Panchvakya)', alpha=0.8)
    
    ax3.set_xlabel('Models')
    ax3.set_ylabel('Average Accuracy')
    ax3.set_title('Model Performance by Question Type\n(With Panchvakya)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Prompt style comparison
    ax4 = axes[1, 1]
    style_comparison = results_df.groupby(['Prompt_Style', 'Question_Type']).agg({
        'Improvement': 'mean'
    }).reset_index()
    
    if not style_comparison.empty:
        style_pivot = style_comparison.pivot(index='Question_Type', 
                                           columns='Prompt_Style', 
                                           values='Improvement')
        style_pivot.plot(kind='bar', ax=ax4)
        ax4.set_title('Panchvakya Improvement by Prompt Style')
        ax4.set_ylabel('Average Improvement')
        ax4.set_xlabel('Question Type')
        ax4.legend(title='Prompt Style')
        ax4.grid(True, alpha=0.3)
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=0)
    
    # Add legend for prompt styles
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='COT'),
                      Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='ZERO_SHOT')]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    
    # Save the plot
    plot_file = os.path.join(FINAL_RESULTS_DIR, "comprehensive_analysis.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Comprehensive analysis plot saved to {plot_file}")

def print_summary_statistics(results_df):
    """Print comprehensive summary statistics"""
    print("\n" + "="*80)
    print("COMPREHENSIVE PERFORMANCE SUMMARY")
    print("="*80)
    
    # Overall statistics
    if not results_df.empty:
        print(f"Total evaluations: {len(results_df)}")
        print(f"Models tested: {results_df['Model'].nunique()}")
        print(f"Question types: {', '.join(results_df['Question_Type'].unique())}")
        print(f"Prompt styles: {', '.join(results_df['Prompt_Style'].unique())}")
        
        print("\n" + "-"*60)
        print("AVERAGE PERFORMANCE BY QUESTION TYPE")
        print("-"*60)
        
        by_type = results_df.groupby('Question_Type').agg({
            'With_Panchvakya': ['mean', 'std'],
            'Without_Panchvakya': ['mean', 'std'],
            'Improvement': ['mean', 'std']
        }).round(4)
        
        for q_type in results_df['Question_Type'].unique():
            type_data = results_df[results_df['Question_Type'] == q_type]
            print(f"\n{q_type} Questions:")
            print(f"  Average accuracy WITH Panchvakya:    {type_data['With_Panchvakya'].mean():.2%} (±{type_data['With_Panchvakya'].std():.2%})")
            print(f"  Average accuracy WITHOUT Panchvakya: {type_data['Without_Panchvakya'].mean():.2%} (±{type_data['Without_Panchvakya'].std():.2%})")
            print(f"  Average improvement:                 {type_data['Improvement'].mean():.2%} (±{type_data['Improvement'].std():.2%})")
            print(f"  Number of evaluations:               {len(type_data)}")
        
        print("\n" + "-"*60)
        print("BEST PERFORMING COMBINATIONS")
        print("-"*60)
        
        # Best with Panchvakya
        best_with = results_df.loc[results_df['With_Panchvakya'].idxmax()]
        print(f"Best WITH Panchvakya: {best_with['Model']} ({best_with['Prompt_Style']}) on {best_with['Question_Type']} - {best_with['With_Panchvakya']:.2%}")
        
        # Best improvement
        best_improvement = results_df.loc[results_df['Improvement'].idxmax()]
        print(f"Best Improvement: {best_improvement['Model']} ({best_improvement['Prompt_Style']}) on {best_improvement['Question_Type']} - +{best_improvement['Improvement']:.2%}")
        
        # Models that benefit most from Panchvakya
        positive_improvements = results_df[results_df['Improvement'] > 0]
        if not positive_improvements.empty:
            avg_improvement_by_model = positive_improvements.groupby('Model')['Improvement'].mean().sort_values(ascending=False)
            print(f"\nModels benefiting most from Panchvakya:")
            for model, improvement in avg_improvement_by_model.head(3).items():
                print(f"  {model}: +{improvement:.2%}")

if __name__ == "__main__":
    create_comprehensive_analysis()
