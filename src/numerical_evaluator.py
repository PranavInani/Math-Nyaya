import pandas as pd
import groq
import time
import json
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from src.config import GROQ_API_KEY, NUMERICAL_COT_SYSTEM_PROMPTS, NUMERICAL_ZERO_SHOT_SYSTEM_PROMPTS, NUMERICAL_PANCHAVAKYA_DATA_PATH, NUMERICAL_RESULTS_DIR, model_name_to_filename
import os

client = groq.Groq(api_key=GROQ_API_KEY)

def make_numerical_evaluation_prompt(df, index, add_panchvakya=True, prompt_style='cot'):
    question = df.loc[index, 'question']
    
    system_prompts = NUMERICAL_COT_SYSTEM_PROMPTS if prompt_style == 'cot' else NUMERICAL_ZERO_SHOT_SYSTEM_PROMPTS
    prompt_template = system_prompts[add_panchvakya]
    
    # Escape any curly braces in the template that aren't meant to be format placeholders
    prompt_template = prompt_template.replace("{", "{{").replace("}", "}}")
    
    # But restore the actual format placeholders
    for placeholder in ["question_text", "hetu", "udaharana", "upanaya"]:
        prompt_template = prompt_template.replace(f"{{{{{placeholder}}}}}", f"{{{placeholder}}}")

    if add_panchvakya:
        prompt = prompt_template.format(
            question_text=question,
            hetu=df.loc[index, 'Hetu'],
            udaharana=df.loc[index, 'Udaharana'],
            upanaya=df.loc[index, 'Upanaya']
        )
    else:
        prompt = prompt_template.format(
            question_text=question
        )
    return prompt

def evaluate_numerical_llm_question(
    user_prompt: str,
    model: str,
    max_tokens: int = 8000,
    temperature: float = 0,
    top_p: float = 0.95,
    debug: bool = False,
):
    messages = [{"role": "user", "content": user_prompt}]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=max_tokens,
        top_p=top_p,
        stream=False,
        stop=None,
        seed=1234,
    )

    content = response.choices[0].message.content
    if debug:
        print(content)
    return content

def extract_numerical_answer_from_text(text):
    """Extract numerical answer from text response"""
    # First try to extract from JSON format
    json_match = re.search(r'"answer"\s*:\s*"([^"]*)"', text)
    if json_match:
        return json_match.group(1).strip()
    
    # Try without quotes
    json_match = re.search(r'"answer"\s*:\s*([^,}]*)', text)
    if json_match:
        return json_match.group(1).strip()
    
    # Try to find numbers in the response
    number_match = re.search(r'(\d+\.?\d*)', text)
    if number_match:
        return number_match.group(1)
    
    return None

def safe_numerical_evaluate(prompt, model, retries=3, delay=2):
    for attempt in range(retries):
        try:
            response = evaluate_numerical_llm_question(prompt, model)
            try:
                parsed_answer = json.loads(response)['answer']
            except (json.JSONDecodeError, KeyError):
                parsed_answer = extract_numerical_answer_from_text(response)
            return parsed_answer, response
        except Exception as e:
            print(f"[Retry {attempt+1}] Error: {e}")
            time.sleep(delay)
    return None, None

def calculate_numerical_accuracy(predicted, actual):
    """Calculate accuracy for numerical answers"""
    try:
        pred_val = float(str(predicted).strip())
        actual_val = float(str(actual).strip())
        
        # Consider answers correct if they match exactly or are very close (within 0.01%)
        if abs(pred_val - actual_val) < 0.0001 * max(abs(actual_val), 1):
            return 1
        else:
            return 0
    except (ValueError, TypeError):
        # If we can't convert to float, check for exact string match
        return 1 if str(predicted).strip() == str(actual).strip() else 0

def run_numerical_evaluation(model_name, prompt_style='cot', num_samples=None):
    """
    Runs the numerical evaluation for a given model and prompt style.
    
    Args:
        model_name: Name of the model to evaluate
        prompt_style: Style of prompting ('cot' or 'zero_shot')
        num_samples: Optional integer to limit processing to the first N samples
    """
    output_filename = os.path.join(NUMERICAL_RESULTS_DIR, f"{model_name_to_filename(model_name)}_{prompt_style}.csv")
    if os.path.exists(output_filename):
        print(f"Numerical results for {model_name} ({prompt_style}) already exist. Skipping.")
        return

    if not os.path.exists(NUMERICAL_PANCHAVAKYA_DATA_PATH):
        print(f"Numerical Panchavakya data not found at: {NUMERICAL_PANCHAVAKYA_DATA_PATH}")
        print("Please run numerical panchavakya generation first.")
        return

    df = pd.read_csv(NUMERICAL_PANCHAVAKYA_DATA_PATH)
    
    # Limit to the specified number of samples if provided
    if num_samples:
        sample_size = min(len(df), num_samples)
        df = df.iloc[:sample_size].copy()
        print(f"Processing only the first {sample_size} numerical questions for testing")

    for col in [f'Output_with_Panchvakya_{prompt_style}', f'Output_without_Panchvakya_{prompt_style}',
                f'Raw_with_Panchvakya_{prompt_style}', f'Raw_without_Panchvakya_{prompt_style}',
                f'Accuracy_with_Panchvakya_{prompt_style}', f'Accuracy_without_Panchvakya_{prompt_style}']:
        if col not in df.columns:
            df[col] = None

    print(f"Running numerical evaluation for model: {model_name} with style: {prompt_style}")
    for i in tqdm(range(len(df))):
        # WITH Panchvakya
        prompt_with = make_numerical_evaluation_prompt(df, i, add_panchvakya=True, prompt_style=prompt_style)
        answer_with, raw_with = safe_numerical_evaluate(prompt_with, model_name)
        df.at[i, f'Output_with_Panchvakya_{prompt_style}'] = answer_with
        df.at[i, f'Raw_with_Panchvakya_{prompt_style}'] = raw_with
        
        # Calculate accuracy
        if answer_with is not None:
            accuracy_with = calculate_numerical_accuracy(answer_with, df.loc[i, 'Answer'])
            df.at[i, f'Accuracy_with_Panchvakya_{prompt_style}'] = accuracy_with

        # WITHOUT Panchvakya
        prompt_without = make_numerical_evaluation_prompt(df, i, add_panchvakya=False, prompt_style=prompt_style)
        answer_without, raw_without = safe_numerical_evaluate(prompt_without, model_name)
        df.at[i, f'Output_without_Panchvakya_{prompt_style}'] = answer_without
        df.at[i, f'Raw_without_Panchvakya_{prompt_style}'] = raw_without
        
        # Calculate accuracy
        if answer_without is not None:
            accuracy_without = calculate_numerical_accuracy(answer_without, df.loc[i, 'Answer'])
            df.at[i, f'Accuracy_without_Panchvakya_{prompt_style}'] = accuracy_without

    if not os.path.exists(NUMERICAL_RESULTS_DIR):
        os.makedirs(NUMERICAL_RESULTS_DIR)
        
    df.to_csv(output_filename, index=False)
    print(f"Numerical evaluation results saved to {output_filename}")

def analyze_numerical_results(model_name, prompt_style='cot'):
    """Analyze and visualize numerical evaluation results"""
    output_filename = os.path.join(NUMERICAL_RESULTS_DIR, f"{model_name_to_filename(model_name)}_{prompt_style}.csv")
    
    if not os.path.exists(output_filename):
        print(f"Results file not found: {output_filename}")
        return
    
    df = pd.read_csv(output_filename)
    
    # Calculate accuracy statistics
    with_panchvakya_col = f'Accuracy_with_Panchvakya_{prompt_style}'
    without_panchvakya_col = f'Accuracy_without_Panchvakya_{prompt_style}'
    
    if with_panchvakya_col in df.columns and without_panchvakya_col in df.columns:
        with_accuracy = df[with_panchvakya_col].mean()
        without_accuracy = df[without_panchvakya_col].mean()
        
        print(f"\n--- Numerical Results for {model_name} ({prompt_style}) ---")
        print(f"Accuracy WITH Panchvakya: {with_accuracy:.2%}")
        print(f"Accuracy WITHOUT Panchvakya: {without_accuracy:.2%}")
        print(f"Improvement: {with_accuracy - without_accuracy:.2%}")
        print(f"Total questions: {len(df)}")
        
        # Create visualizations
        create_numerical_visualizations(df, model_name, prompt_style, with_accuracy, without_accuracy)

def create_numerical_visualizations(df, model_name, prompt_style, with_accuracy, without_accuracy):
    """Create comprehensive visualizations for numerical results"""
    
    # Set up the figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Numerical Analysis: {model_name} ({prompt_style})', fontsize=16, fontweight='bold')
    
    # 1. Accuracy Comparison Bar Chart
    ax1 = axes[0, 0]
    categories = ['With Panchvakya', 'Without Panchvakya']
    accuracies = [with_accuracy, without_accuracy]
    colors = ['#2E8B57', '#CD5C5C']  # Sea green and Indian red
    
    bars = ax1.bar(categories, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Overall Accuracy Comparison')
    ax1.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # Add improvement annotation
    improvement = with_accuracy - without_accuracy
    if improvement > 0:
        ax1.annotate(f'Improvement: +{improvement:.1%}', 
                    xy=(0.5, max(accuracies) + 0.05), ha='center', 
                    fontsize=12, fontweight='bold', color='green')
    elif improvement < 0:
        ax1.annotate(f'Decline: {improvement:.1%}', 
                    xy=(0.5, max(accuracies) + 0.05), ha='center', 
                    fontsize=12, fontweight='bold', color='red')
    
    # 2. Question-by-Question Accuracy
    ax2 = axes[0, 1]
    with_panchvakya_col = f'Accuracy_with_Panchvakya_{prompt_style}'
    without_panchvakya_col = f'Accuracy_without_Panchvakya_{prompt_style}'
    
    question_indices = range(1, len(df) + 1)
    ax2.plot(question_indices, df[with_panchvakya_col], 'o-', color='#2E8B57', 
             linewidth=2, markersize=6, label='With Panchvakya')
    ax2.plot(question_indices, df[without_panchvakya_col], 's-', color='#CD5C5C', 
             linewidth=2, markersize=6, label='Without Panchvakya')
    
    ax2.set_xlabel('Question Number')
    ax2.set_ylabel('Accuracy (0=Wrong, 1=Correct)')
    ax2.set_title('Question-by-Question Performance')
    ax2.set_ylim(-0.1, 1.1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Accuracy Distribution (if we have multiple questions)
    ax3 = axes[1, 0]
    if len(df) > 1:
        # Create bins for accuracy distribution
        with_correct = df[with_panchvakya_col].sum()
        without_correct = df[without_panchvakya_col].sum()
        
        categories = ['Correct\n(With Panchvakya)', 'Wrong\n(With Panchvakya)', 
                     'Correct\n(Without Panchvakya)', 'Wrong\n(Without Panchvakya)']
        values = [with_correct, len(df) - with_correct, without_correct, len(df) - without_correct]
        colors = ['#2E8B57', '#90EE90', '#CD5C5C', '#FFA07A']
        
        bars = ax3.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax3.set_ylabel('Number of Questions')
        ax3.set_title('Correct vs Wrong Distribution')
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(val)}', ha='center', va='bottom', fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'Not enough data\nfor distribution', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Accuracy Distribution')
    
    # 4. Performance Summary Table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create summary statistics
    summary_data = [
        ['Metric', 'With Panchvakya', 'Without Panchvakya', 'Difference'],
        ['Accuracy', f'{with_accuracy:.1%}', f'{without_accuracy:.1%}', f'{improvement:+.1%}'],
        ['Correct Answers', f'{df[with_panchvakya_col].sum():.0f}/{len(df)}', 
         f'{df[without_panchvakya_col].sum():.0f}/{len(df)}', 
         f'{df[with_panchvakya_col].sum() - df[without_panchvakya_col].sum():+.0f}'],
        ['Total Questions', f'{len(df)}', f'{len(df)}', '0']
    ]
    
    # Create table
    table = ax4.table(cellText=summary_data[1:], colLabels=summary_data[0],
                     cellLoc='center', loc='center', bbox=[0, 0.3, 1, 0.6])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(summary_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax4.set_title('Performance Summary', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = os.path.join(NUMERICAL_RESULTS_DIR, f"{model_name_to_filename(model_name)}_{prompt_style}_analysis.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Numerical analysis plot saved to {plot_filename}")
    plt.close()

if __name__ == '__main__':
    from src.config import MODELS_TO_TEST
    for model in MODELS_TO_TEST[:1]:  # Test with first model only
        run_numerical_evaluation(model, 'cot', num_samples=5)
        run_numerical_evaluation(model, 'zero_shot', num_samples=5)
        analyze_numerical_results(model, 'cot')
        analyze_numerical_results(model, 'zero_shot')
