import pandas as pd
import groq
import time
import json
import re
import ast
from tqdm import tqdm
from src.config import GROQ_API_KEY, COT_SYSTEM_PROMPTS, ZERO_SHOT_SYSTEM_PROMPTS, PANCHAVAKYA_DATA_PATH, MCQ_RESULTS_DIR, model_name_to_filename
import os

client = groq.Groq(api_key=GROQ_API_KEY)

def make_evaluation_prompt(df, index, add_panchvakya=True, prompt_style='cot'):
    question = df.loc[index, 'question']
    
    # Safely evaluate the string representation of the list
    try:
        options_list = ast.literal_eval(df.loc[index, 'multiple_choice'])
    except (ValueError, SyntaxError):
        options_list = [] # Handle cases where it's not a valid list

    system_prompts = COT_SYSTEM_PROMPTS if prompt_style == 'cot' else ZERO_SHOT_SYSTEM_PROMPTS
    prompt_template = system_prompts[add_panchvakya]
    
    # Escape any curly braces in the template that aren't meant to be format placeholders
    prompt_template = prompt_template.replace("{", "{{").replace("}", "}}")
    
    # But restore the actual format placeholders
    for placeholder in ["question_text", "optionA", "optionB", "optionC", "optionD", 
                        "hetu", "udaharana", "upanaya"]:
        prompt_template = prompt_template.replace(f"{{{{{placeholder}}}}}", f"{{{placeholder}}}")

    options_dict = {f"option{chr(ord('A') + i)}": option for i, option in enumerate(options_list)}

    if add_panchvakya:
        prompt = prompt_template.format(
            question_text=question,
            **options_dict,
            hetu=df.loc[index, 'Hetu'],
            udaharana=df.loc[index, 'Udaharana'],
            upanaya=df.loc[index, 'Upanaya']
        )
    else:
        prompt = prompt_template.format(
            question_text=question,
            **options_dict
        )
    return prompt

def evaluate_llm_question(
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

def extract_answer_from_text(text):
    match = re.search(r'"answer"\s*:\s*"([A-Z])"', text)
    return match.group(1) if match else None

def safe_evaluate(prompt, model, retries=3, delay=2):
    for attempt in range(retries):
        try:
            response = evaluate_llm_question(prompt, model)
            try:
                parsed_answer = json.loads(response)['answer']
            except (json.JSONDecodeError, KeyError):
                parsed_answer = extract_answer_from_text(response)
            return parsed_answer, response
        except Exception as e:
            print(f"[Retry {attempt+1}] Error: {e}")
            time.sleep(delay)
    return None, None

def run_evaluation(model_name, prompt_style='cot', num_samples=None):
    """
    Runs the comparative evaluation for a given model and prompt style.
    
    Args:
        model_name: Name of the model to evaluate
        prompt_style: Style of prompting ('cot' or 'zero_shot')
        num_samples: Optional integer to limit processing to the first N samples
    """
    output_filename = os.path.join(MCQ_RESULTS_DIR, f"{model_name_to_filename(model_name)}_{prompt_style}.csv")
    if os.path.exists(output_filename):
        print(f"Results for {model_name} ({prompt_style}) already exist. Skipping.")
        return

    df = pd.read_csv(PANCHAVAKYA_DATA_PATH)
    
    # Limit to the specified number of samples if provided
    if num_samples:
        sample_size = min(len(df), num_samples)
        df = df.iloc[:sample_size].copy()
        print(f"Processing only the first {sample_size} questions for testing")

    for col in [f'Output_with_Panchvakya_{prompt_style}', f'Output_without_Panchvakya_{prompt_style}',
                f'Raw_with_Panchvakya_{prompt_style}', f'Raw_without_Panchvakya_{prompt_style}']:
        if col not in df.columns:
            df[col] = None

    print(f"Running evaluation for model: {model_name} with style: {prompt_style}")
    for i in tqdm(range(len(df))):
        # WITH Panchvakya
        prompt_with = make_evaluation_prompt(df, i, add_panchvakya=True, prompt_style=prompt_style)
        answer_with, raw_with = safe_evaluate(prompt_with, model_name)
        df.at[i, f'Output_with_Panchvakya_{prompt_style}'] = answer_with
        df.at[i, f'Raw_with_Panchvakya_{prompt_style}'] = raw_with

        # WITHOUT Panchvakya
        prompt_without = make_evaluation_prompt(df, i, add_panchvakya=False, prompt_style=prompt_style)
        answer_without, raw_without = safe_evaluate(prompt_without, model_name)
        df.at[i, f'Output_without_Panchvakya_{prompt_style}'] = answer_without
        df.at[i, f'Raw_without_Panchvakya_{prompt_style}'] = raw_without

    if not os.path.exists(MCQ_RESULTS_DIR):
        os.makedirs(MCQ_RESULTS_DIR)
        
    df.to_csv(output_filename, index=False)
    print(f"Evaluation results saved to {output_filename}")

if __name__ == '__main__':
    from src.config import MODELS_TO_TEST
    for model in MODELS_TO_TEST:
        run_evaluation(model, 'cot')
        run_evaluation(model, 'zero_shot')
