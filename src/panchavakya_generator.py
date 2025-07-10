import pandas as pd
import groq
import time
import json
from tqdm import tqdm
from src.config import GROQ_API_KEY, PANCHAVAKYA_SYSTEM_PROMPT, RAW_DATA_PATH, PANCHAVAKYA_DATA_PATH
import os

client = groq.Groq(api_key=GROQ_API_KEY)

def make_prompt(df, index, add_system_prompt=True):
    question = df.loc[index, 'question']
    options_list = df.loc[index, 'multiple_choice']

    prompt = f"Question:\n{question}\nOptions:\n{options_list}\n"
    if add_system_prompt:
        prompt = PANCHAVAKYA_SYSTEM_PROMPT + "\n" + prompt
    return prompt

def evaluate_question(
    user_prompt: str,
    model: str = "qwen/qwen3-32b",
    max_tokens: int = 16000,
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
        response_format={"type": "json_object"},
        reasoning_format="parsed",
    )

    content = response.choices[0].message.content
    if debug:
        print(content)
    return content

def generate_panchavakya_arguments(num_samples=None):
    """
    Loads the raw data, generates Panchavakya arguments for each question,
    and saves the results to a new CSV.
    
    Args:
        num_samples: Optional integer to limit processing to the first N samples
    """
    if os.path.exists(PANCHAVAKYA_DATA_PATH):
        print(f"Panchavakya data already exists at {PANCHAVAKYA_DATA_PATH}. Skipping generation.")
        return pd.read_csv(PANCHAVAKYA_DATA_PATH)

    if not os.path.exists(RAW_DATA_PATH):
        print(f"Error: Raw data file not found at {RAW_DATA_PATH}.")
        print("Please place your dataset file in the data/raw directory with the name 'initial_questions.csv'.")
        return None

    df = pd.read_csv(RAW_DATA_PATH)
    df["Panchavakya"] = None
    
    # Limit to the specified number of samples if provided
    sample_size = min(len(df), num_samples) if num_samples else len(df)
    
    print(f"Generating Panchavakya arguments for {sample_size} questions...")
    for i in tqdm(range(sample_size), desc="Evaluating questions"):
        user_prompt = make_prompt(df, i)
        try:
            response = evaluate_question(user_prompt, debug=False)
            df.at[i, "Panchavakya"] = response
        except Exception as e:
            print(f"Error at index {i}: {e}")
            df.at[i, "Panchavakya"] = None

    # Parse the JSON strings into separate columns
    for key in ["Pratijna", "Hetu", "Udaharana", "Upanaya", "Nigamana"]:
        df[key] = None

    for i in tqdm(range(sample_size), desc="Parsing Panchavakya"):
        raw = df.at[i, "Panchavakya"]
        if pd.notna(raw):
            try:
                parsed = json.loads(raw)
                for key in ["Pratijna", "Hetu", "Udaharana", "Upanaya", "Nigamana"]:
                    df.at[i, key] = parsed.get(key)
            except Exception as e:
                print(f"Error parsing JSON at row {i}: {e}")

    df.to_csv(PANCHAVAKYA_DATA_PATH, index=False)
    print(f"Panchavakya arguments saved to {PANCHAVAKYA_DATA_PATH}")
    return df

if __name__ == '__main__':
    generate_panchavakya_arguments()
