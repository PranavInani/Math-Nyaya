import pandas as pd
import groq
import time
import json
from tqdm import tqdm
from src.config import GROQ_API_KEY, PANCHAVAKYA_SYSTEM_PROMPT, GSM8K_DATA_PATH, NUMERICAL_PANCHAVAKYA_DATA_PATH
import os

client = groq.Groq(api_key=GROQ_API_KEY)

def make_numerical_prompt(df, index, add_system_prompt=True):
    question = df.loc[index, 'question']
    solution = df.loc[index, 'Solution']
    answer = df.loc[index, 'Answer']

    prompt = f"Question:\n{question}\nSolution:\n{solution}\nAnswer:\n{answer}\n"
    if add_system_prompt:
        prompt = PANCHAVAKYA_SYSTEM_PROMPT + "\n" + prompt
    return prompt

def evaluate_numerical_question(
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

def generate_numerical_panchavakya_arguments(num_samples=None):
    """
    Loads the GSM8K data, generates Panchavakya arguments for each numerical question,
    and saves the results to a new CSV.
    
    Args:
        num_samples: Optional integer to limit processing to the first N samples
    """
    # Check if the GSM8K data file exists
    if not os.path.exists(GSM8K_DATA_PATH):
        print(f"GSM8K data file not found at: {GSM8K_DATA_PATH}")
        print("Please ensure you have the GSM8K questions in the correct location.")
        return None

    # Read the raw data
    df = pd.read_csv(GSM8K_DATA_PATH)
    print(f"Loaded {len(df)} numerical questions from GSM8K dataset")
    
    # Limit to the specified number of samples if provided
    if num_samples:
        sample_size = min(len(df), num_samples)
        df = df.iloc[:sample_size].copy()
        print(f"Processing only the first {sample_size} questions for testing")

    # Add new columns for Panchavakya elements
    for col in ['Panchavakya', 'Pratijna', 'Hetu', 'Udaharana', 'Upanaya', 'Nigamana']:
        if col not in df.columns:
            df[col] = None

    # Check if processed file already exists and load existing progress
    if os.path.exists(NUMERICAL_PANCHAVAKYA_DATA_PATH):
        existing_df = pd.read_csv(NUMERICAL_PANCHAVAKYA_DATA_PATH)
        print(f"Found existing processed file with {len(existing_df)} rows")
        
        # Merge existing results with current data
        df = df.merge(existing_df[['question', 'Panchavakya', 'Pratijna', 'Hetu', 'Udaharana', 'Upanaya', 'Nigamana']], 
                     on='question', how='left', suffixes=('', '_existing'))
        
        # Copy existing results where available
        for col in ['Panchavakya', 'Pratijna', 'Hetu', 'Udaharana', 'Upanaya', 'Nigamana']:
            existing_col = f"{col}_existing"
            if existing_col in df.columns:
                df[col] = df[col].fillna(df[existing_col])
                df.drop(columns=[existing_col], inplace=True)

    # Process each question
    print("Generating Panchavakya arguments for numerical questions...")
    for i in tqdm(range(len(df))):
        # Skip if already processed
        if pd.notna(df.at[i, 'Panchavakya']):
            continue
        
        try:
            # Generate the prompt for this question
            prompt = make_numerical_prompt(df, i)
            
            # Evaluate the question to get Panchavakya
            response = evaluate_numerical_question(prompt)
            
            # Parse the JSON response
            try:
                panchavakya_data = json.loads(response)
                
                # Store the full response
                df.at[i, 'Panchavakya'] = response
                
                # Extract individual components
                df.at[i, 'Pratijna'] = panchavakya_data.get('Pratijna', '')
                df.at[i, 'Hetu'] = panchavakya_data.get('Hetu', '')
                df.at[i, 'Udaharana'] = panchavakya_data.get('Udaharana', '')
                df.at[i, 'Upanaya'] = panchavakya_data.get('Upanaya', '')
                df.at[i, 'Nigamana'] = panchavakya_data.get('Nigamana', '')
                
            except json.JSONDecodeError:
                print(f"Failed to parse JSON for question {i}: {response}")
                df.at[i, 'Panchavakya'] = response  # Store raw response
                
        except Exception as e:
            print(f"Error processing question {i}: {e}")
            time.sleep(2)  # Wait before retrying
            continue
        
        # Save progress every 10 questions
        if (i + 1) % 10 == 0:
            save_numerical_progress(df)
    
    # Final save
    save_numerical_progress(df)
    print(f"Generated Panchavakya arguments for numerical questions and saved to {NUMERICAL_PANCHAVAKYA_DATA_PATH}")
    return df

def save_numerical_progress(df):
    """Save the current progress to the processed data file"""
    # Ensure the directory exists
    os.makedirs(os.path.dirname(NUMERICAL_PANCHAVAKYA_DATA_PATH), exist_ok=True)
    
    # Save the dataframe
    df.to_csv(NUMERICAL_PANCHAVAKYA_DATA_PATH, index=False)

if __name__ == '__main__':
    generate_numerical_panchavakya_arguments()
