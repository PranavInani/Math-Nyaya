# Math-Nyaya

This project evaluates whether providing a structured logical argument, based on the ancient Indian school of logic called **Nyāya (Panchavakya)**, improves the performance of Large Language Models (LLMs) on multiple-choice questions.

## Project Structure

```
/home/pranav/projects/Math-Nyaya/
├── .env                    # Contains your GROQ API key
├── .gitignore             # Git ignore file
├── requirements.txt       # Python dependencies
├── main.py                # Main script to run the full pipeline
├── test_main.py           # Test script for a limited number of samples
├── data/
│   ├── raw/               # Place your raw CSV data here
│   │   └── initial_questions.csv  # Required input file
│   └── processed/         # Generated output files
│       ├── questions_with_panchavakya.csv
│       └── final_results/ # Results from model evaluations
└── src/                   # Source code
    ├── __init__.py
    ├── config.py          # Configuration settings
    ├── panchavakya_generator.py  # Generates Panchavakya arguments
    ├── evaluator.py       # Evaluates models with/without Panchavakya
    ├── analysis.py        # Analyzes and visualizes results
    └── setup.py           # Sets up the directory structure
```

## How to Use

1. **Setup Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Add API Key**:
   Create a `.env` file in the root directory with your GROQ API key:
   ```
   GROQ_API_KEY=your-api-key-here
   ```

3. **Prepare Data**:
   Place your dataset in the `data/raw` directory with the filename `initial_questions.csv`. The dataset should have these columns:
   - `question`: The question text
   - `multiple_choice`: A list of options
   - `correct_letter`: The correct answer letter (A, B, C, D, etc.)

4. **Run the Pipeline**:
   - For testing with a limited number of questions:
     ```bash
     python test_main.py
     ```
   - For the full dataset:
     ```bash
     python main.py
     ```

5. **View Results**:
   Results will be saved in `data/processed/final_results/` with CSV files and visualizations for each model and prompt style.

6. **Generate Comparison Charts**:
   To create comprehensive comparison charts across all models:
   ```bash
   python compare_all_models.py
   ```
   This will generate:
   - A comprehensive comparison chart saved as `model_comparision.png`
   - Detailed question-by-question results tables for COT and Zero-Shot strategies
   - CSV files with detailed performance data

## Results

![Model Comparison](data/processed/final_results/model_comparision.png)

The comprehensive comparison chart shows the performance of all evaluated models using both Chain-of-Thought (COT) and Zero-Shot prompting, with and without Panchavakya logical reasoning framework.

### Question-by-Question Analysis

The project also generates detailed question-by-question results tables that show exactly which questions each model got correct or incorrect:

- **Chain-of-Thought (COT) Results Table**: `question_results_cot_table.png`
  - Shows performance for each model using COT prompting
  - Displays results both with and without Panchavakya framework
  - ✓ indicates correct answers, ✗ indicates incorrect answers

- **Zero-Shot Results Table**: `question_results_zeroshot_table.png`
  - Shows performance for each model using Zero-Shot prompting
  - Displays results both with and without Panchavakya framework
  - Color-coded cells: Green for correct, Red for incorrect

These tables provide a granular view of model performance, making it easy to identify:
- Which questions are most challenging across all models
- How the Panchavakya framework affects specific question performance
- Model-specific strengths and weaknesses on individual questions

### Data Files

The analysis generates several CSV files for further investigation:
- `detailed_question_results.csv`: Raw question-by-question performance data
- `question_results_table.csv`: Pivot table format for easy analysis
- Individual model result files with `_cot.csv` and `_zero_shot.csv` suffixes

## Models

The following models are being tested:
- qwen/qwen3-32b
- deepseek-r1-distill-llama-70b
- gemma2-9b-it
- llama-3.1-8b-instant
- llama-3.3-70b-versatile
- mistral-saba-24b
