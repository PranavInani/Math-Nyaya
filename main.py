from src.panchavakya_generator import generate_panchavakya_arguments
from src.evaluator import run_evaluation
from src.analysis import analyze_results
from src.config import MODELS_TO_TEST
from src.setup import ensure_directories

def main():
    """
    Main function to run the entire pipeline.
    """
    # Step 0: Setup directories
    print("--- Step 0: Setting Up Directories ---")
    ensure_directories()
    
    # Step 1: Generate Panchavakya arguments
    print("\n--- Step 1: Generating Panchavakya Arguments ---")
    df = generate_panchavakya_arguments()
    
    # Check if the data was loaded successfully
    if df is None:
        print("Exiting due to missing raw data file.")
        return

    # Step 2: Run evaluations for each model and prompt style
    for model in MODELS_TO_TEST:
        print(f"\n--- Step 2: Running Evaluation for {model} ---")
        run_evaluation(model, 'cot')
        run_evaluation(model, 'zero_shot')

    # Step 3: Analyze results for each model and prompt style
    for model in MODELS_TO_TEST:
        print(f"\n--- Step 3: Analyzing Results for {model} ---")
        analyze_results(model, 'cot')
        analyze_results(model, 'zero_shot')

if __name__ == "__main__":
    main()
