from src.panchavakya_generator import generate_panchavakya_arguments
from src.evaluator import run_evaluation
from src.analysis import analyze_results
from src.config import MODELS_TO_TEST
from src.setup import ensure_directories

def main(num_samples=5):
    """
    Main function to run the entire pipeline with a limited number of samples.
    
    Args:
        num_samples: Number of questions to process (default=5)
    """
    # Step 0: Setup directories
    print("--- Step 0: Setting Up Directories ---")
    ensure_directories()
    
    # Step 1: Generate Panchavakya arguments
    print("\n--- Step 1: Generating Panchavakya Arguments ---")
    df = generate_panchavakya_arguments(num_samples=num_samples)
    
    # Check if the data was loaded successfully
    if df is None:
        print("Exiting due to missing raw data file.")
        return

    # Step 2: Run evaluations for each model and prompt style
    # Just use one model for testing to save time
    model = MODELS_TO_TEST[0]
    print(f"\n--- Step 2: Running Evaluation for {model} ---")
    run_evaluation(model, 'cot', num_samples=num_samples)
    run_evaluation(model, 'zero_shot', num_samples=num_samples)

    # Step 3: Analyze results for each model and prompt style
    print(f"\n--- Step 3: Analyzing Results for {model} ---")
    analyze_results(model, 'cot')
    analyze_results(model, 'zero_shot')

if __name__ == "__main__":
    main(num_samples=5)  # Only process the first 5 questions
