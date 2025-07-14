import os
from src.panchavakya_generator import generate_panchavakya_arguments
from src.evaluator import run_evaluation
from src.analysis import analyze_results
from src.config import MODELS_TO_TEST, FINAL_RESULTS_DIR, model_name_to_filename
from src.setup import ensure_directories

# Configuration: Set to True to test all models, False to test only one model first
TEST_ALL_MODELS = True

# If testing single model, specify which one (index from MODELS_TO_TEST)
SINGLE_MODEL_INDEX = 0  # 0 = "qwen/qwen3-32b", 1 = "deepseek-r1-distill-llama-70b", etc.

def check_model_already_tested(model_name):
    """
    Check if a model has already been tested by looking for existing result files.
    Returns True if both COT and Zero-Shot results exist for the model.
    """
    safe_model_name = model_name_to_filename(model_name)
    cot_file = os.path.join(FINAL_RESULTS_DIR, f"{safe_model_name}_cot.csv")
    zero_shot_file = os.path.join(FINAL_RESULTS_DIR, f"{safe_model_name}_zero_shot.csv")
    
    return os.path.exists(cot_file) and os.path.exists(zero_shot_file)

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

    # Determine which models to test
    if TEST_ALL_MODELS:
        models_to_evaluate = MODELS_TO_TEST
        print(f"\n--- Testing ALL {len(models_to_evaluate)} models ---")
        
        # Check which models have already been tested
        already_tested = []
        not_tested = []
        for model in models_to_evaluate:
            if check_model_already_tested(model):
                already_tested.append(model)
            else:
                not_tested.append(model)
        
        if already_tested:
            print(f"--- Found {len(already_tested)} already tested models: {[model.split('/')[-1] for model in already_tested]} ---")
            print("--- Skipping already tested models ---")
        
        if not_tested:
            print(f"--- Will test {len(not_tested)} new models: {[model.split('/')[-1] for model in not_tested]} ---")
            models_to_evaluate = not_tested
        else:
            print("--- All models have already been tested! ---")
            models_to_evaluate = []
    else:
        single_model = MODELS_TO_TEST[SINGLE_MODEL_INDEX]
        if check_model_already_tested(single_model):
            print(f"\n--- Model {single_model} has already been tested ---")
            print("--- Skipping evaluation, proceeding to analysis ---")
            models_to_evaluate = []
            models_to_analyze = [single_model]
        else:
            models_to_evaluate = [single_model]
            models_to_analyze = [single_model]
            print(f"\n--- Testing SINGLE model: {single_model} ---")
        print(f"--- To test all models, set TEST_ALL_MODELS = True ---")

    # Step 2: Run evaluations for selected model(s) and prompt styles
    for model in models_to_evaluate:
        print(f"\n--- Step 2: Running Evaluation for {model} ---")
        run_evaluation(model, 'cot')
        run_evaluation(model, 'zero_shot')

    # Step 3: Analyze results for all relevant models
    if TEST_ALL_MODELS:
        # Analyze all models (both previously tested and newly tested)
        for model in MODELS_TO_TEST:
            if check_model_already_tested(model):
                print(f"\n--- Step 3: Analyzing Results for {model} ---")
                analyze_results(model, 'cot')
                analyze_results(model, 'zero_shot')
    else:
        # Analyze the single model
        for model in models_to_analyze:
            print(f"\n--- Step 3: Analyzing Results for {model} ---")
            analyze_results(model, 'cot')
            analyze_results(model, 'zero_shot')
    
    if not TEST_ALL_MODELS:
        if models_to_evaluate:
            print(f"\n--- Single model testing completed for {models_to_evaluate[0]} ---")
        print("--- If results look good, set TEST_ALL_MODELS = True to test all models ---")
    else:
        tested_count = len([m for m in MODELS_TO_TEST if check_model_already_tested(m)])
        print(f"\n--- Testing pipeline completed. {tested_count}/{len(MODELS_TO_TEST)} models have been tested ---")

if __name__ == "__main__":
    main()
