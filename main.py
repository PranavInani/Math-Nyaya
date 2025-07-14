import os
from src.panchavakya_generator import generate_panchavakya_arguments
from src.numerical_panchavakya_generator import generate_numerical_panchavakya_arguments
from src.evaluator import run_evaluation
from src.numerical_evaluator import run_numerical_evaluation, analyze_numerical_results
from src.analysis import analyze_results
from src.comprehensive_analysis import create_comprehensive_analysis
from src.config import MODELS_TO_TEST, FINAL_RESULTS_DIR, MCQ_RESULTS_DIR, NUMERICAL_RESULTS_DIR, model_name_to_filename
from src.setup import ensure_directories

# Configuration: Set to True to test all models, False to test only one model first
TEST_ALL_MODELS = True

# Configuration: Choose evaluation mode
# Options: "both", "mcq", "numerical"
EVALUATION_MODE = "numerical"  # Change to "mcq" or "numerical" to run only one type

# If testing single model, specify which one (index from MODELS_TO_TEST)
SINGLE_MODEL_INDEX = 0  # 0 = "qwen/qwen3-32b", 1 = "deepseek-r1-distill-llama-70b", etc.

def check_model_already_tested(model_name):
    """
    Check if a model has already been tested by looking for existing result files.
    Returns True if both COT and Zero-Shot results exist for the model.
    """
    safe_model_name = model_name_to_filename(model_name)
    cot_file = os.path.join(MCQ_RESULTS_DIR, f"{safe_model_name}_cot.csv")
    zero_shot_file = os.path.join(MCQ_RESULTS_DIR, f"{safe_model_name}_zero_shot.csv")
    
    return os.path.exists(cot_file) and os.path.exists(zero_shot_file)

def check_numerical_model_already_tested(model_name):
    """
    Check if a model has already been tested for numerical questions.
    Returns True if both COT and Zero-Shot numerical results exist for the model.
    """
    safe_model_name = model_name_to_filename(model_name)
    numerical_cot_file = os.path.join(NUMERICAL_RESULTS_DIR, f"{safe_model_name}_cot.csv")
    numerical_zero_shot_file = os.path.join(NUMERICAL_RESULTS_DIR, f"{safe_model_name}_zero_shot.csv")
    
    return os.path.exists(numerical_cot_file) and os.path.exists(numerical_zero_shot_file)

def main():
    """
    Main function to run the entire pipeline for both MCQ and numerical questions.
    """
    # Step 0: Setup directories
    print("--- Step 0: Setting Up Directories ---")
    ensure_directories()
    
    print(f"\n--- Evaluation Mode: {EVALUATION_MODE.upper()} ---")
    
    # Initialize flags for what to process
    process_mcq = EVALUATION_MODE in ["both", "mcq"]
    process_numerical = EVALUATION_MODE in ["both", "numerical"]
    
    mcq_data_available = False
    numerical_questions_available = False
    
    # Step 1: Generate Panchavakya arguments for MCQ questions
    if process_mcq:
        print("\n--- Step 1: Generating Panchavakya Arguments for MCQ Questions ---")
        df = generate_panchavakya_arguments()
        
        # Check if the MCQ data was loaded successfully
        if df is None:
            print("Warning: Could not load MCQ questions.")
            if EVALUATION_MODE == "mcq":
                print("Exiting due to missing MCQ raw data file.")
                return
        else:
            mcq_data_available = True
    
    # Step 1.5: Generate Panchavakya arguments for numerical questions
    if process_numerical:
        print("\n--- Step 1.5: Generating Panchavakya Arguments for Numerical Questions ---")
        numerical_df = generate_numerical_panchavakya_arguments()
        
        # Check if the numerical data was loaded successfully
        if numerical_df is None:
            print("Warning: Could not load numerical questions.")
            if EVALUATION_MODE == "numerical":
                print("Exiting due to missing numerical raw data file.")
                return
        else:
            numerical_questions_available = True

    # Determine which models to test based on the evaluation mode
    if TEST_ALL_MODELS:
        models_to_evaluate = MODELS_TO_TEST
        print(f"\n--- Testing ALL {len(models_to_evaluate)} models for {EVALUATION_MODE.upper()} evaluation ---")
        
        # Check which models have already been tested based on evaluation mode
        already_tested = []
        not_tested = []
        for model in models_to_evaluate:
            tested = False
            if process_mcq and not process_numerical:
                tested = check_model_already_tested(model)
            elif process_numerical and not process_mcq:
                tested = check_numerical_model_already_tested(model)
            else:  # both
                tested = check_model_already_tested(model) and check_numerical_model_already_tested(model)
            
            if tested:
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
        tested = False
        if process_mcq and not process_numerical:
            tested = check_model_already_tested(single_model)
        elif process_numerical and not process_mcq:
            tested = check_numerical_model_already_tested(single_model)
        else:  # both
            tested = check_model_already_tested(single_model) and check_numerical_model_already_tested(single_model)
        
        if tested:
            print(f"\n--- Model {single_model} has already been tested for {EVALUATION_MODE.upper()} ---")
            print("--- Skipping evaluation, proceeding to analysis ---")
            models_to_evaluate = []
            models_to_analyze = [single_model]
        else:
            models_to_evaluate = [single_model]
            models_to_analyze = [single_model]
            print(f"\n--- Testing SINGLE model: {single_model} for {EVALUATION_MODE.upper()} ---")
        print(f"--- To test all models, set TEST_ALL_MODELS = True ---")

    # Step 2: Run evaluations for selected model(s) and prompt styles
    for model in models_to_evaluate:
        if process_mcq and mcq_data_available:
            print(f"\n--- Step 2: Running MCQ Evaluation for {model} ---")
            run_evaluation(model, 'cot')
            run_evaluation(model, 'zero_shot')
        
        if process_numerical and numerical_questions_available:
            print(f"\n--- Step 2.5: Running Numerical Evaluation for {model} ---")
            run_numerical_evaluation(model, 'cot')
            run_numerical_evaluation(model, 'zero_shot')

    # Step 3: Analyze results for all relevant models
    if TEST_ALL_MODELS:
        # Analyze all models (both previously tested and newly tested)
        for model in MODELS_TO_TEST:
            if process_mcq and mcq_data_available and check_model_already_tested(model):
                print(f"\n--- Step 3: Analyzing MCQ Results for {model} ---")
                analyze_results(model, 'cot')
                analyze_results(model, 'zero_shot')
                
            # Analyze numerical results if available
            if process_numerical and numerical_questions_available and check_numerical_model_already_tested(model):
                print(f"\n--- Step 3.5: Analyzing Numerical Results for {model} ---")
                analyze_numerical_results(model, 'cot')
                analyze_numerical_results(model, 'zero_shot')
    else:
        # Analyze the single model
        for model in models_to_analyze:
            if process_mcq and mcq_data_available and check_model_already_tested(model):
                print(f"\n--- Step 3: Analyzing MCQ Results for {model} ---")
                analyze_results(model, 'cot')
                analyze_results(model, 'zero_shot')
            
            # Analyze numerical results if available
            if process_numerical and numerical_questions_available and check_numerical_model_already_tested(model):
                print(f"\n--- Step 3.5: Analyzing Numerical Results for {model} ---")
                analyze_numerical_results(model, 'cot')
                analyze_numerical_results(model, 'zero_shot')
    
    if not TEST_ALL_MODELS:
        if models_to_evaluate:
            print(f"\n--- Single model testing completed for {models_to_evaluate[0]} ({EVALUATION_MODE.upper()}) ---")
        print("--- If results look good, set TEST_ALL_MODELS = True to test all models ---")
    else:
        if process_mcq:
            mcq_tested_count = len([m for m in MODELS_TO_TEST if check_model_already_tested(m)])
            print(f"\n--- MCQ Testing pipeline completed. {mcq_tested_count}/{len(MODELS_TO_TEST)} models have been tested ---")
        if process_numerical:
            numerical_tested_count = len([m for m in MODELS_TO_TEST if check_numerical_model_already_tested(m)])
            print(f"\n--- Numerical Testing pipeline completed. {numerical_tested_count}/{len(MODELS_TO_TEST)} models have been tested ---")
    
    # Step 4: Create comprehensive analysis if we have results from multiple sources
    if EVALUATION_MODE == "both":
        print(f"\n--- Step 4: Creating Comprehensive Analysis ---")
        create_comprehensive_analysis()
    else:
        print(f"\n--- Step 4: Skipping comprehensive analysis (only running {EVALUATION_MODE.upper()} evaluation) ---")

if __name__ == "__main__":
    main()
