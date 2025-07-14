import os

def ensure_directories():
    """
    Create all the necessary directories for the project if they don't exist.
    """
    from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, FINAL_RESULTS_DIR, MCQ_RESULTS_DIR, NUMERICAL_RESULTS_DIR
    
    directories = [RAW_DATA_DIR, PROCESSED_DATA_DIR, FINAL_RESULTS_DIR, MCQ_RESULTS_DIR, NUMERICAL_RESULTS_DIR]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

if __name__ == "__main__":
    ensure_directories()
