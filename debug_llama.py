import pandas as pd

# Load Llama 3.3 COT results
df = pd.read_csv('data/processed/final_results/numerical/llama-3.3-70b-versatile_cot.csv')
print('Dataset shape:', df.shape)
print('\nColumns:', df.columns.tolist())

# Check the key columns
print('\n--- FIRST 5 ROWS OF KEY COLUMNS ---')
key_cols = ['Answer', 'Output_with_Panchvakya_cot', 'Output_without_Panchvakya_cot', 
            'Accuracy_with_Panchvakya_cot', 'Accuracy_without_Panchvakya_cot']
print(df[key_cols].head())

print('\n--- ACCURACY STATISTICS ---')
print('With Panchvakya accuracy distribution:')
print(df['Accuracy_with_Panchvakya_cot'].value_counts())
print('\nWithout Panchvakya accuracy distribution:')
print(df['Accuracy_without_Panchvakya_cot'].value_counts())

print('\n--- SAMPLE COMPARISONS ---')
for i in range(3):
    print(f'\nQuestion {i+1}:')
    print(f'  Correct Answer: {df.iloc[i]["Answer"]}')
    print(f'  Model Output (with): {df.iloc[i]["Output_with_Panchvakya_cot"]}')
    print(f'  Model Output (without): {df.iloc[i]["Output_without_Panchvakya_cot"]}')
    print(f'  Accuracy (with): {df.iloc[i]["Accuracy_with_Panchvakya_cot"]}')
    print(f'  Accuracy (without): {df.iloc[i]["Accuracy_without_Panchvakya_cot"]}')

# Check if outputs are null
print('\n--- NULL VALUES CHECK ---')
print('Null values in Output_with_Panchvakya_cot:', df['Output_with_Panchvakya_cot'].isnull().sum())
print('Null values in Output_without_Panchvakya_cot:', df['Output_without_Panchvakya_cot'].isnull().sum())
