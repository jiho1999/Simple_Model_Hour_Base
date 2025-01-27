import os
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# File path to your directory
file_path = '/Users/jihopark/Desktop/Jiho_IS/Lung_Epithelial_Simulation/Simple Model'

# Prepare a list to store results
results = []

def div_mig_slope_calculation(file_path):
    # Loop through the files in the directory
    for file in os.listdir(file_path):
        if file.endswith('.xlsx') and 'division_migration_senescence' in file:
            # Read the first sheet of each file using the 'openpyxl' engine
            filepath = os.path.join(file_path, file)
            try:
                df = pd.read_excel(filepath, engine='openpyxl')  # Specify the engine as 'openpyxl'
            except Exception as e:
                print(f"Error reading {file}: {e}")
                continue

            # Extract wound closure column
            wound_closure_step = df['Wound Closure Step'].iloc[0]

            # Calculate the range of steps to consider: wound_closure_step +- 5
            time_steps = range(wound_closure_step - 4, wound_closure_step + 7)
            filtered_df = df[df['Step'].isin(time_steps)]
            
            # Ensure we have 11 time steps (5 before, 1 at closure, 5 after)
            if len(filtered_df) == 11:
                steps = filtered_df['Step'].values.reshape(-1, 1)

                # Calculate slope for Division Count
                division_values = filtered_df['Division Count'].values
                model = LinearRegression()
                model.fit(steps, division_values)
                division_slope = model.coef_[0]

                # Calculate slope for Migration Count
                migration_values = filtered_df['Migration Count'].values
                model.fit(steps, migration_values)
                migration_slope = model.coef_[0]

                # Extract the probability and run number from the file name
                probability = file.split('_')[3]  # e.g. "1.0e-01"
                run_number = file.split('_')[5].split('.')[0]  # e.g. "1"

                # Append the result for this file
                results.append({
                    'File': file,
                    'Senescence Probability': probability,
                    'Run Number': run_number,
                    'Division Slope': division_slope,
                    'Migration Slope': migration_slope
                })

    # Convert the results to a DataFrame
    results_df = pd.DataFrame(results)

    # Save the results to an Excel file
    output_file = '/Users/jihopark/Desktop/Jiho_IS/Lung_Epithelial_Simulation/Simple Model/senescence_slope_results.xlsx'
    results_df.to_excel(output_file, index=False)

    print(f"Results saved to {output_file}")

# div_mig_slope_calculation(file_path)

def senescence_slope_calculation(file_path):
    # Loop through the files in the directory
    for file in os.listdir(file_path):
        if file.endswith('.xlsx') and 'division_migration_senescence' in file:
            # Read the first sheet of each file using the 'openpyxl' engine
            filepath = os.path.join(file_path, file)
            try:
                df = pd.read_excel(filepath, engine='openpyxl')  # Specify the engine as 'openpyxl'
            except Exception as e:
                print(f"Error reading {file}: {e}")
                continue
    
            # Extract wound closure column
            wound_closure_step = df['Wound Closure Step'].iloc[0]

            time_steps = range(wound_closure_step, 100)
            filtered_df = df[df['Step'].isin(time_steps)]

            steps = filtered_df['Step'].values.reshape(-1, 1)

            # Calculate slope for Division Count
            senescent_values = filtered_df['Senescent_Count'].values
            model = LinearRegression()
            model.fit(steps, senescent_values)
            senescent_slope = model.coef_[0]

            # Extract the probability and run number from the file name
            probability = file.split('_')[3]  # e.g. "1.0e-01"
            run_number = file.split('_')[5].split('.')[0]  # e.g. "1"

            # Append the result for this file
            results.append({
                'File': file,
                'Senescence Probability': probability,
                'Run Number': run_number,
                'Senescent Slope': senescent_slope,
            })

    # Convert the results to a DataFrame
    results_df = pd.DataFrame(results)

    # Save the results to an Excel file
    output_file = '/Users/jihopark/Desktop/Jiho_IS/Lung_Epithelial_Simulation/Simple Model/senescence_slope_results.xlsx'
    results_df.to_excel(output_file, index=False)

    print(f"Results saved to {output_file}")

def permeability_slope_calculation(file_path):
    # Loop through the files in the directory
    for file in os.listdir(file_path):
        if file.endswith('.xlsx') and 'division_migration_senescence' in file:
            # Read the first sheet of each file using the 'openpyxl' engine
            filepath = os.path.join(file_path, file)
            try:
                df = pd.read_excel(filepath, engine='openpyxl')  # Specify the engine as 'openpyxl'
            except Exception as e:
                print(f"Error reading {file}: {e}")
                continue
    
            steps = df['Step'].values.reshape(-1, 1)

            # Calculate slope for Division Count
            senescent_values = df['Average Permeability'].values
            model = LinearRegression()
            model.fit(steps, senescent_values)
            senescent_slope = model.coef_[0]

            # Extract the probability and run number from the file name
            probability = file.split('_')[3]  # e.g. "1.0e-01"
            run_number = file.split('_')[5].split('.')[0]  # e.g. "1"

            # Append the result for this file
            results.append({
                'File': file,
                'Senescence Probability': probability,
                'Run Number': run_number,
                'Permeability Slope': senescent_slope,
            })

    # Convert the results to a DataFrame
    results_df = pd.DataFrame(results)

    # Save the results to an Excel file
    output_file = '/Users/jihopark/Desktop/Jiho_IS/Lung_Epithelial_Simulation/Simple Model/permeability_slope_results.xlsx'
    results_df.to_excel(output_file, index=False)

    print(f"Results saved to {output_file}")
