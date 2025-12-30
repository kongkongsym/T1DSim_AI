import os
import sys
import pandas as pd
import numpy as np
import glob
from pathlib import Path

# Add current directory to path to allow importing from trainDigitalTwin
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trainDigitalTwin import trainModel
from t1dsim_ai.options import input_ind

def reproduce_results(n_epochs=150):
    # 1. Setup paths and parameters
    # Path to the T1DEXI subjects data
    base_path = Path("../src/t1dsim_ai/models/IndividualModel/")
    
    # Find all subject folders that look like 'T1DEXI-*'
    subject_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("T1DEXI")]
    
    print(f"Found {len(subject_dirs)} subjects for reproduction: {[d.name for d in subject_dirs]}")
    
    # Hyperparameters (same as trainDigitalTwin.py)
    n_neurons = 128
    hidden_compartments = {
        "models": [5 + len(input_ind), n_neurons, n_neurons // 2, n_neurons // 4, 1]
    }
    lr = 10 ** (-4)
    batch_size = 32
    overlap = 0.9
    seq_len = 60

    results_list = []

    # 2. Train loop
    for subj_dir in subject_dirs:
        subj_id = subj_dir.name
        csv_file = subj_dir / f"T1DEXIMAIN_{subj_id}.csv"
        
        if not csv_file.exists():
            print(f"Skipping {subj_id}: Data file not found at {csv_file}")
            continue
            
        print(f"\n{'='*20}\nProcessing Subject: {subj_id}\n{'='*20}")
        
        # Load data
        df_data_subj = pd.read_csv(csv_file)
        
        # --- Preprocessing: Rename columns to match trainDigitalTwin expectation ---
        # The T1DEXI data has different column names than the example data.
        # We need to map them:
        # 'carbs' -> 'input_meal_carbs'
        # 'cgm' -> 'output_cgm'
        if 'input_meal_carbs' not in df_data_subj.columns:
            if 'carbs' in df_data_subj.columns:
                df_data_subj.rename(columns={'carbs': 'input_meal_carbs'}, inplace=True)
            else:
                print(f"Warning: neither 'input_meal_carbs' nor 'carbs' found for {subj_id}")
                
        if 'output_cgm' not in df_data_subj.columns:
            if 'cgm' in df_data_subj.columns:
                df_data_subj.rename(columns={'cgm': 'output_cgm'}, inplace=True)
            else:
                print(f"Warning: neither 'output_cgm' nor 'cgm' found for {subj_id}")
        
        # Fill NaN in input_meal_carbs with 0 (T1DEXI data might have NaNs for no meal)
        if 'input_meal_carbs' in df_data_subj.columns:
            df_data_subj['input_meal_carbs'] = df_data_subj['input_meal_carbs'].fillna(0)
        # --------------------------------------------------------------------------

        # Train model
        # Note: personalization_path should be the parent dir where the subj folder is
        # trainModel saves results into personalization_path/subj_id/info.csv
        trainModel(
            df_data_subj,
            str(base_path) + "/", # personalization_path
            hidden_compartments,
            lr,
            batch_size,
            n_epochs,
            overlap,
            seq_len,
            subj_id,
        )
        
        # 3. Read back the info.csv to get metrics
        info_path = subj_dir / "info.csv"
        if info_path.exists():
            df_info = pd.read_csv(info_path, index_col=0).T # Transpose back to row
            # Add subject ID column if missing or ensure it's there
            df_info['subject_id'] = subj_id
            results_list.append(df_info)
        else:
            print(f"Warning: No info.csv found for {subj_id} after training.")

    # 4. Aggregate and Report
    if not results_list:
        print("No results collected.")
        return

    all_results = pd.concat(results_list, ignore_index=True)
    
    print("\n" + "="*50)
    print("REPRODUCTION RESULTS (Population Statistics)")
    print("="*50)
    
    # Metrics to summarize (matching Paper Table 3 columns)
    # We will compare Actual vs Population Model (AIPop) vs Digital Twin (AIDT)
    metrics_config = [
        {'label': 'TIR',  'act_col': 'TIR_test', 'pop_col': 'TIR_AIPop_test',  'dt_col': 'TIR_AIDT_test'},
        {'label': 'TAR',  'act_col': 'TAR_test', 'pop_col': 'TAR_AIPop_test',  'dt_col': 'TAR_AIDT_test'},
        {'label': 'TBR',  'act_col': 'TBR_test', 'pop_col': 'TBR_AIPop_test',  'dt_col': 'TBR_AIDT_test'},
        {'label': 'RMSE', 'act_col': None,       'pop_col': 'RMSE_AIPop_test', 'dt_col': 'RMSE_AIDT_test'}
    ]
    
    summary_data = []
    
    for metric in metrics_config:
        row = {'Metric': metric['label']}
        
        # Process Actual
        if metric['act_col'] and metric['act_col'] in all_results.columns:
            all_results[metric['act_col']] = pd.to_numeric(all_results[metric['act_col']], errors='coerce')
            mean_act = all_results[metric['act_col']].mean()
            std_act = all_results[metric['act_col']].std()
            row['Actual (Mean ± SD)'] = f"{mean_act:.1f} ± {std_act:.1f}"
        else:
            row['Actual (Mean ± SD)'] = "-"

        # Process Population Model (AIPop)
        if metric['pop_col'] in all_results.columns:
            all_results[metric['pop_col']] = pd.to_numeric(all_results[metric['pop_col']], errors='coerce')
            mean_pop = all_results[metric['pop_col']].mean()
            std_pop = all_results[metric['pop_col']].std()
            row['NN-based Pop (Mean ± SD)'] = f"{mean_pop:.1f} ± {std_pop:.1f}"
        else:
            row['NN-based Pop (Mean ± SD)'] = "N/A"

        # Process Digital Twin (AIDT)
        if metric['dt_col'] in all_results.columns:
            all_results[metric['dt_col']] = pd.to_numeric(all_results[metric['dt_col']], errors='coerce')
            mean_dt = all_results[metric['dt_col']].mean()
            std_dt = all_results[metric['dt_col']].std()
            row['NN-based DT (Mean ± SD)'] = f"{mean_dt:.1f} ± {std_dt:.1f}"
        else:
            row['NN-based DT (Mean ± SD)'] = "N/A"
            
        summary_data.append(row)
    
    df_summary = pd.DataFrame(summary_data)
    
    print("\n" + "="*80)
    print("REPRODUCTION RESULTS (Comparison with Actual)")
    print("="*80)
    print(df_summary[['Metric', 'Actual (Mean ± SD)', 'NN-based Pop (Mean ± SD)', 'NN-based DT (Mean ± SD)']])
    
    # Detailed breakdown columns to show
    detailed_cols = ['subject_id']
    for m in metrics_config:
        if m.get('act_col') and m['act_col'] in all_results.columns: detailed_cols.append(m['act_col'])
        if m['pop_col'] in all_results.columns: detailed_cols.append(m['pop_col'])
        if m['dt_col'] in all_results.columns: detailed_cols.append(m['dt_col'])
        
    print("\nDetailed breakdown per subject:")
    print(all_results[detailed_cols])

    # Save summary to file
    all_results.to_csv("reproduction_all_subjects_results.csv")
    df_summary.to_csv("reproduction_summary_table.csv")
    print("\nFull results saved to 'reproduction_all_subjects_results.csv'")

if __name__ == "__main__":
    # You can reduce n_epochs for a quick test run (e.g., n_epochs=5)
    # For full reproduction quality, keep n_epochs=150
    epochs = 150
    if len(sys.argv) > 1:
        epochs = int(sys.argv[1])
        
    reproduce_results(n_epochs=epochs)
