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
    
    # Metrics to summarize (matching Paper Table 3 columns roughly)
    metrics = {
        'TIR': 'TIR_test',
        'TAR': 'TAR_test', 
        'TBR': 'TBR_test',
        'RMSE': 'RMSE_AIDT_test'
    }
    
    summary_data = []
    
    for label, col in metrics.items():
        if col in all_results.columns:
            mean_val = all_results[col].mean()
            std_val = all_results[col].std()
            summary_data.append({
                'Metric': label,
                'Mean': mean_val,
                'SD': std_val,
                'Formatted': f"{mean_val:.1f} ± {std_val:.1f}"
            })
    
    df_summary = pd.DataFrame(summary_data)
    print(df_summary[['Metric', 'Formatted']])
    
    print("\nDetailed breakdown per subject:")
    print(all_results[['subject_id'] + list(metrics.values())])

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
