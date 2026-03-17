import pandas as pd
import glob
import os

def load_and_merge_data():
    """
    Robust data loader for Streamlit Cloud.
    """
    # 1. Use the most direct pathing for Streamlit Cloud
    # Streamlit runs from the root folder, so 'data/' should be right there.
    data_dir = 'data'
    
    # DIAGNOSTIC: If it fails, let's see what the server sees.
    if not os.path.exists(data_dir):
        # This will show up in your 'Manage App' logs
        current_folders = os.listdir('.')
        raise FileNotFoundError(f"Folder '{data_dir}' not found. I see these folders: {current_folders}")

    # 2. Find the split files
    search_pattern = os.path.join(data_dir, "encounters_part_*.csv")
    encounter_files = glob.glob(search_pattern)
    
    if not encounter_files:
        raise FileNotFoundError(f"No encounter files found in '{data_dir}'. Files present: {os.listdir(data_dir)}")
    
    # 3. Combine and Merge
    encounter_files.sort()
    encounters = pd.concat((pd.read_csv(f) for f in encounter_files), ignore_index=True)
    patients = pd.read_csv(os.path.join(data_dir, 'patients.csv'))

    # 4. Standard Cleaning (Ensures 0 invalid foreign keys)
    patients['BIRTHDATE'] = pd.to_datetime(patients['BIRTHDATE'])
    patients['AGE'] = (pd.Timestamp.today() - patients['BIRTHDATE']).dt.days // 365
    
    merged_data = pd.merge(encounters, patients, left_on='PATIENT', right_on='Id')

    # 5. The Equity Report Logic
    equity_report = merged_data.groupby(['RACE', 'GENDER']).agg({
        'TOTAL_CLAIM_COST': 'mean',
        'Id': 'count'
    }).rename(columns={'Id': 'Encounter_Count'}).reset_index()

    return merged_data, equity_report
   
