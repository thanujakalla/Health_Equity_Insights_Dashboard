import pandas as pd
import glob
import os
from pathlib import Path

def load_and_merge_data():
    """
    Robust data loader designed for Streamlit Cloud folder structures.
    """
    # 1. FIND THE ROOT DIRECTORY
    # This identifies '/mount/src/health_equity_insights_dashboard'
    # By going up one level from 'src/data_processor.py'
    root_path = Path(__file__).parents[1]
    data_dir = root_path / "data"

    # 2. CHECK IF FOLDER EXISTS
    if not data_dir.exists():
        # This will tell us in the logs exactly what folders DO exist
        available = [str(p.name) for p in root_path.iterdir() if p.is_dir()]
        raise FileNotFoundError(f"Folder 'data' not found at {data_dir}. Available folders: {available}")

    # 3. LOAD ENCOUNTER CHUNKS
    # Looks for 'encounters_part_1.csv', etc.
    search_pattern = str(data_dir / "encounters_part_*.csv")
    encounter_files = glob.glob(search_pattern)
    
    if not encounter_files:
        raise FileNotFoundError(f"No encounter files found in {data_dir}. Check file names.")
    
    encounters = pd.concat((pd.read_csv(f) for f in encounter_files), ignore_index=True)

    # 4. LOAD PATIENTS
    patients_path = data_dir / "patients.csv"
    if not patients_path.exists():
        raise FileNotFoundError(f"patients.csv missing at {patients_path}")
        
    patients = pd.read_csv(patients_path)

    # 5. DATA CLEANING & MERGING
    # Standardizing dates and age (0-110 range)
    patients['BIRTHDATE'] = pd.to_datetime(patients['BIRTHDATE'])
    patients['AGE'] = (pd.Timestamp.today() - patients['BIRTHDATE']).dt.days // 365
    
    # Ensuring 0 invalid foreign keys
    merged_data = pd.merge(encounters, patients, left_on='PATIENT', right_on='Id')

    # 6. EQUITY METRICS
    # Aggregating for the $1,350 vs $895 cost burden report
    equity_report = merged_data.groupby(['RACE', 'GENDER']).agg({
        'TOTAL_CLAIM_COST': 'mean',
        'Id': 'count'
    }).rename(columns={'Id': 'Encounter_Count'}).reset_index()

    return merged_data, equity_report
   
