# Health Equity Insights Dashboard

Streamlit dashboard for exploring intersectional disparities (race/gender/city) using synthetic patient + encounter data, plus a simple cost prediction tool trained from the same dataset.

## Quickstart

Create a virtualenv, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the app from the repo root:

```bash
streamlit run app/main.py
```

## Data files expected

The loader supports either `Data/` (macOS) or `data/` (Linux/Streamlit Cloud).

Required:
- `patients.csv`
- `encounters_part_*.csv` (or `encounters.csv`)

These must include at least:
- From encounters: `PATIENT`, `TOTAL_CLAIM_COST`, `DESCRIPTION`
- From patients: `Id`, `BIRTHDATE`, `RACE`, `GENDER`, `CITY`, `INCOME`

## Model

On first run, the app will train a model from the merged dataset and save it to:
- `models/cost_predictor.pkl`

The prediction is for **per-encounter** `TOTAL_CLAIM_COST` (not a clinical risk score).