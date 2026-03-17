from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import jobos
import pandas as pd

def train_cost_predictor(df):
    # Select features identified in your proposed reports
    features = ['AGE', 'INCOME', 'RACE', 'GENDER']
    target = 'TOTAL_CLAIM_COST'

    # Preprocessing: Encode categorical text into numbers
    le_race = LabelEncoder()
    le_gender = LabelEncoder()
    
    df['RACE_ENC'] = le_race.fit_transform(df['RACE'])
    df['GENDER_ENC'] = le_gender.fit_transform(df['GENDER'])

    X = df[['AGE', 'INCOME', 'RACE_ENC', 'GENDER_ENC']]
    y = df[target]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the model for your Streamlit GUI
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/cost_predictor.pkl')
    print("Predictive model saved to models/cost_predictor.pkl")

# To run:
# data, _ = load_and_merge_data()
# train_cost_predictor(data)
