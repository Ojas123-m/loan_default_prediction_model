
# train.py -- trains a Loan Default prediction pipeline and saves it (joblib)
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

BASE = Path(__file__).resolve().parent.parent
DATA_PATH = BASE / 'data' / 'loan_data.csv'
MODEL_PATH = BASE / 'models' / 'loan_model.joblib'

def load_data(path=DATA_PATH):
    return pd.read_csv(path)

def build_and_train():
    df = load_data()
    df = df.dropna(subset=['Default']).copy()
    X = df.drop(columns=['Loan_ID','Default'])
    y = df['Default']

    numeric_features = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History']
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

    pipe = Pipeline(steps=[('preproc', preprocessor), ('clf', clf)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:,1]

    print('Classification report:\\n', classification_report(y_test, y_pred))
    try:
        print('ROC AUC:', roc_auc_score(y_test, y_proba))
    except Exception as e:
        print('ROC AUC error:', e)

    joblib.dump(pipe, MODEL_PATH)
    print('Saved trained pipeline to', MODEL_PATH)

if __name__ == '__main__':
    build_and_train()
