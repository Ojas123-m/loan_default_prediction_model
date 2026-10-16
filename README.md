
# Loan Default Prediction System

This is a simple end-to-end project scaffold for a Loan Default Prediction System.
It includes a synthetic dataset, training script, a saved trained model, and a Streamlit app.

## Structure
- `data/loan_data.csv` : synthetic dataset (sample)
- `src/train.py` : training script (creates a sklearn Pipeline and saves it to `models/`)
- `models/loan_model.joblib` : trained pipeline saved by the included training run
- `app/app.py` : Streamlit app to interact with the model
- `requirements.txt` : packages to install

## Quick start (run locally)
1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate    
   venv\Scripts\activate    
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Retrain the model:
   ```bash
   python src/train.py
   ```
4. Run the Streamlit app:
   ```bash
   streamlit run app/app.py
   ```

## Notes
- The dataset is synthetic and provided for demonstration. Replace `data/loan_data.csv` with real data when available.
- If you use your own dataset, ensure the same feature names are used in the app or update the app accordingly.
- For production use, consider model explainability (SHAP), fairness checks, and stricter validation.
