import json
import joblib
import pandas as pd

def init():
    global model
    model_path = "model.pkl"  # Azure ML injecte le chemin du modèle ici
    model = joblib.load(model_path)

def run(raw_data):
    try:
        data = pd.read_json(raw_data, orient="records")
        predictions = model.predict(data)
        return json.dumps({"predictions": predictions.tolist()})
    except Exception as e:
        return json.dumps({"error": str(e)})
