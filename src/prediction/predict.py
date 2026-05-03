import os
import mlflow.sklearn
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

# Load production model directly from registry
model = mlflow.sklearn.load_model("models:/titanic-classifier/Production")

# Sample passengers
passengers = pd.DataFrame([
    {
        "Pclass": 1, "Sex": "female", "Age": 29, "SibSp": 0,
        "Parch": 0, "Fare": 211.3, "Embarked": "S",
        "Name": "Cumings, Mrs. John Bradley", "Ticket": "PC 17599", "Cabin": "C85"
    },
    {
        "Pclass": 3, "Sex": "male", "Age": 22, "SibSp": 1,
        "Parch": 0, "Fare": 7.25, "Embarked": "S",
        "Name": "Braund, Mr. Owen Harris", "Ticket": "A/5 21171", "Cabin": ""
    },
])

predictions = model.predict(passengers)
probabilities = model.predict_proba(passengers)[:, 1]

for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    status = "Survived" if pred == 1 else "Did not survive"
    print(f"Passenger {i+1}: {status} (probability: {prob:.4f})")