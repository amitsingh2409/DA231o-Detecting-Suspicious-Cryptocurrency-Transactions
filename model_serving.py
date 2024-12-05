import os
import mlflow
import mlflow.pyfunc
from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
model = mlflow.pyfunc.load_model("models/best_model")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    input_df = pd.DataFrame(data)
    predictions = model.predict(input_df)
    return jsonify(predictions.tolist())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
