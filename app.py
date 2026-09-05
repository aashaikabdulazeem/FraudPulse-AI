from flask import Flask, render_template, jsonify, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load dataset and trained model
data = pd.read_csv("data/transactions.csv")
model = joblib.load("model/fraud_detector.joblib")

FEATURES = [
    "amount",
    "failed",
    "new_device",
    "international",
    "rapid_frequency",
]


@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/api/stats")
def stats():
    total = len(data)
    suspicious = int(data["is_fraud"].sum())

    suspicious_percentage = round((suspicious / total) * 100, 2)

    if suspicious_percentage >= 10:
        risk_level = "HIGH"
    elif suspicious_percentage >= 5:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return jsonify({
        "total_transactions": total,
        "suspicious_transactions": suspicious,
        "suspicious_percentage": suspicious_percentage,
        "risk_level": risk_level
    })


@app.route("/api/recent")
def recent():
    recent_data = data.tail(20).copy()

    predictions = model.predict(recent_data[FEATURES])

    recent_data["prediction"] = predictions

    return jsonify(
        recent_data[
            [
                "transaction_id",
                "amount",
                "failed",
                "new_device",
                "international",
                "rapid_frequency",
                "prediction",
            ]
        ].to_dict(orient="records")
    )



@app.route("/api/predict", methods=["POST"])
def predict():
    request_data = request.get_json()

    transaction = pd.DataFrame([{
        "amount": float(request_data["amount"]),
        "failed": int(request_data["failed"]),
        "new_device": int(request_data["new_device"]),
        "international": int(request_data["international"]),
        "rapid_frequency": int(request_data["rapid_frequency"])
    }])

    prediction = int(model.predict(transaction[FEATURES])[0])

    return jsonify({
        "prediction": prediction,
        "status": "FRAUD" if prediction == 1 else "SAFE"
    })


if __name__ == "__main__":
    app.run(debug=True)