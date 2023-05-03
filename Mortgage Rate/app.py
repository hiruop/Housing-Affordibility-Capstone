import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("rf_model.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods=["POST"])
def predict():
    interest_rate = float(request.form["interest_rate"])
    hpi_change = float(request.form["hpi_change"])
    X = np.array([[interest_rate, hpi_change]])
    prediction = model.predict(X)[0]
    return render_template("index.html", prediction_text="Based on an interest rate of {:.2f}% & a Home Price Index (HPI) of {:.2f} then your mortgage rate will be {:.2f}%".format(interest_rate,hpi_change, prediction))

if __name__ == "__main__":
    flask_app.run(debug=True)