from app import app
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app

model = pickle.load(open("cg.pkl", "rb"))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template("index.html")
