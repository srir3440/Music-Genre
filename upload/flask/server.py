import os
from flask import Flask, jsonify, request
from classificationService import classification_service

myapp = Flask(__name__)


@myapp.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    name = "to_predict"
    file.save(name)
    cs = classification_service()
    prediction = cs.predict(name)
    os.remove(name)
    response = {"Prediction": prediction}
    return jsonify(response)


if __name__ == "__main__":
    myapp.run(debug=False)
