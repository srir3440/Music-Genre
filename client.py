import os
import requests
from model import globals

TEST_FILE = "./dataset/genres_original/jazz/jazz.00003.wav"

def prepare_request():
    file = open(TEST_FILE, "rb")
    values = {"file": (TEST_FILE, file, "audio/wav")}
    response = requests.post(globals.ENDPOINT, files=values)
    data = response.json()
    print(f"Predicted genre: {data['Prediction']}")


if __name__ == "__main__":
    prepare_request()
