import os
import librosa
import numpy as np
from tensorflow import keras

MODEL_PATH = "model.h5"
TEST_FILE = "./dataset/genres_original/jazz/jazz.00003.wav"
SEGMENTS = 10
TRACK_LENGTH = 30
SAMPLE_RATE = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 13
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_LENGTH
NO_OF_SAMPLES_PER_SEGMENT = int(SAMPLES_PER_TRACK / SEGMENTS)
MAPPINGS = ['hiphop', 'rock', 'metal', 'classical', 'country', 'disco', 'pop', 'blues', 'reggae', 'jazz']


class _classificationService:
    _model = None
    _instance = None

    def predict(self, file_path):
        mfcc = self.preprocess(file_path)
        mfcc = mfcc[np.newaxis, ...]
        #print(mfcc.shape)
        predictions = self._model.predict(mfcc).squeeze()
        #print(predictions)
        index = np.argmax(predictions)
        #print(index)
        return MAPPINGS[index]

    def preprocess(self, file_path):
        signal, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=TRACK_LENGTH)
        if len(signal) > NO_OF_SAMPLES_PER_SEGMENT:
            signal = signal[:NO_OF_SAMPLES_PER_SEGMENT]
        mfcc = librosa.feature.mfcc(signal, n_mfcc=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
        #print(mfcc.shape)
        return mfcc.T


def classification_service():
    if _classificationService._instance is None:
        _classificationService._instance = _classificationService()
        _classificationService._model = keras.models.load_model(MODEL_PATH)
    return _classificationService._instance


if __name__ == "__main__":
    csobj = classification_service()
    prediction = csobj.predict(TEST_FILE)
    print(f"Prediction:{prediction}")
