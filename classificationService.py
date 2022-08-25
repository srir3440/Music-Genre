import librosa
import numpy as np
from tensorflow import keras

import globals


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
        return globals.MAPPINGS[index]

    def preprocess(self, file_path):
        signal, sr = librosa.load(file_path, sr=globals.SAMPLE_RATE, duration=globals.TRACK_LENGTH)
        if len(signal) > globals.NO_OF_SAMPLES_PER_SEGMENT:
            signal = signal[:globals.NO_OF_SAMPLES_PER_SEGMENT]
        mfcc = librosa.feature.mfcc(signal, n_mfcc=globals.N_MELS, n_fft=globals.N_FFT, hop_length=globals.HOP_LENGTH)
        #print(mfcc.shape)
        return mfcc.T


def classification_service():
    if _classificationService._instance is None:
        _classificationService._instance = _classificationService()
        _classificationService._model = keras.models.load_model(globals.MODEL_PATH)
    return _classificationService._instance


if __name__ == "__main__":
    csobj = classification_service()
    prediction = csobj.predict("genres_original/jazz/jazz.00003.wav")
    print(f"Prediction:{prediction}")
