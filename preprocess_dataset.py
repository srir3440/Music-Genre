import math
import librosa
import librosa.display
import os
import json
import numpy as np
import globals
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import warnings

warnings.filterwarnings('ignore')


class AudioDataGenerator:

    def __init__(self, directory_path, batch_size):
        self.directory_path = directory_path
        self.batch_size = batch_size
        self.data = {}
        self.data["MFCCS"], self.data["MelSpectrograms"], self.data["labels"] = self.__get_audio_features()

    def __len__(self):
        return 0

    def __getitem__(self, index):
        pass

    def __get_audio_features(self):

        X_mfccs = []
        X_melspecs = []
        y = []

        for i, (dirpath, dirnames, filenames) in enumerate(os.walk(self.directory_path)):
            # print(str(i)+" "+dirpath)
            if i != 0:
                for index in range(1, len(filenames)):
                    signal_list = []
                    # print(filenames[index])
                    signal, sr = librosa.load(os.path.join(dirpath, filenames[index]), sr=globals.SAMPLE_RATE,
                                              duration=globals.TRACK_LENGTH)
                    # audio data augmentation to increase data set size
                    # print(signal.shape)
                    signal_list.append(signal)
                    signal_list.append(self.__add_noise(signal))
                    signal_list.append(self.__pitch_augment(signal))
                    for signal in signal_list:
                        mfccs, melspecs, labels = self.__get_mfcc_melspecs(signal, i - 1)
                        # print(mfccs[0])
                        X_mfccs.extend(mfccs)
                        X_melspecs.extend(melspecs)
                        y.extend(labels)
                    break  # remove to process the whole dataset
                break

        return X_mfccs, X_melspecs, y

    def __add_noise(self, signal):

        noise_amp = 0.04 * np.random.uniform() * np.max(signal)
        signal = signal + noise_amp * np.random.uniform(size=signal.shape[0])
        return signal

    def __pitch_augment(self, signal):

        pitch_ch = 3 * np.random.uniform(low=1, high=2)
        signal_temp = librosa.effects.pitch_shift(signal, sr=globals.SAMPLE_RATE, bins_per_octave=12, n_steps=pitch_ch)
        return signal_temp

    def __get_mfcc_melspecs(self, signal, label):

        mfcc_list = []
        melspec_list = []
        label_list = []
        for i in range(globals.SEGMENTS):
            start = globals.NO_OF_SAMPLES_PER_SEGMENT * i
            end = start + globals.NO_OF_SAMPLES_PER_SEGMENT
            mfcc = librosa.feature.mfcc(signal[start:end], sr=globals.SAMPLE_RATE, n_fft=globals.N_FFT,
                                        hop_length=globals.HOP_LENGTH,
                                        n_mfcc=globals.N_MELS)
            mfcc = mfcc.T
            # print(mfcc.shape)
            if len(mfcc) == globals.EXPECTED_LEN_OF_MFCC_VECTOR:
                melspec = self.__get_melspecs_from_canvas(signal[start:end])
                mfcc_list.append(mfcc.tolist())
                melspec_list.append(melspec.tolist())
                label_list.append(label)
        return mfcc_list, melspec_list, label_list

    def __get_melspecs_from_canvas(self, signal):

        melspec = librosa.feature.melspectrogram(signal, sr=globals.SAMPLE_RATE, n_fft=globals.N_FFT,
                                                 hop_length=globals.HOP_LENGTH,
                                                 n_mels=globals.N_MELS)
        melspec = librosa.power_to_db(melspec, ref=np.max)
        mel_spec_figure = librosa.display.specshow(melspec, hop_length=globals.HOP_LENGTH,
                                                   sr=globals.SAMPLE_RATE).get_figure()
        canvas = FigureCanvasAgg(mel_spec_figure)
        canvas.draw()
        prr, (width, height) = canvas.print_to_buffer()
        X = np.frombuffer(prr, np.uint8).reshape(height, width, 4)
        rgb_weights = [0.2989, 0.5870, 0.1140]
        X_gray = np.dot(X[..., :3], rgb_weights).astype(np.uint8)
        X_gray_img = X_gray[..., np.newaxis]
        plt.close()
        return X_gray_img


if __name__ == "__main__":
    audio_obj = AudioDataGenerator("genres_original", 32)
    print(globals.MAPPINGS)
    print(len(audio_obj.data["MFCCS"]))
    print(len(audio_obj.data["MelSpectrograms"]))
    print(len(audio_obj.data["labels"]))
    with open(globals.JSON_PATH, "w") as fp:
        json.dump(audio_obj.data, fp, indent=4)
