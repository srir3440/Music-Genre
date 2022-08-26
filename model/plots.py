import librosa
import librosa.display
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt

FILE_PATH = "../dataset/genres_original/blues/blues.00000.wav"
SAMPLE_RATE = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MFCC = 26


def plot_wave(signal):
    librosa.display.waveshow(signal, sr=SAMPLE_RATE)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.show()
    print(librosa.core.get_duration(signal, SAMPLE_RATE))


def plot_augmented_signal(signal):
    signal_temp = signal.copy()
    time_ch = np.random.uniform(low=0.5, high=2)
    print(time_ch)
    signal_temp_speed = librosa.effects.time_stretch(signal_temp, time_ch)
    print(signal_temp.shape)
    print(signal_temp_speed.shape)
    minlen = min(signal_temp.shape[0], signal_temp_speed.shape[0])
    signal_temp *= 0
    signal_temp[0:minlen] = signal_temp_speed[0:minlen]
    print(signal_temp.shape)
    #ipd.Audio(signal_temp, rate=SAMPLE_RATE)
    plot_wave(signal_temp)
    return signal_temp


def plot_fft(signal):
    fft = np.fft.fft(signal)
    print(fft.shape, fft[0])
    magnitude = np.abs(fft)
    frequency = np.linspace(0, SAMPLE_RATE, len(magnitude))
    left_freq = frequency[:int(len(magnitude) / 2)]
    left_mag = magnitude[:int(len(magnitude) / 2)]
    plt.plot(left_freq, left_mag)
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.show()


def plot_stft(signal):
    stft = librosa.core.stft(signal, n_fft=N_FFT, hop_length=HOP_LENGTH)
    spectrogram = np.abs(stft)
    print(spectrogram.shape)
    log_spec = librosa.amplitude_to_db(spectrogram, ref=np.max)
    librosa.display.specshow(log_spec, hop_length=HOP_LENGTH, sr=SAMPLE_RATE)
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.colorbar()
    plt.show()


def plot_mfccs(signal):
    mfcc = librosa.feature.mfcc(signal, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mfcc=N_MFCC)
    print(mfcc.shape)
    librosa.display.specshow(mfcc, hop_length=HOP_LENGTH, sr=SAMPLE_RATE)
    plt.xlabel("Time")
    plt.ylabel("MFCC")
    plt.colorbar()
    plt.show()


def plot_melspectrogram(signal):
    melspec = librosa.feature.melspectrogram(signal, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=128)
    melspec = librosa.power_to_db(melspec, ref=np.max)
    print(melspec.shape)
    mel_spec_figure = librosa.display.specshow(melspec, hop_length=HOP_LENGTH, sr=SAMPLE_RATE)
    mel_spec_figure = mel_spec_figure.get_figure()
    canvas = FigureCanvasAgg(mel_spec_figure)
    canvas.draw()
    prr, (width, height) = canvas.print_to_buffer()
    X = np.frombuffer(prr, np.uint8).reshape(height, width, 4)
    rgb_weights = [0.2989, 0.5870, 0.1140]
    X_gray = np.dot(X[..., :3], rgb_weights).astype(np.uint8)
    X_gray_img = X_gray[..., np.newaxis]
    plt.close()


if __name__ == "__main__":
    signal, sr = librosa.load(FILE_PATH, duration=3, sr=SAMPLE_RATE)
    #ipd.Audio(signal, rate=SAMPLE_RATE)
    plot_wave(signal)
    time_stretched_signal = plot_augmented_signal(signal)
    plot_fft(signal)
    plot_fft(time_stretched_signal)
    plot_stft(signal)
    plot_mfccs(signal)
    plot_melspectrogram(signal)
