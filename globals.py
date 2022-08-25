import math

DIRECTORY = "genres_original"
JSON_PATH = "data.json"
MODEL_PATH = "model.h5"
MAPPINGS = ['hiphop', 'rock', 'metal', 'classical', 'country', 'disco', 'pop', 'blues', 'reggae', 'jazz']

NO_OF_LABELS = 10
SEGMENTS = 10
TRACK_LENGTH = 30
SAMPLE_RATE = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 13
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_LENGTH
NO_OF_SAMPLES_PER_SEGMENT = int(SAMPLES_PER_TRACK / SEGMENTS)
EXPECTED_LEN_OF_MFCC_VECTOR = math.ceil(NO_OF_SAMPLES_PER_SEGMENT / HOP_LENGTH)

if __name__ == "__main__":
    print(EXPECTED_LEN_OF_MFCC_VECTOR)
