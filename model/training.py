import json
import numpy as np
import globals
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Input, LSTM, GlobalMaxPool1D
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam


def get_data_splits(json_path):
    with open(json_path, "r") as fp:
        data = json.load(fp)
    X = np.array(data["MFCCS"])
    y = np.array(data["labels"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2)
    return X_train, y_train, X_validation, y_validation, X_test, y_test


def build_model():
    i = Input(shape=(X_train.shape[1], X_train.shape[2]))
    x = LSTM(64, return_sequences=True)(i)
    x = GlobalMaxPool1D()(x)
    o = Dense(globals.NO_OF_LABELS, activation="softmax")(x)
    model = Model(i, o)
    model.compile(optimizer=Adam(lr=0.005), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    return model


def learning_curves(history):
    fig, axs = plt.subplots(2, figsize=(10, 10))
    axs[0].plot(history.history["accuracy"], label="train_acc")
    axs[0].plot(history.history["val_accuracy"], label="validation_acc")
    axs[0].set_ylabel("accuracy")
    axs[0].set_title("accuracy curve")
    axs[0].legend()
    axs[1].plot(history.history["loss"], label="train_loss")
    axs[1].plot(history.history["val_loss"], label="validation_loss")
    axs[1].set_ylabel("loss")
    axs[1].set_title("loss curve")
    axs[1].set_xlabel("epochs")
    axs[1].legend()
    plt.show()


def main():
    X_train, y_train, X_validation, y_validation, X_test, y_test = get_data_splits(globals.JSON_PATH)
    model = build_model()
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), epochs=50)
    learning_curves(history)
    test_error, test_accuracy = model.evaluate(X_test, y_test)
    print(f"test_error:{test_error} , test_accuracy:{test_accuracy}")
    model.save(globals.MODEL_PATH)


if __name__ == "__main__":
    main()
