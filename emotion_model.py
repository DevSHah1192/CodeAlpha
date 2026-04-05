import os
import numpy as np
import pandas as pd
import librosa                        
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical


DATA_PATH = "data"

if not os.path.exists(DATA_PATH):
    print("Data folder not found. Please ensure the 'data' folder with audio files is present.")
    exit()


emotion_dict = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

print("Emotion labels:", list(emotion_dict.values()))




def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, duration=3, offset=0.5)
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None




features = []
labels = []

print("\nLoading audio files...")

for actor in os.listdir(DATA_PATH):

    actor_path = os.path.join(DATA_PATH, actor)

    if not os.path.isdir(actor_path):   # skip if not a folder
        continue

    for file in os.listdir(actor_path):

        if not file.endswith(".wav"):   # only process wav files
            continue

        file_path = os.path.join(actor_path, file)

        try:
            parts = file.split("-")
            emotion_code = parts[2]
            emotion = emotion_dict.get(emotion_code)

            if emotion is None:
                continue

            feature = extract_features(file_path)

            if feature is not None:
                features.append(feature)
                labels.append(emotion)

        except Exception as e:
            print(f"Skipping file {file}: {e}")

print(f"Total samples loaded: {len(features)}")


X = np.array(features)
y = np.array(labels)

# quick check
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Unique emotions found:", np.unique(y))




encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

print("Classes:", encoder.classes_)
print("y_categorical shape:", y_categorical.shape)




X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42
)

print(f"\nTraining samples : {X_train.shape[0]}")
print(f"Testing  samples : {X_test.shape[0]}")




model = Sequential()

model.add(Dense(256, activation="relu", input_shape=(40,)))
model.add(Dropout(0.3))

model.add(Dense(128, activation="relu"))
model.add(Dropout(0.3))

model.add(Dense(64, activation="relu"))     # added one more layer
model.add(Dropout(0.2))

model.add(Dense(y_categorical.shape[1], activation="softmax"))

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.summary()




print("\nTraining model...")

history = model.fit(          
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)



loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Loss     : {loss:.4f}")
print(f"Test Accuracy : {accuracy * 100:.2f}%")




plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"],    label="Train Accuracy")
plt.plot(history.history["val_accuracy"],label="Val Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"],    label="Train Loss")
plt.plot(history.history["val_loss"],label="Val Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.show()
print("Training curves saved.")




y_pred = model.predict(X_test)
y_pred_classes  = np.argmax(y_pred,    axis=1)
y_true_classes  = np.argmax(y_test,    axis=1)

cm = confusion_matrix(y_true_classes, y_pred_classes)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=encoder.classes_,
            yticklabels=encoder.classes_)
plt.title("Confusion Matrix — Emotion Recognition")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("Confusion matrix saved.")

print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes,
                             target_names=encoder.classes_))



model.save("emotion_model.h5")
print("\nModel saved as emotion_model.h5")


np.save("emotion_classes.npy", encoder.classes_)
print("Encoder classes saved as emotion_classes.npy")
