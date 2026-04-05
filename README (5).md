# 🎙️ Mood Recognizer from Speech

**by Dev Shah | ML Engineering Intern Project**

This is my first ML project!! It detects emotions/mood from speech audio using machine learning.

---

## What it does

You give it an audio file (or record your voice), and it tells you what emotion is in the speech.

Supported moods: **neutral, happy, sad, angry, fearful**

---

## Dataset

I used the **RAVDESS dataset** from Kaggle:
👉 https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio

It has 24 actors saying things with different emotions. About 1440 audio files total.

---

## How to run

### 1. Install dependencies
```
pip install -r requirements.txt
```
(warning: takes a while)

### 2. Download dataset
Download from Kaggle and put it in the same folder. The folder should be named `audio_speech_actors_01-24`

### 3. Train the model
```
python train_model.py
```
This will take 10-20 minutes depending on your computer. It saves `model.pkl`, `scaler.pkl`, and `encoder.pkl`

### 4. Start the server
```
python app.py
```

### 5. Open the website
Open `index.html` in your browser

If backend isn't running, it goes into "demo mode" which shows fake results

---

## How it works (what I learned)

1. **Feature extraction** - I extract MFCC (Mel Frequency Cepstral Coefficients) + Chroma + Mel spectrogram features from the audio. I still don't fully understand MFCCs but they're basically a compact representation of audio.

2. **Model** - I used an MLP (Multi Layer Perceptron) from scikit-learn with 3 hidden layers (256, 128, 64 neurons). Originally tried Random Forest but MLP worked better.

3. **Training** - 80/20 train/test split, StandardScaler for normalization

**Accuracy: ~65%** 

That sounds low but apparently human accuracy on emotion recognition is also around 60-70% so it's okay I think. My mentor confirmed this.

---

## Known issues / TODO

- [ ] Only trained on English speech (RAVDESS is English actors)
- [ ] Recording feature might not work on all browsers (Chrome works best)  
- [ ] The model is saved with pickle which is apparently not the best practice (joblib is better?)
- [ ] Need to add proper error handling
- [ ] Mobile UI looks a bit broken
- [ ] The copy-pasted feature extraction between train_model.py and app.py should be in a utils.py
- [ ] confidence scores sometimes seem off (70% confident about wrong prediction)
- [ ] Haven't tested with non-English speech
- [ ] dark/light mode toggle would be cool

---

## Tech Stack

- **Python** - scikit-learn, librosa, Flask
- **Frontend** - HTML, CSS, JS (vanilla), Bootstrap 5
- **Dataset** - RAVDESS (Kaggle)

---

## Lessons learned

- Audio feature extraction is harder than I thought
- Copy-pasting code from training into inference is bad (but I did it anyway)
- 65% accuracy feels bad but is actually normal for this task apparently
- Pickle is fine for learning but not for production
- Need to normalize features or the model performs terribly (found this out the hard way)

---

*This is a learning project. Not production ready. Made mistakes on purpose (and also by accident).*
