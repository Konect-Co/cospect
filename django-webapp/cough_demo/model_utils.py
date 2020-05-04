import keras
import numpy as np
import librosa

def predict(audio_path):
    resample_freq = 8000

    signal, sample_rate = librosa.core.load(audio_path)
    if signal.shape[0] == 2:
        signal = np.mean(signal, axis=0)

    signal_resampled = librosa.core.resample(signal, sample_rate, resample_freq)

    stft = librosa.core.stft(signal_resampled)
    spectrogram = np.power(np.abs(stft), 0.5)

    spectrogram = 1/(1 + np.exp(-spectrogram))

    #model = keras.load()

    #prediction = model.predict(spectrogram)
    #return prediction

    return 0