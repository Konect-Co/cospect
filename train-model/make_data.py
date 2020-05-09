import os

from openpyxl import load_workbook
import numpy as np
import librosa

#utility function to create spectrogram input from an audio file
def make_spectrogram (audio_path, resample_freq):
	# load the signals and resample them
	signal, sample_rate = librosa.core.load(audio_path)
	if signal.shape[0] == 2:
		signal = np.mean(signal, axis=0)

	signal_resampled = librosa.core.resample(signal, sample_rate, resample_freq)

	stft = librosa.core.stft(signal_resampled)
	spectrogram = np.power(np.abs(stft), 0.5)

	# Steps for normalisation===============
	#means = np.mean(spectrogram, 1, keepdims=True)
	#stddevs = np.std(means, 1, keepdims=True)
	#spectrogram = (spectrogram - means)/stddevs
	spectrogram = 1/(1 + np.exp(-spectrogram))
	#=======================================

	# Steps for padding====================== (ignore for now)
	#max_length = 1000
	#spectrogram = np.pad(spectrogram, [[0,0],[spectrogram.shape[1], 0]])
	#=======================================

	#output dimension is now [frequencies of spectrogram, timesteps]
	#switching it to [timesteps, frequencies of spectrogram]
	spectrograms = np.swapaxes(spectrograms, 1, 2)
	
	return spectrogram

def make_data():
	"""
	What needs to be done in this method:
	- Create spectrogram and output data in np array form
	- Random split 90:10 between training and testing data
		- Note: 90:10 split is according to timesteps, not number of samples
		- However, the last training sample should not be cut off before its timesteps are complete
	"""

	"""
	Making spectrogram data
	- Read all the .wav files in Data/YT-Audio Directory
	- For files over duration stored in sample_max_len, first split them into sections of max value stored in sample_max_len
	- For each audio section, generate the spectrogram accordingly
	"""
	spectrogram_data = np.asarray([])
	#end shape: [# samples, timesteps, frequencies of spectrogram]

	#Audio samples over 60 seconds will be split up in sections of max 60 seconds
	sample_max_len = 60

	"""
	Making output data
	- Read all the .json files in Data
	- Each .json file corresponds to a video in Data/YT-Audio
	- .json file has multiple fields
	- For now only one of importance is "symptoms"
	- "symptoms" may have "Wet" "Dry" or neither
	- If dry cough present --> [1 0]. wet cough --> [0 1]. neither --> [0 0]
	- First axis of symptom_data should match length of spectrogram_data (repeat diagnosis for large files that were split up)
	"""
	symptom_data = np.asarray([])
	#end shape: [# samples, 2]

	#decimal between 0 and 1 representing percentage of timesteps
	#1 - training_split is the percentage allocated for testing data
	training_split = 0.9

	#shuffle order of samples in spectrogram_data and symptom data
	#(of course indices should still match each other)

	return (training_x, training_y), (testing_x, testing_y)

if __name__ == "__main__":
	make_data()
