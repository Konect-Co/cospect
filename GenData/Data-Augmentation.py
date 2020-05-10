import os

import numpy as np
import librosa

#utility function to create spectrogram input from an audio file
def make_spectrogram(file_path):
	signal, sample_rate = librosa.core.load(file_path)
	if signal.shape[0] == 2:
		signal = np.mean(signal, axis=0)

	resample_freq = 8000
	signal_resampled = librosa.core.resample(signal, sample_rate, resample_freq)

	stft = librosa.core.stft(signal_resampled)
	spectrogram = np.power(np.abs(stft), 0.5)
	
	return spectrogram

#utility function to write spectrogram input to wav file
def write_spectrogram(spectrogram, file_path):
	
#utility function to change pitch of spectrogram by certain factor
def change_pitch(spectrogram, factor):
	
#utility function to change speed of spectrogram by certain factor
def change_speed(spectrogram, factor):
	
#utility function to add random noise into spectrogram
def inject_noise(spectrogram):
	

def augment():
	"""
	What needs to be done in this method:
	- For each .wav file in Data/YT-Audio Directory
		- load .wav file into librosa
		- Data augmentation can be performed using librosa library
	- Data augmentation
		- Changing pitch of the sound
		- Speeding up/slowing down the sound
		- Injecting noise into the sound
		- Any other changes you recommend?
	- Once sound has been augmented, write spectrogram back to .wav file with "ag" added to end of name
		- For example, after augmenting sound from "7.wav", write to file "7-ag.wav"

	Sources you can consult:
	- https://www.kaggle.com/huseinzol05/sound-augmentation-librosa
	- https://medium.com/@makcedward/data-augmentation-for-audio-76912b01fdf6
	- https://librosa.github.io/librosa/generated/librosa.effects.pitch_shift.html
	- https://librosa.github.io/librosa/generated/librosa.effects.time_stretch.html#librosa.effects.time_stretch
	"""

	"""
	Pseudocode

	for audio_file in directory("Data/YT-Audio Directory"):
		spectrogram = make_spectrogram(audio_file)

		pitch_factor = random number between 0.8 and 1.2
		speed_factor = random number between 0.8 and 1.2

		pitch_threshold = 0.7 #probability of pitch being changed is 1-0.7=0.3
		speed_threshold = 0.7 #probability of speed being changed is 1-0.7=0.3
		noise_threshold = 0.7 #probability of noise being injected is 1-0.7=0.3
		
		if rand(0, 1) exceeds pitch_threshold
			spectrogram = change_pitch(spectrogram, pitch_factor)
		if rand(0, 1) exceeds speed_threshold
			spectrogram = change_speed(spectrogram, speed_factor)
		if rand(0, 1) exceeds noise_threshold
			spectrogram = inject_noise(spectrogram)

		write_spectrogram(spectrogram, new_file_path)
	"""
