"""
Authors: Cassiano Simas, Ravit Sharma
"""
#%%
import numpy as np
import librosa
import random
import os

pitch_factor = random.uniform(0.8, 1.2)
speed_factor = random.uniform(0.8, 1.2)
noise_factor = random.uniform(0.01, 0.03)
pitch_threshold = 0.7 #probability of pitch being changed is 1-0.7=0.3
speed_threshold = 0.7 #probability of speed being changed is 1-0.7=0.3
noise_threshold = 0.7 #probability of noise being injected is 1-0.7=0.3

#utility function to write spectrogram input to wav file
def write_audio(signal, sample_rate, file_path):
	librosa.output.write_wav(file_path, signal, sample_rate)

#utility function to change pitch of spectrogram by certain factor
def change_pitch(signal, sample_rate, pitch_factor):
	return librosa.effects.pitch_shift(signal, sample_rate, pitch_factor)

#utility function to change speed of spectrogram by certain factor
def change_speed(signal, speed_factor):
	return librosa.effects.time_stretch(signal, speed_factor)

#utility function to add random noise into spectrogram
def inject_noise(signal, noise_factor):
	noise = np.random.randn(len(signal))
	augmented_data = signal + noise_factor * noise
	augmented_data = augmented_data.astype(type(signal[0]))
	return augmented_data

#%%
def augment():
	#for each wav file in the folder it will apply all utility functions
	pathAudio = "/home/ravit/Konect-Code/cospect/Data/YT-Audio"
	files = librosa.util.find_files(pathAudio, ext=['wav']) 
	files = np.asarray(files)
	for y in files:
		signal, sample_rate = librosa.load(y, mono=True) 

		rand1 = random.uniform(0, 1)
		rand2 = random.uniform(0, 1)
		rand3 = random.uniform(0, 1)

		if rand1 > pitch_threshold:
			signal = change_pitch(signal, sample_rate, pitch_factor)
		if rand2 > speed_threshold:
			signal = change_speed(signal, speed_factor)
		if rand3 > noise_threshold:
			signal = inject_noise(signal, noise_factor)
		
		file_path = y.split(os.sep)[:-2]
		file_path = os.sep.join(file_path)
		file_path = file_path + os.sep + "Augment" + os.sep + (y.split(os.sep)[-1]).split(".")[0] + "-modified.wav"
		write_audio(signal, sample_rate, file_path)

augment()