# Cough Detection Model
Providing cough sound diagnostics and analytics.

# Installation and usage
Coming soon!

# Dataset gathering
Our dataset consists of a variety of online sources displaying symptoms of whooping cough or inflammation-related cough. See "Data Table.xlsx" for a full list. Additionally, we put the exact range of time (seconds) containing the sound of the coughing patient so that we only train the model on the relevant cough sound.

# Data Pre-processing
Our data pre-processing phase consists of converting our data table with links, labels, and relevant sections of the cough into a series of spectrograms of fixed length for each sample. We went through the following steps to do so:

1. Converting the raw audio (YouTube link, online audio file) into a WAV file and storing these in "Data/raw_audio". We made a simple script "Data/YT-WAV-gen.py" to create the .wav files.
1. Splicing the relevant sections of the 
1. Converting the .wav files to spectrograms after resampling them to 8000Hz. Normalizing the spectrograms and padding them with zeros so there is a fixed number of time steps per sample.

# Model functionality
Conv1d applied on each timestep slice, then passed through an RNN

# Model inputs and outputs
Input is the resampled, normalized, and padded spectrograms corresponding to the audio of the cough. 

Output is the model's prediction for the disease, based upon the cough sound (whooping cough, inflammatory).
