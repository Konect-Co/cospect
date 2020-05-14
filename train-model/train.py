import librosa as lb
import json
import numpy as np
import pandas as pd
import os
import librosa

from openpyxl import load_workbook
pd.DataFrame()

class MakeData:
   def __init__(self,audio_path,json_path):
       self.audio_path=audio_path
       self.json_path=json_path
   
   def is_json(self,filename):
       try:
           x=str(filename)
           y=x.split('.')[1]
           if(y=='json'):
               
               return True
       except IndexError as e:
           print('FILE EXTENSION NOT IN '+'.json')
       
   def make_df(self,csv_name='audio.csv'):
       df=pd.DataFrame()
       json_files=os.listdir(self.json_path)
       audio_files=os.listdir(self.audio_path)
       
       df=pd.DataFrame(columns=['id','age','gender','symptoms','disease','audio','sampling rate','time'])
       i=0
       #getting names of  all the keys in json file
       for j in json_files:
           if(self.is_json(j)==True):
               with open(os.path.join(self.json_path,j)) as f:
                   dict1=json.load(f)
                   id1=j.split('.')[0]
                   age=dict1['age']
                   gender=dict1['gender']
                   symptoms=dict1['symptoms'][0]
                   disease=dict1['disease']
               
                   df.loc[i,'id']=id1
                   df.loc[i,'age']=age
                   df.loc[i,'gender']=gender
                   df.loc[i,'symptoms']=symptoms
                   df.loc[i,'disease']=disease
                   path=self.audio_path+'/'+j.split('.')[0]+'.wav'
                   audio,sample=librosa.core.load(path=os.path.normpath(path))
                   
                   df.loc[i,'audio']=np.asarray(audio)
                   df.loc[i,'sampling rate']=sample
                   
                   duration=librosa.get_duration(y=audio,sr=sample)
                   df.loc[i,'time']=duration
                   
                   i=i+1
       return df
#%%
import ast
from pydub import AudioSegment

def make_df():
    df=pd.DataFrame(columns=['id','age','gender','symptoms','disease','audio','sampling rate','time'])
    
    dirs = "/home/ravit/Konect-Code".split("/")
    base_path = "/".join(dirs)
    
    sheet_path = base_path + "/cospect/GenData/data.xlsx"
    
    wb = load_workbook(sheet_path)
    ws = wb.active
    
    start_row = 7
    end_row = 11

    for row_i in range(start_row, end_row+1):
        name = str(row_i)
        yt_link = ws["H" + str(row_i)].value
        
        file_path = base_path + "/cospect/Data/temp"
        os.system("youtube-dl --extract-audio " + yt_link + " -o " + file_path + ".wav")
        
        opus_ext = os.path.exists(file_path + ".opus")
        m4a_ext = os.path.exists(file_path + ".m4a")
        
        if opus_ext:
            dl_path = file_path + ".opus"
        elif m4a_ext:
            dl_path = file_path + ".m4a"
	
        out_path = file_path + ".wav"

        if opus_ext or m4a_ext:
            cmdstr = "ffmpeg -i \"" + dl_path + "\" -f wav -flags bitexact \"" + out_path + "\""		
            os.system(cmdstr)
            os.remove(dl_path)
            
            section = ws["I" + str(row_i)].value
            if(section != "FULL"):
                sound = AudioSegment.from_wav(out_path)
                
                section = ast.literal_eval(section) #string representation to actual list
                new_sound = AudioSegment.empty()
                section.reverse() #to make intervals from lastest to earliest
                for interval in section: #want to consider the intervals from largest to smallest
                    new_sound += sound[interval[0]*1000:interval[1]*1000] #changing seconds to milliseconds
                
                os.remove(out_path)
                new_sound.export(out_path)
        
        gender = ws["B" + str(row_i)].value
        age = ws["C" + str(row_i)].value
        symptoms = ws["D" + str(row_i)].value.split("_")
        disease = ws["G" + str(row_i)].value
            
        df.loc[row_i,'id']=name
        df.loc[row_i,'age']=age
        df.loc[row_i,'gender']=gender
        df.loc[row_i,'symptoms']=symptoms
        df.loc[row_i,'disease']=disease
        
        audio,sample=librosa.core.load(path=out_path)
                           
        df.loc[row_i,'audio']=np.asarray(audio)
        df.loc[row_i,'sampling rate']=sample
                           
        duration=librosa.get_duration(y=audio,sr=sample)
        df.loc[row_i,'time']=duration
        
        return df
    
#%%
df = make_df()
print(df.head())

#%%
from keras import Model
from keras.layers import Input, Conv1D, RNN, SimpleRNNCell, Dense, Activation, SimpleRNN

def get_model():
    conv_filters = 100
    conv_kernel_size = 100
    rnn_units = 100
    output_size = 1
    
    input_layer = Input(shape=(None, 1025))
    #input size is (None, None, 1025)
    	
    #conv_output = Conv1D(conv_filters, conv_kernel_size, padding='causal')(input_layer)
    #causal padding not implemented for tensorflow js
    conv_output = Conv1D(conv_filters, conv_kernel_size)(input_layer)
    #casual padding preserves time_steps dimension as the same
    	
    #rnn_output = RNN(SimpleRNNCell(rnn_units))(conv_output)
    rnn_output = SimpleRNN(rnn_units)(conv_output)
    #output is (None, rnn_units)
    	
    dense_output = Dense(output_size)(rnn_output)
    #output is (None, 1)
    	
    output_layer = Activation('softmax')(dense_output)
    #output is (None, 1)

    model = Model(inputs=input_layer, outputs=output_layer)
    	
    return model
    
#%%
#TODO: This method is repeated in several locations, including "GenData/DataAugmentation"
#   Consolidate into one Python util file
#utility function to create spectrogram input from an audio file

def make_spectrogram(signal, sample_rate):
	resample_freq = 8000
	signal_resampled = librosa.core.resample(signal, sample_rate, resample_freq)

	stft = librosa.core.stft(signal_resampled)
	spectrogram = np.power(np.abs(stft), 0.5)
	
	spectrogram = np.swapaxes(spectrogram, 0, 1)
    
	return spectrogram

def file2spectrogram(file_path):
	signal, sample_rate = librosa.core.load(file_path)
	if signal.shape[0] == 2:
		signal = np.mean(signal, axis=0)

	spectrogram = make_spectrogram(signal, sample_rate)
	
	return spectrogram
#%%
from keras.optimizers import Adam    

def train (training_x, training_y, testing_x, testing_y):
	lr=1e-4
	decay=1e-6
	epochs=10

	model = get_model()

	opt = Adam(lr=lr, decay=decay)
	model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])
	model.fit(training_x, training_y, epochs=epochs, validation_data=(testing_x, testing_y))

#%%
from sklearn.model_selection import train_test_split

#get defined values for training_x, training_y, testing_x, testing_y from pd dataframe
x_audio_data = df['audio'].to_numpy()
x_sampling_rate = df['sampling rate'].to_numpy()
y_data = df['symptoms'].to_numpy()

x_audio_split = []
y_data_split = []

chunk_len = 10 #seconds each chunk should be

#%%

x_spect_data = []
for (audio_data, sampling_rate) in zip(x_audio_data, x_sampling_rate):
    spectrogram = make_spectrogram(audio_data, sampling_rate)
    x_spect_data.append(spectrogram)


y_num_data = []
for symptoms in y_data:
    if ("Dry" in symptoms):
        y_num_data.append(1)
    else:
        y_num_data.append(0)
y_num_data = np.asarray(y_num_data)
#%%
for i in range(x_audio_data.shape[0]):
    x_audio_sample = x_audio_data[i]
    n = x_sampling_rate[i]*chunk_len
    x_split = [x_audio_sample[t:t + n] for t in range(0, len(x_audio_sample), n)]
    for _ in range(n-len(x_split[-1])):
        x_split[-1] = np.append(x_split[-1], 0)
    for j in range(len(x_split)):
        y_data_split.append(y_data[i])
    x_audio_split += x_split
#%%
num_samples = df.shape[0]

max_len = 0
for x_sample in x_spect_data:
    length = x_sample.shape[0]
    if length > max_len:
        max_len = length

x_padded = np.zeros((num_samples, max_len, 1025))
for i in range(len(x_spect_data)):
    x_sample = x_spect_data[i]
    x_sample_len = x_sample.shape[0]
    x_sample_padded = np.pad(x_sample, [(0, max_len-x_sample_len), (0,0)])
    x_padded[i] = x_sample_padded
#%%
print(y_data_split)
training_x, testing_x, training_y, testing_y = train_test_split(x_spect_data, y_num_data, test_size=0.10)
#%%
print(len(training_x))

#%%
trained_model = train(training_x, training_y, testing_x, testing_y)
#%%
#Saving the model in Keras format
#TODO: Save only the model and weights, not optimizer state or any of that junk
"""trained_model.save("./model.h5")

trained_model.save_weights('./model_weights.h5')
with open("./model.json", 'w') as file:
	file.write(model.to_json())"""

trained_model.save("./tf-model", save_format="tf")

import tensorflow as tf
model_tf = tf.keras.models.load_model("./model.h5")
model_tf.save("./tf-model", save_format="tf")