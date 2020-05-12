import librosa as lb
import json
import numpy as np
import pandas as pd
import os
import librosa
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

sep = os.path.sep
dirs=os.getcwd().split(sep)[:-2]
base_path = sep.join(dirs)
base_path = "/home/ravit/Konect-Code/cospect/"

audio_path = base_path + "Data" + sep + "YT-Audio"
json_path= base_path + "Data"
h = MakeData(audio_path, json_path)
df=h.make_df()
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
def make_spectrogram(file_path):
	signal, sample_rate = librosa.core.load(file_path)
	if signal.shape[0] == 2:
		signal = np.mean(signal, axis=0)

	resample_freq = 8000
	signal_resampled = librosa.core.resample(signal, sample_rate, resample_freq)

	stft = librosa.core.stft(signal_resampled)
	spectrogram = np.power(np.abs(stft), 0.5)
	
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
a = (df['audio'].to_numpy())
#%%
import sklearn

#get defined values for training_x, training_y, testing_x, testing_y from pd dataframe
x_audio_data = df['audio'].to_numpy()
x_spect_data = np.asarray([make_spectrogram(audio_data) for audio_data in x_audio_data])

y_data = np.asarray([]).to_numpy()
y_num_data = np.asarray([])
for symptoms in y_data:
    if ("Dry" in symptoms):
        y_num_data.append(1)
    else:
        y_num_data.append(0)

#x_spect_data.reshape() --> appropriate dimension
#y_num_data.reshape() --> appropriate dimension
        
#split x_spect_data and y_num_data into appropriate variables
#maybe, as you were saying, using sklearn
        
os.chdir("../../..")
trained_model = train(training_x, training_y, testing_x, testing_y)

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