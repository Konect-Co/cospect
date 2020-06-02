#%%
import os

import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Conv1D
from tensorflow.keras.layers import Activation, Lambda, LSTM, GRU
from tensorflow.keras.optimizers import Adam, RMSprop

from matplotlib import pyplot as plt
#%%
base_path = os.path.sep + "home" + os.path.sep + "ravit" + os.path.sep\
     + "Konect-Code" + os.path.sep
#%%
loaded = np.load(base_path + "cospect" + os.path.sep + "GenData" + os.path.sep + "np_data.npz")

x_train = loaded['x_train']
x_test = loaded['x_test']
y_train = loaded['y_train']
y_test = loaded['y_test']
#%%
#create spectrogram
def make_spectrogram(signals):
    frame_length = tf.constant(2048)
    fft_length = tf.constant(512)
    spectrogram_complex = tf.signal.stft(signals, frame_length, fft_length)
    spectrogram = tf.math.abs(spectrogram_complex)
    #spectrogram = tf.transpose(spectrogram, perm=[0,2,1])
    #tf.math.abs
    return spectrogram

#creates model that will be trained with all the layers
def create_model():
    conv_filters = 100
    conv_kernel_size = 100
    rnn_units = 100
    output_size = 1
   
    #forming model using tf.keras.Sequential()
    model = Sequential()
    #forming the input layer for model
    model.add(Input(shape=(None,), dtype=tf.float32))
    #forming the first lambda layer for model
    model.add(Lambda(lambda x:make_spectrogram(x)))
    #forming the first convolutional layer for model
    model.add(Conv1D(conv_filters, conv_kernel_size))
    #forming the simple RNN layer for model where output is fed back to input
    model.add(GRU(rnn_units))
    #forming the Dense layer for model
    model.add(Dense(output_size))
    #forming the activation or output layer for model
    model.add(Activation('sigmoid'))
   
    #returning the final model with all the layers
    return model

#training the model to improve accuracy
def train_model(model,x_train,y_train,x_test,y_test):
    #learning rate(should decrease over time whilst training)
    lr = 1e-3
    #displays how quickly the learning rate is decreasing
    #decay = 1e-6
	
    #sets up optimizer
    optim = RMSprop(lr=lr)
    #compiles the model showing loss and accuracy
    model.compile(optimizer=optim, loss = 'mse', metrics=['accuracy'])
    model.fit(x_train,y_train,validation_data=(x_test, y_test), batch_size=128, epochs = 10)
#%%
plt.specgram(x_train[20], NFFT=1024, Fs=2)
#%%
model = create_model()
train_model(model,x_train,y_train,x_test,y_test)
#%%
import librosa as lib
x, sr = lib.load("/home/ravit/Music/cough-sound.wav")
x = np.expand_dims(x, 0)
print(x.shape)
model.predict(x)
#%%
y_train[10:11]
#%%
model.save(base_path + os.path.sep + "detection-tf-model")
