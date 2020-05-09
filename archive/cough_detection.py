import os
from openpyxl import load_workbook

from make_data import make_data

from keras import Model
from keras.layers import Input, Conv1D, RNN, SimpleRNNCell, Dense, Activation, SimpleRNN
from keras.optimizers import Adam

def get_model():
	conv_filters = 100
	conv_kernel_size = 100
	rnn_units = 100
	output_size = 1

	input_layer = Input(shape=(None, 1025))
	#input size is (None, 1970, 1025)
	
	#conv_output = Conv1D(conv_filters, conv_kernel_size, padding='causal')(input_layer)
	#casual padding not implemented for tensorflow js
	conv_output = Conv1D(conv_filters, conv_kernel_size)(input_layer)
	#output is (None, 1970, conv_filters)
	#casual padding preserves time_steps as 1970
	
	#rnn_output = RNN(SimpleRNNCell(rnn_units))(conv_output)
	rnn_output = SimpleRNN(rnn_units)(conv_output)
	#output is (None, rnn_units)
	
	dense_output = Dense(output_size)(rnn_output)
	#output is (None, 1)
	
	output_layer = Activation('softmax')(dense_output)
	#output is (None, 1)
	
	model = Model(inputs=input_layer, outputs=output_layer)
	
	return model

def train (training_x, training_y, testing_x, testing_y):
	lr=1e-4
	decay=1e-6
	epochs=10

	model = get_model()

	opt = Adam(lr=lr, decay=decay)
	model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])
	model.fit(training_x, training_y, epochs=epochs, validation_data=(testing_x, testing_y))

	return model

if __name__ == '__main__':
	(training_x, training_y), (testing_x, testing_y) = make_data()

	os.chdir("../../..")
	trained_model = train(training_x, training_y, testing_x, testing_y)

	#TODO: Save only the model and weights, not optimizer state or any of that junk
	"""trained_model.save("./model.h5")

	trained_model.save_weights('./model_weights.h5')
	with open("./model.json", 'w') as file:
		file.write(model.to_json())"""

	trained_model.save("./tf-model", save_format="tf")

	import tensorflow as tf
	model_tf = tf.keras.models.load_model("./model.h5")
	model_tf.save("./tf-model", save_format="tf")
