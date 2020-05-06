from django.shortcuts import render
import requests
import os
import json
from django.http import HttpResponse
import tempfile

"""
	Left to do:
- Add functionality for passing file through python
- Obtain file from post request

- Add check for file size
- Check mobile version
"""

def home(request):
	return render(request,'home.html')

def process(request):
	"""if (request.method == "POST"):
		print("Post request")
		url = "./" +  + ".wav"
		with open(url, 'wb') as file:
			for chunk in request.FILES['UploadedFile'].chunks():
				file.write(chunk)
		prediction = predict(url)
		os.remove(url)"""

	audio_file = tempfile.NamedTemporaryFile()
	for chunk in request.FILES['UploadedFile'].chunks():
		audio_file.write(chunk)
	prediction = predict(audio_file.name)
	audio_file.close()

	data = getData(prediction)
	return render(request,'home.html',{'data':data}, status=200)

from keras.models import load_model
from keras.models import model_from_json
import tensorflow as tf
import numpy as np
import librosa

def getData(prediction):
	threshold = 0.75
	data = "<script>\n\
alert('Prediction is " + str(prediction) + "');\n"
	if (prediction>threshold):
		data += "document.getElementById(\"pos-diagnosis\").style.visibility = \"visible\";\n"
	else:
		data += "document.getElementById(\"neg-diagnosis\").style.visibility = \"visible\";\n"
	data += "document.getElementById(\"loading\").style.visibility = \"hidden\";\n</script>\n"
	return data

def predict(audio_path):
	resample_freq = 8000

	signal, sample_rate = librosa.core.load(audio_path)
	if signal.shape[0] == 2:
		signal = np.mean(signal, axis=0)

	signal_resampled = librosa.core.resample(signal, sample_rate, resample_freq)

	stft = librosa.core.stft(signal_resampled)
	spectrogram = np.power(np.abs(stft), 0.5)

	spectrogram = 1/(1 + np.exp(-spectrogram))
	spectrogram = np.expand_dims(spectrogram, axis=0)

	#TODO: Change directory for web app deployment
	#os.chdir("/home/ravit/Konect-Code/django-broswer-demo/cough_demo/static_files")

	"""
	model_arch = "/home/ravit/Konect-Code/django-broswer-demo/cough_demo/static_files/model.json"
	model_weights = "/home/ravit/Konect-Code/django-broswer-demo/cough_demo/static_files/model_weights.h5"

	with open(model_arch, 'r') as file:
		model = model_from_json(file.read())
		model.load_weights(model_weights)"""
	#model = load_model()

	model = tf.keras.models.load_model("./static_files/tf-model")
	spectrogram = np.swapaxes(spectrogram, 1, 2)
	prediction = model.predict(spectrogram)[0][0]
	return prediction