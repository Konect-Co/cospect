var fft = require('fft-js').fft;
var dsp = require('dsp.js-browser');
var decode = require('audio-decode');

/*
https://developer.mozilla.org/en-US/docs/Web/API/AudioContext
https://developer.mozilla.org/en-US/docs/Web/API/FileReader
https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/ArrayBuffer
https://developer.mozilla.org/en-US/docs/Web/API/BaseAudioContext/decodeAudioData
https://developer.mozilla.org/en-US/docs/Web/API/File/Using_files_from_web_applications

This guy has the same problem of making the spectrogram
https://stackoverflow.com/questions/54726349/how-can-i-do-fft-analysis-of-audio-file-in-chrome-without-need-for-playback/54735593#54735593
*/

document.getElementById("upload-input").addEventListener('change', function() {
	const file = this.files[0];
	processFile(file);
});

document.getElementById("record").addEventListener('click', function() {
	let downloadButton = document.getElementById("download");
	let preview = document.getElementById("preview");
	let recording = document.getElementById("recording");
	recordingTimeMS = 5000;

	var getAccess = navigator.mediaDevices.getUserMedia({video: false, audio: true}).then(stream => {		
			preview.srcObject = stream;
			downloadButton.href = stream;
			preview.captureStream = preview.captureStream || preview.mozCaptureStream;
			return new Promise(resolve => preview.onplaying = resolve);
	});
	
	var record = getAccess.then(() => startRecording(preview.captureStream(), recordingTimeMS));
	
	record.then (recordedChunks =>
	{
		let recordedBlob = new Blob(recordedChunks, { type: "video/webm" });
		recording.src = URL.createObjectURL(recordedBlob);
		
		downloadButton.href = recording.src;
		downloadButton.download = "RecordedVideo.WAV";

		processFile(recordedBlob);
	});
});

document.getElementById("stop").addEventListener("click", function() { stopRecording(preview.srcObject); });

function wait(delayInMS) {
  return new Promise(resolve => setTimeout(resolve, delayInMS));
}

function startRecording (stream, lengthInMS) {
	let recorder = new MediaRecorder(stream);
	let data = [];

	recorder.ondataavailable = event => data.push(event.data);
	recorder.start();

	let stopped = new Promise((resolve, reject) => {
		recorder.onstop = resolve;
		recorder.onerror = event => reject(event.name);
	});

	let recorded = wait(lengthInMS).then(() => recorder.state == "recording" && recorder.stop());

	return Promise.all([
	stopped,
	recorded
	])
	.then(() => data);
}

function stopRecording(stream) {
	stream.getTracks().forEach(track => track.stop());
}

function processFile(file) {
	var spectrogram = new Array();
	var reader = new FileReader();
	reader.readAsArrayBuffer(file);
	
	reader.onloadend = function (evt) {
		if (evt.target.readyState == FileReader.DONE) {
			var arrayBuffer = evt.target.result;
			typedArray = new Int8Array(arrayBuffer);
			var signals = Array.from(typedArray);

			//padding function for even window size
			var window_size = 2048;
			if (signals.length%window_size != 0) {
				var difference = window_size-(signals.length%window_size);
				for (let i = 0; i < difference; i++) {
					signals.push(0);
				}
			}

			//processing signals in array
			var add_to_array = function(spectrum) {
				var spectrum_arr = [];
				for (let i = 0; i < spectrum.length; i++) {
					spectrum_arr.push(spectrum[i]);
				}
				spectrum_arr.push(0);
				spectrogram.push(spectrum_arr);
			};
			
			for (var i = 0; i < signals.length; i+= window_size) {
				var signal_wind = signals.slice(i, i+window_size);
				var fft = new dsp.FFT(signal_wind.length, 8000);
				
				fft.forward(signal_wind);
				add_to_array(fft.spectrum);
			}

			var spectrogram_tensor = tf.tensor(spectrogram);
			spectrogram_tensor = spectrogram_tensor.reshape([1,spectrogram.length,1025]);
			console.log(spectrogram_tensor.shape);
			const model_promise = tf.loadLayersModel("./tfjs-model/model.json");
			model_promise.then(function(model){
				prediction = model.predict(spectrogram_tensor).flatten().dataSync()[0];
				console.log(prediction);
				var display;
				if (prediction <= 0) {
					display = "No COVID-19 Symptoms Detected. Recommendation: Continue to monitor health and check for symptoms";
				} else {
					display = "COVID-19 Dry Cough Detected. Recommendation: Follow up with a doctor appointment. In case of severe symptoms or difficulty breathing, call 911."
				}
				document.getElementById("diagnosis").innerHTML = display;
				document.getElementById("diagnosis").style.visibility = "visible";
			});
		}
	}
}

