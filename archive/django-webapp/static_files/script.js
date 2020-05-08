document.getElementById("upload-input").addEventListener('change', function() {
	const file = this.files[0];
	document.getElementById("buttons").style.visibility = "hidden";
	document.getElementById("loading").style.visibility = "visible";
	processFile(file)
});

document.getElementById("record").addEventListener('click', function() {
	let preview = document.getElementById("preview");
	let recording = document.getElementById("recording");
	recordingTimeMS = 5000;

	var getAccess = navigator.mediaDevices.getUserMedia({video: false, audio: true}).then(stream => {		
			preview.srcObject = stream;
			preview.captureStream = preview.captureStream || preview.mozCaptureStream;
			return new Promise(resolve => preview.onplaying = resolve);
	});
	
	var record = getAccess.then(() => startRecording(preview.captureStream(), recordingTimeMS));
	
	record.then (recordedChunks =>
	{
		let recordedBlob = new Blob(recordedChunks, { type: "video/webm" });
		recording.src = URL.createObjectURL(recordedBlob);

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
	/*let xhr = new XMLHttpRequest();
	xhr.open("POST", "process/");
	xhr.setRequestHeader("X-CSRFToken", csrftoken);
	var reqBody = {"file":file};
	xhr.send(JSON.stringify(reqBody));*/

	document.getElementById("upload-form").submit();
	console.log("POST file for processing");
}