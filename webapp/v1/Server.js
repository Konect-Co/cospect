/*#!/usr/bin/env node

//Following two are used in creating the server and GET/POST requests
var express = require('express');
var formidable = require('formidable');
var app = express();

var fs = require('fs');
var uuid = require("uuid");


app.use(express.urlencoded());
app.use(express.json());

app.post('/submit', function(req, res){
	var id = uuid.v4();
	console.log("POST " + req.url);

	var form = new formidable.IncomingForm();

    form.parse(req);

    form.on('fileBegin', function (name, file){
        file.path = __dirname + '/uploads/' + file.name;
    });

    form.on('file', function (name, file){
        console.log('Uploaded ' + file.name);
    });
	//form.on('error', function(err) { console.log(err); });
	//form.on('aborted', function() { console.log('Aborted'); });
	//form.on('fileBegin', function (name, file){ file.path = __dirname + '/uploads/' + file.name; });
	//form.on('file', function(name, file) { console.log(file); });
	
	//TODO: Get audio body from req
	//TODO: write audio file to audioPath
	//TODO: Save data into MySQL Database

	fs.readFile("./result.html", function (err, html) {
		if (err) {
			throw(err);
		}
		res.writeHeader(200, {"Content-Type": "text/html"});
		html +=
"<script>\
document.getElementById(\"cough-symptom\").innerHTML = \"something for cough symptom\";\
document.getElementById(\"id\").innerHTML = \"" + id + "\";\
</script>";
		res.write(html);
		res.end();
	});
});

var server = function(req, res, next) {
	if (req.method == "GET") {
		console.log("GET " + req.url);
		if (req.url == "/" || req.url == "") {
			fs.readFile("./index.html", function (err, html) {
				if (err) {
					throw(err);
				}
				res.writeHeader(200, {"Content-Type": "text/html"});
				res.write(html);
				res.end();
			});
		}
		else {
			fs.readFile("." + req.url, function (err, content) {
				if (err) { res.end(); return; }
				res.writeHeader(200);
				res.write(content);
				res.end();
			});
		}
	}
}

app.use(server);

var PORT=8000;
app.listen(PORT, function() {
	console.log("Listening on port " + PORT);
});*/

var fs = require('fs');
var uuid = require("uuid");

var express = require('express');
const {spawn} = require('child_process');
var formidable = require('formidable');

var app = express();

app.get('/', function (req, res){
    res.sendFile(__dirname + '/index.html');
});

app.get('/audio.html', function (req, res){
	res.sendFile(__dirname + '/audio.html');
});

app.get('/style.css', function (req, res){
	fs.readFile("./style.css", function (err, content) {
		if (err) { res.end(); return; }
		res.writeHeader(200, {"Content-Type": "text/css"});
		res.write(content);
		res.end();
	});
});

app.post('/submit', function (req, res){
	var id = uuid.v4();
	var file_path = __dirname + '/uploads/' + id;

    var form = new formidable.IncomingForm();

    form.parse(req);

    form.on('fileBegin', function (name, file){
        file.path = file_path;
    });

    form.on('file', function (name, file){
        console.log('Uploaded ' + file.name);
    });

    fs.readFile("./result.html", function (err, html) {
		if (err) {
			throw(err);
		}	

		const python = spawn('python', ['modelInference.py', file_path]);
		
		python.stdout.on('data', function (data) {
			var coughSymptom = data.toString().replace(/\n/g, '');
			//trimming off the last \n character
			html +=
"<script>\
document.getElementById(\"cough-symptom\").innerHTML = \"" + coughSymptom + "\";\
document.getElementById(\"id\").innerHTML = \"" + id + "\";\
</script>";

			res.writeHeader(200, {"Content-Type": "text/html"});

			res.write(html);
			res.end();

		});
		
		/*python.on('close', (code) => { });

		res.writeHeader(200, {"Content-Type": "text/html"});

		res.write(html);
		res.end();*/
	});
});

app.listen(3000);
