#!/usr/bin/env node

//https://bulma.io/ for CSS

//Following two are used in creating the server and GET/POST requests
var express = require('express');
var app = express();

var fs = require('fs');
var uuid = require("uuid");
var formidable = require('formidable');

app.use(express.urlencoded());
app.use(express.json());

var server = function(req, res, next) {
	if (req.method == 'POST') {
		var form = new formidable.IncomingForm();

		form.parse(req, function(err, fields, files) {
			console.log(fields);
			console.log(files);
		});

		console.log("POST " + req.url);
		console.log(req.body.status);									
		
		var name = "";
		var sex = "";
		var age = "";
		var status = "";
		var comments = "";

		console.log(req.body);
		console.log(name);
		console.log(sex);
		console.log(age);
		console.log(status);
		console.log(comments);

		var id = uuid.v4();

		//TODO: mkdir id
		fs.mkdirSync("audio_files/" + id);

		var audioPath = "audio_files/" + id + "/" + id + ".wav";
		//TODO: Get audio body from req
		//TODO: write audio file to audioPath

		var json_obj = {
			"id": id,
			"name": name,
			"sex": sex,
			"age": age,
			"status": status,
			"comments": comments
		};

		var jsonPath = "audio_files/" + id + "/" + "info.json";
		fs.writeFileSync(jsonPath, JSON.stringify(json_obj));

		res.end();
	}
	else if (req.method == "GET") {
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
			fs.readFile(req.url, function (err, html) {
				if (err) {	return;	}
				res.writeHeader(200, {"Content-Type": "text/html"});
				res.write(html);
				res.end();
			});
		}
	}
}



app.use(server);

var PORT=8000;
app.listen(PORT, function() {
	console.log("Listening on port " + PORT);
});
