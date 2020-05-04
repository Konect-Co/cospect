const http = require('http');
const fs = require('fs');
var mime = require('mime');

const hostname = '127.0.0.1';
const port = 3000;

const server = http.createServer((req, res) => {
	if (req.method == 'GET') {
	    if (req.url == '/') {
	        fs.readFile("index.html", function(err, data){
	            res.writeHead(200, {'Content-Type': 'text/html'});
	            res.write(data);
	            res.end();
	        });
	    }
	    else {
			console.log (req.url);
	        var file = req.url.substr(1);
	        if (fs.existsSync(file)) {
	            fs.readFile(file, function(err, data){
	                res.writeHead(200, {'Content-Type': mime.getType(req.url)});
	                res.write(data);
	                res.end();
	            });
	        }
	    }
	}
});

server.listen(port, hostname, () => {
  console.log(`Server running at http://${hostname}:${port}/`);
});
