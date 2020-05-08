data: GenData/YT-WAV-gen.py
	mkdir -p Data/YT-Audio
	pip install audiosegment
	python3 GenData/YT-WAV-gen.py

clean: 
	rm -rf Data
