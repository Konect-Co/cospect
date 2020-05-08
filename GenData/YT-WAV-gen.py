import os
from openpyxl import load_workbook
import ast
from pydub import AudioSegment
import json

base_path = "/home/ubuntu/"
sheet_path = base_path + "cospect/data.xlsx"

def splice_interval(input_path, output_path, interval_list):
	sound = AudioSegment.from_wav(input_path)
	
	new_sound = AudioSegment.empty()
	interval_list.reverse() #to make intervals from lastest to earliest
	for interval in interval_list: #want to consider the intervals from largest to smallest
		    new_sound += sound[interval[0]*1000:interval[1]*1000] #changing seconds to milliseconds

	os.remove(input_path)
	new_sound.export(output_path)

def gen_vid(filename, link, section, path):
	file_path = path + filename
	
	os.system("youtube-dl --extract-audio " + link + " -o " + file_path + ".wav")

	opus_ext = os.path.exists(file_path + ".opus")
	m4a_ext = os.path.exists(file_path + ".m4a")

	if opus_ext:
		dl_path = file_path + ".opus"
	elif m4a_ext:
		dl_path = file_path + ".m4a"
	
	out_path = file_path + ".wav"

	if opus_ext or m4a_ext:
		cmdstr = "ffmpeg -i \"" + dl_path + "\" -f wav -flags bitexact \"" + out_path + "\""		
		os.system(cmdstr)
		os.remove(dl_path)

		if(section != "FULL"):
			section = ast.literal_eval(section) #string representation to actual list
			splice_interval(out_path, out_path, section)

def gen_vids (path):
	wb = load_workbook(sheet_path)
	ws = wb.active

	start_row = 7
	end_row = 10

	for row_i in range(start_row, end_row+1):
		name = str(row_i)
		yt_link = ws["H" + str(row_i)].value
		section = ws["I" + str(row_i)].value

		gender = ws["B" + str(row_i)].value
		age = ws["C" + str(row_i)].value
		symptoms = ws["D" + str(row_i)].value.split("_")
		disease = ws["G" + str(row_i)].value

		cough_data = {"gender":gender, "age":age, "symptoms":symptoms, "disease":disease}
		json_cough_data = json.dumps(cough_data)
		with open(base_path + "cospect/Data/" + name + ".json", 'w') as file:
			file.write(json_cough_data)
		
		gen_vid(name, yt_link, section, base_path + "cospect/Data/YT-Audio/")

path = base_path + "cospect/Data/YT-Audio/"
gen_vids(path)
