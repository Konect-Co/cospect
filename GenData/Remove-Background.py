import os
import wave
from pydub import AudioSegment

train_intervals = {
"TR1":[[12,28]],
"TR3":[[0,126]],
"TR8":[[0,30],[55,61],[86,89]],
"TR9":[[5,65]],
"TR11":[[107,112]],
"TR12":[[7,13]],
"TR13":[[0,5]],
"TR17":[[55,60]]}

test_intervals = {
"TE2":[[0,48]],
"TE4":[[0,18]],
"TE5":[[2,20]],
"TE6":[[0,89]],
"TE7":[[4,26],[72,84]],
"TE8":[[4,12], [30,37]],
"TE9":[[4,25]],
"TE10":[[0,9]],
"TE16":[[14,45]],
"TE17":[[10,13], [20,23], [28,35]],
"TE20":[[4,33]],
"TE21":[[3,13]]}


def splice_interval(name, interval_list, output_dir):
	sound = AudioSegment.from_wav(name + ".wav")
	new_sound = AudioSegment.empty()

	for interval in interval_list: #want to consider the intervals from largest to smallest
		new_sound += sound[interval[0]*1000:interval[1]*1000] #changing seconds to milliseconds

	new_sound.export(output_dir + name + ".wav")

def splice_intervals (intervals, output_dir):
	for name in intervals:
		interval_list = intervals[name]
		splice_interval(name, interval_list, output_dir)

os.chdir("./raw/train/")
mypath = os.getcwd()
splice_intervals(train_intervals, "../../processed/train/")
	
os.chdir("../test/")
mypath = os.getcwd()
splice_intervals(test_intervals, "../../processed/test/")

