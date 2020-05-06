"""
FOR RICHARD:

- Replace Data Table in prev. directory with new directory
- Remove train_vids and test_vids and instead read from Data_Table in parent directory
- Modify script to store audio files appropriately with their row name in spreadsheet

"""


import os

train_vids = {
"TR1":"https://www.youtube.com/watch?v=l-sNgKgAucI",
"TR3":"https://www.youtube.com/watch?v=TIV460AQUWk",
"TR4":"https://www.youtube.com/watch?v=wuvn-vp5InE",
"TR7":"https://www.youtube.com/watch?v=31tnXPlhA7w",
"TR8":"https://www.youtube.com/watch?v=FIsQjsUJSiM",
"TR9":"https://www.youtube.com/watch?v=Rmlo2to0ogs",
"TR11":"http://www.youtube.com/watch?v=xwOfOgY8Ye8",
"TR12":"http://www.youtube.com/watch?v=ID5KlHVJ91M",
"TR13":"http://www.youtube.com/watch?v=Qbn1Zw5CTbA",
"TR16":"https://www.youtube.com/watch?v=pAHDqQRDPCk	",
"TR17":"https://www.youtube.com/watch?v=RFwr_zbgJII"}

test_vids = {
"TE2":"https://www.youtube.com/watch?v=KZV4IAHbC48",
"TE7":"https://www.youtube.com/watch?v=VX98aiYpmW4",
"TE8":"https://www.youtube.com/watch?v=yv4GUrI0Cw4",
"TE9":"https://www.youtube.com/watch?v=zuK4honWVsE",
"TE10":"https://www.youtube.com/watch?v=PFNvGqw9HKY",
"TE16":"https://www.youtube.com/watch?v=IYllzXfvkmY",
"TE17":"https://www.youtube.com/watch?v=IE_6K-ZfI64",
"TE20":"https://www.youtube.com/watch?v=5kAWlNZ-I_I",
"TE21":"https://www.youtube.com/watch?v=SsxsiISkLZA"}

def gen_vid(name, link, mypath):
	os.system("youtube-dl --extract-audio " + link + " -o " + name + ".wav")
	vidID= link.split("=")[1]

	opus_ext = os.path.exists(mypath + "/" + name + ".opus")
	m4a_ext = os.path.exists(mypath + "/" + name + ".m4a")


	filename = name
	if opus_ext:
		filename += ".opus"
	elif m4a_ext:
		filename += ".m4a"
	
	if opus_ext or m4a_ext:
		cmdstr = "ffmpeg -i \"" + filename + "\" -f wav -flags bitexact \"" + name + ".wav\""
		os.system(cmdstr)
	os.remove(filename)

def gen_vids (vids, mypath):
	for name in vids:
		link = vids[name]
		gen_vid(name, link, mypath)

os.chdir("./train/")
mypath = os.getcwd()
gen_vids(train_vids, mypath)
	
os.chdir("../test/")
mypath = os.getcwd()
gen_vids(test_vids, mypath)

