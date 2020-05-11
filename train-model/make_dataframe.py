import librosa as lb
import json
import numpy as np
import pandas as pd
import os
import librosa
pd.DataFrame()

class MakeData:
    def __init__(self,audio_path,json_path):
        self.audio_path=audio_path
        self.json_path=json_path
    
    def is_json(self,filename):
        try:
            x=str(filename)
            y=x.split('.')[1]
            if(y=='json'):
                
                return True
        except IndexError as e:
            print('FILE EXTENSION NOT IN '+'.json')
            
        
        
    def make_df(self,csv_name='audio.csv'):
        
        df=pd.DataFrame()
        json_files=os.listdir(self.json_path)
        audio_files=os.listdir(self.audio_path)
        
        
        
        df=pd.DataFrame(columns=['id','age','gender','symptoms','disease','audio','sampling rate','time'])
        i=0
        #getting names of  all the keys in json file
        for j in json_files:
            if(self.is_json(j)==True):
                with open(os.path.join(self.json_path,j)) as f:
                    
                    dict1=json.load(f)
                    id1=j.split('.')[0]
                    age=dict1['age']
                    gender=dict1['gender']
                    symptoms=dict1['symptoms'][0]
                    disease=dict1['disease']
                    
                
                    df.loc[i,'id']=id1
                    df.loc[i,'age']=age
                    df.loc[i,'gender']=gender
                    df.loc[i,'symptoms']=symptoms
                    df.loc[i,'disease']=disease
                    path=self.audio_path+'/'+j.split('.')[0]+'.wav'
                    audio,sample=librosa.core.load(path=os.path.normpath(path))
                    
                    df.loc[i,'audio']=np.asarray(audio)
                    df.loc[i,'sampling rate']=sample
                    
                    duration=librosa.get_duration(y=audio,sr=sample)
                    df.loc[i,'time']=duration
                    
                    
                    
                    
                    i=i+1

        return df
            
                
#%%

'''
h=MakeData(audio_path='F:/PROJECTS/cospect-master/Data/YT-Audio',json_path='F:/PROJECTS/cospect-master/Data')
df=h.make_df()
print(df.head())
'''
