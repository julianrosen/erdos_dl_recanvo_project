#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 14:51:24 2020

@author: jnarain
"""


import pandas as pd
from datetime import datetime, timedelta
from mutagen.mp3 import MP3
from pytz import timezone as timezonepytz
from preprocessing_general import *

#This function finds groups of labels that occur in 'chunks' (i.e., are closely spaced together) in order to extract shorter audio segments
##that enclode all the labels from the longer audio file

#Input todays_data: pandas Dataframe with the labels from the selected day
#Returns chunkDF: pandas Dataframe with timings for audio chunks that encompass all the labels
def find_label_chunks(todays_data):
    #to minimize processing time, we will only look at segments of the audio near labels
    #Define audio chunks that have labels in them
    todays_data['Next Label Time'] = todays_data['Event Created Time'].shift(-1)
    todays_data['Time Between Labels'] = todays_data['Next Label Time']- todays_data['Event Created Time'] #Find the time delay between consecutive labels so we don't have to look at the whole file
    todays_data_indices = list(todays_data.index)
    todays_data_indices.sort() #Makes sure the indices are in numerical order
    
    todays_data.at[todays_data_indices[-1],'Time Between Labels'] = timedelta(0) #Convert the last number in the Time Between Labels Column to a 0
    todays_data['Time Between Labels (s)']=todays_data['Time Between Labels'].apply(seconds)

    max_time_between_labels = 100 #in s #Create a new group if labels are this many seconds apart or more
    

    chunks_near_labels_df = todays_data.loc[todays_data['Time Between Labels (s)'] > max_time_between_labels] #Look for transition points that meet the new chunk criteria
    
    #Create a list of the end and start times for the chunks, based on the max time between labels (later shifting will make this work)
    chunks = list(chunks_near_labels_df.index)
    chunks_end = list(chunks_near_labels_df['Event End']) #These times correspond to when a chunk should end 
    chunks_start = list(chunks_near_labels_df['Next Label Time']) #These times correspond to when a chunk should start
    
    corresponding_starts = list(chunks_near_labels_df['Event Created Time'])
    
    for i in range(len(chunks_end)): #Set the duration for a note added to the app as 5s
        if pd.isna(chunks_end[i]): #if the participant added a note, the end time will appear as nan
            chunks_end[i] = corresponding_starts[i] + timedelta(seconds=5)
    
    
    num_padding_seconds = 20 #Pad the specified chunks by this amount (s) to make sure we are getting the sounds of interest
    K=timedelta(seconds=num_padding_seconds)
    
    first_chunk_start_df = todays_data.loc[ [todays_data_indices[0]] , : ] #The first chunk should always start the defined delta away from the last label
    first_chunk_start = list(first_chunk_start_df['Event Created Time'])
    first_chunk_start[0]=first_chunk_start[0]-2*K
    
    last_chunk_end_df=todays_data.loc[ [todays_data_indices[-1]] , : ] #The last chunk should end the defined delta away from the last label
    last_chunk_end = list(last_chunk_end_df['Event End'])
    last_chunk_end[0]=last_chunk_end[0]+2*K
    
    chunks_end = [x + K for x in chunks_end] #Add the time delta on the end time
    chunks_start = [x - K for x in chunks_start] #Subtract the time delta from the start time
    
    chunks_start.insert(0, first_chunk_start[0])
    chunks_end.insert(len(chunks_end), last_chunk_end[0])
    
    chunk_data = {'Chunk Start':chunks_start,'Chunk End': chunks_end}
    chunkDF = pd.DataFrame(chunk_data)
    
    return chunkDF #Dataframe with start and end times for the chunks

#Inputs
#file_start_time start time of recorder file
#file_end_time end time of recorder file
#chunks_start list of chunk start times in UTC based on labeling info
#chunks_end list of chunk end times in UTC based on labeling info

#Returns
#f_chunks_start List of chunk start times that occur within the file, relative to the file start time
#f_chunks_end Listof chunk end times that occur within the file, relative to the file end time

#Given an audio file beginning at file_start_time and file_end_time, and chunk information, this function returns the chunks that fall within the specified event
def get_corresponding_chunks(file_start_time, file_end_time, chunks_start, chunks_end): #Given a list of targeted label-focused chunks, this function finds the chunks in corresponding to an event beginning at f_start_time and ending at f_end_time
    #These lists will become lists of only the label-defined chunks that go with the given audio file
    
    f_chunks_start = np.array(chunks_start) 
    f_chunks_end = np.array(chunks_end)

    #Define boolean masks based on the chunk start and end time

    #Check if the chunk started within the given file            
    start_and_start = f_chunks_start > file_start_time #The chunk starts after the file starts
    start_and_end = f_chunks_start < file_end_time #The chunk starts before the file ends
    chunk_start_in_file = [a and b for a,b in zip(start_and_start, start_and_end)]

    #Check if the chunk ends within the given file
    end_and_start = f_chunks_end > file_start_time #The chunk ends after the file starts
    end_and_end = f_chunks_end < file_end_time #The chunk ends before the file ends
    chunk_end_in_file = [a and b for a,b in zip(end_and_start, end_and_end)] 

    chunk_part_in_file = np.array([a or b for a,b in zip(chunk_start_in_file, chunk_end_in_file)])
    chunk_full_in_file = np.array([a and b for a,b in zip(chunk_start_in_file, chunk_end_in_file)])
    no_chunk_here = np.array([not(a) and not(b) for a,b, in zip(chunk_part_in_file, chunk_full_in_file)])

    #If the chunk was partially in the file, adjust the chunk start and end time to stay within the file bounds
    for i, (s, e) in enumerate(zip(f_chunks_start, f_chunks_end)): #Define appropraite start and end times if only part of the chunk is in the file
        if (not(chunk_full_in_file[i]) and chunk_start_in_file[i]):
            f_chunks_end[i] = pd.Timestamp(datetime.timestamp(file_end_time), unit='s', tz='UTC')
        elif (not(chunk_full_in_file[i]) and chunk_end_in_file[i]):
            f_chunks_start[i]=pd.Timestamp(datetime.timestamp(file_start_time), unit='s', tz='UTC')

    not_in_file = np.where(no_chunk_here == True)[0]
    f_chunks_start = np.delete(f_chunks_start, not_in_file)
    f_chunks_end = np.delete(f_chunks_end, not_in_file)
                        
    return f_chunks_start, f_chunks_end


#This exports the audio segments that contain the labels (from within a larger recording file) as .wav files
#Inputs
#chunkDF: pandas dataframe with timing information for chunks in UTC, returned by the function find_label_chunks
#data_path: path to the folder containing the full audio files
#utc_offset: time offset between the recorder clock and UTC (labels are timestamped in UTC) in hours
#drift: manually calculated recorder clock drift (s)

#Returns
#chunk_offset_dict
#chunk_duration_dict

#Exports
#.wav files that have the sub-chunks of audio segments that contain all the labels present in all the recording files in data_path
#Thes files are exported to a subdirectory of data_path called "AudioChunsksByLabel".

def convert_chunks_to_wav(chunkDF, data_path, utc_offset, drift):

    files = []
    for filename in os.listdir(data_path):
        if filename.endswith(".mp3"): #The vidoes that have been rotated have the file extension c
            files.append(os.path.join(data_path, filename))
    print("found %d .mp3 files in %s"%(len(files),data_path))
    
    chunks_start = list(chunkDF['Chunk Start'])  
    chunks_end = list(chunkDF['Chunk End'])
    
    chunk_offset_dict = {} #Will be used to relate the start times of identified vocalization segments from label chunks to the file start time
    chunk_duration_dict = {}
    
    for j,f in enumerate(files): #For each audio file in the directory, calculate which chunks started and ended in the file and export
        mod_time=os.path.getmtime(f)
        audio = MP3(f)
        
        ##Convert the recorder time stamp to UTC, using information in the participant database
        utc_offset_seconds = utc_offset*3600+drift #convert utc_offset (given in hours, to seconds)
        seg_start = mod_time - utc_offset_seconds
  
        seg_start=datetime.fromtimestamp(seg_start)
        file_start_time= seg_start.replace(tzinfo=timezonepytz('UTC'))
        file_end_time = file_start_time+timedelta(seconds=audio.info.length)
        
        f_chunks_start, f_chunks_end = get_corresponding_chunks(file_start_time, file_end_time, chunks_start, chunks_end)    
        f_chunks_duration = [a_i - b_i for a_i, b_i in zip(f_chunks_end, f_chunks_start)]

        #Create the AudioChunksByLabel directory if it doesn't exist
        if not(os.path.isdir(data_path+"/AudioChunksByLabel")):
            os.mkdir(data_path+"/AudioChunksByLabel")
        
        #Export the defined audio chunks from the file
        print("Exporting audio chunks around data labels for", f)
        for i, (chunk_start, chunk_end) in enumerate(zip(f_chunks_start, f_chunks_end)):
            fname = os.path.basename(f)

            #Start and end times of chun relative to file
            #Start and end itme in seconds
            this_chunk_starts = (timedelta.total_seconds(chunk_start-file_start_time)) #chunk start time relative to file, in ms
            this_chunk_ends = (timedelta.total_seconds(chunk_end-file_start_time)) #chunk end time relative to file, in ms

            string_start = hms_string(this_chunk_starts, include_sec_frac=True)
            string_end = hms_string(this_chunk_ends, include_sec_frac=True)

            file_extension = string_start + '--' + string_end
            export_path = data_path + "/AudioChunksByLabel/" + fname[:-4] + '_'+file_extension + '.wav'
            print(export_path)

            y, sr = librosa.load(f, sr = 44100)
            chunk, sr = crop_audio(y, sr, this_chunk_starts, this_chunk_ends)
            soundfile.write(export_path, chunk, sr)
            
            #Add the offset and duration values to the corresponding dictionaries, in seconds
            chunk_offset_dict[export_path[:-4]] = this_chunk_starts #Store the start time of the chunk relative to the file time
            chunk_duration_dict[export_path[:-4]] = (this_chunk_ends)-(this_chunk_starts) #Store the chunk duration
            
    return chunk_offset_dict, chunk_duration_dict

#Find sub-sections of long audio recordings that contain labels and export them.  Calls the other subfunctions in this package in sequence.

#Inputs
#Inputs
#chunkDF: pandas dataframe with timing information for chunks in UTC, returned by the function find_label_chunks
#data_path: path to the folder containing the full audio files
#utc_offset: time offset between the recorder clock and UTC (labels are timestamped in UTC) in hours
#drift: manually calculated recorder clock drift (s)

#Returns
#chunk_offset_dict
#chunk_duration_dict

#Exports
#.wav files that have the sub-chunks of audio segments that contain all the labels present in all the recording files in data_path
#Thes files are exported to a subdirectory of data_path called "AudioChunsksByLabel".

def get_chunks(todays_data, data_path, utc_offset, drift):
    chunkDF = find_label_chunks(todays_data)
    chunk_offset_dict, chunk_duration_dict= convert_chunks_to_wav(chunkDF, data_path, utc_offset, drift)
    return chunkDF, chunk_offset_dict
