#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 14:52:21 2020

@author: jnarain
"""

import pandas as pd
from preprocessing.align_labels import *
from preprocessing.get_audio_chunks import *
from preprocessing.find_vocalizations import *
from preprocessing.assign_labels import *

##Change these based on what day and whose data you are processing        
data_path = '' #Path to a folder with the data you are processing.  The folder name should be the date of data collection in format: YYYYMMDD

participant_id = '' #Should match the participant IDs in the labeling file and the participant_database file
labels_path = 'label_data.csv'
pt_dir_path = 'participant_database.csv'

pt_db = pd.read_csv(pt_dir_path) #Database that contains DST start information for all participants
pt_db=pt_db.set_index('Participant')
selected_pt=pt_db.loc[participant_id]
utc_offset = selected_pt['UTC_offset']

drift = 0

parsedLabelDF, alignedLabelDF = align_data(labels_path, participant_id, data_path, utc_offset, drift = drift) #Align the label and recorder data.  The returned dataframes have the timings of labels relative to the recorder files
chunkDF, chunk_offset_dict = get_chunks(parsedLabelDF, data_path, utc_offset, drift = drift) #Remove any sections of audio that don't have labels to speed up later steps

token = segment_data(data_path, final_padding = 200, min_silence_len=300, silence_thresh=-24) #Segment the data using volume
token = 'Volume' #Added to incorporate in folder name for reference
path_to_df =  data_path + '/AudioSegments_' + token + '_' + os.path.split(data_path)[1] + '.csv'
path_to_segs = data_path + "/AudioSegments_" + token
assign_labels(data_path, path_to_df, data_path+'/formattedLabels'+os.path.split(data_path)[-1]+'.csv', token) #Match the segments with labels

