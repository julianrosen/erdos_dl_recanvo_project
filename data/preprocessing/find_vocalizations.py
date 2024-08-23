#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 13:19:18 2020

@author: jnarain
"""
import pandas as pd
from pydub.silence import detect_nonsilent
from preprocessing_general import *

#All times in ms

#Return the times to clip the segment including padding
#Inputs
#start_time time the segment starts relative to the file
#end_time time the segment ends relative to the file
#padding: the amount of padding to apply to the segment
#file_length: the total file length

#Returns
#start_time start time of the segment with padding
#end_time end time of the segment with padding
def pad_segment(start_time, end_time, padding, file_length):
    start_time = start_time - padding
    end_time = end_time + padding

    # make sure the start time and end time didn't go beyond the file limits
    if start_time < 0:
        start_time = 0
    if end_time > file_length:
        end_time = file_length

    return start_time, end_time

#This function exports each audio segment that meets the specified volume thresholds.  It is called by the function segment_data

#Inputs
#f: the audio file that the segment is being extracted from
#voc_start: the start of the segment relatie to f
#voc_duration: the duration of the segment
#padding: the amount of padding to add to the segment
#export_folder: path to which the segment is exported to
#using_raw_files: Boolean specifying whether f is an audio chunk file or a raw file

#Outputs
#orig_file_name: the name of the original .mp3 file fom the recorder
#label_chunk_file: the name of the audio chunk file from which the segment is being cut (if using_raw_files this is the original file name)
# start_rel_label_chunk: The start time of the segment relative to the label chunk file (label_chunk_file) in s
# start_rel_label_chunk_hms: The start time of the segment relative to the label chunk file (label_chunk_file) in hours, minutes, and second format (HH_MM_SS.SS)
# voc_start_rel_orig/1000: The start time of the segment relative to the original file (orig_file_name)
# string_start_csv: A string with the start time of the segment relative to the original file (orig_file_name) in hours, minutes, and second format (HH_MM_SS.SS)
# seg_duration: The duratino of the segment in s
# export_name: The name of the exported segment

def export_segment(f, voc_start, voc_duration, padding, export_folder, using_raw_files):
    sound = AudioSegment.from_wav(f)
    file_length = sound.duration_seconds  # the length of the label chunnk file, in seconds
    voc_end=voc_start+voc_duration

    ##Pad the segments, but make sure the results aren't out of bounds
    # Start times relative to f
    voc_start, voc_end = pad_segment(voc_start, voc_end, padding, file_length*1000)

    f_basename = os.path.basename(f)

    if using_raw_files:
        f_orig = os.path.splitext(f_basename.split('_')[0] + '_' + f_basename.split('_')[
            1])[0]  # The name of the original .mp3 file from the recorder
        total_offset = 0
    else:
        f_orig = f_basename.split('_')[0] + '_' + f_basename.split('_')[
            1]  # The name of the original .mp3 file from the recorder
        f_time_range = f_basename.split('_')[2]  # Extract the time range of the chunk being analyzed
        hour_offset, minute_offset, sec_offset, total_offset = find_offset_from_orig_file(f_time_range) #total_offset in seconds

    voc_start_rel_orig = total_offset*1000 + voc_start
    voc_end_rel_orig = total_offset*1000 + voc_end

    string_start = hms_string(voc_start_rel_orig/1000)
    string_start_csv = hms_string(voc_start_rel_orig/1000, use_colon=True)  # colon format is easier to view in excel
    string_end = hms_string(voc_end_rel_orig/1000, include_sec_frac=True)
    file_extension = '/' + f_orig + '_' + string_start + '--' + string_end

    if not(os.path.isdir(export_folder)):
       os.mkdir(export_folder)

    export_name = export_folder + file_extension + '.wav'
    audio_chunk = sound[voc_start: voc_end]  # AudioSegment indexing is in ms
    audio_chunk.export(export_name, format='wav')

    # Return information regarding the exported volumes segment to be saved in a csv
    orig_file_name = f_orig+'.mp3'
    label_chunk_file = f
    start_rel_label_chunk = voc_start/1000
    start_rel_label_chunk_hms=hms_string(voc_start/1000, use_colon=True)
    seg_duration = (voc_end-voc_start)/1000

    return orig_file_name, label_chunk_file, start_rel_label_chunk, start_rel_label_chunk_hms, voc_start_rel_orig/1000, string_start_csv, seg_duration, export_name

#This function segments the audio files in data_path using the inputted thresholds

#Inputs
#data_path: path to folder containing the audio files being processed
#silence_thresh: audio that is quieter than this is considered silence (dB)
#min silence length: the minimum length between non-silent regions (ms)
#final padding: the amount to pad segments (ms)

#Exports
#Creates a folder called AudioChunksByLabel within data_path that contains short audio segments based on volume thresholds
#Exports a directory of timing information for the extracted segments to the data_path folder called 'AudioSegments..."

def segment_data(data_path, min_silence_len=150, silence_thresh=-50, final_padding = 50, token="Volume"):
    path_to_chunks = data_path + "/AudioChunksByLabel" #location of the audio data trimmed to include only labeled areas

    if not (os.path.isdir(path_to_chunks)): #If the chunks have not been created, use the original files
        path_to_chunks = data_path
        using_raw_files = True
    else:
        using_raw_files = False

    ##Initialize lists to store segment information to export to a .csv file
    orig_file_list = []
    label_chunk_list = []
    start_rel_label_chunk_list = []
    start_rel_label_chunk_hms_list = []
    start_rel_orig_file_list=[]
    start_rel_orig_file_hms_list = []
    vol_seg_duration_list = []
    segment_path_list= []

    files = []
    for filename in os.listdir(path_to_chunks):
        if filename.endswith(".wav"):
            files.append(os.path.join(path_to_chunks, filename))
        if filename.endswith(".mp3") and not(os.path.exists(os.path.splitext(os.path.join(path_to_chunks, filename))[0]+'.wav')): #If using the original files, they will be converted to mp3 first; check to make sure it hasn't already been converted first
            wav_path = convert_wav(os.path.join(path_to_chunks, filename))
            files.append(wav_path)
    print("found %d .wav files in %s" % (len(files), path_to_chunks))

    for j, f in enumerate(files):
        f_basename = os.path.basename(f)

        sound=AudioSegment.from_wav(f)
        file_length = sound.duration_seconds

        print("Segmenting by volume: ", f)
        splitdata = detect_nonsilent(sound, min_silence_len,
                                     silence_thresh)  # list of pairs [start, end] of non-silent segments in ms
        # must be silent for at least min_silence_length; silence is anything quieter than -20 dB

        for vol_chunk in splitdata:
            start_time = vol_chunk[0]  # pad the start time by 500 ms
            end_time = vol_chunk[1]  # pad the end time by 500 ms
            duration = end_time-start_time

            orig_file_name, label_chunk_file, start_rel_label_chunk, start_rel_label_chunk_hms, start_rel_orig_file, start_rel_orig_file_hms, seg_duration, export_name = export_segment(f, start_time, duration, padding=final_padding, export_folder = data_path+"/AudioSegments_"+token, using_raw_files = using_raw_files)

            orig_file_list.append(orig_file_name)
            label_chunk_list.append(f)
            start_rel_label_chunk_list.append(start_rel_label_chunk)
            start_rel_label_chunk_hms_list.append(start_rel_label_chunk_hms)
            start_rel_orig_file_list.append(start_rel_orig_file)
            start_rel_orig_file_hms_list.append(start_rel_orig_file_hms)
            vol_seg_duration_list.append(seg_duration)
            segment_path_list.append(export_name)

    data = {'Recorder file': orig_file_list, 'Segment path': segment_path_list,
            'Label chunk file': label_chunk_list,
            'Start relative recorder (hh:mm:ss)': start_rel_orig_file_hms_list,
            'Start relative recorder (s)': start_rel_orig_file_list,
            'Segment duration': vol_seg_duration_list,
            'Start relative label chunk file (hh:mm:ss)': start_rel_label_chunk_hms_list,
            'Start relative label chunk file (s)': start_rel_label_chunk_list}
    df = pd.DataFrame(data)

    try:
        df_filename = data_path + '/AudioSegments_' + token + '_' + os.path.split(data_path)[1] + '.csv'
        df.to_csv(df_filename, index=None, header = True)
        return token
    except:
        print('No segments found with specified parameters')
        return None








    