import numpy as np
import math
import librosa
import soundfile
import os
from pydub import AudioSegment

#Functions to allow for easy time manipulations with pandas library
def hour(x):
    return(x.hour)
def day(x):
    return(x.day)
def month(x):
    return(x.month)
def year(x):
    return(x.year)
def seconds(x):
    return x.seconds

#Function converts an audio file to a .wav file (PCM_16)
#Inputs
#Filename: path the audio file
#output_path: path specifying where to output the converted .wav file
#Returns:
#Path to which the converted file was saved
#Exports
#Converted wave_file to output_path.  If output_path is not specified, the converted file is saved to the same directory as the original
def convert_wav(filename, output_path=None):
    sound = AudioSegment.from_file(filename)
    sr = sound.frame_rate
    y, lsr = librosa.load(filename, sr = sr)
    if output_path == None:
        output_path = os.path.splitext(filename)[0]+'.wav'
        print(output_path)
    soundfile.write(output_path, y, samplerate=sr, subtype='PCM_16')
    return output_path

#Function crops an array y sampled at sr
#Inputs
#y: array of time series data to crop
#sr: samping rate at which y was recorded
#t_start: start time for crop in seconds
#t_end: end_time to crop in seconds

#Returns:
#y_crop: array of cropped time series data
#sr: sampling rate of y_crop

def crop_audio(y, sr, t_start, t_end):
    n = len(y) #the number of indices in the array
    t = range(n) #create a time array of the corresponding size
    t = [t_val/sr for t_val in t] #create a time vector given the sampling rate information
    #Find the indicies corresponding to the given start and end times
    t=np.array(t)
    start_ind=np.argmin(np.absolute(t-t_start))
    end_ind = np.argmin(np.absolute(t-t_end))

    y_crop = y[start_ind: end_ind]
    return y_crop, sr

#This function finds the time offsets relative to the original file for an outputted audio chunk file
#Input f_time_range: an exported audio chunk file that has time information in the file name
#Returns
#hour_offset: hour offset from original file (along with minute_offset, sec_offset)
#minute_offset: minute offset from original file (along with hour_offset, sec_offset)
#sec_offset: second offset from original file (along with hour_offset, minute_offset)
#total_offset: total offset from original file in seconds
def find_offset_from_orig_file(f_time_range):
    hour_offset = float(f_time_range.split('-')[0])
    minute_offset = float(f_time_range.split('-')[1])
    sec_offset = float(f_time_range.split('-')[2])
    total_offset = hour_offset * 3600 + minute_offset * 60 + sec_offset  # total offset from original file, in s
    return hour_offset, minute_offset, sec_offset, total_offset

#This function converts a total number of seconds to hours, minutes, and seconds
#Input x: time in seconds to convert
#Returns
#num_hour, num_min, num_sec: number of hours, minutes, and seconds respectively in x (x = num_hour+num_min+num_sec)
def hms(x):
    num_hour= math.floor(x/3600)
    num_min = math.floor((x-(num_hour*3600)) / 60)
    num_sec = math.floor(x - ((num_hour * 3600) + (num_min*60)))
    return num_hour, num_min, num_sec

#This function creates a string that has the hour minute and second components of x, where x is a total number of seconds.

#Inputs
#x: time in seconds to convert
#include_sec_frac: Boolean specifying whether to include fractional seconds (SS.SS instead of SS)
#use_colon: Boolean specifying whether to use a colon in the file string.  If false, a hyphen is used

#Returns
#A string that has the hours minutes and seconds in x (such that hours+minutes+seconds = x)
#The outputted string has the format HH-MM-SS.SS if use_colon = False.  The outputted string has the format HH:MM:SS.SS if use_colon = True

def hms_string(x, include_sec_frac = True, use_colon=False):
    num_hour= (math.floor(x/3600))
    num_min = (math.floor((x-(num_hour*3600)) / 60))
    num_sec_base = (math.floor(x - ((num_hour * 3600) + (num_min*60))))
    num_sec = round((x - ((num_hour * 3600) + (num_min*60))),2)

    num_hour = str(num_hour); num_min = str(num_min); num_sec_base = str(num_sec_base); num_sec = str(num_sec)
    if len(num_hour) < 2:
        num_hour = "0"+num_hour
    if len(num_min) < 2:
        num_min = "0"+num_min
    if len(num_sec_base) < 2:
        num_sec = "0"+num_sec
        num_sec_base = "0"+num_sec_base

    if use_colon:
        if include_sec_frac:
            return ":"+num_hour + ':' + num_min + ':' + num_sec #added in ":" string because otherwise excel deletes leader 0s
        else:
            return ":"+num_hour + ':' + num_min + ':' + num_sec_base
    else:
        if include_sec_frac:
            return num_hour+'-'+num_min+'-'+num_sec
        else:
            return num_hour + '-' + num_min + '-' + num_sec_base

