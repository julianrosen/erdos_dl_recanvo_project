import pandas as pd
from preprocessing_general import *
from shutil import copyfile

#This function using the timing information to match vocalizations to labels

#Inputs:
#data_path: path to the folder with the data being processed
#path_to_segment_df: Path to .csv with segment timing information.  This dataframe is exported by find_vocalizations.segment_data
#path_to_label_df: Path to .csv with label timing information.  This dataframe is exported by align_labels.align_data
#token: the token that was provided to the function find_vocalizations.segment_data as a short descriptor of the method for reference

#Exports
#The folder AutoSegments which contains subfolders of labels that contain all the audio segments matched with that label
#Rewrite the dataframe in path_to_segment df to include the matched label

def assign_labels(data_path, path_to_segment_df, path_to_label_df, token):
    segment_folder = '/AutoSegments_' + token #Where to export the segments to

    #Import and sort information on the identified audio segments
    segDF = pd.read_csv(path_to_segment_df)
    labelDF = pd.read_csv(path_to_label_df)

    segDF['End relative recorder (s)'] = segDF['Start relative recorder (s)'] + segDF['Segment duration']
    segDF.sort_values(['Recorder file', 'Start relative recorder (s)'], ascending=[True, True], inplace=True)

    labelDF['End relative recorder (s)'] = labelDF['Start relative recorder (s)'] + labelDF['Label duration']
    labelDF.sort_values(['Recorder file', 'Start relative recorder (s)'], ascending=[True, True], inplace=True)

    files_with_labels = np.array(list(labelDF['Recorder file']))
    files_with_labels = np.unique(files_with_labels)

    if not (os.path.isdir(data_path + segment_folder)):
        os.mkdir(data_path + segment_folder)

    #Initialize dictionary to store the assigned label for every file
    matchedLabels = dict()

    for f in files_with_labels:  # Go through the label-matching process separately for each file that has corresponding labels
        fileLabelDF = labelDF.loc[labelDF['Recorder file'] == f] #Find the lables corresponding to the file
        fileSegDF = segDF.loc[segDF['Recorder file'] == f] #Find the volume segments corresponding to the file

        seg_start_list = np.array(list(fileSegDF['Start relative recorder (s)']))
        seg_end_list = np.array(list(fileSegDF['End relative recorder (s)']))
        vol_file_list = list(fileSegDF['Recorder file'])
        vol_duration_list = list(fileSegDF['Segment duration'])
        segment_path_list = list(fileSegDF['Segment path'])

        label_start_list = np.array(list(fileLabelDF['Start relative recorder (s)']))
        label_end_list = np.array(list(fileLabelDF['End relative recorder (s)']))
        label_list = np.array(list(fileLabelDF['Label']))

        # These data structures are used to evaluate if there is a label near the start or end of a volume segment
        # Each row in start_comparison corresponds to an isolated segment; each column for that row is the time difference between the segment and the label
        # This matrix contains the distance from each segment to every label for the given file
        start_comparison = np.zeros((len(seg_start_list), len(label_list)))
        end_comparison = np.zeros((len(seg_start_list), len(label_list)))

        for i,label in enumerate(label_list):
            label_start_comparison = seg_start_list - label_start_list[
                i]  # Compare the entire list of volume segments to when each label started
            label_end_comparison = seg_start_list - label_end_list[
                i]  # Compare the entire list of volume segments to when each label ended
            #If negative, the segment started after the label ended; #if positive, the segment started before the label ended

            start_comparison[:,
            i] = label_start_comparison  # Put the results for comparing with the label start in column i of this array
            end_comparison[:,
            i] = label_end_comparison  # Put the results for comparing with the label end in column i of this array

        col_index_start = np.argmin(np.absolute(start_comparison),
                                    axis=1)  # The index corresponds to the column (label) with the start time closest to the segment

        min_value_start= np.array([start_comparison[i, col_ind] for i, col_ind in enumerate(col_index_start)])
        # The value of the time difference between the segment and the closes label start time (with index given in col_index_start)

        col_index_end = np.argmin(np.absolute(end_comparison),
                                  axis=1)  # This is the minimum value in the row that represents all the labels
        min_value_end = np.array([end_comparison[i, col_ind] for i, col_ind in enumerate(col_index_end)])

        #These values should be adjusted depending on the associated labeling accuracy for the participant,
        #and based on the accuracy of the segments outputted by find_vocalizations
        allowedDelayConfident = 15; #to be used if the label started after the segment
        allowedDelayTentative = 3;  # To be used if the label started before the segment; allows for a small amount of drift


        ##In-progress rules for assigning labels
        for i, seg in enumerate(segment_path_list):
            seg_start = seg_start_list[i]
            # The above lists were used to calculated the closest position of a label to a segment
            # Depending on the labeling style, there may be no labels close to a segment if the label spanned a long time duration
            # Here we check, for a given segment, if there are any labels that contain the segment entirely or partially
            #
            # Arrays to compare the volume segment start and end times to the label start and end times, for every label
            start_and_start = seg_start_list[i] > label_start_list  # The segment starts after the label starts; list has length of the number of labels in the file
            start_and_end = seg_start_list[i] < label_end_list  # The segment starts before the label ends; list has length of the number of labels in the file
            end_and_start = seg_end_list[i] > label_start_list  # The segment ends after the label starts; list has length of the number of labels in the file
            end_and_end = seg_end_list[i] < label_end_list  # The segment ends before the label ends; list has length of the number of labels in the file

            seg_start_in_label = [a and b for a, b in zip(start_and_start,
                                                            start_and_end)]  # The start of the segment is within the label bounds (for every label)
            seg_end_in_label = [a and b for a, b in
                                  zip(end_and_start, end_and_end)]  # The end of the segment is within the label bounds (for every label)

            seg_full_in_label = np.array([a and b for a, b in zip(seg_start_in_label,
                                                                    seg_end_in_label)])  # The start and end of the chunk is within the label bounds

            #Find the indices of those segments that fell partially or fully within the label
            seg_fully_in_label_indices = [j for j, x in enumerate(seg_full_in_label) if x]
            seg_end_in_label_indices = [j for j,x in enumerate(seg_end_in_label) if x]
            seg_start_in_label_indices = [j for j, x in enumerate(seg_start_in_label) if x]

            ##First check if a segment lies entirely within a label (e.g., a person labeled a long span with the same label, and the segment is entirely within it

            ##Match a segment with a label if:
            ## 1. the segment is fully or partially within the bounds of a label
            ## 2. the end of the segment occured during a label (suggesting that the label was pressed after the user heard the content of the segment)
            ## 3. The label and segment starts are close together.
            # 4. Segment started soon after a label ended
            ##5. The segment started within a label and ended in close proximity to a label
            ###This is the least ideal labeling situation because it could mean that  the label was not accurate for that segment so the user ended it
            #But, it could also mean that the labeler thought the segment was going to end and it didn't, so allow for some of these if there is a high threshold (very close proximity)
            if len(seg_fully_in_label_indices) > 0: #Condition 1
                label_index = seg_fully_in_label_indices[0] #the way the app label data is stored, the labels cannot overlap so the segment can be fully in at most exactly one label
                matchedLabels[seg] = label_list[label_index]
            elif len(seg_end_in_label_indices) > 0: #Condition2, The segment end time fell within a label, but not both
                label_index = seg_end_in_label_indices[0]
                matchedLabels[seg] = label_list[label_index]
            elif (min_value_start[i] < 0) and (abs(min_value_start[ #Condition 3, Less than zero means that the label started after the segment started
                                                       i]) <= allowedDelayConfident):  # if the sign is greater than 0, the segment started after a label
                matchedLabels[seg] = label_list[col_index_start[i]]
            elif (abs(min_value_end[i]) <= allowedDelayTentative): #Condition 4
                matchedLabels[seg] = label_list[col_index_start[i]]
            elif len(seg_start_in_label_indices) > 0: #Condition 5
                label_index = seg_start_in_label_indices[0]
                end_diff = end_comparison[i,label_index]
                if (abs(end_diff) <= allowedDelayTentative):
                    matchedLabels[seg] = label_list[label_index]
                else:
                    matchedLabels[seg] = ""
            else: #There was no label close enough to the segment to make an assignment
                matchedLabels[seg] = ""

        for key in matchedLabels:
            if (len(matchedLabels[key]) != 0):  # If there was a matched label
                labelFolder = data_path + segment_folder +"/" + matchedLabels[key]  # The folder name is the label
                if not (os.path.isdir(labelFolder)):  # If a folder for the label doesn't exist
                    os.mkdir(labelFolder)  # make it
                path_to_file = key
                segment_basename=os.path.basename(path_to_file)
                copydest = labelFolder + '/' + segment_basename
                copyfile(path_to_file, copydest)

    segDF['Possible Label'] = segDF['Segment path'].map(matchedLabels)
    segDF.to_csv( path_to_segment_df, index=None, header=True)

