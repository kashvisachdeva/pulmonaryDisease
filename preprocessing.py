import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
def getPureSample(raw_data, start, end, sr=22050):
    '''
    Extracts a segment of audio data from a raw numpy array based on specified start and end times.
    
    Parameters:
    raw_data (numpy array): The full audio data array to extract the segment from.
    start (float): The start time (in seconds) of the segment to extract.
    end (float): The end time (in seconds) of the segment to extract.
    sr (int): Sampling rate of the audio data in samples per second. Default is 22050.
    
    Returns:
    numpy array: A segment of the audio data between the start and end times.
    '''
    
    # Calculate the maximum valid index in the raw_data array.
    max_ind = len(raw_data)
    
    # Convert the start time (in seconds) to an index in the raw_data array.
    # Ensure the index does not exceed the length of the array (max_ind).
    start_ind = min(int(start * sr), max_ind)
    
    # Convert the end time (in seconds) to an index in the raw_data array.
    # Ensure the index does not exceed the length of the array (max_ind).
    end_ind = min(int(end * sr), max_ind)
    
    # Return the segment of the array between the calculated start and end indices.
    return raw_data[start_ind: end_ind]

data=pd.read_csv('csv_data/data.csv')

'''
#os.makedirs('processed_audio_files')
for index,row in data.iterrows():
    print("Index ->",index)
    print("Data->\n",row)
    break

import librosa as lb  # For audio loading, processing, and padding.
import soundfile as sf  # For saving processed audio files.

# Initialize counters for file indexing and total processed files.
i, c = 0, 0
path='data/'
# Iterate over each row in the `data` DataFrame.
for index, row in data.iterrows():
    maxLen = 6  # Maximum allowed length of an audio segment (in seconds).
    start = row['start']  # Start time of the segment.
    end = row['end']  # End time of the segment.
    filename = row['filename']  # filename for the audio file.
    
    # If the duration of the segment exceeds the maximum length, truncate it.
    if end - start > maxLen:
        end = start + maxLen
    
    # Construct the full path to the original audio file.
    audio_file_loc = path + filename + '.wav'
    
    # Adjust the filename if the same patient has multiple cycles.
    if index > 0:
        # Check if the current file is the same as the previous file in the DataFrame.
        if data.iloc[index - 1]['filename'] == filename:
            i += 1  # Increment the counter to differentiate the file.
        else:
            i = 0  # Reset the counter for a new file.
    filename = filename + '_' + str(i) + '.wav'  # Append the counter to the filename.
    
    # Define the save path for the processed audio file.
    save_path = 'processed_audio_files/' + filename
    c += 1  # Increment the count of processed files.
    
    # Load the audio file using Librosa.
    audioArr, sampleRate = lb.load(audio_file_loc)
    
    # Extract the desired audio segment using the `getPureSample` function.
    pureSample = getPureSample(audioArr, start, end, sampleRate)
    
    # If the extracted segment is shorter than `maxLen`, pad it to the required length.
    reqLen = 6 * sampleRate  # Required length in samples for a 6-second segment.
    padded_data = lb.util.pad_center(pureSample, size=reqLen)   #ِAdd padding if the audio is less than required length
    
    # Save the padded audio segment to the specified path.
    sf.write(file=save_path, data=padded_data, samplerate=sampleRate)

# Print the total number of files processed.
print('Total Files Processed: ', c)
'''

diagnosis=pd.read_csv('data/patient_diagnosis.csv',names=['pid','disease'])
print(diagnosis.head())
#In the following plot we can see that 
# classes are imbalanced so we must split them into 
# train and validation set via stratify


'''
sns.barplot(x = diagnosis["disease"].value_counts().index,y=diagnosis["disease"].value_counts().values)
plt.xticks(rotation=90);
plt.title("Classes Vs No. of patients")
plt.show()'''

#We will try to extract Id of each processed audio file and 
# then merge them with their respective class label so we 
# can split files in to train and validation folder in stratified manner

def extractId(filename):
    return filename.split('_')[0]

path='processed_audio_files/'
length = len(os.listdir(path))
index = range(length)

files_df = pd.DataFrame(index=index, columns=['pid', 'filename'])
for i, f in enumerate(os.listdir(path)):
    files_df.loc[i, 'pid'] = extractId(f)
    files_df.loc[i, 'filename'] = f

#print(files_df.tail())

files_df.pid=files_df.pid.astype('int64') # both pid's must be of same dtype for them to merge
#print(files_df.info())

data=pd.merge(files_df,diagnosis,on='pid')
print(data.head())
data.to_csv('csv_data/data1.csv')
'''
sns.barplot(x = data["disease"].value_counts().index,y=data["disease"].value_counts().values)
plt.xticks(rotation=90)
plt.title("Classes Vs No. of Audios")
plt.show()'''
#classes are skewed
from sklearn.model_selection import train_test_split
Xtrain,Xval,ytrain,yval=train_test_split(data,data.disease,stratify=data.disease,random_state=42,test_size=0.25)

#Above i used the stratify arg of train_test_split and 
# set it to disease to stratify data based on class labels

print(Xtrain.disease.value_counts()/Xtrain.shape[0])
print(Xval.disease.value_counts()/Xval.shape[0])

#percentage of class labels is same in both train and val as we can see above

# We did this because this will help our model to learn and validate classes ,
# it will not be like we are training only on COPD disease and 
# there is no COPD in our validation
