import matplotlib.pyplot as plt
import seaborn as sns
import librosa.display
import pandas as pd
import os
import numpy as np
data=pd.read_csv('csv_data/data.csv')
diagnosis=pd.read_csv('data/patient_diagnosis.csv',names=['pid','disease'])

#In the following plot we can see that 
# classes are imbalanced so we must split them into 
# train and validation set via stratify
'''
sns.barplot(x = diagnosis["disease"].value_counts().index,y=diagnosis["disease"].value_counts().values)
plt.xticks(rotation=90)
plt.title("Classes Vs No. of patients")
plt.show()
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
'''


df = pd.read_csv('csv_data/data.csv')
audio_dir='data/'
extension='.wav'
# Define a function to plot spectrogram

# Function to plot the spectrogram
def plot_spectrogram(audio_file, disease, ax):
    audio_path = os.path.join(audio_dir, audio_file)
    print(f"Trying to load file: {audio_path}")  # Debugging line

    if not os.path.exists(audio_path):
        print(f"File not found: {audio_path}")
        return  # Skip this file if it's missing

    # Load audio file
    y, sr = librosa.load(audio_path)

    # Create a spectrogram (Short-time Fourier Transform)
    S = np.abs(librosa.stft(y))  # Compute the magnitude
    D = librosa.amplitude_to_db(S, ref=np.max)  # Convert to dB scale

    # Display the spectrogram
    img = librosa.display.specshow(D, x_axis='time', y_axis='log', sr=sr, ax=ax)

    # Add labels and title
    ax.set_title(f'Spectrogram of {audio_file} ({disease})')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    
    return img  # Return the image for colorbar handling


# Path to the audio files and disease labels
audio_dir = 'data/'
disease_labels = df['disease'].unique()

# Create subplots
fig, axes = plt.subplots(len(disease_labels), 1, figsize=(12, 8))

if len(disease_labels) == 1:
    axes = [axes]  # Make it iterable if only one subplot

# Plot each spectrogram
for idx, disease in enumerate(disease_labels):
    disease_data = df[df['disease'] == disease]
    
    for _, row in disease_data.iterrows():
        audio_file = row['filename']
        img = plot_spectrogram(audio_file, disease, axes[idx])

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(hspace=0.5)  # Adjust spacing between subplots

# Add a single colorbar for the entire figure
cbar = fig.colorbar(img, ax=axes, orientation='horizontal', format='%+2.0f dB', pad=0.1)
cbar.set_label('Amplitude (dB)')

# Show the plot
plt.show()
# Assuming your dataframe has columns 'crackles', 'wheezes', and 'Disease'
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
df = pd.read_csv('csv_data/data.csv')

# Set plot style
sns.set_theme(style="whitegrid")

# Plot histograms for 'crackles' and 'wheezes' for each disease type
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot Histogram for Crackles
sns.histplot(data=df, x='crackles', hue='disease', kde=True, multiple="stack", ax=axes[0])
axes[0].set_title('Distribution of Crackles for Different Diseases')
axes[0].set_xlabel('Crackles')
axes[0].set_ylabel('Frequency')

# Plot Histogram for Wheezes
sns.histplot(data=df, x='wheezes', hue='disease', kde=True, multiple="stack", ax=axes[1])
axes[1].set_title('Distribution of Wheezes for Different Diseases')
axes[1].set_xlabel('Wheezes')
axes[1].set_ylabel('Frequency')

# Adjust layout
plt.tight_layout()
plt.show()

# Alternatively, Boxplots can also be plotted:
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Boxplot for Crackles
sns.boxplot(x='disease', y='crackles', data=df, ax=axes[0])
axes[0].set_title('Boxplot of Crackles for Different Diseases')
axes[0].set_xlabel('Disease')
axes[0].set_ylabel('Crackles')

# Boxplot for Wheezes
sns.boxplot(x='disease', y='wheezes', data=df, ax=axes[1])
axes[1].set_title('Boxplot of Wheezes for Different Diseases')
axes[1].set_xlabel('Disease')
axes[1].set_ylabel('Wheezes')

# Adjust layout
plt.tight_layout()
plt.show()
