import streamlit as st
import numpy as np
import librosa
import os
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from ftExtraction import InstantiateAttributes  # Import your feature extraction module
import csv
import matplotlib.pyplot as plt

class ModelWrapper:
    def __init__(self, model):
        self.model = model

    def predict(self, inputs):
        # Flatten inputs to ensure compatibility
        
        inputs = inputs.reshape((-1, 1, 40))
        st.write(f"input shape: {inputs.shape}")
        return self.model.predict(inputs)


# Load pre-trained model
@st.cache_resource
def load_trained_model():
    return load_model("best_model_22.keras")

# Function to preprocess uploaded audio
def preprocess_audio(file):
    # Convert uploaded file to audio waveform
    data_x, sampling_rate = librosa.load(file,res_type='kaiser_fast')
    # Extract features
    mfccs = np.mean(librosa.feature.mfcc(y=data_x, sr=sampling_rate, n_mfcc=40).T, axis=0)
    return np.expand_dims(np.expand_dims(mfccs, axis=0), axis=1)  # Reshape to match model input

# Function to predict disease
def predict_disease(model, features):
    prediction = model.predict(features)
    return np.argmax(prediction), prediction

# Load the class mapping from CSV file
def load_class_mapping(file_path):
    class_mapping = {}
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header row
        for row in reader:
            class_mapping[int(row[0])] = row[1]  # map index to label
    return class_mapping

# Load the mapping
class_mapping = load_class_mapping('class_mapping.csv')
# Map class indices to disease labels
#DISEASE_CLASSES = ["COPD", "Healthy", "Pneumonia", "Asthma", "Bronchitis", "Other"]

def explain_prediction(model, features):
    # Reshape features to match the expected input
    reshaped_features = features.reshape((-1, *features.shape[2:]))
    st.write(f"Reshaped feature shape: {reshaped_features.shape}")

    # Create a model wrapper to pass to SHAP's KernelExplainer
    model_wrapper = ModelWrapper(model)
    
    # Use KernelExplainer to compute SHAP values
    explainer = shap.KernelExplainer(model_wrapper.predict, reshaped_features)
    shap_values = explainer.shap_values(reshaped_features)
    if isinstance(shap_values, list):
        st.write("Detected multiple SHAP values, using the first class for explanation.")
        shap_values_instance = shap_values[0]  # Use the SHAP values for the first class
    else:
        st.write("Detected single SHAP value object.")
        shap_values_instance = shap_values  # In case there's only one class output
    if isinstance(shap_values_instance, np.ndarray):
        # Reshape if needed: SHAP requires (num_features,) shape for the waterfall plot
        shap_values_instance = shap_values_instance.flatten()
    base_values = np.zeros_like(shap_values_instance) 
    st.write("base value shape", base_values) # For bina
    shap_values_instance = shap.Explanation(values=shap_values_instance,
                                             base_values=base_values,
                                             data=reshaped_features)
    # Convert shap_values_instance to the correct format if needed

    return shap_values_instance



def display_explanation(shap_values, features):
    st.subheader("Prediction Explanation")

    # Visualize SHAP values using a waterfall plot for the first prediction
    st.write("Waterfall plot for feature contributions:")
    st.write(f"shape shape;{shap_values.shape}")
    
    # Pass the explanation object directly to the waterfall plot
    shap.waterfall_plot(shap_values)
    st.pyplot()  # Render the plot in Streamlit

    # Summary plot for global importance
    features = features.reshape((-1, *features.shape[2:]))
    # Summary plot for global importance
    st.write("Summary plot for feature contributions:")
    shap.summary_plot(shap_values, features, show=False)
    st.pyplot() 

def show_prediction_details(predictions, class_mapping):
     # Flatten the 3D predictions to get the 1D array of probabilities
    predictions_flat = predictions[0, 0, :]  # Extract the probabilities from the shape (1, 1, 6)

    # Prepare data for the bar chart
    labels = [class_mapping[i] for i in range(len(predictions_flat))]
    probabilities = predictions_flat

    # Display the bar chart for prediction probabilities
    st.write("Prediction Probabilities:")
    fig1, ax1 = plt.subplots(figsize=(10, 5)) 
    ax1.barh(labels, probabilities, color='skyblue')
    ax1.set_xlabel("Probability")
    ax1.set_title("Prediction Probabilities for Each Disease")
    st.pyplot(fig1)   # Render the plot in Streamlit

    # Display the probabilities in a table with the predicted class highlighted
    
    st.markdown("---")
    st.subheader("Resources for Further Reading")
    predicted_class = class_mapping[np.argmax(predictions)]
    resources = {
        "Bronchiectasis": "https://www.mayoclinic.org/diseases-conditions/bronchiectasis/symptoms-causes/syc-20355339",
        "Bronchiolitis": "https://www.mayoclinic.org/diseases-conditions/bronchiolitis/symptoms-causes/syc-20351565",
        "COPD": "https://www.lung.org/lung-health-diseases/lung-disease-lookup/copd",
        "Healthy": "https://www.cdc.gov/physicalactivity/basics/index.htm",
        "Pneumonia": "https://www.mayoclinic.org/diseases-conditions/pneumonia/symptoms-causes/syc-20354204",
        "URTI": "https://www.cdc.gov/groupastrep/diseases-public/upper-respiratory-infections.html"
    }
    
    # Display the resource link for the predicted class
    if predicted_class in resources:
        st.markdown(f"[Learn more about {predicted_class}]({resources[predicted_class]})")
    else:
        st.write("No additional resources available.")
    
# Streamlit app
def main():
    st.title("Pulmonary Disorder Classifier Leveraging Sequential Deep Learning")
    st.image(
        "images.jpg",  # Replace with the path to your image
        caption="Analyze audio files to predict respiratory conditions",
        use_column_width=True
    )
    st.write("""
        Upload an audio file (e.g., MP3, WAV etc.) to predict the associated respiratory disease.
    """)

    # File uploader
    uploaded_file = st.file_uploader("Upload your audio file", type=["wav", "mp3"])
    
    if uploaded_file:
        st.audio(uploaded_file, format="audio/wav")

        with st.spinner("Processing the audio file..."):
            # Convert audio to WAV if needed
            file_path = uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Preprocess the audio
            features = preprocess_audio(file_path)

            # Load model and predict
            model = load_trained_model()
            predicted_class, prediction_probabilities = predict_disease(model, features)
            predicted_disease = class_mapping.get(predicted_class, 'Unknown')  # Use 'Unknown' if the class index is not found
            # Display results
              # Print the type to understand if it's an array or scalar

            st.success(f"The model predicts: **{predicted_disease}**")
            #st.success(f"the model class:**{predicted_class}**")
            #st.write(f"Features shape: {features.shape}")
            #st.write(f"Raw prediction probabilities: {prediction_probabilities}")
            
            #shap=explain_prediction(model, features)
            #display_explanation(shap, features)
            #st.write("Prediction Probabilities:")
            show_prediction_details(prediction_probabilities,class_mapping)
            
if __name__ == '__main__':
    main()
