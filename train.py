from keras import backend as K
from model import InstantiateModel
from keras.models import Model
from keras.optimizers import Adamax
from keras.layers import Input
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
from keras.optimizers import Adam
from keras.layers import Dropout
# Split data

def trainModel(X, y):
    print(
    '''
        Training the Neural Network model against the data.
        Args: 
            X: Array of features to be trained.
            y: Array of Target attribute.

        Returns: Save Trained model weights.
        ''')
    K.clear_session()
    X = X.reshape(X.shape[0], 1, X.shape[1])  # Shape: (797, 1, 40)

    num_batch_size=X.shape[0]
    time_steps=X.shape[1]
    data_dim=X.shape[2]
    Input_Sample = Input(shape=(time_steps,data_dim))
    Output_ = InstantiateModel(Input_Sample)
    Model_Enhancer = Model(inputs=Input_Sample, outputs=Output_)

    # Optimizer and compile
    optimizer = Adam(learning_rate=0.001)
    Model_Enhancer.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Callbacks
    LR = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)  # Reduced patience to 5 epochs
    
    ES = EarlyStopping(monitor='val_loss', min_delta=0.5, patience=200, verbose=1, mode='auto', baseline=None,
                              restore_best_weights=True)
    MC = ModelCheckpoint('best_model_22.keras', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
    
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Reshape target labels to be 3D with shape (None, 1, 6)
 

# Initialize the LabelEncoder
    encoder = LabelEncoder()

# Fit the encoder and transform your string labels (e.g., 'Pneumonia', 'COPD', etc.) into integers
    y_train_encoded = encoder.fit_transform(y_train)
    y_test_encoded = encoder.transform(y_test)

# Now convert the integer labels to categorical (one-hot encoding)
    y_train = to_categorical(y_train_encoded, num_classes=6)
    y_test = to_categorical(y_test_encoded, num_classes=6)
    y_train = np.expand_dims(y_train, axis=1)
    y_test = np.expand_dims(y_test, axis=1)
    # Get the mapping of encoded values to original labels
    class_mapping = dict(enumerate(encoder.classes_))
    import csv

    # Save the mapping as a CSV file
    with open('class_mapping.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Class Index', 'Label'])
        for index, label in class_mapping.items():
            writer.writerow([index, label])
    from sklearn.utils import class_weight
        # Compute class weights
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_encoded), y=y_train_encoded)
    class_weights_dict = dict(enumerate(class_weights))

    #class_weights = class_weight.compute_sample_weight('balanced',
	#                                                 np.unique(y[:,0],axis=0),
	#                                                 y[:,0])

    ModelHistory = Model_Enhancer.fit(x_train, y_train, batch_size=32, epochs=1000,
                                  validation_data=(x_test, y_test),
                                  callbacks = [MC  ,ES],
                                  #class_weight=class_weights_dict,
                                  verbose=1)
    print("History keys:", ModelHistory.history.keys())

# Check if ModelCheckpoint is monitoring the correct metric
    if 'val_accuracy' in ModelHistory.history:
        print("Validation accuracy is being computed and monitored.")
    else:
        print("Validation accuracy not found. Check your metrics configuration.")

    # Check validation accuracy history
    val_acc_history = ModelHistory.history.get('val_accuracy', [])
    if val_acc_history:
        print("Validation accuracy history:", val_acc_history)
        print("Best validation accuracy:", max(val_acc_history))
    else:
        print("No validation accuracy history available.")

    print(ModelHistory)
    print(ModelHistory.history)
    return x_train, x_test, y_train, y_test, ModelHistory
    