# Import necessary libraries
import os  # For file and directory operations
import numpy as np  # For numerical operations
import tensorflow as tf  # Main library for machine learning
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # For data augmentation and preprocessing
from tensorflow.keras.applications import MobileNet  # Pre-trained model architecture
from tensorflow.keras.models import Sequential  # For creating the model
from tensorflow.keras.layers import Flatten, Dense, Dropout  # Layers for the model
import matplotlib.pyplot as plt  # For plotting graphs
from sklearn.metrics import confusion_matrix, classification_report  # For model evaluation

# Set the path to your dataset
dataset_path = r'C:\Users\HP\OneDrive\Desktop\RICE DATASET\1.Rice_Dataset_Original'

# Set the image dimensions (adjust as needed)
img_width, img_height = 150, 150  # Width and height of input images

# Set the number of classes (change this based on your dataset)
num_classes = 9  # Number of different rice disease classes

# Set other hyperparameters
batch_size = 32  # Number of samples per gradient update
epochs = 1  # Number of times to iterate over the entire dataset (increase for better training)

# Data augmentation to increase the diversity of training examples
datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Normalize pixel values to [0,1]
    shear_range=0.2,  # Shear angle in counter-clockwise direction
    zoom_range=0.2,  # Range for random zoom
    horizontal_flip=True,  # Randomly flip inputs horizontally
    validation_split=0.2  # 20% of the data will be used for validation
)

# Load and augment the training data
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_width, img_height),  # Resize images to this size
    batch_size=batch_size,
    class_mode='categorical',  # Multi-class labels
    subset='training'  # Specify this is the training set
)

# Get the class indices and corresponding class labels
class_indices = train_generator.class_indices  # Dictionary mapping class names to class indices
class_labels = list(class_indices.keys())  # List of class names

# Define the class names explicitly
class_names = [
    "Hispa",
    "bacterial_leaf_blight",
    "leaf_blast",
    "Brown_spot",
    "Healthy",
    "Shath_Blight",
    "leaf_scald",
    "narrow_brown_spot",
    "Tungro"
]

# Function to build the MobileNet model
def build_mobilenet_model():
    # Load the pre-trained MobileNet model without the top layers
    base_model = MobileNet(input_shape=(img_width, img_height, 3), include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze the pre-trained weights

    # Create a new model
    model = Sequential()
    model.add(base_model)  # Add the base model
    model.add(Flatten())  # Flatten the output
    model.add(Dense(128, activation='relu'))  # Add a dense layer with 128 units and ReLU activation
    model.add(Dropout(0.3))  # Add dropout for regularization
    model.add(Dense(num_classes, activation='softmax'))  # Output layer with softmax activation

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Build and train the model
best_model_mobilenet = build_mobilenet_model()

# Train the model
history_mobilenet = best_model_mobilenet.fit(
    train_generator,
    epochs=epochs,
    validation_data=train_generator,  # Use the same generator for validation (not ideal, but works for this example)
    validation_steps=train_generator.samples // batch_size  # Number of validation steps
)

# Make predictions on a sample image
sample_image_path = r'C:\Users\HP\OneDrive\Desktop\EXAMPLE.jpg'
sample_image = tf.keras.preprocessing.image.load_img(sample_image_path, target_size=(img_width, img_height))
sample_image = tf.keras.preprocessing.image.img_to_array(sample_image)
sample_image = np.expand_dims(sample_image, axis=0)  # Add batch dimension
sample_image = sample_image / 255.0  # Rescale the image
predictions = best_model_mobilenet.predict(sample_image)
predicted_class_index = np.argmax(predictions)  # Get the index of the highest probability

# Get the predicted class label
predicted_class_label = class_names[predicted_class_index]
print(f'Predicted class label using MobileNet: {predicted_class_label}')

# Create a generator for the validation data
validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'  # Specify this is the validation set
)

# Get true labels and predictions for the validation set
y_true = validation_generator.classes  # True labels
y_pred = best_model_mobilenet.predict(validation_generator)  # Predictions
y_pred = np.argmax(y_pred, axis=1)  # Convert from one-hot to class indices

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Display the confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix (MobileNet)')
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# Add text annotations to the confusion matrix
for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(j, i, str(cm[i][j]), horizontalalignment='center', verticalalignment='center')

plt.show()

# Print Classification Report
print("Classification Report (MobileNet):")
print(classification_report(y_true, y_pred, target_names=class_names))

# Plot accuracy during training for MobileNet
plt.figure()
plt.plot(history_mobilenet.history['accuracy'], label='Training Accuracy (MobileNet)')
plt.plot(history_mobilenet.history['val_accuracy'], label='Validation Accuracy (MobileNet)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy during Training (MobileNet)')

plt.show()
