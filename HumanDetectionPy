import cv2
from skimage import exposure
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import os
import numpy as np
import matplotlib.pyplot as plt

# Function to extract HOG features from an image
def extract_hog_features(image):
    resized_image = cv2.resize(image, (64, 128))  # resize
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    hog = cv2.HOGDescriptor()
    features = hog.compute(gray)
    features = exposure.rescale_intensity(features, in_range=(0, 10))
    return features.flatten()

# Function to load and process images in batches
def process_images_in_batches(folder_path, label, batch_size):
    data = []
    labels = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            print(f"Loading image: {image_path}")
            image = cv2.imread(image_path)
            hog_features = extract_hog_features(image)
            data.append(hog_features)
            labels.append(label)
            
            if len(data) >= batch_size:
                yield data, labels
                data = []
                labels = []
    # remaining images
    if data:
        yield data, labels

# Function to display image with prediction result
def display_result(image_path, prediction):
    print(f"Loading image: {image_path}")

    # Try displaying the image with matplotlib as an alternative
    img = plt.imread(image_path)
    plt.imshow(img)
    plt.title(f"Prediction: {'Person' if prediction == 1 else 'Not a Person'}")
    plt.show()

# Path to folders containing images with people, validation set, and test set
positive_folder_learning = "D:\\TIA-Python3\\Proiect_Human_Det\\Dataset\\learn\\Human"
negative_folder_learning = "D:\\TIA-Python3\\Proiect_Human_Det\\Dataset\\learn\\Not_Human"

positive_folder_test = "D:\\TIA-Python3\\Proiect_Human_Det\\Dataset\\test\\Human"
negative_folder_test = "D:\\TIA-Python3\\Proiect_Human_Det\\Dataset\\test\\Not_Human"

# Batch size for processing images
batch_size = 50  # You can adjust this based on your memory constraints

# Load positive and negative samples for learning using batch processing
data_learning = []
labels_learning = []
for batch_data, batch_labels in process_images_in_batches(positive_folder_learning, 1, batch_size):
    data_learning.extend(batch_data)
    labels_learning.extend(batch_labels)

for batch_data, batch_labels in process_images_in_batches(negative_folder_learning, 0, batch_size):
    data_learning.extend(batch_data)
    labels_learning.extend(batch_labels)

# Split data into training and validation sets
X_train, X_verify, y_train, y_verify = train_test_split(data_learning, labels_learning, test_size=0.2, random_state=42)

# Load positive and negative samples for test using batch processing
data_test = []
labels_test = []
for batch_data, batch_labels in process_images_in_batches(positive_folder_test, 1, batch_size):
    data_test.extend(batch_data)
    labels_test.extend(batch_labels)

for batch_data, batch_labels in process_images_in_batches(negative_folder_test, 0, batch_size):
    data_test.extend(batch_data)
    labels_test.extend(batch_labels)

# Train a Support Vector Machine (SVM) classifier on the learning set
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

y_verify_pred = clf.predict(X_verify)

accuracy_verify = accuracy_score(y_verify, y_verify_pred)
print(f"Validation Accuracy: {accuracy_verify}")

y_test_pred = clf.predict(data_test)

accuracy_test = accuracy_score(labels_test, y_test_pred)
print(f"Test Accuracy: {accuracy_test}")

# Display test images with predictions
positive_files = sorted(os.listdir(positive_folder_test))
negative_files = sorted(os.listdir(negative_folder_test))

# Display test images with predictions
for filename, prediction in zip(positive_files + negative_files, y_test_pred):
    if filename in positive_files:
        image_path = os.path.join(positive_folder_test, filename)
    else:
        image_path = os.path.join(negative_folder_test, filename)
    
    display_result(image_path, prediction)
