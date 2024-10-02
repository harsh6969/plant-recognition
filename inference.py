import argparse
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("Model.h5")

# Define class names
class_names = ['Aloevera', 'Amaranthus Viridis', 'Amruthabali', 'Arali', 'Castor', 'Mango', 'Mint', 'Neem', 'Sandalwood', 'Turmeric']

# Function to preprocess the image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0  # Normalize
    img = img.reshape(1, 224, 224, 3)  # Reshape for the model
    return img

# Main function for inference
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plant Recognition Inference')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the image file')
    args = parser.parse_args()

    # Preprocess the image
    img = preprocess_image(args.image_path)

    # Make predictions
    predictions = model.predict(img)
    predicted_class = class_names[np.argmax(predictions)]
    print(f'The predicted class is: {predicted_class}')
