# Assigment
This project implements a waste classification system using PyTorch and transfer learning with ResNet18.
The trained model is exported to ONNX format and deployed through a Streamlit web app for real-time inference.

#Dataset

We use the TrashNet Dataset
 which contains images of:

cardboard

glass

metal

paper

plastic

trash

Download the dataset from https://github.com/garythung/trashnet

#Installation
Install the required dependencies:
pip install torch torchvision matplotlib scikit-learn streamlit onnxruntime pillow

# Training the Model
Update dataset path inside the training script:
data_dir = "/content/drive/MyDrive/dataset-resized"


Run the training code (train.py or notebook).
The script:
Splits dataset into 80% training / 20% validation
Loads ResNet18 (pretrained on ImageNet) and replaces the final layer
Trains with Adam optimizer and CrossEntropyLoss
Evaluates with Accuracy, Precision, Recall, and Confusion Matrix
Exports the trained model to ONNX

Example output:

Epoch 10/10 - Train Loss: 0.2450, Val Loss: 0.3201, Val Acc: 0.8912
Best Validation Accuracy: 0.8912
Final Accuracy: 0.8900
Precision: 0.8820
Recall: 0.8835

# Model Export (ONNX)
After training, the model is exported:
onnx_file = "trained_model.onnx"


This ONNX file will be used by the Streamlit app.

#Running the Streamlit App

Ensure trained_model.onnx is in your project folder.
Run the Streamlit app:
streamlit run app.py


Upload an image (JPG/PNG).
The app will display:
The uploaded image
Predicted class name
Class probabilities

# Example Output (Streamlit)
Prediction: paper (index 3)

Probabilities:
cardboard: 0.0512
glass:     0.1023
metal:     0.0801
paper:     0.7214
plastic:   0.0345
trash:     0.0105

# Project Structure
.
├── train.py              # Training + evaluation + ONNX export
├── app.py                # Streamlit inference app
├── trained_model.onnx    # Exported ONNX model (after training)
├── dataset-resized/      # Dataset folder (TrashNet images)
└── README.md             # Project documentation

 Future Work
Fine-tune the entire ResNet18 instead of only classifier head
Try deeper architectures (ResNet50, EfficientNet)
Deploy app on HuggingFace Spaces / Docker for easy access
