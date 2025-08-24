

import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort
import torch
import torchvision.transforms as transforms

# ---------------- Load ONNX Model ----------------
session = ort.InferenceSession("trained_model.onnx", providers=["CPUExecutionProvider"])

# ---------------- Class Labels ----------------
# Make sure these are in the same order as during training
class_names = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "metal",
    "trash"
    # add more if needed
]

# ---------------- Image Preprocessing ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------------- Streamlit UI ----------------
st.title("🖼️ Image Classification with ONNX")
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = transform(image).unsqueeze(0).numpy()

    # Get input name for ONNX
    input_name = session.get_inputs()[0].name

    # Run inference
    outputs = session.run(None, {input_name: img})[0]

    # Get prediction index
    pred_idx = np.argmax(outputs, axis=1)[0]

    # Map to class name
    pred_class = class_names[pred_idx]

    # Show results
    st.subheader("🔮 Prediction Result")
    st.write(f"**Predicted Class:** {pred_class} (index {pred_idx})")

    # Optionally show raw probabilities
    probs = torch.nn.functional.softmax(torch.tensor(outputs[0]), dim=0).numpy()
    for i, p in enumerate(probs):
        st.write(f"{class_names[i]}: {p:.4f}")
