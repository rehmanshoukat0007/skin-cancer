import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from pathlib import Path

# Keep this order identical to the alphabetical ImageFolder class order
# used while training the uploaded model.
CLASS_NAMES = ["Herpes", "Melanoma", "Monkeypox", "Sarampion", "Varicela"]
MODEL_PATH = Path(__file__).parent / "custom_skin_cnn.pth"
IMAGE_SIZE = (224, 224)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class CustomCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 14 * 14, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "custom_skin_cnn.pth nahi mili. Is file ko app.py ke sath "
            "GitHub repository ke root folder mein upload karein."
        )
    model = CustomCNN(num_classes=len(CLASS_NAMES))
    state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def preprocess_image(image):
    image = image.convert("RGB").resize(IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - MEAN) / STD
    tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)
    return tensor


st.set_page_config(
    page_title="Skin Condition Image Classifier",
    page_icon="🔬",
    layout="centered",
)

st.title("Skin Condition Image Classifier")
st.write(
    "Upload a skin image to see the class predicted by the trained CNN model."
)
st.warning(
    "This is an educational machine-learning demonstration, not a medical "
    "diagnosis. Image predictions can be wrong. Please consult a qualified "
    "healthcare professional for medical advice."
)

uploaded_file = st.file_uploader(
    "Choose an image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

        with st.spinner("Analyzing image..."):
            model = load_model()
            input_tensor = preprocess_image(image)
            with torch.inference_mode():
                probabilities = torch.softmax(model(input_tensor), dim=1)[0]
            confidence, predicted_index = torch.max(probabilities, dim=0)

        st.subheader("Model prediction")
        st.write(f"**Predicted class:** {CLASS_NAMES[predicted_index.item()]}")
        st.write(f"**Model confidence:** {confidence.item() * 100:.2f}%")

        st.subheader("Class scores")
        for name, score in zip(CLASS_NAMES, probabilities.tolist()):
            st.write(f"{name}: {score * 100:.2f}%")
            st.progress(float(score))

    except FileNotFoundError as error:
        st.error(str(error))
    except Exception as error:
        st.error(f"Could not process this image: {error}")
