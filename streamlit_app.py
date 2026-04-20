import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import timm
import os
import gdown   # ✅ NEW

# ================= CONFIG =================
IMAGE_SIZE = 224
CLASSES = ["Healthy", "Mosaic", "RedRot", "Rust", "Yellow"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "best_vit_cnn.pth"
FILE_ID = "1HtGwAPqvKtB3Aa-e_xFnIVB8_ep814QB"   # ✅ your drive id

# ================= DOWNLOAD MODEL =================
def download_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

# ================= MODEL =================

class CNNBranch(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet50(weights=None)
        self.backbone = nn.Sequential(*list(base.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x


class ViTBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=False,
            num_classes=0
        )

    def forward(self, x):
        return self.vit(x)


class HybridViTCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        self.vit = ViTBranch()
        self.cnn = CNNBranch()

        self.vit_proj = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512)
        )

        self.cnn_proj = nn.Sequential(
            nn.Linear(2048, 512),
            nn.LayerNorm(512)
        )

        self.head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        v = self.vit(x)
        c = self.cnn(x)

        v = F.gelu(self.vit_proj(v))
        c = F.gelu(self.cnn_proj(c))

        fused = torch.cat([v, c], dim=1)
        return self.head(fused)

# ================= LOAD MODEL =================

@st.cache_resource
def load_model():
    download_model()   # ✅ MUST

    model = HybridViTCNN(num_classes=len(CLASSES))
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()
    return model


model = load_model()

# ================= TRANSFORM =================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])

# ================= UI =================

st.title("🌿 Sugarcane Disease Classifier (ViT + CNN)")
st.write("Upload a leaf image to classify disease")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()

    pred_class = CLASSES[pred_idx]
    confidence = probs[pred_idx].item() * 100

    st.success(f"Prediction: {pred_class}")
    st.info(f"Confidence: {confidence:.2f}%")

    st.subheader("Class Probabilities")
    for i, cls in enumerate(CLASSES):
        st.write(f"{cls}: {probs[i].item()*100:.2f}%")
