import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from transformers import AutoImageProcessor, Dinov2Model

# Load DINOv2 (ViT-large by default, you can change to vit-small, vit-base, etc.)
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model = Dinov2Model.from_pretrained("facebook/dinov2-base")
model.eval()

# Preprocessing
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=processor.image_mean, std=processor.image_std)
])

def get_embedding(img_path):
    image = Image.open(img_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)  # Shape: (1, 3, 224, 224)

    with torch.no_grad():
        outputs = model(tensor)
        last_hidden_state = outputs.last_hidden_state  # Shape: (1, seq_len, dim)
        cls_embedding = last_hidden_state[:, 0, :]  # Take [CLS] token

    embedding = cls_embedding.squeeze().numpy()
    return embedding / np.linalg.norm(embedding)  # Normalize