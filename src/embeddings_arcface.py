import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from insightface.app import FaceAnalysis

# Initialize InsightFace app (ArcFace-based)
face_app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(224, 224))

# Preprocessing (InsightFace handles resizing internally)
def get_embedding(img_path):
    image = Image.open(img_path).convert("RGB")
    image_np = np.array(image)
    faces = face_app.get(image_np)
    if not faces:
        raise ValueError(f"No face found in image: {img_path}")
    emb = faces[0].embedding
    return emb / np.linalg.norm(emb)  # Normalize