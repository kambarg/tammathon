import numpy as np
from collections import defaultdict
from itertools import combinations
import math
import pandas as pd
from tqdm import tqdm

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def build_embeddings(df, folder_path, get_embedding):
    embeddings_by_id = defaultdict(list)
    for _, row in tqdm(df.iterrows(), total=len(df)):
        image_path = f"{folder_path}/{row['filename']}"
        emb = get_embedding(image_path)
        embeddings_by_id[row['label']].append((row['filename'], emb))
    return embeddings_by_id

def clean_embeddings(embeddings_by_id):
    single_ids = {k: v for k, v in embeddings_by_id.items() if len(v) == 1}
    multi_ids = {k: v for k, v in embeddings_by_id.items() if len(v) > 1}

    mean_embs = [np.mean([emb for _, emb in v], axis=0) for v in multi_ids.values()]
    singles_for_inference = []
    for label, [(img_id, emb)] in single_ids.items():
        if any(cosine_sim(emb, mean) > 0.95 for mean in mean_embs):
            continue
        singles_for_inference.append((img_id, label))

    clean_2images, clean_3to7, clean_8plus = {}, {}, {}
    for label, items in multi_ids.items():
        n = len(items)
        if n == 2:
            sim = cosine_sim(items[0][1], items[1][1])
            if sim > 0.8:
                clean_2images[label] = items
        elif 3 <= n <= 7:
            for group in combinations(items, math.ceil(n / 2)):
                embs = [emb for _, emb in group]
                sims = [cosine_sim(a, b) for i, a in enumerate(embs)
                        for j, b in enumerate(embs) if i < j]
                if len(sims) >= 3 and np.mean(sims) > 0.4:
                    clean_3to7[label] = list(group)
                    break
        elif n >= 8:
            embs = [emb for _, emb in items]
            sims = [cosine_sim(a, b) for i, a in enumerate(embs)
                    for j, b in enumerate(embs) if i < j]
            mean_sim = np.mean(sims)
            std_sim = np.std(sims)
            if mean_sim > 0.4 and std_sim < 0.2:
                clean_8plus[label] = items
            else:
                for group in combinations(items, math.ceil(n / 2)):
                    embs = [emb for _, emb in group]
                    sims = [cosine_sim(a, b) for i, a in enumerate(embs)
                            for j, b in enumerate(embs) if i < j]
                    if len(sims) >= 3 and np.mean(sims) > 0.4:
                        clean_8plus[label] = list(group)
                        break

    return clean_2images, clean_3to7, clean_8plus, singles_for_inference

def save_cleaned_csv(clean_2images, clean_3to7, clean_8plus, singles_for_inference, clean_path, single_path):
    clean_images = []
    for clean_group in [clean_2images, clean_3to7, clean_8plus]:
        for label, group in clean_group.items():
            for img_id, _ in group:
                clean_images.append((img_id, label))

    train_clean_df = pd.DataFrame(clean_images, columns=["image_id", "label"])
    singles_df = pd.DataFrame(singles_for_inference, columns=["image_id", "label"])

    train_clean_df.to_csv(clean_path, index=False)
    singles_df.to_csv(single_path, index=False)