from src.data_loader import load_data
from src.embeddings_dinov2 import get_embedding # choose embedding function: resnet, arcface, dinov2
from src.cleaning import build_embeddings, clean_embeddings, save_cleaned_csv

# Load dataset
df = load_data("snippet.csv")

# Build embeddings
embeddings_by_id = build_embeddings(df, folder_path="snippet", get_embedding=get_embedding)

# Clean
clean_2, clean_3to7, clean_8plus, singles = clean_embeddings(embeddings_by_id)

# Save
save_cleaned_csv(clean_2, clean_3to7, clean_8plus, singles, "snippet_clean.csv", "singles_for_inference.csv")