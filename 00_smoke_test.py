import torch
from sentence_transformers import SentenceTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# loads (downloads once) a small English-centric model
model = SentenceTransformer("intfloat/e5-base-v2", device=device)

emb = model.encode(["What causes diabetes?"])
print("Embedding shape:", emb.shape)   # expect something like (1, 768)
print("Smoke test passed.")