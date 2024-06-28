# from .SFR import SFR
# from .NVEmbed import NVEmbed
import torch
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

SUPPORTED_MODELS = [
    "sentence-transformers/gtr-t5-base",
]

from v2t.overwatch import initialize_overwatch
overwatch = initialize_overwatch(__name__)

def load_embedder(
    model_name_or_path: str,
):
    overwatch.info(f"Loading Retriever from: {model_name_or_path}")
    if model_name_or_path == "sentence-transformers/gtr-t5-base":
        embedder = SentenceTransformer(model_name_or_path,device='cpu')
        tokenizer = embedder.tokenizer
    return embedder