PDF_DIR = "data/pdfs"
CHUNK_SIZE = 800  # Increased for better heading-aware chunks
JSON_CHUNKS = "data/pdf_chunks.json"
NPZ_EMBED = "data/pdf_chunk_embeddings.npy.npz"
MODEL_PATH = "multi-qa-MiniLM-L6-cos-v1"
TOP_K = 5
MAX_CHUNK_LENGTH = 1200  # Increased for complete sections
CHUNK_OVERLAP = 100  # Reduced since we use heading boundaries
HASH_CACHE_PATH = "data/chunk_hashes.json"

# New heading-aware settings
MIN_HEADING_CONFIDENCE = 0.4
PRESERVE_HEADING_HIERARCHY = True
INCLUDE_HEADING_CONTEXT = True
MAX_CHUNK_WITHOUT_HEADING = 600