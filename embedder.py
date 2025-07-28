import os
import numpy as np
import json
import hashlib
from sentence_transformers import SentenceTransformer
from config import NPZ_EMBED, HASH_CACHE_PATH, MODEL_PATH

def hash_chunk_enhanced(doc):
    """Enhanced hash that includes heading metadata"""
    text = doc.get("chunk", "")
    heading_info = ""
    
    if doc.get('has_heading'):
        heading_info = f"{doc.get('heading_text', '')}_{doc.get('heading_level', '')}"
    
    combined = f"{text}_{heading_info}"
    return hashlib.md5(combined.encode("utf-8")).hexdigest()

def get_embeddings(docs, force=False):
    """Enhanced embedding generation with heading-aware caching"""
    chunk_hashes = [hash_chunk_enhanced(doc) for doc in docs]

    if not force and os.path.exists(NPZ_EMBED) and os.path.exists(HASH_CACHE_PATH):
        try:
            saved = np.load(NPZ_EMBED, allow_pickle=True)
            with open(HASH_CACHE_PATH, "r", encoding="utf-8") as f:
                cached_hashes = json.load(f)
                if cached_hashes == chunk_hashes:
                    print("Using cached embeddings...")
                    embeddings = saved["embeddings"]
                    for i, doc in enumerate(docs):
                        doc["embedding"] = embeddings[i].astype(float).tolist()
                    return docs
        except Exception as e:
            print(f"Error loading cached embeddings: {e}")
            pass

    print("Generating new embeddings...")
    model = SentenceTransformer(MODEL_PATH)
    
    # Prepare text for embedding - include heading context
    chunks_for_embedding = []
    for doc in docs:
        text = doc.get("chunk", "")
        
        # If chunk has heading, give it more weight in embedding
        if doc.get('has_heading') and doc.get('heading_text'):
            heading = doc['heading_text']
            # Repeat heading to give it more weight in the embedding
            enhanced_text = f"{heading}. {heading}. {text}"
        else:
            enhanced_text = text
        
        chunks_for_embedding.append(enhanced_text)
    
    batch_size = 100
    all_embeddings = []
    
    print(f"Processing {len(chunks_for_embedding)} chunks in batches of {batch_size}")
    
    for i in range(0, len(chunks_for_embedding), batch_size):
        batch = chunks_for_embedding[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(chunks_for_embedding) + batch_size - 1)//batch_size}")
        
        batch_embeddings = model.encode(batch, normalize_embeddings=True)
        all_embeddings.extend(batch_embeddings)
    
    embeddings = np.array(all_embeddings)

    # Add embeddings to documents
    for i, doc in enumerate(docs):
        doc["embedding"] = embeddings[i].astype(float).tolist()

    # Save embeddings and hashes
    os.makedirs(os.path.dirname(NPZ_EMBED), exist_ok=True)
    np.savez(NPZ_EMBED, embeddings=embeddings)
    
    with open(HASH_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(chunk_hashes, f)

    print(f"Generated and cached embeddings for {len(docs)} documents")
    return docs