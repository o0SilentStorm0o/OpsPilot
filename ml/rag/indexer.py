"""
OpsPilot - RAG Knowledge Base Indexer (LOCAL MODELS)
Creates FAISS vector index from IT incident knowledge base using local embeddings

Usage:
    # Download embeddings first: python ml/download_models.py
    # Then run indexer:
    python ml/rag/indexer.py --embeddings ./models/embeddings
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import pandas as pd
from datetime import datetime

# Configuration
DEFAULT_EMBEDDINGS_PATH = "../models/embeddings_rag"  # Updated to correct path
KNOWLEDGE_BASE_PATH = "./datasets/internal_docs.jsonl"
CSV_FALLBACK_PATH = "./datasets/incident_logs.csv"
INDEX_OUTPUT_DIR = "./outputs/faiss_index"
BATCH_SIZE = 32


def check_embeddings_model(embeddings_path: Path) -> bool:
    """Verify embeddings model exists locally."""
    if not embeddings_path.exists():
        print(f"❌ Embeddings not found: {embeddings_path}")
        print("   Download with: python ml/download_models.py")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="OpsPilot RAG Indexer (Local Models)")
    parser.add_argument(
        "--embeddings",
        type=str,
        default=DEFAULT_EMBEDDINGS_PATH,
        help="Path to local all-MiniLM-L6-v2 model"
    )
    parser.add_argument(
        "--knowledge-base",
        type=str,
        default=KNOWLEDGE_BASE_PATH,
        help="Path to JSONL knowledge base"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=INDEX_OUTPUT_DIR,
        help="Output directory for FAISS index"
    )
    args = parser.parse_args()
    
    embeddings_path = Path(args.embeddings)
    
    print("=" * 80)
    print("📚 OpsPilot RAG Indexer - Building Knowledge Base (Local Models)")
    print("=" * 80)
    print(f"Embeddings: {embeddings_path}")
    print(f"Knowledge base: {args.knowledge_base}")
    print(f"Output: {args.output}")
    
    # Verify embeddings exist
    if not check_embeddings_model(embeddings_path):
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load embedding model from LOCAL path
    print(f"\n🔧 Loading embedding model from {embeddings_path}...")
    model = SentenceTransformer(str(embeddings_path), device='cpu')  # Use CPU for indexing
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"   ✅ Model loaded")
    print(f"   Embedding dimension: {embedding_dim}")
    
    # Load knowledge base
    print(f"\n📖 Loading knowledge base...")
    documents = []
    
    # Try JSONL first (internal_docs.jsonl)
    if os.path.exists(args.knowledge_base):
        with open(args.knowledge_base, 'r', encoding='utf-8') as f:
            for line in f:
                documents.append(json.loads(line))
        print(f"   ✅ Loaded {len(documents)} documents from JSONL")
    else:
        # Fallback: Create from CSV
        print(f"   JSONL not found, creating from CSV: {CSV_FALLBACK_PATH}")
        df = pd.read_csv(CSV_FALLBACK_PATH)
        
        for idx, row in df.iterrows():
            doc = {
                "id": f"doc_{idx}",
                "text": f"{row['prompt']}\n\n{row['response']}",
                "category": row['category'],
                "severity": row['severity'],
                "metadata": {
                    "source": "training_dataset",
                    "created_at": datetime.now().isoformat()
                }
            }
            documents.append(doc)
        
        # Save as JSONL for future use
        jsonl_output = Path(args.output) / "incident_knowledge.jsonl"
        with open(jsonl_output, 'w', encoding='utf-8') as f:
            for doc in documents:
                f.write(json.dumps(doc) + '\n')
        
        print(f"   Created {len(documents)} documents from CSV")
        print(f"   Saved to: {jsonl_output}")
    
    # Extract texts for embedding
    texts = [doc.get("content", doc.get("text", "")) for doc in documents]
    print(f"\n🧮 Generating embeddings for {len(texts)} documents...")
    print(f"   Batch size: {BATCH_SIZE}")
    
    # CRITICAL: Generate embeddings with normalization for COSINE similarity
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # CRITICAL for cosine similarity!
    )
    
    print(f"   Embeddings shape: {embeddings.shape}")
    print(f"   Embeddings normalized: True (for cosine similarity)")
    assert embeddings.shape == (len(documents), embedding_dim)
    
    # Create FAISS index
    print(f"\n🗂️  Building FAISS index...")
    embeddings = embeddings.astype('float32')
    
    # CRITICAL: Use IndexFlatIP for COSINE similarity with normalized vectors
    # (Inner Product = Cosine similarity when vectors are L2-normalized)
    # Previously used IndexFlatL2 which is Euclidean distance (incorrect for cosine embeddings)
    index = faiss.IndexFlatIP(embedding_dim)  # IP = Inner Product (cosine after normalization)
    index.add(embeddings)
    
    print(f"   Index type: IndexFlatIP (cosine similarity)")
    print(f"   Total vectors: {index.ntotal}")
    print(f"   Index trained: {index.is_trained}")
    
    # Save index
    index_path = os.path.join(args.output, "faiss.index")
    faiss.write_index(index, index_path)
    print(f"\n💾 FAISS index saved to: {index_path}")
    
    # Save document metadata
    metadata_path = os.path.join(args.output, "documents.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(documents, f, indent=2)
    print(f"   Document metadata saved to: {metadata_path}")
    
    # Save configuration
    config = {
        "model_name": str(embeddings_path),
        "embedding_dim": embedding_dim,
        "num_documents": len(documents),
        "index_type": "IndexFlatIP",  # Updated to reflect cosine similarity
        "similarity_metric": "cosine",  # NEW: Explicitly document the metric
        "embeddings_normalized": True,  # NEW: Important for reproducibility
        "created_at": datetime.now().isoformat(),
        "categories": list(set(doc.get("category", "unknown") for doc in documents)),
    }
    
    config_path = os.path.join(args.output, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"   Configuration saved to: {config_path}")
    
    # Test retrieval
    print(f"\n🧪 Testing retrieval...")
    test_query = "Database connection pool exhausted"
    
    # CRITICAL: Normalize query embedding for cosine similarity (must match indexing)
    query_embedding = model.encode(
        [test_query], 
        convert_to_numpy=True,
        normalize_embeddings=True  # CRITICAL: Must match indexing normalization!
    ).astype('float32')
    
    k = 3  # Top 3 results
    distances, indices = index.search(query_embedding, k)
    
    print(f"\n   Query: '{test_query}'")
    print(f"   Top {k} results (cosine similarity - higher is better):")
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        doc = documents[idx]
        content = doc.get('content', doc.get('text', 'N/A'))
        snippet = content[:100].replace('\n', ' ')
        # For IndexFlatIP, distance is the inner product (cosine similarity for normalized vectors)
        # Range: -1 to 1, where 1 = identical, 0 = orthogonal, -1 = opposite
        print(f"   {i+1}. [Similarity: {dist:.4f}] {snippet}...")
        print(f"      Category: {doc.get('category', 'N/A')}")
    
    print("\n" + "=" * 80)
    print("✅ RAG Index Build Complete!")
    print("=" * 80)
    print(f"\n📊 Statistics:")
    print(f"   Documents indexed: {len(documents)}")
    print(f"   Embedding model: {embeddings_path}")
    print(f"   Vector dimensions: {embedding_dim}")
    print(f"   Index size: {os.path.getsize(index_path) / 1024:.2f} KB")
    print(f"\n🎯 Next steps:")
    print(f"   1. Test retrieval: python ml/rag/test_retriever.py")
    print(f"   2. Integrate with backend: ml/rag/retriever.py")
    print(f"   3. Monitor performance: faiss_latency metric in Prometheus")
    print("=" * 80)


if __name__ == "__main__":
    main()
