"""
OpsPilot - RAG Retriever
Retrieves relevant context from FAISS index for incident analysis
"""

import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import time

# Configuration
INDEX_DIR = "outputs/faiss_index"  # Relative to ml/ directory
INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
DOCUMENTS_PATH = os.path.join(INDEX_DIR, "documents.json")
CONFIG_PATH = os.path.join(INDEX_DIR, "config.json")

class RAGRetriever:
    """Retrieval-Augmented Generation retriever for incident knowledge"""
    
    def __init__(self, index_dir: str = INDEX_DIR):
        """Initialize retriever with FAISS index"""
        self.index_dir = index_dir
        self.index_path = os.path.join(index_dir, "faiss.index")
        self.documents_path = os.path.join(index_dir, "documents.json")
        self.config_path = os.path.join(index_dir, "config.json")
        
        # Load configuration
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        
        # Load embedding model
        self.model_name = self.config['model_name']
        self.model = SentenceTransformer(self.model_name)
        
        # Load FAISS index
        self.index = faiss.read_index(self.index_path)
        
        # Load documents
        with open(self.documents_path, 'r', encoding='utf-8') as f:
            self.documents = json.load(f)
        
        print(f"✅ RAG Retriever initialized")
        print(f"   Model: {self.model_name}")
        print(f"   Documents: {len(self.documents)}")
        print(f"   Index type: {self.config['index_type']}")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        category_filter: str = None
    ) -> List[Dict]:
        """
        Retrieve top-k most relevant documents for query
        
        Args:
            query: Search query (incident description)
            top_k: Number of results to return
            category_filter: Optional category filter (Database, Network, etc.)
        
        Returns:
            List of documents with relevance scores
        """
        start_time = time.time()
        
        # Generate query embedding
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True
        ).astype('float32')
        
        # Search index
        distances, indices = self.index.search(query_embedding, top_k * 2)  # Fetch more for filtering
        
        # Retrieve documents
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            doc = self.documents[idx].copy()
            
            # Apply category filter if specified
            if category_filter and doc.get('category') != category_filter:
                continue
            
            # Calculate relevance score (convert L2 distance to similarity)
            doc['relevance_score'] = 1.0 / (1.0 + dist)
            doc['distance'] = float(dist)
            doc['index'] = int(idx)
            
            results.append(doc)
            
            if len(results) >= top_k:
                break
        
        retrieval_time = time.time() - start_time
        
        return {
            "results": results,
            "query": query,
            "retrieval_time_ms": retrieval_time * 1000,
            "total_documents": len(self.documents),
            "returned_count": len(results)
        }
    
    def get_context(
        self,
        query: str,
        top_k: int = 3,
        max_context_length: int = 2000
    ) -> str:
        """
        Get concatenated context for LLM prompt
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
            max_context_length: Maximum context characters
        
        Returns:
            Formatted context string
        """
        retrieval = self.retrieve(query, top_k)
        results = retrieval['results']
        
        context_parts = []
        total_length = 0
        
        for i, doc in enumerate(results):
            # Format document
            doc_text = f"""
## Relevant Context {i+1} (Relevance: {doc['relevance_score']:.3f})
**Category:** {doc.get('category', 'Unknown')}
**Severity:** {doc.get('severity', 'Unknown')}

{doc['text']}
"""
            doc_length = len(doc_text)
            
            # Check if adding this would exceed limit
            if total_length + doc_length > max_context_length:
                # Truncate if it's the first document
                if i == 0:
                    context_parts.append(doc_text[:max_context_length])
                break
            
            context_parts.append(doc_text)
            total_length += doc_length
        
        return "\n".join(context_parts)


def test_retriever():
    """Test RAG retriever with sample queries"""
    print("\n" + "=" * 80)
    print("🧪 Testing RAG Retriever")
    print("=" * 80)
    
    retriever = RAGRetriever()
    
    test_queries = [
        "Database connection pool exhausted",
        "High memory usage causing swap",
        "SSL certificate expiring soon",
        "Kubernetes pod crash loop"
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: '{query}'")
        print("-" * 80)
        
        retrieval = retriever.retrieve(query, top_k=2)
        
        print(f"   Retrieval time: {retrieval['retrieval_time_ms']:.2f}ms")
        print(f"   Results found: {retrieval['returned_count']}")
        print()
        
        for i, doc in enumerate(retrieval['results']):
            print(f"   {i+1}. Relevance: {doc['relevance_score']:.3f}")
            print(f"      Category: {doc.get('category', 'N/A')}")
            print(f"      Severity: {doc.get('severity', 'N/A')}")
            snippet = doc['text'][:150].replace('\n', ' ')
            print(f"      Text: {snippet}...")
            print()


if __name__ == "__main__":
    test_retriever()
