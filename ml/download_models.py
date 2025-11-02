"""
OpsPilot - Model Downloader
Downloads all required models locally for offline training/inference.

Models downloaded:
1. microsoft/Phi-3-mini-4k-instruct (LLM for fine-tuning)
2. openai/clip-vit-base-patch32 (Visual embeddings)
3. sentence-transformers/all-MiniLM-L6-v2 (Text embeddings for RAG)

Usage:
    python ml/download_models.py [--models-dir ./models]
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        CLIPProcessor,
        CLIPModel,
    )
    from sentence_transformers import SentenceTransformer
    import torch
except ImportError as e:
    logger.error(f"Missing dependencies: {e}")
    logger.error("Install with: pip install transformers sentence-transformers torch")
    sys.exit(1)


MODEL_CONFIGS: List[Dict[str, str]] = [
    {
        "name": "Phi-3-mini-4k-instruct",
        "hf_id": "microsoft/Phi-3-mini-4k-instruct",
        "local_dir": "phi3",
        "type": "llm",
        "description": "Main LLM for fine-tuning and inference"
    },
    {
        "name": "CLIP-ViT-Base-Patch32",
        "hf_id": "openai/clip-vit-base-patch32",
        "local_dir": "clip",
        "type": "vision",
        "description": "Visual embeddings for multimodal analysis"
    },
    {
        "name": "all-MiniLM-L6-v2 (baseline)",
        "hf_id": "sentence-transformers/all-MiniLM-L6-v2",
        "local_dir": "embeddings",
        "type": "embeddings",
        "description": "Baseline text embeddings (384 dim)"
    },
    {
        "name": "multi-qa-MiniLM-L6-cos-v1 (improved RAG)",
        "hf_id": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        "local_dir": "embeddings_rag",
        "type": "embeddings",
        "description": "Optimized for Q&A retrieval (384 dim, cosine-similarity trained)"
    }
]


def download_llm(hf_id: str, local_path: Path) -> None:
    """Download LLM (Phi-3) with tokenizer."""
    logger.info(f"📥 Downloading LLM: {hf_id}")
    
    # Download tokenizer
    logger.info("  ├─ Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        hf_id,
        trust_remote_code=True
    )
    tokenizer.save_pretrained(local_path)
    
    # Download model (use CPU to avoid CUDA issues during download)
    logger.info("  ├─ Model weights...")
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        trust_remote_code=True,
        torch_dtype=torch.float16,  # Save space
        device_map="cpu"  # Download to CPU first
    )
    model.save_pretrained(local_path)
    
    logger.info(f"  ✅ Saved to {local_path}")


def download_clip(hf_id: str, local_path: Path) -> None:
    """Download CLIP model with processor."""
    logger.info(f"📥 Downloading Vision Model: {hf_id}")
    
    # Download processor
    logger.info("  ├─ Processor...")
    processor = CLIPProcessor.from_pretrained(hf_id)
    processor.save_pretrained(local_path)
    
    # Download model
    logger.info("  ├─ Model weights...")
    model = CLIPModel.from_pretrained(hf_id)
    model.save_pretrained(local_path)
    
    logger.info(f"  ✅ Saved to {local_path}")


def download_embeddings(hf_id: str, local_path: Path) -> None:
    """Download sentence-transformers embedding model."""
    logger.info(f"📥 Downloading Embedding Model: {hf_id}")
    
    model = SentenceTransformer(hf_id)
    model.save(str(local_path))
    
    logger.info(f"  ✅ Saved to {local_path}")


def check_disk_space(models_dir: Path, required_gb: float = 10.0) -> bool:
    """Check if sufficient disk space is available."""
    import shutil
    
    stat = shutil.disk_usage(models_dir.parent)
    free_gb = stat.free / (1024 ** 3)
    
    if free_gb < required_gb:
        logger.error(f"❌ Insufficient disk space: {free_gb:.1f}GB free (need {required_gb}GB)")
        return False
    
    logger.info(f"✅ Disk space OK: {free_gb:.1f}GB free")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download OpsPilot models locally")
    parser.add_argument(
        "--models-dir",
        type=str,
        default="./models",
        help="Directory to save models (default: ./models)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip models that already exist locally"
    )
    args = parser.parse_args()
    
    models_dir = Path(args.models_dir).resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("OpsPilot Model Downloader")
    logger.info("=" * 60)
    logger.info(f"Target directory: {models_dir}")
    logger.info(f"Models to download: {len(MODEL_CONFIGS)}")
    logger.info("")
    
    # Check disk space (estimated ~8GB total)
    if not check_disk_space(models_dir, required_gb=10.0):
        sys.exit(1)
    
    # Download each model
    for idx, config in enumerate(MODEL_CONFIGS, 1):
        logger.info(f"\n[{idx}/{len(MODEL_CONFIGS)}] {config['name']}")
        logger.info(f"Description: {config['description']}")
        
        local_path = models_dir / config['local_dir']
        
        # Skip if exists
        if args.skip_existing and local_path.exists():
            logger.info(f"⏭️  Already exists, skipping...")
            continue
        
        local_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Download based on type
            if config['type'] == 'llm':
                download_llm(config['hf_id'], local_path)
            elif config['type'] == 'vision':
                download_clip(config['hf_id'], local_path)
            elif config['type'] == 'embeddings':
                download_embeddings(config['hf_id'], local_path)
            else:
                logger.warning(f"⚠️  Unknown model type: {config['type']}")
        
        except Exception as e:
            logger.error(f"❌ Failed to download {config['name']}: {e}")
            logger.error("Continuing with next model...")
            continue
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 Download Summary")
    logger.info("=" * 60)
    
    for config in MODEL_CONFIGS:
        local_path = models_dir / config['local_dir']
        status = "✅ Downloaded" if local_path.exists() else "❌ Missing"
        
        # Calculate size
        size_mb = 0
        if local_path.exists():
            size_mb = sum(f.stat().st_size for f in local_path.rglob('*') if f.is_file()) / (1024 ** 2)
        
        logger.info(f"{status} {config['name']:30s} ({size_mb:,.0f} MB)")
    
    logger.info("\n✅ Model download complete!")
    logger.info(f"📁 Models saved to: {models_dir}")
    logger.info("\nNext steps:")
    logger.info("  1. Update .gitignore to exclude models/")
    logger.info("  2. Run: python ml/train_lora.py --model models/phi3")
    logger.info("  3. Build RAG index: python ml/rag/indexer.py")


if __name__ == "__main__":
    main()
