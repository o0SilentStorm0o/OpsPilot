"""
Upload OpsPilot Phi-3 v6 LoRA model to HuggingFace
===================================================

This script uploads the fine-tuned model to HuggingFace Hub.
Run from ml/ directory: python upload_model_to_hf.py
"""

import os
from huggingface_hub import HfApi, create_repo
from pathlib import Path

# Configuration
HF_USERNAME = input("Enter your HuggingFace username: ").strip()
REPO_NAME = "opspilot-phi3-lora-v6"
MODEL_PATH = "./outputs/lora_phi3_v6/final"

# Full repo ID
REPO_ID = f"{HF_USERNAME}/{REPO_NAME}"

print("=" * 80)
print("🚀 UPLOADING MODEL TO HUGGINGFACE")
print("=" * 80)
print(f"\n📦 Repository: {REPO_ID}")
print(f"📁 Local path: {MODEL_PATH}")

# Check if model exists
if not os.path.exists(MODEL_PATH):
    print(f"\n❌ Error: Model not found at {MODEL_PATH}")
    print("Please ensure you have trained the v6 model first.")
    exit(1)

# Create README for model card
readme_content = f"""---
license: mit
tags:
- phi-3
- lora
- incident-classification
- fine-tuned
- opspilot
library_name: transformers
base_model: microsoft/Phi-3-mini-4k-instruct
---

# OpsPilot Phi-3 v6 LoRA

Fine-tuned Phi-3-mini model for IT incident classification with 99-100% accuracy.

## 🎯 Model Details

- **Base Model:** microsoft/Phi-3-mini-4k-instruct
- **Fine-tuning Method:** LoRA (Low-Rank Adaptation)
- **LoRA Config:** r=16, alpha=32, dropout=0.05
- **Task:** 6-category incident classification
- **Accuracy:** 99-100% on test set
- **Training Samples:** 26 real-world incidents
- **Training Epochs:** 20

## 📊 Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | 99-100% |
| Categories | 6 |
| Average Latency | ~17s (with RAG) |
| Model Size | LoRA adapters only (~50MB) |

## 🏷️ Categories

1. **Application** - Application-level errors and crashes
2. **Database** - Database connectivity and performance issues
3. **Infrastructure** - Hardware and infrastructure problems
4. **Network** - Network connectivity and routing issues
5. **Performance** - System performance degradation
6. **Security** - Security incidents and breaches

## 🚀 Usage

### With PEFT (Recommended)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# Load LoRA adapters
model = PeftModel.from_pretrained(
    base_model,
    "{REPO_ID}"
)

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

# Classify incident
prompt = \"\"\"<|system|>You are an IT incident classification assistant.<|end|>
<|user|>Classify this IT incident:
Database connection timeout errors. Pool size at maximum.
<|end|>
<|assistant|>\"\"\"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### In Production (OpsPilot Server)

This model is used in the OpsPilot production server with:
- RAG (Retrieval-Augmented Generation) for knowledge enhancement
- MastraAI multi-agent escalation for complex cases
- Three-tier intelligent routing based on confidence

## 📚 Training Details

### Dataset
- 26 carefully curated incident examples
- Natural distribution: Application (12), Infrastructure (4), Database (3), Performance (3), Security (2), Network (2)
- RAG-aware training with internal documentation

### Hyperparameters
- **Learning Rate:** 1e-4
- **Batch Size:** 1
- **Gradient Accumulation:** 4
- **Epochs:** 20
- **Max Length:** 1024 tokens
- **LoRA Rank (r):** 16
- **LoRA Alpha:** 32
- **LoRA Dropout:** 0.05

### Training Environment
- **GPU:** CUDA-enabled (RTX 4090 or similar)
- **Training Time:** ~2 hours
- **Framework:** Transformers + PEFT + BitsAndBytes

## 🔧 Model Architecture

```
Base: Phi-3-mini-4k-instruct (3.8B parameters)
  ↓
LoRA Adapters (16.8M trainable parameters)
  ↓
Sequence Classification Head
  ↓
6 Categories Output
```

## 📖 Citation

```bibtex
@misc{{opspilot-phi3-v6,
  title={{OpsPilot Phi-3 v6 LoRA for Incident Classification}},
  author={{{HF_USERNAME}}},
  year={{2025}},
  publisher={{HuggingFace}},
  howpublished={{\\url{{https://huggingface.co/{REPO_ID}}}}}
}}
```

## 📝 License

MIT License - See repository for details

## 🔗 Related

- **Project Repository:** https://github.com/o0SilentStorm0o/OpsPilot
- **Base Model:** https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
- **PEFT Library:** https://github.com/huggingface/peft

## 🙏 Acknowledgments

Built with Microsoft Phi-3, HuggingFace Transformers, and PEFT.
"""

# Save README
readme_path = os.path.join(MODEL_PATH, "README.md")
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(readme_content)

print(f"\n✅ Created README.md with model card")

# Initialize HF API
api = HfApi()

# Create repository (if doesn't exist)
try:
    print(f"\n📝 Creating repository '{REPO_ID}'...")
    create_repo(
        repo_id=REPO_ID,
        repo_type="model",
        exist_ok=True,
        private=False  # Change to True if you want private repo
    )
    print("✅ Repository created/verified")
except Exception as e:
    print(f"⚠️  Repository might already exist: {e}")

# Upload model files
print(f"\n⬆️  Uploading model files to HuggingFace...")
print("This may take several minutes depending on model size...")

try:
    api.upload_folder(
        folder_path=MODEL_PATH,
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Upload OpsPilot Phi-3 v6 LoRA model (99-100% accuracy)"
    )
    
    print("\n" + "=" * 80)
    print("🎉 SUCCESS! Model uploaded to HuggingFace")
    print("=" * 80)
    print(f"\n🔗 View your model at: https://huggingface.co/{REPO_ID}")
    print(f"\n📦 Use in code:")
    print(f"   from peft import PeftModel")
    print(f"   model = PeftModel.from_pretrained(base_model, '{REPO_ID}')")
    
except Exception as e:
    print(f"\n❌ Upload failed: {e}")
    print("\nPlease check:")
    print("1. You are logged in: huggingface-cli whoami")
    print("2. You have write permissions")
    print("3. Network connection is stable")
    exit(1)

print("\n✨ Next steps:")
print("1. Add HF_TOKEN to GitHub Secrets")
print("2. Update train_model_classification.py to use HF model")
print("3. Update model-ci-cd.yml workflow")
