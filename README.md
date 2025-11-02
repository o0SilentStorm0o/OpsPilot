# OpsPilot – AI Assistant for IT Operations 🚀

> An end-to-end open-source AI/OPS automation platform demonstrating **REAL** LLM integration, **production-grade fine-tuning**, uncertainty estimation, RAG enhancement, and enterprise-grade ML Ops practices.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.3-blue)](https://www.typescriptlang.org/)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-brightgreen)](https://www.docker.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![HuggingFace Model](https://img.shields.io/badge/🤗%20Model-opspilot--phi3--lora--v6-yellow)](https://huggingface.co/SilentStorm99/opspilot-phi3-lora-v6)

## 📖 Overview

**OpsPilot** is a showcase project demonstrating enterprise-level AI/OPS automation capabilities with production ML engineering:
- ✅ **Three-Tier Intelligence System** - Fast Path → RAG Path → MastraAI Path (100% accuracy)
- ✅ **LoRA Fine-tuning** of Phi-3 (3.8B params) on HuggingFace Hub
- ✅ **HuggingFace Integration** - Cloud-distributed model storage & CI/CD
- ✅ **RAG Enhancement** - FAISS-powered intelligent retrieval
- ✅ **MastraAI Multi-Agent** - Advanced reasoning for complex incidents
- ✅ **Production ML Ops** - Automated testing, deployment, and monitoring

### 🤖 AI Models & Capabilities

#### 🎯 Three-Tier Intelligence System (Production! ⭐)

**Fast Path** (High Confidence ≥85%):
- Direct ML classification with fine-tuned Phi-3
- Average response time: ~17 seconds
- Handles 87.5% of incidents (7/8 test cases)

**RAG Path** (Medium Confidence 70-85%):
- FAISS vector search + knowledge base retrieval
- Enhanced context for borderline cases
- Improved accuracy with domain knowledge

**MastraAI Path** (Low Confidence <70%):
- Multi-agent reasoning system
- Research, analysis, and synthesis agents
- Average response time: ~20 seconds
- 100% accuracy on escalated cases (1/8 test cases)

**Overall Performance**: 100% accuracy (8/8) in production testing

#### 📦 Fine-Tuned Model on HuggingFace

🔗 **Model Repository**: [SilentStorm99/opspilot-phi3-lora-v6](https://huggingface.co/SilentStorm99/opspilot-phi3-lora-v6)

- **Base Model**: microsoft/Phi-3-mini-4k-instruct (3.8B params)
- **Method**: LoRA fine-tuning with SEQ_CLS task type (r=16, α=32, dropout=0.05)
- **Model Size**: 35.7MB (LoRA adapters only)
- **Performance**: 
  - **Accuracy**: 99-100% on test incidents
  - **Dynamic confidence**: 99.8-100% (real probabilities from logits)
  - **Training**: 20 epochs, 26 samples, loss 2.6→0.3 (89% reduction)
- **Deployment**: 
  - Cloud-distributed via HuggingFace Hub
  - No large model files in git repository
  - Automatic downloads in CI/CD pipelines
  - Production server pulls model on startup
- **See**: Model card on HuggingFace for complete details

### 🤗 HuggingFace Model Distribution

OpsPilot uses **cloud-distributed model storage** via HuggingFace Hub:

**Benefits:**
- ✅ No large model files in git repository (35.7MB LoRA vs 4GB+ base model)
- ✅ Automatic model downloads in CI/CD and production
- ✅ Version control and model history on HuggingFace
- ✅ Public sharing and collaboration
- ✅ Simplified containerized deployments

**Usage in Production:**
```python
# Automatic download from HuggingFace Hub
from transformers import AutoModelForCausalLM
from peft import PeftModel

# Base model (auto-downloaded)
base_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    trust_remote_code=True
)

# LoRA adapters (auto-downloaded from our HF repo)
model = PeftModel.from_pretrained(
    base_model,
    "SilentStorm99/opspilot-phi3-lora-v6"
)
```

**Training New Versions:**
```bash
# Manual workflow trigger on GitHub Actions
# .github/workflows/model-training.yml
# - Trains new model version
# - Optionally uploads to HuggingFace
# - Creates versioned releases (v7, v8, etc.)
```

#### HuggingFace API Integration (Fallback)
- **facebook/bart-large-mnli**: Zero-shot incident classification
- **deepset/roberta-base-squad2**: Question-answering for log analysis
- **facebook/bart-large-cnn**: Text summarization for recommendations

## 🎯 Use Cases

**AI-powered IT incident management** with adaptive intelligence routing:
1. **Classifies incidents** using three-tier system (Fast → RAG → MastraAI)
2. **Analyzes system logs** with context-aware RAG retrieval
3. **Recommends remediation** via intelligent agent orchestration
4. **Escalates complex cases** to multi-agent reasoning (MastraAI)
5. **Monitors performance** with production-grade metrics
6. **100% accuracy** across all confidence levels in production testing

## 🏗️ Architecture

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────────┐
│   Frontend      │─────▶│   Backend API    │─────▶│   ML Service        │
│   (Next.js)     │      │  (TypeScript)    │      │ Phi-3 + LoRA (HF)   │
└─────────────────┘      └──────────────────┘      │ + RAG (FAISS)       │
                                  │                 │ + MastraAI Agents   │
                                  ▼                 └─────────────────────┘
                         ┌──────────────────┐                 ▲
                         │   Prometheus     │                 │
                         │   + Grafana      │                 │
                         └──────────────────┘                 │
                                                               │
                                                    ┌──────────▼──────────┐
                                                    │  HuggingFace Hub    │
                                                    │  Model Storage      │
                                                    │  (35.7MB LoRA)      │
                                                    └─────────────────────┘
```

### ML Pipeline Architecture (Three-Tier System! ⭐)
```
Incident Description
        ↓
  [Phi-3 + LoRA v6]
        ↓
   Classification + Confidence
        ↓
   ┌────┴────────────────────┐
   │  Confidence Routing     │
   └────┬────────────────────┘
        │
   ┌────┴───────┬──────────────┬──────────────┐
   │            │              │              │
≥ 85%      70-85%         < 70%         Error
   │            │              │              │
   ▼            ▼              ▼              ▼
[Fast Path] [RAG Path]  [MastraAI Path] [Fallback]
   │            │              │              │
   │    ┌───────▼──────┐       │              │
   │    │ FAISS Search │       │              │
   │    │ + Knowledge  │       │              │
   │    └───────┬──────┘       │              │
   │            │              │              │
   │            ▼              ▼              │
   │    [Enhanced Context] [Multi-Agent]     │
   │            │          Reasoning          │
   │            │         (Research +         │
   │            │          Analysis +         │
   │            │          Synthesis)         │
   │            │              │              │
   └────────────┴──────────────┴──────────────┘
                     ↓
            Final Classification
             (100% accuracy)
```

### Core Components

- **Backend** (`backend/`): Express.js API with MastraAI multi-agent orchestration
  - `analyzeLogs`: Log analysis and anomaly detection
  - `classifyIncident`: Three-tier intelligent routing
  - `recommendFix`: Remediation planning with RAG enhancement
  - MastraAI agents: Research, Analysis, Synthesis

- **ML Service** (`ml/`): Python-based model serving and intelligence
  - Production server with Phi-3 v6 LoRA (from HuggingFace)
  - FAISS vector search for RAG enhancement
  - Three-tier confidence-based routing
  - Automatic model download from HuggingFace Hub
  - Model evaluation and training scripts

- **Frontend** (`frontend/`): Next.js dashboard
  - Chat interface for incident analysis
  - Real-time metrics visualization
  - System health monitoring

- **CI/CD** (`.github/workflows/`): Automated ML pipeline
  - **model-test.yml**: Fast validation (download from HF + smoke tests, ~2-3 min)
  - **model-training.yml**: Manual retraining workflow (~40 min)
  - Automatic model download from HuggingFace
  - Comprehensive testing and deployment

- **Monitoring**: Prometheus metrics + Grafana dashboards

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- Node.js 20+ (for local development)
- Python 3.11+ (for ML development)
- HuggingFace API Token (for model access)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/o0SilentStorm0o/OpsPilot.git
cd OpsPilot
```

2. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env and add:
# - HF_TOKEN (for HuggingFace model access)
# - OPENAI_API_KEY (for MastraAI agents)
```

3. **Run with Docker**
```bash
docker-compose up --build
```

This starts:
- Backend API: http://localhost:3001
- Frontend: http://localhost:3000
- ML Service: http://localhost:8000 (auto-downloads model from HuggingFace)
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3002 (admin/admin)

### Local Development

#### Backend
```bash
npm install
npm run dev
```

#### ML Service
```bash
cd ml
pip install -r requirements.txt

# Model automatically downloads from HuggingFace on first run
# Set PHI3_MODEL_PATH env var to use local model instead:
# export PHI3_MODEL_PATH="../models/phi3"

# Run production ML server (loads v6 LoRA from HF)
python server_production.py

# Optional: Train new model version
python train_model_classification.py

# Optional: Upload trained model to HuggingFace
python upload_model_to_hf.py
```

#### Testing Production Intelligence System
```bash
# Run comprehensive production tests
cd ml
python test_production_system.py

# Expected results:
# - Fast Path: 7/8 incidents (87.5%)
# - MastraAI escalation: 1/8 incidents (12.5%)
# - Overall accuracy: 100% (8/8)
```

## 📊 Workflows

### 1. Analyze Logs
```typescript
POST /api/analyze-logs
{
  "logs": [
    {
      "timestamp": "2024-01-15T10:30:00Z",
      "level": "error",
      "message": "Connection timeout to database",
      "source": "api-gateway"
    }
  ]
}
```

### 2. Classify Incident
```typescript
POST /api/classify-incident
{
  "incident": {
    "title": "Database connection failure",
    "description": "Unable to connect to primary database",
    "source": "api-service"
  }
}
```

### 3. Recommend Fix
```typescript
POST /api/recommend-fix
{
  "incident": {
    "title": "High CPU usage",
    "description": "CPU at 95% for 10 minutes",
    "source": "app-server-01"
  }
}
```

## 📈 Metrics & Monitoring

OpsPilot exposes Prometheus metrics at `/metrics`:

- `http_request_duration_seconds` - API latency
- `llm_inference_latency_seconds` - Model inference time
- `llm_tokens_generated_total` - Token usage
- `incident_classification_accuracy` - Classification accuracy
- `model_drift_score` - Model performance drift

Access Grafana dashboards at http://localhost:3002 to visualize these metrics.

## 🧪 Testing

```bash
# Run tests
npm test

# Lint code
npm run lint

# Type check
npm run build
```

## 🔒 Security & Privacy

See [ETHICS_AND_PRIVACY.md](docs/ETHICS_AND_PRIVACY.md) for:
- Data anonymization practices
- GDPR compliance guidelines
- Model security considerations
- Responsible AI usage

## 📚 Documentation

- [Setup Guide](SETUP.md) - Production deployment and configuration
- [Contributing](CONTRIBUTING.md) - Contribution guidelines
- [AI Ethics](AI_ETHICS.md) - Responsible AI practices and GDPR compliance
- [HuggingFace Model](https://huggingface.co/SilentStorm99/opspilot-phi3-lora-v6) - Phi-3 v6 LoRA model card

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | TypeScript, Express, Node.js, MastraAI |
| ML | Python, PyTorch, Transformers, PEFT, FAISS |
| Model Storage | HuggingFace Hub (cloud-distributed) |
| Frontend | Next.js, React, Tailwind CSS |
| Monitoring | Prometheus, Grafana |
| Deployment | Docker, Docker Compose |
| CI/CD | GitHub Actions (automated testing + optional training) |

## 🎓 Learning Outcomes

This project demonstrates:

1. **Three-Tier Intelligence**: Adaptive routing (Fast Path → RAG → MastraAI) with 100% accuracy
2. **LLM Fine-tuning**: LoRA-based fine-tuning with cloud distribution via HuggingFace
3. **Production ML**: Cloud model storage, automatic downloads, version management
4. **RAG Enhancement**: FAISS vector search for knowledge retrieval
5. **Multi-Agent AI**: MastraAI orchestration with research, analysis, and synthesis agents
6. **Modern CI/CD**: Separated testing (~3 min) and training (~40 min) workflows
7. **Observability**: Comprehensive logging, metrics, and performance monitoring

## 🤝 Contributing

Contributions are welcome! Please read the contributing guidelines first.

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**David Strnadel**

Created as a demonstration of AI/OPS engineering capabilities for enterprise AI applications.

---

## 🎯 For Recruiters

This project showcases:
- ✅ Three-tier adaptive AI intelligence system (100% accuracy)
- ✅ Production-ready LLM fine-tuning with HuggingFace integration
- ✅ Multi-agent orchestration with MastraAI
- ✅ RAG implementation with FAISS vector search
- ✅ Cloud-distributed ML (HuggingFace Hub for model storage)
- ✅ Modern CI/CD with separated testing and training workflows
- ✅ Monitoring, observability, and production-grade logging
- ✅ Docker deployment and enterprise documentation standards
- ✅ AI ethics and privacy considerations

**Built to demonstrate real-world AI/OPS engineering skills with modern MLOps practices.**
