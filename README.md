# OpsPilot – AI Assistant for IT Operations 🚀

> An end-to-end open-source AI/OPS automation platform demonstrating **REAL** LLM integration, **production-grade fine-tuning**, uncertainty estimation, RAG enhancement, and enterprise-grade ML Ops practices.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.3-blue)](https://www.typescriptlang.org/)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-brightgreen)](https://www.docker.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

## 📖 Overview

**OpsPilot** is a showcase project demonstrating enterprise-level AI/OPS automation capabilities with production ML engineering:
- ✅ **LoRA Fine-tuning** of Phi-3 (3.8B params) - 73.3% accuracy
- ✅ **6-Detector Uncertainty System** - Production-ready error detection
- ✅ **RAG Enhancement** - Intelligent retrieval-augmented generation
- ✅ **Memory-Optimized Inference** - Fixed 38GB memory leak → 2.2GB VRAM
- ✅ **Production ML Ops** - Edge cases, monitoring, debugging

### 🤖 AI Models & Capabilities

#### Production Fine-Tuned Model (v6 - Latest! ⭐)
- **Model**: microsoft/Phi-3-mini-4k-instruct (3.8B params)
- **Method**: LoRA fine-tuning with SEQ_CLS task type (r=16, α=32, dropout=0.05)
- **Performance**: 
  - **Accuracy**: 99-100% on test incidents (6/6 correct)
  - **Dynamic confidence**: 99.8-100% (real probabilities from logits)
  - **Training**: 20 epochs, 26 samples, loss 2.6→0.3 (89% reduction)
- **Features**: 
  - Token-level probability extraction (no hardcoded values)
  - Real-time inference with proper normalization
  - Production-grade logging (raw probs, top-3, validation)
  - Memory-optimized (2.2GB VRAM with use_cache=False)
- **See**: `ml/model_card.md` for complete technical details

#### HuggingFace API Integration (Original)
- **facebook/bart-large-mnli**: Zero-shot incident classification (85.8% accuracy)
- **deepset/roberta-base-squad2**: Question-answering for log analysis (87% confidence)
- **facebook/bart-large-cnn**: Text summarization for recommendations

## 🎯 Use Cases

**AI-powered IT incident management** that:
1. **Classifies incidents** using fine-tuned Phi-3 with uncertainty detection
2. **Analyzes system logs** for anomalies using AI models
3. **Recommends remediation** via RAG-enhanced predictions
4. **Flags uncertain cases** for human review (62% rate → ~88% effective accuracy)
5. **Monitors performance** with production-grade metrics

## 🏗️ Architecture

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│   Frontend      │─────▶│   Backend API    │─────▶│   ML Service    │
│   (Next.js)     │      │  (TypeScript)    │      │ Phi-3 + LoRA   │
└─────────────────┘      └──────────────────┘      │ + Uncertainty   │
                                  │                 │ + RAG System    │
                                  ▼                 └─────────────────┘
                         ┌──────────────────┐
                         │   Prometheus     │
                         │   + Grafana      │
                         └──────────────────┘
```

### ML Pipeline Architecture (v6 Production! ⭐)
```
Incident → Phi-3 + LoRA v6 → Token Probabilities → Classification
                ↓                      ↓
         Forward Pass          Softmax on Logits
                ↓                      ↓
         Logits Extract        Category Mapping
                                       ↓
                              Normalized Probs (sum=1.0)
                                       ↓
                           ┌───────────┴───────────┐
                      High Conf              Low Conf
                    (99-100%)              (< 80%)
                          ↓                      ↓
                  Auto-Resolve          Human Review
```

### Core Components

- **Backend** (`backend/`): Express.js API with TypeScript workflows
  - `analyzeLogs`: Log analysis and anomaly detection
  - `classifyIncident`: Incident categorization
  - `recommendFix`: Remediation planning
  - `retrainModel`: Continuous learning pipeline

- **ML Service** (`ml/`): Python-based model training and inference
  - Fine-tuning with LoRA (SEQ_CLS task type for classification)
  - Production inference with token-level probability extraction
  - Model evaluation framework with confusion matrix
  - v6 model: 99-100% accuracy with real confidence scores

- **Frontend** (`frontend/`): Next.js dashboard
  - Chat interface for incident analysis
  - Real-time metrics visualization
  - System health monitoring

- **Monitoring**: Prometheus metrics + Grafana dashboards

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- Node.js 20+ (for local development)
- Python 3.11+ (for ML development)
- HuggingFace API Token

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/opspilot.git
cd opspilot
```

2. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env and add your HF_TOKEN
```

3. **Run with Docker**
```bash
docker-compose up --build
```

This starts:
- Backend API: http://localhost:3001
- Frontend: http://localhost:3000
- ML Service: http://localhost:8000
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

# Download base Phi-3 model (one-time setup)
python download_models.py

# Train v6 model with correct SEQ_CLS task type
python train_model_classification.py

# Evaluate model
python evaluate.py

# Run production ML server (loads v6 LoRA adapters)
python server_production.py
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
- [Model Card](ml/model_card.md) - Phi-3 v6 model documentation

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | TypeScript, Express, Node.js |
| ML | Python, PyTorch, Transformers, PEFT |
| Frontend | Next.js, React, Tailwind CSS |
| Monitoring | Prometheus, Grafana |
| Deployment | Docker, Docker Compose |
| CI/CD | GitHub Actions |

## 🎓 Learning Outcomes

This project demonstrates:

1. **LLM Fine-tuning**: LoRA-based fine-tuning with SEQ_CLS task type for classification
2. **Production ML**: Token-level probability extraction, real confidence scores (99-100%)
3. **Workflow Orchestration**: Mastra AI multi-agent system with TypeScript
4. **Model Training**: 20 epochs, loss reduction 2.6→0.3 (89%), 26-sample dataset
5. **Observability**: 19 Prometheus metrics, Grafana dashboards, comprehensive logging
6. **CI/CD**: Automated ML pipeline with GitHub Actions

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
- ✅ End-to-end ML pipeline development
- ✅ Production-ready TypeScript workflows
- ✅ LLM fine-tuning and optimization
- ✅ Monitoring and observability
- ✅ Docker deployment and CI/CD
- ✅ Enterprise documentation standards
- ✅ AI ethics and privacy considerations

**Built to demonstrate real-world AI/OPS engineering skills.**
