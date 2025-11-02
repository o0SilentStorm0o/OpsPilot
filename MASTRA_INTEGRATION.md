# MastraAI Integration

## 🎯 Overview

OpsPilot ML Server integrates with MastraAI backend for **intelligent escalation** of complex incidents that require multi-agent orchestration.

## 🔄 Decision Flow

```
Incident → ML Server (Phi-3 v6)
    ↓
Step 1: Fast Classification (NO RAG)
    ↓
IF confidence ≥ 85% AND low uncertainty:
    → ✅ Return classification (FAST PATH - ~2-3s)
    
ELSE IF confidence < 85% OR moderate uncertainty:
    → 🔍 Invoke RAG (retrieve knowledge base)
    → Re-run classification with RAG context
    → IF confidence ≥ 70%:
        → ✅ Return RAG-enhanced result (~15-17s)
    
ELSE (confidence < 70% OR high uncertainty):
    → 🚀 Escalate to MastraAI Backend
    → Multi-agent workflow orchestration
    → ✅ Return MastraAI analysis (~20-30s)
```

## 📊 Thresholds

| Metric | Threshold | Action |
|--------|-----------|--------|
| **RAG Activation** | confidence < 85% | Retrieve relevant docs from knowledge base |
| **MastraAI Escalation** | confidence < 70% **OR** `should_escalate == true` | Invoke multi-agent workflow |
| **Uncertainty Escalation** | `mc_dropout > 0.3` **OR** `margin < 0.2` **OR** `entropy > 1.0` | Flag for escalation |

## 🔌 MastraAI Backend API

### Endpoint
```
POST {MASTRA_BACKEND_URL}/api/workflows/classify
```

### Request
```json
{
  "incident": {
    "title": "Intermittent 500 errors",
    "description": "Users report occasional errors...",
    "source": "monitoring",
    "timestamp": "2025-11-02T20:30:00Z"
  },
  "ml_classification": {
    "category": "Application",
    "confidence": 0.65,
    "severity": "medium",
    "probabilities": {
      "Application": 0.65,
      "Performance": 0.20,
      "Infrastructure": 0.10,
      ...
    },
    "uncertainty": {
      "mc_dropout": 0.35,
      "margin": 0.15,
      "entropy": 1.2,
      "should_escalate": true
    }
  },
  "escalation_reason": "high_uncertainty"
}
```

### Response
```json
{
  "workflow_id": "wf_abc123",
  "category": "Performance",
  "confidence": 0.92,
  "severity": "high",
  "probabilities": {...},
  "reasoning": "Multi-agent analysis suggests performance degradation...",
  "recommendations": [
    "Check database query performance",
    "Review connection pool settings",
    "Monitor memory usage during peak hours"
  ],
  "analysis": "Detailed analysis from MastraAI agents..."
}
```

## 📈 Response Models

### ClassifyResponse (Standard)
```python
{
  "category": "Database",
  "severity": "high",
  "confidence": 0.995,
  "reasoning": "Top predictions: Database (99.5%), ...",
  "uncertainty": {...},
  "all_probabilities": {...},
  "model_version": "phi3-lora-v6-rag",
  "escalated_to_mastra": false,
  "mastra_workflow_id": null
}
```

### MastraEscalationResponse (Escalated)
```python
{
  "category": "Performance",
  "severity": "high",
  "confidence": 0.92,
  "reasoning": "Multi-agent analysis...",
  "uncertainty": {...},
  "all_probabilities": {...},
  "model_version": "phi3-lora-v6-rag-mastra",
  "escalated_to_mastra": true,
  "mastra_workflow_id": "wf_abc123",
  "mastra_recommendations": [
    "Check database performance",
    "Review connection pools"
  ],
  "mastra_analysis": "Detailed MastraAI analysis..."
}
```

## 🛠️ Configuration

### Environment Variables
```bash
# MastraAI Backend URL (default: http://localhost:3001)
MASTRA_BACKEND_URL=http://localhost:3001

# LoRA Model Path
LORA_PATH=/path/to/lora_phi3_v6/final

# Base Model Path
BASE_MODEL_PATH=/path/to/models/phi3
```

## 📊 Performance Metrics

| Path | Avg Latency | Use Case |
|------|-------------|----------|
| **Fast Path** | 2-3s | High confidence (≥85%), no RAG |
| **RAG Path** | 15-17s | Medium confidence (70-85%), RAG enhancement |
| **MastraAI Path** | 20-30s | Low confidence (<70%) or high uncertainty |

## ✅ Benefits

1. **Intelligent Routing**: Automatic escalation based on confidence/uncertainty
2. **Graceful Degradation**: Falls back to ML result if MastraAI unavailable
3. **Multi-tier Analysis**: Fast → RAG → MastraAI as complexity increases
4. **Production Ready**: Async HTTP with timeouts and error handling

## 🔍 Monitoring

Check `/model/info` endpoint for MastraAI status:
```bash
curl http://localhost:8000/model/info
```

```json
{
  "model_version": "v6 (fine-tuned)",
  "expected_accuracy": "95%+",
  "rag_enabled": true,
  "mastra_backend_url": "http://localhost:3001",
  "mastra_escalation_enabled": true,
  "mastra_confidence_threshold": 0.70,
  "rag_confidence_threshold": 0.85
}
```

## 🚀 Next Steps

To fully enable MastraAI workflows:
1. Start MastraAI backend: `cd backend && npm run dev`
2. Implement `/api/workflows/classify` endpoint in backend
3. Test with low-confidence incidents
4. Monitor escalation rates and performance
