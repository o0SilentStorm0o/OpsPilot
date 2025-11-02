# System Architecture

## Overview

OpsPilot follows a microservices architecture with three main components:

```
┌────────────────────────────────────────────────────────────────┐
│                        User Interface                           │
│                      (Next.js Frontend)                         │
└───────────────────────────┬────────────────────────────────────┘
                            │
                            │ HTTP/REST
                            │
┌───────────────────────────▼────────────────────────────────────┐
│                      Backend API Server                         │
│                    (TypeScript + Express)                       │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ analyzeLogs  │  │  classify    │  │ recommendFix │         │
│  │              │  │  Incident    │  │              │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│                                                                 │
│  ┌──────────────┐                                              │
│  │ retrainModel │  Workflows                                   │
│  └──────────────┘                                              │
└─────────────┬───────────────────────────┬──────────────────────┘
              │                           │
              │                           │
              │ HuggingFace API           │ HTTP
              │                           │
┌─────────────▼───────────┐   ┌───────────▼──────────────────────┐
│  HuggingFace Inference  │   │      ML Service (Python)         │
│   (Mistral-7B / Phi-3)  │   │                                  │
│                         │   │  ┌────────────────────────────┐  │
└─────────────────────────┘   │  │  Fine-tuning Engine        │  │
                              │  │  (LoRA + PEFT)             │  │
                              │  └────────────────────────────┘  │
                              │                                  │
                              │  ┌────────────────────────────┐  │
                              │  │  Model Evaluation          │  │
                              │  └────────────────────────────┘  │
                              └──────────────────────────────────┘
                                            │
                                            │
                              ┌─────────────▼────────────────────┐
                              │      Monitoring Stack            │
                              │   (Prometheus + Grafana)         │
                              └──────────────────────────────────┘
```

## Component Breakdown

### 1. Frontend (Next.js)

**Purpose**: User interface for interacting with OpsPilot

**Key Features**:
- Chat interface for incident analysis
- Real-time metrics dashboard
- System health visualization
- Incident history and tracking

**Technology**:
- Next.js 14 with App Router
- React 18
- Tailwind CSS
- Shadcn UI components

**API Integration**:
```typescript
// Example: Analyze logs
const response = await fetch('/api/analyze-logs', {
  method: 'POST',
  body: JSON.stringify({ logs })
});
```

### 2. Backend API (TypeScript + Express)

**Purpose**: Orchestration layer and workflow management

**Core Responsibilities**:
- Request handling and validation
- Workflow orchestration
- LLM interaction management
- Metrics collection
- Error handling

**Workflows**:

#### a) analyzeLogs
```typescript
Input: LogEntry[]
Output: LogAnalysisResult {
  summary, anomalies, patterns, severity, suggestedActions
}
```
- Aggregates log entries
- Calls LLM for analysis
- Extracts patterns using heuristics
- Records metrics

#### b) classifyIncident
```typescript
Input: Incident
Output: ClassificationResult {
  category, subcategory, severity, priority, confidence
}
```
- Categorizes incidents (Network/Database/Application/Security/Hardware/Performance)
- Determines severity and priority
- Validates against known patterns

#### c) recommendFix
```typescript
Input: Incident + RemediationContext
Output: RemediationPlan {
  steps, estimatedTime, risks, successCriteria, rollbackPlan
}
```
- Generates step-by-step remediation
- Includes verification methods
- Provides rollback procedures

#### d) retrainModel
```typescript
Input: FeedbackData
Output: RetrainingResult {
  triggered, reason, feedbackCount, scheduledTime
}
```
- Collects user feedback
- Calculates model drift
- Triggers ML service for retraining

### 3. ML Service (Python)

**Purpose**: Model training, fine-tuning, and evaluation

**Components**:

#### Training Pipeline (`train.py`)
- Loads base model (Mistral-7B or Phi-3)
- Applies LoRA configuration for efficient fine-tuning
- Uses 4-bit quantization for memory efficiency
- Trains on synthetic IT operations data
- Saves fine-tuned model

**LoRA Configuration**:
```python
LoraConfig(
  r=16,              # Rank
  lora_alpha=32,
  target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
  lora_dropout=0.05
)
```

#### Evaluation (`eval.py`)
- Measures classification accuracy
- Calculates precision, recall, F1
- Benchmarks inference latency
- Generates evaluation reports

#### Dataset Generation (`generate_dataset.py`)
- Creates synthetic IT incident logs
- Balanced across 6 categories
- Realistic error messages and patterns
- Train/test/validation splits

### 4. Monitoring (Prometheus + Grafana)

**Metrics Collected**:

| Metric | Type | Description |
|--------|------|-------------|
| `http_request_duration_seconds` | Histogram | API response time |
| `llm_inference_latency_seconds` | Histogram | LLM processing time |
| `llm_tokens_generated_total` | Counter | Total tokens used |
| `incident_classification_accuracy` | Gauge | Classification accuracy |
| `model_drift_score` | Gauge | Performance drift (0-1) |
| `active_incidents_total` | Gauge | Incidents being processed |

**Grafana Dashboards**:
- API Performance Overview
- LLM Metrics & Token Usage
- Model Quality & Drift
- System Health

## Data Flow

### Typical Request Flow

1. **User submits incident via Frontend**
   ```
   POST /api/classify-incident
   ```

2. **Backend validates request**
   ```typescript
   if (!incident) return 400;
   ```

3. **Workflow executes**
   ```typescript
   classifyIncident(incident) →
     chatCompletion(prompt) →
       HuggingFace API
   ```

4. **Response processed**
   - Parse LLM output
   - Validate classification
   - Record metrics

5. **Metrics updated**
   ```typescript
   incidentClassificationAccuracy.set(confidence);
   llmInferenceLatency.observe(latency);
   ```

6. **Response returned to frontend**
   ```json
   {
     "category": "Database",
     "severity": "high",
     "confidence": 0.89
   }
   ```

## Model Fine-tuning Pipeline

```
┌──────────────────┐
│  Collect User    │
│  Feedback        │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Store Feedback  │
│  (50+ samples)   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Calculate Drift │
│  (Error Rate)    │
└────────┬─────────┘
         │
         ▼ (drift > 0.2)
┌──────────────────┐
│  Trigger ML      │
│  Service         │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Fine-tune with  │
│  LoRA            │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Evaluate Model  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Deploy Updated  │
│  Model           │
└──────────────────┘
```

## Security Considerations

1. **API Security**
   - Input validation with Zod schemas
   - Rate limiting
   - Authentication tokens (HF_TOKEN)

2. **Data Privacy**
   - Log anonymization
   - No PII in training data
   - Synthetic data only

3. **Model Security**
   - Prompt injection prevention
   - Output sanitization
   - Command validation (prevent `rm -rf`)

## Scalability

**Current Architecture**: Single-node deployment

**Production Scaling**:
- Backend: Horizontal scaling with load balancer
- ML Service: GPU instance pool
- Monitoring: Separate Prometheus cluster
- Database: Redis for feedback storage

## Deployment

**Development**:
```bash
docker-compose up
```

**Production**:
- Kubernetes manifests
- Helm charts
- GitHub Actions CI/CD
- Infrastructure as Code (Terraform)

## Future Enhancements

1. **RAG Integration**: Vector database for knowledge retrieval
2. **Multi-modal**: Image analysis (screenshots, graphs)
3. **Agent Framework**: LangChain/AutoGen integration
4. **Real-time Processing**: WebSocket for live log streaming
5. **A/B Testing**: Compare model versions
