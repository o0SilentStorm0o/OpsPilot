# 🚀 Production Setup Guide

Complete guide to deploy OpsPilot in production.

---

## ✅ Pre-Deployment Checklist

- [ ] Models downloaded (`models/phi3/`)
- [ ] v6 LoRA adapters trained (`ml/outputs/lora_phi3_v6/final/`)
- [ ] Environment variables configured (`.env`)
- [ ] Docker images built and tested
- [ ] SSL certificates obtained
- [ ] Monitoring stack configured
- [ ] Backup strategy defined

---

## 🔧 Environment Configuration

Create `.env` file in project root:

```bash
# Backend
NODE_ENV=production
PORT=3001
HF_TOKEN=your_huggingface_token_here

# ML Service  
ML_SERVICE_URL=http://localhost:8000
MODEL_PATH=./models/phi3
LORA_PATH=./ml/outputs/lora_phi3_v6/final
DEVICE=cuda

# Monitoring
PROMETHEUS_URL=http://prometheus:9090
GRAFANA_ADMIN_PASSWORD=your_secure_password_here

# Optional: Cloud Storage
AZURE_STORAGE_CONNECTION_STRING=your_connection_string
AWS_S3_BUCKET=your_bucket_name
```

---

## 🐳 Docker Production Deployment

### 1. Build Images

```bash
# ML Service
docker build -t opspilot-ml:v1.0.0 -f ml/Dockerfile .

# Backend
docker build -t opspilot-backend:v1.0.0 -f backend/Dockerfile .
```

### 2. Start Production Stack

```bash
docker-compose -f docker-compose.prod.yml up -d
```

### 3. Verify Health

```bash
# Check all containers
docker-compose ps

# Check ML service
curl http://localhost:8000/health

# Check backend
curl http://localhost:3001/health

# Check metrics
curl http://localhost:3001/metrics
```

---

## ☁️ Cloud Deployment (Azure)

### Prerequisites
- Azure CLI installed
- Azure subscription
- Container Registry created

### Step 1: Push Images to ACR

```bash
# Login to Azure
az login

# Create container registry (if needed)
az acr create --name opspilotregistry --resource-group opspilot-rg --sku Basic

# Login to ACR
az acr login --name opspilotregistry

# Tag images
docker tag opspilot-ml:v1.0.0 opspilotregistry.azurecr.io/opspilot-ml:v1.0.0
docker tag opspilot-backend:v1.0.0 opspilotregistry.azurecr.io/opspilot-backend:v1.0.0

# Push images
docker push opspilotregistry.azurecr.io/opspilot-ml:v1.0.0
docker push opspilotregistry.azurecr.io/opspilot-backend:v1.0.0
```

### Step 2: Deploy to Azure Container Apps

```bash
# Create container app environment
az containerapp env create \
  --name opspilot-env \
  --resource-group opspilot-rg \
  --location eastus

# Deploy ML service
az containerapp create \
  --name opspilot-ml \
  --resource-group opspilot-rg \
  --environment opspilot-env \
  --image opspilotregistry.azurecr.io/opspilot-ml:v1.0.0 \
  --target-port 8000 \
  --ingress external \
  --cpu 2 \
  --memory 8Gi \
  --min-replicas 1 \
  --max-replicas 5

# Deploy backend
az containerapp create \
  --name opspilot-backend \
  --resource-group opspilot-rg \
  --environment opspilot-env \
  --image opspilotregistry.azurecr.io/opspilot-backend:v1.0.0 \
  --target-port 3001 \
  --ingress external \
  --cpu 1 \
  --memory 2Gi \
  --min-replicas 1 \
  --max-replicas 10
```

---

## 🔒 Security Hardening

### 1. Enable HTTPS

Add SSL certificate to `docker-compose.prod.yml`:

```yaml
services:
  nginx:
    image: nginx:alpine
    ports:
      - "443:443"
    volumes:
      - ./ssl/cert.pem:/etc/nginx/ssl/cert.pem
      - ./ssl/key.pem:/etc/nginx/ssl/key.pem
      - ./nginx.conf:/etc/nginx/nginx.conf
```

### 2. Add API Authentication

Update `backend/src/middleware/auth.ts`:

```typescript
export const apiKeyAuth = (req, res, next) => {
  const apiKey = req.headers['x-api-key'];
  if (!apiKey || !validateApiKey(apiKey)) {
    return res.status(401).json({ error: 'Unauthorized' });
  }
  next();
};
```

### 3. Rate Limiting

```typescript
import rateLimit from 'express-rate-limit';

const limiter = rateLimit({
  windowMs: 60 * 1000, // 1 minute
  max: 100 // 100 requests per minute
});

app.use('/api/', limiter);
```

---

## 📊 Monitoring Setup

### 1. Configure Prometheus Alerts

Create `monitoring/alerts.yml`:

```yaml
groups:
  - name: opspilot
    rules:
      - alert: HighLatency
        expr: phi3_inference_latency_seconds > 2
        for: 5m
        annotations:
          summary: "ML service latency too high"
          
      - alert: HighErrorRate
        expr: rate(http_request_duration_seconds{status_code=~"5.."}[5m]) > 0.05
        for: 5m
        annotations:
          summary: "Error rate above 5%"
```

### 2. Grafana Dashboard Backup

```bash
# Export dashboard
curl http://localhost:3002/api/dashboards/uid/phi3-mastra > dashboard-backup.json

# Import on new instance
curl -X POST http://new-grafana:3002/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @dashboard-backup.json
```

---

## 🔄 CI/CD Pipeline (GitHub Actions)

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to Production

on:
  push:
    tags:
      - 'v*'

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Build Docker images
        run: |
          docker build -t opspilot-ml:${{ github.ref_name }} -f ml/Dockerfile .
          docker build -t opspilot-backend:${{ github.ref_name }} -f backend/Dockerfile .
      
      - name: Push to ACR
        run: |
          az acr login --name ${{ secrets.ACR_NAME }}
          docker push opspilotregistry.azurecr.io/opspilot-ml:${{ github.ref_name }}
          docker push opspilotregistry.azurecr.io/opspilot-backend:${{ github.ref_name }}
      
      - name: Deploy to Azure
        run: |
          az containerapp update \
            --name opspilot-ml \
            --resource-group opspilot-rg \
            --image opspilotregistry.azurecr.io/opspilot-ml:${{ github.ref_name }}
```

---

## 📦 Backup & Recovery

### Database Backup (if using PostgreSQL)

```bash
# Backup
docker exec opspilot-postgres pg_dump -U postgres opspilot > backup.sql

# Restore
docker exec -i opspilot-postgres psql -U postgres opspilot < backup.sql
```

### Model Backup

```bash
# Backup models to Azure Blob Storage
az storage blob upload-batch \
  --source ./models \
  --destination opspilot-models \
  --account-name opspilot
```

---

## 🔍 Troubleshooting

### ML Service won't start

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Check model files
ls -lh models/phi3/

# Check logs
docker logs opspilot-ml
```

### High latency

```bash
# Check GPU memory
nvidia-smi

# Check Prometheus metrics
curl http://localhost:3001/metrics | grep phi3_inference_latency
```

### Metrics not showing in Grafana

```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Restart Prometheus
docker-compose restart prometheus
```

---

## 📞 Production Support

- **On-call**: ops-oncall@company.com
- **Slack**: #opspilot-production
- **Runbook**: https://wiki.company.com/opspilot

---

**Last updated:** 2025-11-02
