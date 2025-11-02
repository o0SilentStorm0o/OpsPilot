# Model Card - OpsPilot IT Operations Model

## Model Details

**Model Name**: OpsPilot v6 IT Incident Classification Model  
**Base Model**: microsoft/Phi-3-mini-4k-instruct (3.8B parameters)  
**Fine-tuning Method**: LoRA (Low-Rank Adaptation) with SEQ_CLS task type  
**Version**: 6.0 (Production)  
**Date**: November 2025  
**License**: MIT  
**Training**: 20 epochs, 26 samples, loss 2.596→0.277 (89% reduction)  
**Performance**: 99-100% accuracy on test set with dynamic confidence scores  

## Intended Use

### Primary Use Cases

This model is designed for **real-time IT incident classification** in enterprise environments:

1. **Incident Classification**: Categorizing IT incidents into 6 categories (Database, Network, Security, Performance, Application, Infrastructure)
2. **Confidence Scoring**: Providing dynamic probability-based confidence scores (99-100% on clear cases)
3. **Production Inference**: Fast token-level classification with proper probability normalization

### Intended Users

- IT Operations Teams
- DevOps Engineers
- Site Reliability Engineers (SREs)
- System Administrators
- NOC (Network Operations Center) Staff

### Out-of-Scope Uses

❌ **Not suitable for**:
- Medical diagnosis or healthcare decisions
- Legal advice or compliance decisions
- Financial trading or investment recommendations
- Safety-critical systems (aviation, automotive)
- Real-time emergency response (should involve human oversight)
- Customer-facing production use without human review

## Training Data

### Dataset Characteristics

- **Type**: Real-world inspired IT incident logs (incident_logs_v5_train.csv)
- **Size**: 26 training samples
- **Categories**: Database, Network, Security, Performance, Application, Infrastructure (6 categories)
- **Distribution**: Balanced across categories
- **Language**: English
- **Privacy**: Synthetic data, no real customer data, no PII

### Data Generation Process

Training data was created using historical incident patterns:
- Realistic IT incident descriptions
- Diverse error scenarios (connection timeouts, security breaches, performance issues)
- Balanced category distribution
- Production-ready incident formats

### Known Limitations

- **English-only**: Model not trained on other languages
- **Small dataset**: 26 samples (production v6 trained for demonstration)
- **Limited domain**: Specific to IT operations scenarios (6 categories)
- **Edge cases**: May show uncertainty on ambiguous incidents (e.g., CPU issues can be Performance or Infrastructure)

## Model Architecture

### Base Model

- **Phi-3-mini-4k-instruct**: 3.8 billion parameter transformer model from Microsoft
- **Context length**: 4096 tokens
- **Optimized**: For efficient inference on consumer GPUs

### Fine-tuning (v6 Production)

- **Method**: LoRA (Low-Rank Adaptation)
- **Task Type**: **SEQ_CLS** (sequence classification) - CRITICAL FIX from v5
- **Configuration**:
  - Rank (r): 16
  - Alpha: 32
  - Dropout: 0.05
  - Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Trainable params**: 8.9M / 3.8B = 0.44%
- **Training**:
  - Epochs: 20
  - Batch size: 1
  - Learning rate: 1e-4
  - Optimizer: AdamW
  - Loss: Cross-entropy with SEQ_CLS head

### Training Hardware

- **Recommended**: NVIDIA GPU with 8GB+ VRAM (RTX 3060 or better)
- **Memory optimization**: use_cache=False to avoid DynamicCache bugs
- **Training time**: ~3.6 minutes for 20 epochs (26 samples)

## Performance

### Evaluation Metrics (v6 Production)

Measured on 6-sample test set (real-world incidents):

| Metric | Score |
|--------|-------|
| **Classification Accuracy** | **100%** (6/6 correct) |
| **Confidence Range** | 99.8-100% (dynamic, real probabilities) |
| **Training Loss** | 2.596 → 0.277 (89% reduction) |
| **Inference Latency** | ~200-500ms (10 tokens vs 50 in v5) |
| **Memory Usage** | 2.2GB VRAM (with use_cache=False) |

### Test Results (Production Validation)

```
Database connection timeout    → Database (100.0%) ✅
Network packet loss detected   → Network (100.0%) ✅
Security breach attempt        → Security (100.0%) ✅
CPU usage at 95%              → Performance (99.8%) ✅
Application crashed           → Application (100.0%) ✅
Disk full on server           → Infrastructure (99.9%) ✅
```

**Key Improvements over v5:**
- ✅ Real probabilities (vs hardcoded 85% confidence)
- ✅ SEQ_CLS task type (vs CAUSAL_LM in v5)
- ✅ Token-level extraction (proper classification head)
- ✅ 5x faster inference (10 tokens vs 50)
- ✅ Production-grade logging (raw probs, top-3, validation)

### Per-Category Performance

Expected variation across categories:
- **Network**: High accuracy (clear keywords)
- **Database**: High accuracy (distinct error patterns)
- **Application**: Moderate accuracy (broader category)
- **Security**: High accuracy (specific threat patterns)
- **Hardware**: Moderate accuracy (overlaps with performance)
- **Performance**: Moderate accuracy (overlaps with other categories)

## Limitations and Risks

### Technical Limitations

1. **Context Window**: Limited by base model (4k-32k tokens)
2. **Hallucination**: May generate plausible but incorrect recommendations
3. **Outdated Knowledge**: Cutoff date of base model training
4. **Language**: English-only, no multilingual support
5. **Confidence Calibration**: Confidence scores may not always reflect accuracy

### Known Failure Modes

- **Novel Incident Types**: Poor performance on completely new incident categories
- **Ambiguous Descriptions**: Struggles with vague or incomplete incident reports
- **Multi-category Incidents**: May misclassify incidents spanning multiple domains
- **Complex Dependencies**: May oversimplify intricate system interactions

### Safety Considerations

⚠️ **Important Safety Notes**:

1. **Human Review Required**: All recommendations should be reviewed by qualified personnel
2. **No Automated Execution**: Do not auto-execute commands without verification
3. **Dangerous Commands**: System blocks known dangerous operations (rm -rf, DROP TABLE)
4. **Test Environment First**: Always test recommendations in non-production environments
5. **Rollback Plans**: Review and prepare rollback procedures before applying fixes

### Bias and Fairness

**Potential Biases**:
- **Technology Stack Bias**: May favor common stacks (Linux, PostgreSQL, etc.)
- **Severity Bias**: May under/over-estimate severity for rare incident types
- **Temporal Bias**: Recommendations based on 2023-2024 best practices

**Mitigation Strategies**:
- Balanced training data across categories
- Regular evaluation on diverse incident types
- Feedback loop for continuous improvement
- Human oversight for all recommendations

## Ethical Considerations

### Privacy

✅ **Privacy-Preserving Design**:
- No real user data in training
- Synthetic data only
- No PII collection
- Anonymization recommended for production use

### Transparency

✅ **Model provides**:
- Confidence scores for predictions
- Reasoning for classifications
- Explainable remediation steps
- Clear limitations disclosure

### Responsible AI Principles

1. **Human Oversight**: AI assists, humans decide
2. **Transparency**: Clear indication of AI-generated content
3. **Accountability**: Audit trail for all recommendations
4. **Safety**: Built-in safeguards against dangerous operations
5. **Privacy**: No unauthorized data collection
6. **Fairness**: Balanced performance across incident types

## Usage Guidelines

### Best Practices

✅ **Do**:
- Review all recommendations before implementation
- Verify confidence scores (>0.7 recommended)
- Test in non-production environments first
- Provide feedback for incorrect classifications
- Monitor model performance regularly
- Maintain human oversight

❌ **Don't**:
- Auto-execute recommendations without review
- Use for safety-critical systems
- Trust low-confidence predictions (<0.5)
- Deploy without proper monitoring
- Use with sensitive data without anonymization
- Ignore security warnings

### Deployment Checklist

- [ ] Human review process in place
- [ ] Monitoring and alerting configured
- [ ] Data anonymization implemented
- [ ] Rollback procedures documented
- [ ] Team training completed
- [ ] Feedback collection mechanism active
- [ ] Regular model evaluation scheduled

## Maintenance and Updates

### Continuous Improvement

**Retraining Triggers**:
- Model drift score > 0.2
- 50+ feedback samples collected
- Classification accuracy drops below 0.7
- New incident categories emerge

**Update Process**:
1. Collect user feedback
2. Calculate drift metrics
3. Retrain with LoRA on new data
4. Evaluate on validation set
5. A/B test new model version
6. Deploy if improved

### Monitoring

**Key Metrics to Track**:
- Classification accuracy
- Inference latency
- Token usage
- User feedback scores
- Model drift
- Error rates per category

## Contact and Support

**Model Maintainer**: David Strnadel  
**Repository**: https://github.com/davidstrnadel/opspilot  
**Issues**: https://github.com/davidstrnadel/opspilot/issues  
**Documentation**: See docs/ folder  

## Version History

### v1.0 (January 2024)
- Initial release
- Mistral-7B-Instruct-v0.2 base model
- LoRA fine-tuning on synthetic IT operations data
- 6 incident categories supported
- Basic remediation recommendations

## Acknowledgments

- **HuggingFace**: Transformers library and model hosting
- **Mistral AI**: Base model (Mistral-7B)
- **Microsoft**: Alternative base model (Phi-3)
- **PEFT**: Parameter-efficient fine-tuning library

## License

MIT License - See LICENSE file for details

---

**Disclaimer**: This model is a demonstration project. While designed with best practices, it should be thoroughly evaluated before production use. Always maintain human oversight for IT operations decisions.
