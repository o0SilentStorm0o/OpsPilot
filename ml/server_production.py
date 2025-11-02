"""
🚀 PRODUCTION-READY ML SERVER
FastAPI server s real Phi-3 inference + uncertainty quantification + RAG
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import logging
import httpx
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="OpsPilot ML Service - Phi-3 + RAG",
    description="Production ML service with fine-tuned Phi-3 model and RAG enhancement",
    version="3.0.0"
)

# ============================================================================
# MODEL LOADING (at startup)
# ============================================================================

# Get absolute paths
import pathlib
ML_DIR = pathlib.Path(__file__).parent.resolve()
PROJECT_ROOT = ML_DIR.parent

# v6 LoRA adapters (classification task type - FIXED!)
LORA_PATH = os.getenv("LORA_PATH", str(ML_DIR / "outputs" / "lora_phi3_v6" / "final"))
BASE_MODEL_PATH = str(PROJECT_ROOT / "models" / "phi3")
MASTRA_BACKEND_URL = os.getenv("MASTRA_BACKEND_URL", "http://localhost:3001")
model = None
tokenizer = None
rag_retriever = None
device = "cuda" if torch.cuda.is_available() else "cpu"

@app.on_event("startup")
async def load_model():
    """Load model and RAG system at startup"""
    global model, tokenizer, rag_retriever
    
    try:
        logger.info(f"🚀 Loading fine-tuned Phi-3 with v6 LoRA adapters + RAG")
        logger.info(f"📁 LoRA adapters: {LORA_PATH}")
        logger.info(f"📁 Base model: {BASE_MODEL_PATH}")
        logger.info(f"🔧 Device: {device}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL_PATH,
            local_files_only=True,
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Check if v6 LoRA adapters exist
        import os
        if os.path.exists(LORA_PATH):
            logger.info("✅ v6 LoRA adapters found - loading fine-tuned model")
            
            # Import PEFT for LoRA loading
            from peft import PeftModel
            
            # Load base model with classification head
            from transformers import AutoModelForCausalLM
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_PATH,
                local_files_only=True,
                trust_remote_code=True,
                device_map="auto" if device == "cuda" else None
            )
            
            # Load LoRA adapters
            model = PeftModel.from_pretrained(
                base_model,
                LORA_PATH,
                is_trainable=False
            )
            
            # Merge adapters into base model for faster inference
            model = model.merge_and_unload()
            
            logger.info("✅ v6 LoRA adapters loaded and merged!")
            logger.info("🎯 Model: Fine-tuned Phi-3 (95%+ accuracy expected)")
            
        else:
            logger.warning(f"⚠️ v6 LoRA adapters not found at {LORA_PATH}")
            logger.info("📦 Falling back to base model with classification head")
            
            # Load base model with classification head
            model = AutoModelForSequenceClassification.from_pretrained(
                BASE_MODEL_PATH,
                num_labels=6,
                local_files_only=True,
                trust_remote_code=True,
                device_map="auto" if device == "cuda" else None
            )
            logger.info("⚠️ Using base model (83% accuracy)")
        
        model.eval()  # Set to evaluation mode
        logger.info(f"✅ Model ready on {device}!")
        
        # Load RAG retriever (optional - graceful degradation if fails)
        try:
            from rag.retriever import RAGRetriever
            rag_retriever = RAGRetriever()
            logger.info("✅ RAG retriever loaded successfully!")
        except Exception as rag_error:
            logger.warning(f"⚠️ RAG retriever failed to load: {rag_error}")
            logger.warning("⚠️ Continuing without RAG enhancement")
            rag_retriever = None
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.error("Server will run but /classify will return 503 errors")
        model = None
        tokenizer = None
        rag_retriever = None


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class Incident(BaseModel):
    title: str
    description: str
    source: Optional[str] = "unknown"


class ClassifyRequest(BaseModel):
    title: str
    description: str
    source: Optional[str] = "unknown"


class UncertaintyMetrics(BaseModel):
    mc_dropout: float
    margin: Optional[float] = None
    entropy: Optional[float] = None
    energy: Optional[float] = None
    knn_distance: Optional[float] = None
    should_escalate: bool


class ClassifyResponse(BaseModel):
    model_config = {"protected_namespaces": ()}  # Allow model_* fields
    
    category: str
    severity: str
    confidence: float
    reasoning: str
    uncertainty: UncertaintyMetrics
    all_probabilities: Dict[str, float]
    model_version: str = "phi3-lora-v6-rag"
    escalated_to_mastra: bool = False
    mastra_workflow_id: Optional[str] = None


class MastraEscalationResponse(BaseModel):
    """Response from MastraAI backend workflow"""
    model_config = {"protected_namespaces": ()}
    
    category: str
    severity: str
    confidence: float
    reasoning: str
    uncertainty: UncertaintyMetrics
    all_probabilities: Dict[str, float]
    model_version: str = "phi3-lora-v6-rag-mastra"
    escalated_to_mastra: bool = True
    mastra_workflow_id: str
    mastra_recommendations: Optional[List[str]] = None
    mastra_analysis: Optional[str] = None


# ============================================================================
# INFERENCE HELPERS
# ============================================================================

# Category mapping (as used in training)
CATEGORY_NAMES = [
    "Database",
    "Network", 
    "Security",
    "Performance",
    "Application",
    "Infrastructure"
]

# Severity mapping (based on confidence)
def get_severity(confidence: float, category: str) -> str:
    """Map confidence to severity"""
    if category == "Security":
        # Security incidents are always higher severity
        if confidence > 0.8:
            return "critical"
        elif confidence > 0.6:
            return "high"
        else:
            return "medium"
    else:
        if confidence > 0.9:
            return "high"
        elif confidence > 0.7:
            return "medium"
        else:
            return "low"


def calculate_uncertainty(probs: torch.Tensor) -> UncertaintyMetrics:
    """Calculate uncertainty metrics"""
    
    # MC Dropout uncertainty (approximated from single pass)
    top_prob = probs.max().item()
    mc_dropout = 1.0 - top_prob
    
    # Margin uncertainty (gap between top 2 predictions)
    top_2 = torch.topk(probs, 2)
    margin = (top_2.values[0] - top_2.values[1]).item() if len(top_2.values) >= 2 else 1.0
    
    # Entropy uncertainty
    entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
    
    # Energy uncertainty (lower is more certain)
    logits_approx = torch.log(probs + 1e-10)
    energy = -torch.logsumexp(logits_approx, dim=0).item()
    
    # Decision: should escalate?
    should_escalate = (
        mc_dropout > 0.3 or  # High MC Dropout
        margin < 0.2 or      # Low margin
        entropy > 1.0        # High entropy
    )
    
    return UncertaintyMetrics(
        mc_dropout=mc_dropout,
        margin=margin,
        entropy=entropy,
        energy=energy,
        knn_distance=None,  # Would require embeddings + kNN index
        should_escalate=should_escalate
    )


def generate_reasoning(category: str, confidence: float, text: str) -> str:
    """Generate human-readable reasoning"""
    
    keywords = {
        "Database": ["database", "connection", "pool", "query", "timeout", "deadlock"],
        "Network": ["network", "latency", "packet", "dns", "firewall", "connection"],
        "Security": ["security", "breach", "attack", "vulnerability", "unauthorized", "malicious"],
        "Performance": ["performance", "slow", "cpu", "memory", "latency", "degradation"],
        "Application": ["application", "error", "crash", "exception", "bug"],
        "Infrastructure": ["infrastructure", "server", "disk", "hardware", "outage"]
    }
    
    found_keywords = [kw for kw in keywords.get(category, []) if kw.lower() in text.lower()]
    
    if found_keywords:
        return f"Pattern matches {category} incidents based on keywords: {', '.join(found_keywords[:3])}"
    else:
        return f"Model classified as {category} with {confidence:.1%} confidence"


async def escalate_to_mastra(
    incident_data: Dict,
    ml_result: Dict,
    uncertainty: UncertaintyMetrics
) -> Optional[Dict]:
    """
    Escalate incident to MastraAI backend for complex multi-agent workflow
    
    Args:
        incident_data: Original incident (title, description, source)
        ml_result: ML classification result
        uncertainty: Uncertainty metrics from ML model
    
    Returns:
        MastraAI workflow result or None if failed
    """
    try:
        logger.info("🚀 Escalating to MastraAI backend for complex analysis...")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Call MastraAI backend workflow endpoint
            response = await client.post(
                f"{MASTRA_BACKEND_URL}/api/workflows/classify",
                json={
                    "incident": incident_data,
                    "ml_classification": {
                        "category": ml_result["category"],
                        "confidence": ml_result["confidence"],
                        "severity": ml_result["severity"],
                        "probabilities": ml_result["probabilities"],
                        "uncertainty": {
                            "mc_dropout": uncertainty.mc_dropout,
                            "margin": uncertainty.margin,
                            "entropy": uncertainty.entropy,
                            "should_escalate": uncertainty.should_escalate
                        }
                    },
                    "timestamp": datetime.utcnow().isoformat(),
                    "escalation_reason": "high_uncertainty" if uncertainty.should_escalate else "low_confidence"
                }
            )
            
            if response.status_code == 200:
                mastra_result = response.json()
                logger.info(f"✅ MastraAI workflow completed: {mastra_result.get('workflow_id')}")
                return mastra_result
            else:
                logger.warning(f"⚠️ MastraAI backend returned {response.status_code}: {response.text}")
                return None
                
    except httpx.TimeoutException:
        logger.error("❌ MastraAI backend timeout (30s)")
        return None
    except httpx.ConnectError:
        logger.warning(f"⚠️ MastraAI backend not available at {MASTRA_BACKEND_URL}")
        return None
    except Exception as e:
        logger.error(f"❌ MastraAI escalation failed: {e}")
        return None


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {
        "service": "OpsPilot ML Service",
        "model": "Phi-3 LoRA v5",
        "status": "running" if model is not None else "model_loading_failed",
        "version": "2.0.0",
        "device": device
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "device": device
    }


@app.post("/classify", response_model=ClassifyResponse)
async def classify_incident(request: ClassifyRequest):
    """
    Classify incident using fine-tuned Phi-3 model + RAG enhancement
    
    Returns:
    - category: Database|Network|Security|Performance|Application|Infrastructure
    - severity: low|medium|high|critical
    - confidence: 0-1 score
    - uncertainty: Multiple uncertainty metrics
    """
    
    # Check if model is loaded
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check server logs."
        )
    
    try:
        # Step 1: Initial classification WITHOUT RAG (fast path)
        prompt_initial = f"""<|system|>You are an IT incident classification assistant. Analyze the incident and classify it into EXACTLY ONE category: Application, Infrastructure, Database, Performance, Security, or Network.<|end|>
<|user|>Classify this IT incident:
Title: {request.title}
Description: {request.description}<|end|>
<|assistant|>**Category:** """
        
        
        # Tokenize initial prompt
        inputs = tokenizer(
            prompt_initial,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # PRODUCTION INFERENCE - Get real probabilities from fine-tuned model
        with torch.no_grad():
            # Step 1: Forward pass to get logits (for TRUE confidence scores)
            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                use_cache=False,  # Fix for DynamicCache incompatibility
                return_dict=True
            )
            
            # Step 2: Extract probabilities for each category from next-token prediction
            last_token_logits = outputs.logits[0, -1, :]  # Shape: [vocab_size]
            vocab_probs = torch.softmax(last_token_logits, dim=-1)
            
            # Map category names to their token IDs and extract probabilities
            category_token_ids = {}
            category_probs_raw = {}
            
            for cat in CATEGORY_NAMES:
                # Get first token of category name (e.g., "Database" → token ID)
                cat_tokens = tokenizer.encode(cat, add_special_tokens=False)
                if not cat_tokens:
                    logger.warning(f"Failed to tokenize category: {cat}")
                    category_probs_raw[cat] = 1e-10  # Minimal probability
                    continue
                
                cat_token_id = cat_tokens[0]
                category_token_ids[cat] = cat_token_id
                category_probs_raw[cat] = vocab_probs[cat_token_id].item()
            
            # Normalize to sum to 1.0 (proper probability distribution)
            total_prob = sum(category_probs_raw.values())
            if total_prob > 0:
                category_probs = {k: v/total_prob for k, v in category_probs_raw.items()}
            else:
                # Emergency fallback (should never happen)
                logger.error("All category probabilities are zero! Using uniform distribution.")
                category_probs = {k: 1.0/len(CATEGORY_NAMES) for k in CATEGORY_NAMES}
        
        # Get initial prediction
        predicted_category = max(category_probs, key=category_probs.get)
        confidence = category_probs[predicted_category]
        
        # Calculate uncertainty metrics to decide if RAG is needed
        probs_tensor = torch.tensor([category_probs[name] for name in CATEGORY_NAMES])
        uncertainty = calculate_uncertainty(probs_tensor)
        
        # RAG DECISION LOGIC: Use RAG only if confidence is low or uncertainty is high
        RAG_CONFIDENCE_THRESHOLD = 0.85
        rag_context = ""
        rag_used = False
        
        if rag_retriever is not None and (
            confidence < RAG_CONFIDENCE_THRESHOLD or 
            uncertainty.should_escalate
        ):
            logger.info(f"⚠️ Low confidence ({confidence:.2%}) or high uncertainty - invoking RAG")
            try:
                # Retrieve relevant context from knowledge base
                query = f"{request.title} {request.description}"
                rag_response = rag_retriever.retrieve(query, top_k=3)
                
                # Format RAG context
                if rag_response and rag_response.get("results"):
                    results = rag_response["results"]
                    rag_context = "\n\nRelevant Knowledge:\n"
                    for i, doc in enumerate(results, 1):
                        text = doc.get('text', '')
                        rag_context += f"- {text[:200]}...\n"
                    
                    logger.info(f"✅ RAG: Retrieved {len(results)} relevant documents")
                    
                    # Re-run inference with RAG context
                    prompt_with_rag = f"""<|system|>You are an IT incident classification assistant. Analyze the incident and classify it into EXACTLY ONE category: Application, Infrastructure, Database, Performance, Security, or Network.<|end|>
<|user|>Classify this IT incident:
{rag_context}
Title: {request.title}
Description: {request.description}<|end|>
<|assistant|>**Category:** """
                    
                    # Re-tokenize with RAG context
                    inputs_rag = tokenizer(
                        prompt_with_rag,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512
                    )
                    inputs_rag = {k: v.to(device) for k, v in inputs_rag.items()}
                    
                    # Re-run inference
                    with torch.no_grad():
                        outputs_rag = model(
                            input_ids=inputs_rag['input_ids'],
                            attention_mask=inputs_rag['attention_mask'],
                            use_cache=False,
                            return_dict=True
                        )
                        
                        last_token_logits_rag = outputs_rag.logits[0, -1, :]
                        vocab_probs_rag = torch.softmax(last_token_logits_rag, dim=-1)
                        
                        category_probs_rag_raw = {}
                        for cat in CATEGORY_NAMES:
                            cat_tokens = tokenizer.encode(cat, add_special_tokens=False)
                            if not cat_tokens:
                                category_probs_rag_raw[cat] = 1e-10
                                continue
                            cat_token_id = cat_tokens[0]
                            category_probs_rag_raw[cat] = vocab_probs_rag[cat_token_id].item()
                        
                        total_prob_rag = sum(category_probs_rag_raw.values())
                        if total_prob_rag > 0:
                            category_probs = {k: v/total_prob_rag for k, v in category_probs_rag_raw.items()}
                        
                        # Update prediction with RAG-enhanced probabilities
                        predicted_category = max(category_probs, key=category_probs.get)
                        confidence = category_probs[predicted_category]
                        
                        # Recalculate uncertainty
                        probs_tensor = torch.tensor([category_probs[name] for name in CATEGORY_NAMES])
                        uncertainty = calculate_uncertainty(probs_tensor)
                    
                    rag_used = True
                    
            except Exception as rag_error:
                logger.warning(f"⚠️ RAG retrieval failed: {rag_error}")
                # Continue with original prediction
        else:
            logger.info(f"✅ High confidence ({confidence:.2%}) - skipping RAG (fast path)")
        
        # Log probabilities for debugging
        logger.info(f"Raw probabilities: {[(k, f'{v:.4f}') for k, v in sorted(category_probs.items(), key=lambda x: x[1], reverse=True)[:3]]}")
        
        # Generate text for validation (optional, but good for monitoring)
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=10,  # Just category name
                do_sample=False,    # Deterministic (greedy decoding)
                use_cache=False,    # Fix for DynamicCache
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Parse generated text (for validation/logging)
        response_text = tokenizer.decode(
            output_ids[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        # Validate that generated text matches prediction (sanity check)
        text_matches = predicted_category.lower() in response_text.lower()[:30]
        if not text_matches:
            logger.warning(f"Probability says '{predicted_category}' ({confidence:.2%}) but text says '{response_text[:50]}' - using probabilities")
        
        # Determine severity
        severity = get_severity(confidence, predicted_category)
        
        # Generate reasoning with actual probabilities and RAG context
        top_3 = sorted(category_probs.items(), key=lambda x: x[1], reverse=True)[:3]
        reasoning = f"Top predictions: {', '.join([f'{cat} ({prob:.1%})' for cat, prob in top_3])}. Model response: {response_text[:100]}"
        
        # Add RAG context to reasoning if used
        if rag_used:
            reasoning += " | RAG: Enhanced with relevant knowledge base documents"
        
        # All probabilities (already normalized)
        all_probs = category_probs
        
        logger.info(f"✅ Classified: {predicted_category} ({confidence:.2%}) | Top-3: {[(c, f'{p:.1%}') for c, p in top_3]}")
        
        # MASTRA ESCALATION: Check if we should escalate to MastraAI backend
        # Escalate if: uncertainty is high OR confidence is very low (even after RAG)
        MASTRA_ESCALATION_THRESHOLD = 0.70
        should_escalate_to_mastra = (
            uncertainty.should_escalate or 
            confidence < MASTRA_ESCALATION_THRESHOLD
        )
        
        if should_escalate_to_mastra:
            logger.warning(f"⚠️ High uncertainty or low confidence ({confidence:.2%}) - escalating to MastraAI")
            
            # Prepare incident data for MastraAI
            incident_data = {
                "title": request.title,
                "description": request.description,
                "source": request.source,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            ml_result = {
                "category": predicted_category,
                "confidence": confidence,
                "severity": severity,
                "probabilities": all_probs
            }
            
            # Escalate to MastraAI (async call)
            mastra_result = await escalate_to_mastra(incident_data, ml_result, uncertainty)
            
            if mastra_result:
                # MastraAI provided enhanced analysis
                logger.info("✅ MastraAI workflow completed successfully")
                
                return MastraEscalationResponse(
                    category=mastra_result.get("category", predicted_category),
                    severity=mastra_result.get("severity", severity),
                    confidence=mastra_result.get("confidence", confidence),
                    reasoning=mastra_result.get("reasoning", reasoning),
                    uncertainty=uncertainty,
                    all_probabilities=mastra_result.get("probabilities", all_probs),
                    escalated_to_mastra=True,
                    mastra_workflow_id=mastra_result.get("workflow_id", "unknown"),
                    mastra_recommendations=mastra_result.get("recommendations"),
                    mastra_analysis=mastra_result.get("analysis")
                )
            else:
                # MastraAI failed - return ML result with escalation flag
                logger.warning("⚠️ MastraAI escalation failed - returning ML result")
                return ClassifyResponse(
                    category=predicted_category,
                    severity=severity,
                    confidence=confidence,
                    reasoning=reasoning + " | ⚠️ MastraAI escalation attempted but failed",
                    uncertainty=uncertainty,
                    all_probabilities=all_probs,
                    escalated_to_mastra=False
                )
        else:
            # High confidence - no escalation needed
            return ClassifyResponse(
                category=predicted_category,
                severity=severity,
                confidence=confidence,
                reasoning=reasoning,
                uncertainty=uncertainty,
                all_probabilities=all_probs,
                escalated_to_mastra=False
            )
        
    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info")
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Check if LoRA adapters are loaded
    import os
    using_lora = os.path.exists(LORA_PATH)
    
    return {
        "lora_path": LORA_PATH if using_lora else None,
        "base_model_path": BASE_MODEL_PATH,
        "model_version": "v6 (fine-tuned)" if using_lora else "base",
        "model_type": "Phi-3 with v6 LoRA adapters (SEQ_CLS)" if using_lora else "Phi-3 base",
        "expected_accuracy": "95%+" if using_lora else "83%",
        "num_labels": 6,
        "categories": CATEGORY_NAMES,
        "device": device,
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "rag_enabled": rag_retriever is not None,
        "mastra_backend_url": MASTRA_BACKEND_URL,
        "mastra_escalation_enabled": True,
        "mastra_confidence_threshold": 0.70,
        "rag_confidence_threshold": 0.85
    }


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", "8000"))
    
    uvicorn.run(
        "server_production:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # Disable in production
        log_level="info"
    )
