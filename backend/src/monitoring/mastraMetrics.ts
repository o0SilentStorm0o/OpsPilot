import promClient from 'prom-client';

/**
 * MASTRA + PHI-3 SPECIFIC METRICS
 * 
 * Tracks performance and effectiveness of the hybrid AI system
 */

// ============================================================================
// PHI-3 MODEL METRICS
// ============================================================================

/**
 * Phi-3 inference latency (should be ~50-60ms)
 */
export const phi3InferenceLatency = new promClient.Histogram({
  name: 'phi3_inference_latency_seconds',
  help: 'Latency of Phi-3 model inference in seconds',
  labelNames: ['endpoint', 'status'],
  buckets: [0.01, 0.05, 0.1, 0.2, 0.5, 1, 2], // 10ms to 2s
});

/**
 * Phi-3 classification confidence distribution
 */
export const phi3Confidence = new promClient.Histogram({
  name: 'phi3_classification_confidence',
  help: 'Confidence scores from Phi-3 classification (0-1)',
  labelNames: ['category', 'severity'],
  buckets: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
});

/**
 * Phi-3 uncertainty detector scores
 */
export const phi3UncertaintyScores = new promClient.Gauge({
  name: 'phi3_uncertainty_detector_score',
  help: 'Scores from individual uncertainty detectors',
  labelNames: ['detector'], // mc_dropout, margin, entropy, energy, knn_distance
});

/**
 * Counter for Phi-3 classifications by category
 */
export const phi3ClassificationsByCategory = new promClient.Counter({
  name: 'phi3_classifications_total',
  help: 'Total number of Phi-3 classifications by category',
  labelNames: ['category', 'severity'],
});

// ============================================================================
// MASTRA WORKFLOW METRICS
// ============================================================================

/**
 * Mastra workflow execution duration
 */
export const mastraWorkflowDuration = new promClient.Histogram({
  name: 'mastra_workflow_duration_seconds',
  help: 'Duration of complete Mastra workflow execution',
  labelNames: ['workflow_type', 'decision'], // workflow_type: single-agent|multi-agent, decision: auto-resolved|human-escalated
  buckets: [0.5, 1, 2, 5, 10, 20, 30, 60],
});

/**
 * Mastra tool execution latency
 */
export const mastraToolLatency = new promClient.Histogram({
  name: 'mastra_tool_execution_latency_seconds',
  help: 'Latency of individual Mastra tool execution',
  labelNames: ['tool_id', 'status'],
  buckets: [0.1, 0.5, 1, 2, 5, 10],
});

/**
 * Mastra token usage per workflow
 */
export const mastraTokenUsage = new promClient.Counter({
  name: 'mastra_tokens_used_total',
  help: 'Total tokens used by Mastra workflows',
  labelNames: ['model', 'workflow_type'],
});

/**
 * Mastra cost per workflow (in USD)
 */
export const mastraWorkflowCost = new promClient.Histogram({
  name: 'mastra_workflow_cost_usd',
  help: 'Cost of Mastra workflow in USD',
  labelNames: ['workflow_type', 'model'],
  buckets: [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
});

// ============================================================================
// DECISION ROUTING METRICS
// ============================================================================

/**
 * Auto-resolve vs human-escalate decisions
 */
export const incidentDecisions = new promClient.Counter({
  name: 'incident_decisions_total',
  help: 'Total number of incident routing decisions',
  labelNames: ['decision', 'category', 'severity'], // decision: auto-resolved|human-escalated
});

/**
 * Auto-resolve success rate (gauge updated periodically)
 */
export const autoResolveSuccessRate = new promClient.Gauge({
  name: 'auto_resolve_success_rate',
  help: 'Success rate of auto-resolved incidents (0-1)',
  labelNames: ['category'],
});

/**
 * Human escalation rate by confidence threshold
 */
export const humanEscalationRate = new promClient.Gauge({
  name: 'human_escalation_rate',
  help: 'Rate of human escalations (0-1)',
});

/**
 * Uncertainty detector triggers (how often each detector flags for escalation)
 */
export const uncertaintyDetectorTriggers = new promClient.Counter({
  name: 'uncertainty_detector_triggers_total',
  help: 'Number of times each uncertainty detector triggered escalation',
  labelNames: ['detector', 'category'],
});

// ============================================================================
// BUSINESS METRICS
// ============================================================================

/**
 * Time saved per incident (in seconds)
 */
export const timeSavedPerIncident = new promClient.Histogram({
  name: 'incident_time_saved_seconds',
  help: 'Time saved per incident compared to manual processing',
  labelNames: ['decision'],
  buckets: [60, 300, 600, 1200, 1800, 3600], // 1 min to 1 hour
});

/**
 * Total incidents processed
 */
export const incidentsProcessed = new promClient.Counter({
  name: 'incidents_processed_total',
  help: 'Total number of incidents processed by the system',
  labelNames: ['source', 'category'],
});

/**
 * SRE time freed (in hours per day)
 */
export const sreTimeFreeGauge = new promClient.Gauge({
  name: 'sre_time_freed_hours_daily',
  help: 'Estimated SRE hours freed per day by automation',
});

// ============================================================================
// ACCURACY & QUALITY METRICS
// ============================================================================

/**
 * Effective accuracy (Phi-3 + HITL)
 */
export const effectiveAccuracy = new promClient.Gauge({
  name: 'effective_classification_accuracy',
  help: 'Effective accuracy combining Phi-3 and HITL (0-1)',
});

/**
 * Confusion matrix elements (for tracking classification quality)
 */
export const confusionMatrixElements = new promClient.Counter({
  name: 'classification_confusion_matrix_total',
  help: 'Confusion matrix elements for classification quality',
  labelNames: ['true_category', 'predicted_category'],
});

/**
 * False positive/negative rates by category
 */
export const classificationErrorRate = new promClient.Gauge({
  name: 'classification_error_rate',
  help: 'Error rate by category (false positives + false negatives)',
  labelNames: ['category', 'error_type'], // error_type: false_positive|false_negative
});

// ============================================================================
// EXPORT ALL METRICS
// ============================================================================

export const mastraMetrics = {
  // Phi-3
  phi3InferenceLatency,
  phi3Confidence,
  phi3UncertaintyScores,
  phi3ClassificationsByCategory,
  
  // Mastra
  mastraWorkflowDuration,
  mastraToolLatency,
  mastraTokenUsage,
  mastraWorkflowCost,
  
  // Routing
  incidentDecisions,
  autoResolveSuccessRate,
  humanEscalationRate,
  uncertaintyDetectorTriggers,
  
  // Business
  timeSavedPerIncident,
  incidentsProcessed,
  sreTimeFreeGauge,
  
  // Quality
  effectiveAccuracy,
  confusionMatrixElements,
  classificationErrorRate,
};

/**
 * Register all Mastra metrics with Prometheus
 */
export function registerMastraMetrics(register: promClient.Registry) {
  Object.values(mastraMetrics).forEach(metric => {
    register.registerMetric(metric as any); // Type widening needed for metric registration
  });
}
