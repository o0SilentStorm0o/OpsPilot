import { logger } from '../utils/logger';
import { 
  phi3InferenceLatency,
  phi3Confidence,
  phi3UncertaintyScores,
  phi3ClassificationsByCategory,
  incidentDecisions
} from '../monitoring/mastraMetrics';

interface MLServiceResponse {
  category: string;
  confidence: number;
  top_k: Array<{ label: string; score: number }>;
  uncertainty: {
    mc_dropout: number;
    margin: number;
    entropy: number;
    energy: number;
    should_escalate: boolean;
  };
}

export interface Incident {
  id?: string;
  title: string;
  description: string;
  source: string;
  timestamp: string;
  metadata?: Record<string, any>;
}

export interface ClassificationResult {
  category: string;
  subcategory: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  priority: number;
  affectedSystems: string[];
  confidence: number;
  reasoning: string;
}

/**
 * Classifies IT incidents using LLM into appropriate categories
 * Categories: Network, Database, Application, Security, Hardware, Performance
 */
export async function classifyIncident(incident: Incident): Promise<ClassificationResult> {
  logger.info(`Classifying incident: ${incident.title}`);

  try {
    const startTime = Date.now();
    
    // Call Phi-3 ML service
    const mlResponse = await fetch('http://localhost:8000/classify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        title: incident.title,
        description: incident.description,
        source: incident.source || 'unknown'
      })
    });

    if (!mlResponse.ok) {
      throw new Error(`ML service error: ${mlResponse.status} ${mlResponse.statusText}`);
    }

    const mlResult = await mlResponse.json() as MLServiceResponse;
    const latency = (Date.now() - startTime) / 1000;

    // Record Phi-3 metrics
    phi3InferenceLatency.observe({ endpoint: 'classify', status: 'success' }, latency);
    
    // Map ML service response to ClassificationResult
    const severityMap: Record<string, 'low' | 'medium' | 'high' | 'critical'> = {
      'Database': 'high',
      'Network': 'medium',
      'Security': 'critical',
      'Performance': 'high',
      'Application': 'medium',
      'Infrastructure': 'high'
    };

    const severity = severityMap[mlResult.category] || 'medium';
    
    // Record confidence
    phi3Confidence.observe(
      { category: mlResult.category, severity },
      mlResult.confidence
    );
    
    // Record classification count
    phi3ClassificationsByCategory.inc({ category: mlResult.category, severity });
    
    // Record uncertainty scores
    phi3UncertaintyScores.set({ detector: 'mc_dropout' }, mlResult.uncertainty.mc_dropout);
    phi3UncertaintyScores.set({ detector: 'margin' }, mlResult.uncertainty.margin);
    phi3UncertaintyScores.set({ detector: 'entropy' }, mlResult.uncertainty.entropy);
    phi3UncertaintyScores.set({ detector: 'energy' }, mlResult.uncertainty.energy);
    
    // Record routing decision
    const decision = mlResult.uncertainty.should_escalate ? 'human-escalated' : 'auto-resolved';
    incidentDecisions.inc({ decision, category: mlResult.category });

    const classification: ClassificationResult = {
      category: mlResult.category,
      subcategory: mlResult.category, // ML service doesn't provide subcategory yet
      severity,
      priority: mlResult.confidence > 0.8 ? 1 : mlResult.confidence > 0.6 ? 2 : 3,
      affectedSystems: [incident.source],
      confidence: mlResult.confidence,
      reasoning: `Classified by Phi-3 model. Uncertainty: ${mlResult.uncertainty.should_escalate ? 'HIGH (escalate to human)' : 'LOW (auto-resolve)'}`
    };

    logger.info(`Incident classified as ${classification.category} with ${(classification.confidence * 100).toFixed(1)}% confidence`);
    return classification;

  } catch (error) {
    logger.error('Error in classifyIncident workflow:', error);
    throw error;
  }
}

/**
 * Validates classification results against known patterns
 */
export function validateClassification(
  incident: Incident,
  classification: ClassificationResult
): boolean {
  // Simple validation rules
  const description = incident.description.toLowerCase();
  
  // Network-related keywords
  if (description.includes('network') || description.includes('connection')) {
    return classification.category === 'Network';
  }
  
  // Database-related keywords
  if (description.includes('database') || description.includes('sql')) {
    return classification.category === 'Database';
  }
  
  // Security-related keywords
  if (description.includes('security') || description.includes('unauthorized')) {
    return classification.category === 'Security';
  }

  // If no specific match, accept any classification
  return true;
}
