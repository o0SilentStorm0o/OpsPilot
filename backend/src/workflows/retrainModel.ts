import { logger } from '../utils/logger';
import { modelDriftScore } from '../monitoring/metrics';

export interface FeedbackData {
  incidentId: string;
  classification?: {
    predicted: string;
    actual: string;
    correct: boolean;
  };
  remediation?: {
    planId: string;
    successful: boolean;
    userFeedback: string;
  };
  timestamp: string;
}

export interface RetrainingResult {
  triggered: boolean;
  reason: string;
  feedbackCount: number;
  scheduledTime?: string;
  estimatedDuration?: string;
  status: 'scheduled' | 'in-progress' | 'completed' | 'not-needed';
}

// In-memory storage for feedback (in production, use database)
const feedbackStore: FeedbackData[] = [];
const RETRAINING_THRESHOLD = 50; // Retrain after 50 feedback items

/**
 * Workflow for model retraining based on collected feedback
 * This integrates with the Python ML service for actual retraining
 */
export async function retrainModel(
  feedbackData?: FeedbackData
): Promise<RetrainingResult> {
  logger.info('Model retraining workflow initiated');

  try {
    // Store new feedback if provided
    if (feedbackData) {
      feedbackStore.push(feedbackData);
      logger.info(`Stored feedback for incident ${feedbackData.incidentId}. Total feedback: ${feedbackStore.length}`);
    }

    // Calculate model drift score
    const driftScore = calculateDriftScore();
    modelDriftScore.set(driftScore);

    // Determine if retraining is needed
    const needsRetraining = shouldRetrain(driftScore);

    if (!needsRetraining) {
      return {
        triggered: false,
        reason: `Insufficient data or drift. Current: ${feedbackStore.length} samples, drift: ${driftScore.toFixed(2)}`,
        feedbackCount: feedbackStore.length,
        status: 'not-needed'
      };
    }

    // In production, this would trigger the ML service
    logger.info('Retraining conditions met. Scheduling model retraining...');
    
    // Schedule retraining job
    const scheduledTime = new Date(Date.now() + 5 * 60 * 1000).toISOString(); // 5 minutes from now
    
    // Trigger ML service (this would be an actual API call in production)
    await triggerMLServiceRetraining(feedbackStore);

    return {
      triggered: true,
      reason: `Retraining triggered: ${feedbackStore.length} feedback samples collected, drift score: ${driftScore.toFixed(2)}`,
      feedbackCount: feedbackStore.length,
      scheduledTime,
      estimatedDuration: '30-60 minutes',
      status: 'scheduled'
    };

  } catch (error) {
    logger.error('Error in retrainModel workflow:', error);
    throw error;
  }
}

/**
 * Calculate model drift score based on feedback accuracy
 */
function calculateDriftScore(): number {
  if (feedbackStore.length === 0) {
    return 0;
  }

  // Count incorrect classifications
  const classificationFeedback = feedbackStore.filter(f => f.classification);
  if (classificationFeedback.length === 0) {
    return 0;
  }

  const incorrectCount = classificationFeedback.filter(f => !f.classification?.correct).length;
  const errorRate = incorrectCount / classificationFeedback.length;

  // Drift score is error rate (0-1)
  return errorRate;
}

/**
 * Determine if model should be retrained
 */
function shouldRetrain(driftScore: number): boolean {
  // Retrain if:
  // 1. We have enough feedback samples
  const hasEnoughSamples = feedbackStore.length >= RETRAINING_THRESHOLD;
  
  // 2. Model drift exceeds threshold (>20% error rate)
  const hasSignificantDrift = driftScore > 0.2;

  return hasEnoughSamples && hasSignificantDrift;
}

/**
 * Trigger the ML service to perform retraining
 * In production, this would make an HTTP request to the Python ML service
 */
async function triggerMLServiceRetraining(feedback: FeedbackData[]): Promise<void> {
  logger.info('Triggering ML service retraining...');
  
  // This is a placeholder - in production, you would:
  // 1. Send feedback data to ML service
  // 2. ML service would fine-tune the model
  // 3. Deploy updated model
  // 4. Clear feedback store after successful retraining
  
  const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://ml-service:8000';
  
  try {
    // Simulated API call
    logger.info(`Would send ${feedback.length} feedback samples to ${ML_SERVICE_URL}/retrain`);
    
    // In real implementation:
    // const response = await fetch(`${ML_SERVICE_URL}/retrain`, {
    //   method: 'POST',
    //   headers: { 'Content-Type': 'application/json' },
    //   body: JSON.stringify({ feedback })
    // });
    
    logger.info('Retraining job submitted successfully');
  } catch (error) {
    logger.error('Failed to trigger ML service retraining:', error);
    throw error;
  }
}

/**
 * Get current feedback statistics
 */
export function getFeedbackStats(): {
  totalFeedback: number;
  classificationAccuracy: number;
  remediationSuccessRate: number;
  driftScore: number;
} {
  const classificationFeedback = feedbackStore.filter(f => f.classification);
  const remediationFeedback = feedbackStore.filter(f => f.remediation);

  const classificationCorrect = classificationFeedback.filter(f => f.classification?.correct).length;
  const remediationSuccessful = remediationFeedback.filter(f => f.remediation?.successful).length;

  return {
    totalFeedback: feedbackStore.length,
    classificationAccuracy: classificationFeedback.length > 0 
      ? classificationCorrect / classificationFeedback.length 
      : 0,
    remediationSuccessRate: remediationFeedback.length > 0
      ? remediationSuccessful / remediationFeedback.length
      : 0,
    driftScore: calculateDriftScore()
  };
}
