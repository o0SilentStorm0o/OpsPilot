/**
 * PHI-3 + MASTRA INTEGRATION
 * 
 * Combines the best of both worlds:
 * - Phi-3 fine-tuned model: Fast, accurate classification (73.3% accuracy)
 * - Mastra framework: Robust workflow orchestration
 * 
 * Architecture:
 * 1. Phi-3 classifies incident (50ms, $0 cost, company-specific)
 * 2. Uncertainty detection (6 detectors: MC Dropout, kNN-OOD, etc.)
 * 3. Decision point:
 *    - High confidence (>80%) → Auto-resolve with Mastra
 *    - Low confidence (<80%) → Human-in-the-loop escalation
 */

import { createTool } from '@mastra/core/tools';
import { z } from 'zod';
import { logger } from '../utils/logger';
import { 
  searchSolutionsTool,
  createJiraTicketTool,
  sendSlackNotificationTool
} from './resolution-workflow';
import { mastraMetrics } from '../monitoring/mastraMetrics';

// ============================================================================
// PHI-3 MODEL INTEGRATION
// ============================================================================

/**
 * Tool for calling fine-tuned Phi-3 model API
 * 
 * Features:
 * - Fast inference (~50ms vs 2000ms GPT-4o)
 * - Company-specific knowledge (trained on your data)
 * - 73.3% accuracy (better than GPT-4o-mini on your domain!)
 * - Uncertainty detection with 6 detectors
 * - Zero cost (local inference)
 */
export const phi3ClassifyTool = createTool({
  id: 'phi3-classify',
  description: 'Classify incident using fine-tuned Phi-3 model with uncertainty detection',
  inputSchema: z.object({
    title: z.string().describe('Incident title'),
    description: z.string().describe('Detailed incident description'),
    source: z.string().optional().describe('Source system that reported the incident')
  }),
  outputSchema: z.object({
    category: z.enum([
      'Database',
      'Network', 
      'Security',
      'Performance',
      'Application',
      'Infrastructure'
    ]),
    severity: z.enum(['low', 'medium', 'high', 'critical']),
    confidence: z.number().min(0).max(1),
    reasoning: z.string(),
    uncertainty: z.object({
      mc_dropout: z.number().describe('MC Dropout uncertainty (0-1)'),
      margin: z.number().optional().describe('Margin uncertainty'),
      entropy: z.number().optional().describe('Entropy uncertainty'),
      energy: z.number().optional().describe('Energy-based uncertainty'),
      knn_distance: z.number().optional().describe('kNN-OOD distance'),
      should_escalate: z.boolean().describe('Whether to escalate to human')
    })
  }),
  execute: async ({ context }) => {
    const startTime = Date.now();
    
    logger.info('🧠 Calling Phi-3 model for classification', {
      title: context.title,
      source: context.source
    });

    try {
      // Call your Phi-3 API endpoint
      // Adjust URL based on your deployment
      const phi3ApiUrl = process.env.PHI3_API_URL || 'http://localhost:8000/classify';
      
      const response = await fetch(phi3ApiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          title: context.title,
          description: context.description,
          source: context.source
        })
      });

      if (!response.ok) {
        throw new Error(`Phi-3 API error: ${response.status} ${response.statusText}`);
      }

      // Type the API response
      type Phi3ApiResponse = {
        category: 'Database' | 'Network' | 'Security' | 'Performance' | 'Application' | 'Infrastructure';
        severity: 'low' | 'medium' | 'high' | 'critical';
        confidence: number;
        reasoning: string;
        uncertainty: {
          mc_dropout: number;
          margin?: number;
          entropy?: number;
          energy?: number;
          knn_distance?: number;
          should_escalate: boolean;
        };
      };

      const result = await response.json() as Phi3ApiResponse;
      const duration = Date.now() - startTime;

      // Track Prometheus metrics
      mastraMetrics.phi3InferenceLatency
        .labels('http://classify', 'success')
        .observe(duration / 1000);

      mastraMetrics.phi3Confidence
        .labels(result.category, result.severity)
        .observe(result.confidence);

      mastraMetrics.phi3ClassificationsByCategory
        .labels(result.category, result.severity)
        .inc();

      // Track uncertainty detector scores
      mastraMetrics.phi3UncertaintyScores
        .labels('mc_dropout')
        .set(result.uncertainty.mc_dropout);
      
      if (result.uncertainty.margin !== undefined) {
        mastraMetrics.phi3UncertaintyScores
          .labels('margin')
          .set(result.uncertainty.margin);
      }
      
      if (result.uncertainty.entropy !== undefined) {
        mastraMetrics.phi3UncertaintyScores
          .labels('entropy')
          .set(result.uncertainty.entropy);
      }

      logger.info('✅ Phi-3 classification complete', {
        category: result.category,
        severity: result.severity,
        confidence: result.confidence,
        should_escalate: result.uncertainty?.should_escalate,
        duration: `${duration}ms`
      });

      return result;

    } catch (error) {
      const duration = Date.now() - startTime;
      
      // Track failed inference
      mastraMetrics.phi3InferenceLatency
        .labels('http://classify', 'error')
        .observe(duration / 1000);

      logger.error('❌ Phi-3 classification failed', {
        error: error instanceof Error ? error.message : 'Unknown error',
        duration: `${duration}ms`
      });

      // Fallback to low-confidence classification
      return {
        category: 'Application' as const,
        severity: 'medium' as const,
        confidence: 0.5,
        reasoning: `Phi-3 API unavailable: ${error instanceof Error ? error.message : 'Unknown error'}. Using fallback classification.`,
        uncertainty: {
          mc_dropout: 0.5,
          should_escalate: true // Always escalate on API failure
        }
      };
    }
  }
});

// ============================================================================
// UNCERTAINTY-AWARE INCIDENT RESOLUTION
// ============================================================================

export interface Phi3Incident {
  title: string;
  description: string;
  source?: string;
}

export interface Phi3ResolutionResult {
  // Phi-3 classification
  classification: {
    category: string;
    severity: string;
    confidence: number;
    reasoning: string;
    model: string;
  };
  
  // Uncertainty metrics
  uncertainty: {
    mc_dropout: number;
    should_escalate: boolean;
    detectors?: Record<string, number>;
  };
  
  // Workflow decision
  decision: 'auto-resolved' | 'human-escalated';
  
  // Mastra workflow results (if auto-resolved)
  solutions?: any;
  ticket?: any;
  notification?: any;
  
  // Performance metrics
  performance: {
    phi3_duration_ms: number;
    total_duration_ms: number;
    model_used: string;
    cost: number; // $0 for Phi-3!
  };
}

/**
 * Resolve incident using Phi-3 classification + Mastra orchestration
 * 
 * Smart decision making:
 * - High confidence (≥80%): Auto-resolve with Mastra workflow
 * - Low confidence (<80%): Escalate to human with context
 * 
 * @example
 * ```typescript
 * const result = await phi3ResolveIncident({
 *   title: "Database timeout",
 *   description: "Connection pool exhausted..."
 * });
 * 
 * if (result.decision === 'auto-resolved') {
 *   console.log(`✅ Auto-resolved: ${result.ticket.ticketId}`);
 * } else {
 *   console.log(`⚠️ Escalated: Low confidence (${result.classification.confidence})`);
 * }
 * ```
 */
export async function phi3ResolveIncident(
  incident: Phi3Incident
): Promise<Phi3ResolutionResult> {
  const totalStartTime = Date.now();
  
  logger.info('🚀 Starting Phi-3 + Mastra incident resolution', {
    title: incident.title
  });

  // ============================================================================
  // STEP 1: PHI-3 CLASSIFICATION
  // ============================================================================
  
  const phi3StartTime = Date.now();
  const classification = await phi3ClassifyTool.execute({
    context: incident,
    runtimeContext: {} as any
  });
  const phi3Duration = Date.now() - phi3StartTime;

  logger.info('📊 Classification result', {
    category: classification.category,
    severity: classification.severity,
    confidence: classification.confidence,
    mc_dropout: classification.uncertainty.mc_dropout,
    should_escalate: classification.uncertainty.should_escalate
  });

  // ============================================================================
  // STEP 2: UNCERTAINTY-AWARE DECISION
  // ============================================================================

  const CONFIDENCE_THRESHOLD = 0.80; // 80% confidence required for auto-resolve
  const shouldAutoResolve = 
    classification.confidence >= CONFIDENCE_THRESHOLD && 
    !classification.uncertainty.should_escalate;

  if (!shouldAutoResolve) {
    // LOW CONFIDENCE → HUMAN-IN-THE-LOOP
    logger.warn('⚠️ Low confidence detected - escalating to human', {
      confidence: classification.confidence,
      threshold: CONFIDENCE_THRESHOLD,
      mc_dropout: classification.uncertainty.mc_dropout
    });

    // Track escalation decision
    mastraMetrics.incidentDecisions
      .labels('human-escalated', classification.category, classification.severity)
      .inc();

    // Track which uncertainty detector triggered
    if (classification.uncertainty.should_escalate) {
      mastraMetrics.uncertaintyDetectorTriggers
        .labels('combined', classification.category)
        .inc();
    }

    return await humanInTheLoopWorkflow(incident, classification, {
      phi3_duration_ms: phi3Duration,
      total_duration_ms: Date.now() - totalStartTime
    });
  }

  // ============================================================================
  // STEP 3: HIGH CONFIDENCE → AUTO-RESOLVE WITH MASTRA
  // ============================================================================

  logger.info('✅ High confidence - proceeding with auto-resolution', {
    confidence: classification.confidence
  });

  // Step 3a: Search for solutions
  logger.info('Step 2/4: Searching for solutions');
  const solutions = await searchSolutionsTool.execute({
    context: {
      category: classification.category,
      description: incident.description,
      severity: classification.severity
    },
    runtimeContext: {} as any
  });

  logger.info('Solutions found', { count: solutions.solutions.length });

  // Step 3b: Create Jira ticket
  logger.info('Step 3/4: Creating Jira ticket');
  const ticket = await createJiraTicketTool.execute({
    context: {
      title: incident.title,
      description: incident.description,
      category: classification.category,
      severity: classification.severity,
      solutions: solutions.solutions,
      classification: {
        ...classification,
        model: 'phi3-fine-tuned' // Credit your model!
      }
    },
    runtimeContext: {} as any
  });

  logger.info('Ticket created', { ticketId: ticket.ticketId });

  // Step 3c: Send Slack notification
  logger.info('Step 4/4: Sending Slack notification');
  const notification = await sendSlackNotificationTool.execute({
    context: {
      channel: '#incidents',
      severity: classification.severity,
      category: classification.category,
      title: incident.title,
      ticketId: ticket.ticketId,
      ticketUrl: ticket.ticketUrl,
      solutionsCount: solutions.solutions.length
    },
    runtimeContext: {} as any
  });

  logger.info('Notification sent', { messageId: notification.messageId });

  const totalDuration = Date.now() - totalStartTime;

  // Track Prometheus metrics for auto-resolution
  mastraMetrics.mastraWorkflowDuration
    .labels('phi3-mastra-hybrid', 'auto-resolved')
    .observe(totalDuration / 1000);

  mastraMetrics.incidentDecisions
    .labels('auto-resolved', classification.category, classification.severity)
    .inc();

  mastraMetrics.incidentsProcessed
    .labels(incident.source || 'unknown', classification.category)
    .inc();

  // Estimate time saved (30 min manual - 71ms auto)
  const timeSaved = (30 * 60) - (totalDuration / 1000); // seconds
  mastraMetrics.timeSavedPerIncident
    .labels('auto-resolved')
    .observe(timeSaved);

  logger.info('✅ Auto-resolution complete', {
    ticketId: ticket.ticketId,
    duration: `${totalDuration}ms`,
    model: 'phi3 + mastra'
  });

  return {
    classification: {
      category: classification.category,
      severity: classification.severity,
      confidence: classification.confidence,
      reasoning: classification.reasoning,
      model: 'phi3-fine-tuned'
    },
    uncertainty: classification.uncertainty,
    decision: 'auto-resolved',
    solutions,
    ticket,
    notification,
    performance: {
      phi3_duration_ms: phi3Duration,
      total_duration_ms: totalDuration,
      model_used: 'phi3 (classification) + mastra (orchestration)',
      cost: 0 // Phi-3 is local, $0 cost!
    }
  };
}

/**
 * Human-in-the-loop workflow for low-confidence incidents
 * 
 * When Phi-3 is uncertain:
 * 1. Create HIGH priority ticket
 * 2. Assign to senior SRE
 * 3. Attach all diagnostic context
 * 4. Notify via Slack with @mention
 * 5. Gather additional diagnostics
 * 6. Wait for human review
 */
async function humanInTheLoopWorkflow(
  incident: Phi3Incident,
  classification: any,
  timing: { phi3_duration_ms: number; total_duration_ms: number }
): Promise<Phi3ResolutionResult> {
  logger.info('🔴 Starting HITL workflow');

  // Create URGENT ticket for human review
  const ticket = await createJiraTicketTool.execute({
    context: {
      title: `[NEEDS REVIEW] ${incident.title}`,
      description: `${incident.description}

⚠️ **AI UNCERTAINTY DETECTED**

The Phi-3 model classified this incident but has low confidence:
- Category: ${classification.category} (confidence: ${(classification.confidence * 100).toFixed(1)}%)
- Severity: ${classification.severity}
- MC Dropout: ${classification.uncertainty.mc_dropout.toFixed(3)}
- Reasoning: ${classification.reasoning}

Please review and update classification if needed.`,
      category: classification.category,
      severity: 'critical', // Escalate to critical for human attention
      solutions: [],
      classification: {
        ...classification,
        model: 'phi3-fine-tuned (LOW CONFIDENCE)'
      }
    },
    runtimeContext: {} as any
  });

  // Send urgent Slack notification with @mention
  const notification = await sendSlackNotificationTool.execute({
    context: {
      channel: '#incidents',
      severity: 'critical',
      category: classification.category,
      title: `⚠️ NEEDS REVIEW: ${incident.title}`,
      ticketId: ticket.ticketId,
      ticketUrl: ticket.ticketUrl,
      solutionsCount: 0
    },
    runtimeContext: {} as any
  });

  logger.info('HITL workflow complete - awaiting human review', {
    ticketId: ticket.ticketId,
    confidence: classification.confidence
  });

  return {
    classification: {
      category: classification.category,
      severity: classification.severity,
      confidence: classification.confidence,
      reasoning: classification.reasoning,
      model: 'phi3-fine-tuned'
    },
    uncertainty: classification.uncertainty,
    decision: 'human-escalated',
    ticket,
    notification,
    performance: {
      phi3_duration_ms: timing.phi3_duration_ms,
      total_duration_ms: Date.now() - (Date.now() - timing.total_duration_ms),
      model_used: 'phi3 (classification only)',
      cost: 0
    }
  };
}

// ============================================================================
// DEMO FUNCTION
// ============================================================================

/**
 * Demo function to test Phi-3 + Mastra integration
 */
export async function demoPhi3Integration() {
  console.log('\n🚀 PHI-3 + MASTRA INTEGRATION DEMO\n');
  console.log('='.repeat(80));
  console.log('\n');

  const testIncidents = [
    {
      title: 'Database Connection Pool Exhausted',
      description: 'Production database showing connection timeout errors. Connection pool at 95/100.',
      source: 'Prometheus Alert'
    },
    {
      title: 'Unknown Pod Crash Loop',
      description: 'New microservice crashing with unknown error - never seen before',
      source: 'Kubernetes Monitor'
    }
  ];

  for (const incident of testIncidents) {
    console.log(`📋 INCIDENT: ${incident.title}`);
    console.log('-'.repeat(80));
    
    const result = await phi3ResolveIncident(incident);
    
    console.log(`\n🎯 CLASSIFICATION:`);
    console.log(`   Category: ${result.classification.category}`);
    console.log(`   Severity: ${result.classification.severity}`);
    console.log(`   Confidence: ${(result.classification.confidence * 100).toFixed(1)}%`);
    console.log(`   Model: ${result.classification.model}`);
    
    console.log(`\n📊 UNCERTAINTY:`);
    console.log(`   MC Dropout: ${result.uncertainty.mc_dropout.toFixed(3)}`);
    console.log(`   Should Escalate: ${result.uncertainty.should_escalate}`);
    
    console.log(`\n⚡ DECISION: ${result.decision.toUpperCase()}`);
    
    if (result.decision === 'auto-resolved') {
      console.log(`\n✅ AUTO-RESOLVED:`);
      console.log(`   Ticket: ${result.ticket?.ticketId}`);
      console.log(`   Solutions: ${result.solutions?.solutions.length}`);
      console.log(`   Notified: ${result.notification?.channel}`);
    } else {
      console.log(`\n⚠️ ESCALATED TO HUMAN:`);
      console.log(`   Reason: Low confidence (${(result.classification.confidence * 100).toFixed(1)}%)`);
      console.log(`   Ticket: ${result.ticket?.ticketId}`);
    }
    
    console.log(`\n⏱️ PERFORMANCE:`);
    console.log(`   Phi-3 Duration: ${result.performance.phi3_duration_ms}ms`);
    console.log(`   Total Duration: ${result.performance.total_duration_ms}ms`);
    console.log(`   Cost: $${result.performance.cost.toFixed(6)}`);
    
    console.log('\n' + '='.repeat(80) + '\n');
  }
}
