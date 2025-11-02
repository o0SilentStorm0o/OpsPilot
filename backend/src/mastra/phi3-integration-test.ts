/**
 * TEST PHI-3 INTEGRATION WITH MOCK DATA
 * 
 * This tests the integration logic without requiring the Phi-3 API to be running.
 * We simulate Phi-3 responses to test the decision-making logic.
 */

import { logger } from '../utils/logger';
import { 
  searchSolutionsTool,
  createJiraTicketTool,
  sendSlackNotificationTool
} from './resolution-workflow';

// Mock Phi-3 classification responses
const mockPhi3Classify = async (incident: { title: string; description: string; source?: string }) => {
  // Simulate inference time
  await new Promise(resolve => setTimeout(resolve, 50));

  if (incident.title.toLowerCase().includes('database')) {
    // High confidence - known pattern
    return {
      category: 'Database' as const,
      severity: 'high' as const,
      confidence: 0.87,
      reasoning: 'Pattern matches database connection pool exhaustion incidents in training data',
      uncertainty: {
        mc_dropout: 0.13,
        margin: 0.67,
        entropy: 0.23,
        should_escalate: false
      }
    };
  } else {
    // Low confidence - unknown pattern
    return {
      category: 'Application' as const,
      severity: 'medium' as const,
      confidence: 0.45,
      reasoning: 'Novel pattern not well represented in training data - high uncertainty detected',
      uncertainty: {
        mc_dropout: 0.65,
        margin: 0.12,
        entropy: 0.89,
        should_escalate: true
      }
    };
  }
};

/**
 * Phi-3 + Mastra resolution with mock data
 */
async function mockPhi3ResolveIncident(incident: { title: string; description: string; source?: string }) {
  const startTime = Date.now();
  
  // Step 1: Phi-3 classification
  const classifyStart = Date.now();
  const classification = await mockPhi3Classify(incident);
  const phi3Duration = Date.now() - classifyStart;

  logger.info('🧠 Phi-3 classification complete', {
    category: classification.category,
    confidence: classification.confidence,
    should_escalate: classification.uncertainty.should_escalate,
    duration: `${phi3Duration}ms`
  });

  // Step 2: Decision point - uncertainty-aware routing
  if (classification.confidence >= 0.80 && !classification.uncertainty.should_escalate) {
    logger.info('✅ HIGH CONFIDENCE - Auto-resolving with Mastra workflow');

    // Step 3a: Search solutions
    const solutions = await searchSolutionsTool.execute({
      context: {
        category: classification.category,
        description: incident.description,
        severity: classification.severity
      },
      runtimeContext: {} as any
    });

    // Step 3b: Create ticket
    const ticket = await createJiraTicketTool.execute({
      context: {
        title: incident.title,
        description: incident.description,
        category: classification.category,
        severity: classification.severity,
        solutions: solutions.solutions,
        classification: {
          ...classification,
          model: 'phi3-fine-tuned'
        }
      },
      runtimeContext: {} as any
    });

    // Step 3c: Notify team
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

    const totalDuration = Date.now() - startTime;

    return {
      classification: {
        category: classification.category,
        severity: classification.severity,
        confidence: classification.confidence,
        reasoning: classification.reasoning,
        model: 'phi3-fine-tuned'
      },
      uncertainty: classification.uncertainty,
      decision: 'auto-resolved' as const,
      solutions,
      ticket,
      notification,
      performance: {
        phi3_duration_ms: phi3Duration,
        total_duration_ms: totalDuration,
        model_used: 'phi3 + mastra',
        cost: 0
      }
    };

  } else {
    logger.info('⚠️ LOW CONFIDENCE - Escalating to human-in-the-loop');

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
        severity: 'critical',
        solutions: [],
        classification: {
          ...classification,
          model: 'phi3-fine-tuned (LOW CONFIDENCE)'
        }
      },
      runtimeContext: {} as any
    });

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

    const totalDuration = Date.now() - startTime;

    return {
      classification: {
        category: classification.category,
        severity: classification.severity,
        confidence: classification.confidence,
        reasoning: classification.reasoning,
        model: 'phi3-fine-tuned'
      },
      uncertainty: classification.uncertainty,
      decision: 'human-escalated' as const,
      ticket,
      notification,
      performance: {
        phi3_duration_ms: phi3Duration,
        total_duration_ms: totalDuration,
        model_used: 'phi3 (classification only)',
        cost: 0
      }
    };
  }
}

/**
 * Run the mock integration test
 */
async function runMockTest() {
  console.log('\n🚀 PHI-3 + MASTRA INTEGRATION TEST (MOCK DATA)\n');
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

  const results = [];

  for (const incident of testIncidents) {
    console.log(`\n📋 INCIDENT: ${incident.title}`);
    console.log('-'.repeat(80));
    
    const result = await mockPhi3ResolveIncident(incident);
    results.push(result);
    
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
      console.log(`   Notification: Sent to #incidents`);
      console.log(`\n💡 TOP SOLUTION:`);
      if (result.solutions && result.solutions.solutions.length > 0) {
        const topSolution = result.solutions.solutions[0];
        console.log(`   ${topSolution.title}`);
        console.log(`   Success Rate: ${(topSolution.successRate * 100).toFixed(0)}%`);
      }
    } else {
      console.log(`\n🔴 ESCALATED TO HUMAN:`);
      console.log(`   Ticket: ${result.ticket?.ticketId} (CRITICAL priority)`);
      console.log(`   Notification: Urgent alert sent`);
      console.log(`   Reason: ${result.classification.reasoning}`);
    }
    
    console.log(`\n⚡ PERFORMANCE:`);
    console.log(`   Phi-3 inference: ${result.performance.phi3_duration_ms}ms`);
    console.log(`   Total duration: ${result.performance.total_duration_ms}ms`);
    console.log(`   Cost: $${result.performance.cost}`);
    
    console.log('\n' + '='.repeat(80));
  }

  // Summary
  console.log('\n📊 SUMMARY:');
  console.log('-'.repeat(80));
  
  const autoResolved = results.filter(r => r.decision === 'auto-resolved').length;
  const escalated = results.filter(r => r.decision === 'human-escalated').length;
  const avgConfidence = results.reduce((sum, r) => sum + r.classification.confidence, 0) / results.length;
  const avgDuration = results.reduce((sum, r) => sum + r.performance.total_duration_ms, 0) / results.length;
  const avgPhi3Duration = results.reduce((sum, r) => sum + r.performance.phi3_duration_ms, 0) / results.length;
  
  console.log(`Total incidents: ${results.length}`);
  console.log(`Auto-resolved: ${autoResolved} (${(autoResolved/results.length*100).toFixed(0)}%)`);
  console.log(`Human-escalated: ${escalated} (${(escalated/results.length*100).toFixed(0)}%)`);
  console.log(`Average confidence: ${(avgConfidence * 100).toFixed(1)}%`);
  console.log(`Average Phi-3 inference: ${avgPhi3Duration.toFixed(0)}ms`);
  console.log(`Average total duration: ${avgDuration.toFixed(0)}ms`);
  console.log(`Total cost: $0 (local inference)`);
  
  console.log('\n✅ PHI-3 + MASTRA INTEGRATION TEST COMPLETE\n');
}

// Run the test
runMockTest().catch(console.error);
