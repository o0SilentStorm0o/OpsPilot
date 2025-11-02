/**
 * Multi-Agent Incident Resolution Workflow
 * 
 * Automated end-to-end incident handling:
 * 1. Classify incident (AI-powered)
 * 2. Search for solutions (RAG)
 * 3. Create tracking ticket (Jira)
 * 4. Notify team (Slack)
 * 
 * Impact: 30 min manual work → 15 seconds automated
 * 
 * Note: Using direct orchestration instead of Workflow API
 * for better compatibility with current Mastra version
 */

import { createTool } from '@mastra/core/tools';
import { z } from 'zod';
import { incidentAgent } from './incident-agent';
import { logger } from '../utils/logger';

// ============================================================================
// TOOLS - Integration Points
// ============================================================================

/**
 * Search knowledge base for solutions
 */
export const searchSolutionsTool = createTool({
  id: 'search-solutions',
  description: 'Search knowledge base for incident solutions and best practices',
  inputSchema: z.object({
    category: z.string().describe('Incident category'),
    description: z.string().describe('Incident description'),
    severity: z.string().describe('Incident severity')
  }),
  outputSchema: z.object({
    solutions: z.array(z.object({
      title: z.string(),
      description: z.string(),
      steps: z.array(z.string()),
      successRate: z.number()
    })),
    relatedIncidents: z.array(z.string())
  }),
  execute: async ({ context }) => {
    logger.info('Searching for solutions', { 
      category: context.category,
      severity: context.severity 
    });

    // Mock solutions based on category
    // In production: Query vector DB with incident embeddings
    const solutionDB: Record<string, any[]> = {
      'Database': [
        {
          title: 'Increase Connection Pool Size',
          description: 'Scale up database connection pool to handle increased load',
          steps: [
            'Check current pool size and utilization',
            'Calculate required pool size based on traffic',
            'Update database configuration',
            'Restart database service',
            'Monitor connection metrics'
          ],
          successRate: 0.92
        },
        {
          title: 'Optimize Slow Queries',
          description: 'Identify and fix slow database queries causing timeouts',
          steps: [
            'Enable slow query log',
            'Identify queries > 1 second',
            'Add missing indexes',
            'Optimize query logic',
            'Test query performance'
          ],
          successRate: 0.85
        }
      ],
      'Security': [
        {
          title: 'Block Malicious IP',
          description: 'Add IP to firewall blocklist to prevent further attacks',
          steps: [
            'Identify attacking IP address',
            'Verify IP is not legitimate user',
            'Add IP to firewall rules',
            'Monitor for bypass attempts',
            'Review logs for similar patterns'
          ],
          successRate: 0.95
        },
        {
          title: 'Reset User Credentials',
          description: 'Force password reset for compromised accounts',
          steps: [
            'Identify compromised accounts',
            'Invalidate active sessions',
            'Force password reset',
            'Enable 2FA if available',
            'Notify affected users'
          ],
          successRate: 0.88
        }
      ],
      'Performance': [
        {
          title: 'Scale Application Pods',
          description: 'Increase number of application instances to handle load',
          steps: [
            'Check current CPU/Memory usage',
            'Calculate required replicas',
            'Scale deployment',
            'Verify load distribution',
            'Monitor performance metrics'
          ],
          successRate: 0.90
        },
        {
          title: 'Enable Caching',
          description: 'Add caching layer to reduce backend load',
          steps: [
            'Identify cacheable endpoints',
            'Configure Redis/Memcached',
            'Update application to use cache',
            'Set appropriate TTL values',
            'Monitor cache hit rate'
          ],
          successRate: 0.87
        }
      ]
    };

    const solutions = solutionDB[context.category] || [
      {
        title: 'Generic Troubleshooting',
        description: 'Standard incident investigation procedure',
        steps: [
          'Check application logs',
          'Review metrics and alerts',
          'Verify recent deployments',
          'Test system components',
          'Escalate if needed'
        ],
        successRate: 0.70
      }
    ];

    return {
      solutions: solutions.slice(0, 3), // Top 3 solutions
      relatedIncidents: [
        `INC-2024-${Math.floor(Math.random() * 1000)}`,
        `INC-2024-${Math.floor(Math.random() * 1000)}`
      ]
    };
  }
});

/**
 * Create Jira ticket (mock implementation)
 */
export const createJiraTicketTool = createTool({
  id: 'create-jira-ticket',
  description: 'Create Jira ticket for incident tracking',
  inputSchema: z.object({
    title: z.string(),
    description: z.string(),
    category: z.string(),
    severity: z.string(),
    solutions: z.array(z.any()).optional(),
    classification: z.any().optional()
  }),
  outputSchema: z.object({
    ticketId: z.string(),
    ticketUrl: z.string(),
    status: z.string()
  }),
  execute: async ({ context }) => {
    logger.info('Creating Jira ticket', { 
      title: context.title,
      severity: context.severity 
    });

    // Mock Jira ticket creation
    // In production: Call Jira REST API
    const ticketId = `INC-2025-${Math.floor(1000 + Math.random() * 9000)}`;
    
    return {
      ticketId,
      ticketUrl: `https://jira.company.com/browse/${ticketId}`,
      status: 'Open'
    };
  }
});

/**
 * Send Slack notification (mock implementation)
 */
export const sendSlackNotificationTool = createTool({
  id: 'send-slack-notification',
  description: 'Send Slack notification to incident response team',
  inputSchema: z.object({
    channel: z.string(),
    severity: z.string(),
    category: z.string(),
    title: z.string(),
    ticketId: z.string().optional(),
    ticketUrl: z.string().optional(),
    solutionsCount: z.number().optional()
  }),
  outputSchema: z.object({
    sent: z.boolean(),
    messageId: z.string(),
    channel: z.string()
  }),
  execute: async ({ context }) => {
    logger.info('Sending Slack notification', { 
      channel: context.channel,
      severity: context.severity 
    });

    // Mock Slack notification
    // In production: Call Slack Web API
    const severityEmoji = {
      'low': '🟢',
      'medium': '🟡',
      'high': '🟠',
      'critical': '🔴'
    }[context.severity.toLowerCase()] || '⚪';

    const message = `${severityEmoji} **${context.severity.toUpperCase()}** ${context.category} Incident
    
📋 ${context.title}
${context.ticketId ? `🎫 Ticket: ${context.ticketUrl}` : ''}
${context.solutionsCount ? `💡 Solutions found: ${context.solutionsCount}` : ''}

🤖 Auto-triaged by OpsPilot AI`;

    logger.info('Slack message would be sent', { message });

    return {
      sent: true,
      messageId: `msg_${Date.now()}`,
      channel: context.channel
    };
  }
});

// ============================================================================
// DIRECT ORCHESTRATION (Workflow API removed for compatibility)
// ============================================================================

// ============================================================================
// CONVENIENCE FUNCTION
// ============================================================================

/**
 * Execute the complete incident resolution workflow
 * 
 * Simplified implementation without Workflow API
 * (Direct orchestration for better compatibility)
 * 
 * @example
 * ```typescript
 * const result = await resolveIncident({
 *   title: "Database connection timeout",
 *   description: "Users getting 500 errors...",
 *   source: "monitoring"
 * });
 * 
 * console.log(result.ticket.ticketUrl);
 * console.log(result.solutionResults.solutions);
 * ```
 */
export async function resolveIncident(incident: {
  title: string;
  description: string;
  source?: string;
}) {
  const startTime = Date.now();
  
  try {
    logger.info('Starting incident resolution workflow', { title: incident.title });

    // Step 1: Classify incident using AI agent
    logger.info('Step 1/4: Classifying incident');
    const classifyResult = await incidentAgent.generate(`
      Analyze and classify this incident:
      
      Title: ${incident.title}
      Description: ${incident.description}
      ${incident.source ? `Source: ${incident.source}` : ''}
      
      Use searchKnowledgeBase and classifyIncident tools to provide accurate classification.
    `);

    // Extract classification from tool results
    // Agent calls two tools: [0] searchKnowledgeBase, [1] classifyIncident
    const allToolResults = classifyResult.toolResults || [];
    const classifyToolResult = allToolResults[1]; // Second tool is classifyIncident

    // Extract classification from tool result
    let classification;
    if (classifyToolResult) {
      const toolResult = classifyToolResult as any;
      classification = toolResult.payload?.result?.classification;
      
      if (!classification) {
        logger.warn('Classification not found in tool result, using fallback');
        classification = {
          category: 'Application',
          severity: 'medium',
          confidence: 0.7,
          reasoning: 'Unable to extract classification from tool result'
        };
      }
    } else {
      // Fallback if tool wasn't called
      logger.warn('Classification tool not called, using fallback');
      classification = {
        category: 'Application',
        severity: 'medium',
        confidence: 0.7,
        reasoning: 'Unable to classify with high confidence - tool not called'
      };
    }

    logger.info('Classification complete', { 
      category: classification.category,
      severity: classification.severity,
      confidence: classification.confidence
    });

    // Step 2: Search for solutions
    logger.info('Step 2/4: Searching for solutions');
    const solutionResults = await searchSolutionsTool.execute({
      context: {
        category: classification.category,
        description: incident.description,
        severity: classification.severity
      },
      runtimeContext: {} as any
    });

    logger.info('Solutions found', { count: solutionResults.solutions.length });

    // Step 3: Create Jira ticket
    logger.info('Step 3/4: Creating Jira ticket');
    const ticket = await createJiraTicketTool.execute({
      context: {
        title: incident.title,
        description: incident.description,
        category: classification.category,
        severity: classification.severity,
        solutions: solutionResults.solutions,
        classification
      },
      runtimeContext: {} as any
    });

    logger.info('Ticket created', { ticketId: ticket.ticketId });

    // Step 4: Send Slack notification
    logger.info('Step 4/4: Sending Slack notification');
    const notification = await sendSlackNotificationTool.execute({
      context: {
        channel: '#incidents',
        severity: classification.severity,
        category: classification.category,
        title: incident.title,
        ticketId: ticket.ticketId,
        ticketUrl: ticket.ticketUrl,
        solutionsCount: solutionResults.solutions.length
      },
      runtimeContext: {} as any
    });

    logger.info('Notification sent', { messageId: notification.messageId });

    const duration = Date.now() - startTime;
    
    logger.info('Incident resolution workflow completed', {
      ticketId: ticket.ticketId,
      category: classification.category,
      severity: classification.severity,
      solutionsFound: solutionResults.solutions.length,
      duration: `${duration}ms`
    });
    
    return {
      success: true,
      duration,
      title: incident.title,
      description: incident.description,
      source: incident.source,
      classification,
      solutionResults,
      ticket,
      notification
    };
  } catch (error) {
    logger.error('Incident resolution workflow failed', error);
    throw error;
  }
}
