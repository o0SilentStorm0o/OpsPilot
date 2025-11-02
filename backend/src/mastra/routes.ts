/**
 * Mastra Integration - API Endpoints
 * 
 * Provides endpoints for:
 * 1. Incident classification (single agent)
 * 2. Incident resolution (multi-agent workflow)
 */

import express from 'express';
import { analyzeIncidentWithMastra } from './incident-agent';
import { resolveIncident } from './resolution-workflow';
import { logger } from '../utils/logger';

export const mastraRouter = express.Router();

/**
 * POST /api/mastra/analyze-incident
 * 
 * Analyzes an incident using the Mastra AI agent
 * 
 * Body:
 * {
 *   "title": "Database connection timeout",
 *   "description": "PostgreSQL connection pool exhausted after traffic spike",
 *   "source": "api-gateway"
 * }
 */
mastraRouter.post('/analyze-incident', async (req, res, next): Promise<void> => {
  try {
    const { title, description, source } = req.body;

    // Validation
    if (!title || !description) {
      res.status(400).json({
        error: 'Missing required fields: title and description'
      });
      return;
    }

    logger.info('Mastra endpoint called', { title, source });

    // Call the Mastra agent
    const result = await analyzeIncidentWithMastra({
      title,
      description,
      source
    });

    res.json({
      agent: 'Mastra AI',
      ...result,
      // Add helpful metadata
      meta: {
        toolsUsed: result.steps?.flatMap((s: any) => 
          s.toolCalls?.map((tc: any) => tc.toolName) || []
        ) || [],
        totalTokens: result.usage?.totalTokens || 0,
        model: 'gpt-4o-mini'
      }
    });

  } catch (error) {
    logger.error('Mastra endpoint error', error);
    next(error);
  }
});

/**
 * POST /api/mastra/resolve-incident
 * 
 * Executes complete incident resolution workflow:
 * classify → search solutions → create ticket → notify team
 * 
 * Body:
 * {
 *   "title": "API response time degradation",
 *   "description": "Response time increased from 200ms to 3s...",
 *   "source": "APM Monitor"
 * }
 * 
 * Response:
 * {
 *   "success": true,
 *   "duration": 12500,
 *   "classification": { category, severity, confidence, reasoning },
 *   "solutionResults": { solutions: [...], relatedIncidents: [...] },
 *   "ticket": { ticketId, ticketUrl, status },
 *   "notification": { sent, messageId, channel }
 * }
 */
mastraRouter.post('/resolve-incident', async (req, res, next): Promise<void> => {
  try {
    const { title, description, source } = req.body;

    // Validation
    if (!title || !description) {
      res.status(400).json({
        error: 'Missing required fields: title and description'
      });
      return;
    }

    logger.info('Incident resolution workflow started via API', { title, source });

    // Execute the multi-agent workflow
    const result = await resolveIncident({
      title,
      description,
      source
    });

    res.json({
      workflow: 'incident-resolution',
      ...result,
      meta: {
        stepsCompleted: 4,
        category: result.classification.category,
        severity: result.classification.severity,
        ticketId: result.ticket.ticketId,
        solutionsFound: result.solutionResults.solutions.length
      }
    });

  } catch (error) {
    logger.error('Incident resolution workflow error', error);
    next(error);
  }
});

/**
 * GET /api/mastra/health
 * 
 * Health check for Mastra integration
 */
mastraRouter.get('/health', (_req, res) => {
  res.json({
    status: 'healthy',
    features: {
      incidentClassification: {
        agent: 'OpsPilot Incident Classifier',
        model: 'gpt-4o-mini',
        tools: ['search_knowledge_base', 'classify_incident']
      },
      resolutionWorkflow: {
        name: 'Multi-Agent Resolution',
        steps: ['classify', 'search-solutions', 'create-ticket', 'notify-team'],
        integrations: ['Jira', 'Slack']
      }
    }
  });
});
