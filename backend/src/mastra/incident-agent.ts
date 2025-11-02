/**
 * Mastra AI Agent for Incident Classification
 * 
 * Using official Mastra Agent API with createTool from @mastra/core
 */

import { Agent } from '@mastra/core/agent';
import { createTool } from '@mastra/core/tools';
import { openai } from '@ai-sdk/openai';
import { z } from 'zod';
import { logger } from '../utils/logger';

// Mock RAG search function
async function searchKnowledgeBase(query: string, category?: string): Promise<string> {
  logger.info('RAG tool called', { query, category });
  
  const mockKB: Record<string, string> = {
    'database': 'Common DB issues: connection timeouts (increase pool), slow queries (add indexes), deadlocks (optimize transactions)',
    'network': 'Network issues: packet loss (check cables), DNS failures (verify nameservers), latency (check routing)',
    'security': 'Security incidents: unauthorized access (review IAM), SQL injection (use parameterized queries), DDoS (enable WAF)',
    'performance': 'Performance issues: high CPU (optimize algorithms), memory leaks (profile heap), slow response (add caching)',
    'application': 'App errors: null pointer (add validation), timeout (increase limits), crash (check logs)',
    'infrastructure': 'Infra problems: disk full (clean logs), OOM (increase memory), network partition (check connectivity)'
  };
  
  const lowerCategory = category?.toLowerCase() || 'unknown';
  return mockKB[lowerCategory] || 'No specific guidance found. Analyze logs and metrics.';
}

// Define tools using Mastra's createTool
const searchKnowledgeBaseTool = createTool({
  id: 'search-knowledge-base',
  description: 'Search the incident knowledge base for relevant solutions and best practices',
  inputSchema: z.object({
    query: z.string().describe('Search query describing the incident'),
    category: z.enum([
      'Database',
      'Network',
      'Security',
      'Performance',
      'Application',
      'Infrastructure'
    ]).optional().describe('Optional category to narrow search')
  }),
  outputSchema: z.object({
    results: z.string(),
    source: z.string()
  }),
  execute: async ({ context }) => {
    const results = await searchKnowledgeBase(context.query, context.category);
    return { results, source: 'knowledge_base' };
  }
});

const classifyIncidentTool = createTool({
  id: 'classify-incident',
  description: 'Classify an IT incident with confidence score',
  inputSchema: z.object({
    title: z.string().describe('The incident title'),
    description: z.string().describe('Detailed incident description'),
    category: z.enum([
      'Database',
      'Network',
      'Security',
      'Performance',
      'Application',
      'Infrastructure'
    ]).describe('The incident category'),
    severity: z.enum(['low', 'medium', 'high', 'critical']).describe('Incident severity'),
    confidence: z.number().min(0).max(1).describe('Classification confidence (0-1)'),
    reasoning: z.string().describe('Explanation for the classification')
  }),
  outputSchema: z.object({
    success: z.boolean(),
    classification: z.any()
  }),
  execute: async ({ context }) => {
    logger.info('Classification completed', context);
    return { success: true, classification: context };
  }
});

// Create the Mastra Agent
export const incidentAgent = new Agent({
  id: 'incident-agent',
  name: 'Incident Classification Agent',
  instructions: `You are an expert IT incident analyst. Analyze incidents and classify them accurately.

When analyzing an incident:
1. First, search the knowledge base using search-knowledge-base tool to get relevant context
2. Consider the incident details, severity indicators, and best practices
3. Use classify-incident tool to provide your final classification

Categories: Database, Network, Security, Performance, Application, Infrastructure
Severities: low, medium, high, critical

Be thorough and explain your reasoning.`,
  model: openai('gpt-4o-mini'),
  tools: {
    searchKnowledgeBase: searchKnowledgeBaseTool,
    classifyIncident: classifyIncidentTool
  }
});

/**
 * Analyze incident using Mastra Agent
 */
export async function analyzeIncidentWithMastra(incident: {
  title: string;
  description: string;
  source?: string;
}) {
  try {
    logger.info('Starting Mastra AI analysis', { 
      title: incident.title,
      source: incident.source 
    });

    const prompt = `Analyze this IT incident and classify it:

Title: ${incident.title}
Description: ${incident.description}
${incident.source ? `Source: ${incident.source}` : ''}

Please:
1. First use the searchKnowledgeBase tool to gather relevant context
2. Then use the classifyIncident tool to provide your final classification

Classify into: Database, Network, Security, Performance, Application, or Infrastructure
Determine severity: low, medium, high, or critical
Provide confidence score (0-1) and reasoning.`;

    const result = await incidentAgent.generate(prompt);

    logger.info('Mastra AI analysis completed', { 
      text: result.text?.substring(0, 100),
      usage: result.usage
    });

    // Extract tool results from the agent response
    const steps = result.steps || [];
    const toolResults = steps.flatMap(step => step.toolResults || []);
    
    const classifyResult = toolResults.find(
      (tr: any) => tr.toolName === 'classify-incident'
    );
    
    const classification = classifyResult ? (classifyResult as any).result?.classification : undefined;

    return {
      success: true,
      text: result.text,
      steps: steps.map(step => ({
        toolCalls: step.toolCalls?.map((tc: any) => ({
          toolName: tc.toolName,
          args: tc.args
        })) || [],
        toolResults: (step.toolResults || []).map((tr: any) => ({
          toolName: tr.toolName,
          result: tr.result
        }))
      })),
      usage: result.usage,
      classification
    };

  } catch (error) {
    logger.error('Mastra AI analysis error', error);
    throw error;
  }
}

