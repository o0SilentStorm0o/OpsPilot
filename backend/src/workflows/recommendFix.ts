import { logger } from '../utils/logger';
import { chatCompletion } from '../utils/hfClient';
import { recordLLMMetrics } from '../monitoring/metrics';
import { Incident } from './classifyIncident';

export interface RemediationContext {
  historicalIncidents?: Incident[];
  systemConfiguration?: Record<string, any>;
  availableResources?: string[];
}

export interface RemediationPlan {
  steps: RemediationStep[];
  estimatedTime: string;
  requiredPermissions: string[];
  risks: string[];
  successCriteria: string[];
  rollbackPlan: string[];
}

export interface RemediationStep {
  order: number;
  description: string;
  command?: string;
  expectedOutcome: string;
  verificationMethod: string;
}

/**
 * Generates remediation recommendations for IT incidents using LLM
 */
export async function recommendFix(
  incident: Incident,
  context?: RemediationContext
): Promise<RemediationPlan> {
  logger.info(`Generating remediation plan for incident: ${incident.title}`);

  try {
    // Build context information
    let contextInfo = '';
    if (context?.historicalIncidents && context.historicalIncidents.length > 0) {
      contextInfo += '\n\nSimilar past incidents:\n';
      contextInfo += context.historicalIncidents
        .slice(0, 3)
        .map(inc => `- ${inc.title}: ${inc.description}`)
        .join('\n');
    }

    const prompt = `You are an expert IT operations engineer. Generate a detailed remediation plan for the following incident:

Title: ${incident.title}
Description: ${incident.description}
Source: ${incident.source}
${contextInfo}

Provide a structured remediation plan with:
1. steps: Array of ordered remediation steps (each with: order, description, command if applicable, expectedOutcome, verificationMethod)
2. estimatedTime: How long this will take (e.g., "15-30 minutes")
3. requiredPermissions: What access/permissions are needed
4. risks: Potential risks of this remediation
5. successCriteria: How to verify the fix worked
6. rollbackPlan: Steps to undo changes if needed

Focus on safe, tested procedures. Respond in JSON format.`;

    const response = await chatCompletion([
      { role: 'system', content: 'You are an expert IT operations and incident remediation specialist with years of experience in enterprise systems.' },
      { role: 'user', content: prompt }
    ], 1000);

    recordLLMMetrics('mistral-7b', 'recommend-fix', response.latency, response.tokens);

    // Parse response
    let plan: RemediationPlan;
    try {
      const jsonMatch = response.text.match(/\{[\s\S]*\}/);
      const parsed = jsonMatch ? JSON.parse(jsonMatch[0]) : null;
      
      plan = {
        steps: parsed?.steps || generateDefaultSteps(incident),
        estimatedTime: parsed?.estimatedTime || '30-60 minutes',
        requiredPermissions: parsed?.requiredPermissions || ['admin'],
        risks: parsed?.risks || ['Service interruption possible'],
        successCriteria: parsed?.successCriteria || ['Issue resolved', 'No error logs'],
        rollbackPlan: parsed?.rollbackPlan || ['Restore from backup', 'Revert configuration']
      };
    } catch (e) {
      logger.warn('Failed to parse remediation response, using defaults');
      plan = {
        steps: generateDefaultSteps(incident),
        estimatedTime: '30-60 minutes',
        requiredPermissions: ['admin'],
        risks: ['Manual review required'],
        successCriteria: ['Issue investigation complete'],
        rollbackPlan: ['Document all changes before proceeding']
      };
    }

    logger.info(`Generated remediation plan with ${plan.steps.length} steps`);
    return plan;

  } catch (error) {
    logger.error('Error in recommendFix workflow:', error);
    throw error;
  }
}

/**
 * Generate default remediation steps when LLM parsing fails
 */
function generateDefaultSteps(incident: Incident): RemediationStep[] {
  return [
    {
      order: 1,
      description: 'Review incident logs and gather diagnostic information',
      expectedOutcome: 'Complete understanding of the issue',
      verificationMethod: 'Log analysis complete'
    },
    {
      order: 2,
      description: `Investigate ${incident.source} system status`,
      command: 'systemctl status <service-name>',
      expectedOutcome: 'System status identified',
      verificationMethod: 'Status check returns valid data'
    },
    {
      order: 3,
      description: 'Apply recommended fix based on incident type',
      expectedOutcome: 'Issue resolved',
      verificationMethod: 'Error no longer appears in logs'
    },
    {
      order: 4,
      description: 'Monitor system for 15 minutes to confirm stability',
      expectedOutcome: 'System operating normally',
      verificationMethod: 'No new errors in monitoring dashboard'
    }
  ];
}

/**
 * Validates a remediation plan for safety and completeness
 */
export function validateRemediationPlan(plan: RemediationPlan): {
  valid: boolean;
  issues: string[];
} {
  const issues: string[] = [];

  if (!plan.steps || plan.steps.length === 0) {
    issues.push('No remediation steps provided');
  }

  if (!plan.rollbackPlan || plan.rollbackPlan.length === 0) {
    issues.push('No rollback plan provided');
  }

  if (!plan.successCriteria || plan.successCriteria.length === 0) {
    issues.push('No success criteria defined');
  }

  // Check for dangerous commands
  const dangerousCommands = ['rm -rf', 'DROP TABLE', 'DELETE FROM', 'format'];
  for (const step of plan.steps) {
    if (step.command) {
      for (const dangerous of dangerousCommands) {
        if (step.command.includes(dangerous)) {
          issues.push(`Potentially dangerous command detected in step ${step.order}: ${dangerous}`);
        }
      }
    }
  }

  return {
    valid: issues.length === 0,
    issues
  };
}
