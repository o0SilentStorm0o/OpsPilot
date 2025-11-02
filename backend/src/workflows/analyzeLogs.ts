import { logger } from '../utils/logger';
import { chatCompletion } from '../utils/hfClient';
import { recordLLMMetrics } from '../monitoring/metrics';
import { activeIncidents } from '../monitoring/metrics';

export interface LogEntry {
  timestamp: string;
  level: string;
  message: string;
  source?: string;
  metadata?: Record<string, any>;
}

export interface LogAnalysisResult {
  summary: string;
  anomalies: string[];
  patterns: string[];
  severity: 'low' | 'medium' | 'high' | 'critical';
  suggestedActions: string[];
  processingTime: number;
}

/**
 * Analyzes IT logs using LLM to identify patterns, anomalies, and potential incidents
 */
export async function analyzeLogs(logs: LogEntry[] | string[]): Promise<LogAnalysisResult> {
  const startTime = Date.now();
  activeIncidents.inc();

  try {
    logger.info(`Starting log analysis for ${logs.length} entries`);

    // Normalize logs to LogEntry format if they're strings
    const normalizedLogs: LogEntry[] = logs.map(log => {
      if (typeof log === 'string') {
        return {
          timestamp: new Date().toISOString(),
          level: log.includes('ERROR') ? 'ERROR' : log.includes('WARN') ? 'WARN' : 'INFO',
          message: log
        };
      }
      return log as LogEntry;
    });

    // Prepare logs for analysis
    const logSummary = normalizedLogs
      .slice(0, 50) // Limit to prevent token overflow
      .map(log => `[${log.timestamp}] ${log.level}: ${log.message}`)
      .join('\n');

    const prompt = `You are an IT operations expert analyzing system logs. Analyze the following logs and provide:
1. A brief summary of what's happening
2. Any anomalies or unusual patterns
3. Severity level (low/medium/high/critical)
4. Suggested actions

Logs:
${logSummary}

Respond in JSON format with fields: summary, anomalies (array), severity, suggestedActions (array).`;

    const response = await chatCompletion([
      { role: 'system', content: 'You are an expert IT operations analyst.' },
      { role: 'user', content: prompt }
    ], 800);

    // Record metrics
    recordLLMMetrics('mistral-7b', 'analyze-logs', response.latency, response.tokens);

    // Parse response (in production, you'd want better error handling)
    let parsedResult;
    try {
      // Extract JSON from response
      const jsonMatch = response.text.match(/\{[\s\S]*\}/);
      parsedResult = jsonMatch ? JSON.parse(jsonMatch[0]) : null;
    } catch (e) {
      logger.warn('Failed to parse LLM response as JSON, using fallback');
      parsedResult = {
        summary: response.text.substring(0, 200),
        anomalies: [],
        severity: 'medium',
        suggestedActions: ['Review logs manually']
      };
    }

    const result: LogAnalysisResult = {
      summary: parsedResult?.summary || 'Analysis complete',
      anomalies: parsedResult?.anomalies || [],
      patterns: extractPatterns(normalizedLogs),
      severity: parsedResult?.severity || 'medium',
      suggestedActions: parsedResult?.suggestedActions || [],
      processingTime: Date.now() - startTime
    };

    logger.info(`Log analysis completed in ${result.processingTime}ms`);
    return result;

  } catch (error) {
    logger.error('Error in analyzeLogs workflow:', error);
    throw error;
  } finally {
    activeIncidents.dec();
  }
}

/**
 * Extract common patterns from logs using simple heuristics
 */
function extractPatterns(logs: LogEntry[]): string[] {
  const patterns: string[] = [];
  
  // Count error levels
  const errorCount = logs.filter(l => l.level.toLowerCase() === 'error').length;
  const warnCount = logs.filter(l => l.level.toLowerCase() === 'warn').length;
  
  if (errorCount > 5) {
    patterns.push(`High error rate: ${errorCount} errors detected`);
  }
  if (warnCount > 10) {
    patterns.push(`Elevated warning level: ${warnCount} warnings`);
  }

  // Common error messages
  const messages = logs.map(l => l.message.toLowerCase());
  const connectionErrors = messages.filter(m => m.includes('connection') || m.includes('timeout')).length;
  const memoryErrors = messages.filter(m => m.includes('memory') || m.includes('oom')).length;
  
  if (connectionErrors > 3) {
    patterns.push('Recurring connection issues detected');
  }
  if (memoryErrors > 0) {
    patterns.push('Memory-related problems identified');
  }

  return patterns;
}
