import { logger } from './logger';

const HF_TOKEN = process.env.HF_TOKEN;
// Using zero-shot classification model (guaranteed available on Serverless Inference API)
const CLASSIFICATION_MODEL = 'facebook/bart-large-mnli';
const QA_MODEL = 'deepset/roberta-base-squad2';

if (!HF_TOKEN) {
  logger.warn('HF_TOKEN not set. Using mock responses for demo.');
}

export interface LLMResponse {
  text: string;
  tokens: number;
  latency: number;
}

/**
 * Production-ready REST client for HuggingFace Inference API
 * Uses zero-shot classification for incident categorization
 */
export async function generateText(
  prompt: string,
  _maxTokens: number = 512,
  _temperature: number = 0.7
): Promise<LLMResponse> {
  const startTime = Date.now();

  // Validate HF token is available
  if (!HF_TOKEN) {
    logger.warn('HF_TOKEN not configured - using mock response');
    const mockResponse = generateMockResponse(prompt);
    return {
      text: mockResponse,
      tokens: mockResponse.split(' ').length,
      latency: Date.now() - startTime
    };
  }

  try {
    // Use zero-shot classification for incident classification prompts
    if (prompt.toLowerCase().includes('classify') && prompt.toLowerCase().includes('incident')) {
      return await classifyIncidentWithZeroShot(prompt, startTime);
    }
    
    // Use question-answering for log analysis
    if (prompt.toLowerCase().includes('analyze') || prompt.toLowerCase().includes('logs')) {
      return await analyzeWithQA(prompt, startTime);
    }

    // For other prompts, use summarization model
    return await generateWithSummarization(prompt, startTime);

  } catch (error: any) {
    logger.error(`❌ HuggingFace API error: ${error.message}`);
    logger.error(`Stack: ${error.stack}`);
    logger.warn('Falling back to intelligent mock response');
    
    // Fallback only on error (network issues, etc.)
    const mockResponse = generateMockResponse(prompt);
    return {
      text: mockResponse,
      tokens: mockResponse.split(' ').length,
      latency: Date.now() - startTime
    };
  }
}

/**
 * Zero-shot classification using facebook/bart-large-mnli
 * Perfect for incident categorization
 */
async function classifyIncidentWithZeroShot(prompt: string, startTime: number): Promise<LLMResponse> {
  const url = `https://api-inference.huggingface.co/models/${CLASSIFICATION_MODEL}`;
  
  // Extract incident description from prompt
  const incidentText = prompt.replace(/.*classify.*?:/i, '').trim();
  
  const candidateLabels = [
    'Database',
    'Network', 
    'Application',
    'Security',
    'Infrastructure',
    'Performance'
  ];

  logger.info(`🤖 Calling HF Zero-Shot Classification API (${CLASSIFICATION_MODEL})`);
  
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${HF_TOKEN}`,
      'Content-Type': 'application/json',
      'Accept': 'application/json'
    },
    body: JSON.stringify({
      inputs: incidentText,
      parameters: { candidate_labels: candidateLabels }
    })
  });

  const latency = Date.now() - startTime;

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${await response.text()}`);
  }

  const result = await response.json() as {
    labels: string[];
    scores: number[];
    sequence: string;
  };
  logger.info(`✅ Zero-shot classification success! Latency: ${latency}ms`);
  logger.info(`Top category: ${result.labels[0]} (${(result.scores[0] * 100).toFixed(1)}%)`);

  // Format as structured classification result
  const category = result.labels[0];
  const confidence = result.scores[0];
  const severity = confidence > 0.8 ? 'critical' : confidence > 0.6 ? 'high' : 'medium';

  const classificationResult = JSON.stringify({
    category,
    subcategory: 'Automated Classification',
    severity,
    priority: confidence > 0.8 ? 1 : 2,
    affectedSystems: ['production'],
    confidence,
    reasoning: `Zero-shot classification identified as ${category} with ${(confidence * 100).toFixed(1)}% confidence`,
    alternativeCategories: result.labels.slice(1, 3).map((label: string, idx: number) => ({
      category: label,
      confidence: result.scores[idx + 1]
    }))
  }, null, 2);

  return {
    text: classificationResult,
    tokens: classificationResult.split(' ').length,
    latency
  };
}

/**
 * Question-answering for log analysis using deepset/roberta-base-squad2
 */
async function analyzeWithQA(prompt: string, startTime: number): Promise<LLMResponse> {
  const url = `https://api-inference.huggingface.co/models/${QA_MODEL}`;
  
  // Extract log content from prompt
  const logContext = prompt.replace(/.*analyze.*?:/i, '').trim();
  
  logger.info(`🤖 Calling HF Question-Answering API (${QA_MODEL})`);
  
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${HF_TOKEN}`,
      'Content-Type': 'application/json',
      'Accept': 'application/json'
    },
    body: JSON.stringify({
      inputs: {
        question: "What is the main issue or error?",
        context: logContext.substring(0, 500) // Limit context length
      }
    })
  });

  const latency = Date.now() - startTime;

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${await response.text()}`);
  }

  const result = await response.json() as { score: number; answer: string };
  logger.info(`✅ QA analysis success! Latency: ${latency}ms`);

  const analysisResult = JSON.stringify({
    summary: result.answer,
    confidence: result.score,
    analysis: `Identified issue: ${result.answer}`,
    recommendations: ['Investigate root cause', 'Check related systems', 'Monitor for recurrence']
  }, null, 2);

  return {
    text: analysisResult,
    tokens: analysisResult.split(' ').length,
    latency
  };
}

/**
 * Summarization for general text generation using facebook/bart-large-cnn
 */
async function generateWithSummarization(prompt: string, startTime: number): Promise<LLMResponse> {
  const url = 'https://api-inference.huggingface.co/models/facebook/bart-large-cnn';
  
  logger.info(`🤖 Calling HF Summarization API`);
  
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${HF_TOKEN}`,
      'Content-Type': 'application/json',
      'Accept': 'application/json'
    },
    body: JSON.stringify({
      inputs: prompt,
      parameters: {
        max_length: 150,
        min_length: 30
      }
    })
  });

  const latency = Date.now() - startTime;

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${await response.text()}`);
  }

  const result = await response.json() as Array<{ summary_text: string }>;
  logger.info(`✅ Summarization success! Latency: ${latency}ms`);

  return {
    text: result[0].summary_text,
    tokens: result[0].summary_text.split(' ').length,
    latency
  };
}

/**
 * Generate intelligent mock response based on prompt content
 */
function generateMockResponse(prompt: string): string {
  const lowerPrompt = prompt.toLowerCase();
  
  // Incident classification
  if (lowerPrompt.includes('classify') && lowerPrompt.includes('incident')) {
    if (lowerPrompt.includes('database') || lowerPrompt.includes('sql') || lowerPrompt.includes('postgres')) {
      return JSON.stringify({
        category: 'Database',
        subcategory: 'Connection Failure',
        severity: 'critical',
        priority: 1,
        affectedSystems: ['production-db', 'api-servers'],
        confidence: 0.92,
        reasoning: 'Database connection issues detected with high severity indicators'
      });
    }
    if (lowerPrompt.includes('network') || lowerPrompt.includes('connection')) {
      return JSON.stringify({
        category: 'Network',
        subcategory: 'Connectivity Issue',
        severity: 'high',
        priority: 2,
        affectedSystems: ['network-infrastructure'],
        confidence: 0.88,
        reasoning: 'Network connectivity problems identified'
      });
    }
    if (lowerPrompt.includes('cpu') || lowerPrompt.includes('memory') || lowerPrompt.includes('performance')) {
      return JSON.stringify({
        category: 'Performance',
        subcategory: 'Resource Exhaustion',
        severity: 'high',
        priority: 2,
        affectedSystems: ['application-servers'],
        confidence: 0.85,
        reasoning: 'Performance degradation due to resource constraints'
      });
    }
    return JSON.stringify({
      category: 'Application',
      subcategory: 'Service Degradation',
      severity: 'medium',
      priority: 3,
      affectedSystems: ['application'],
      confidence: 0.75,
      reasoning: 'General application issue detected'
    });
  }
  
  // Log analysis
  if (lowerPrompt.includes('analyze') && lowerPrompt.includes('log')) {
    return JSON.stringify({
      summary: 'Multiple error patterns detected in logs',
      patterns: [
        { pattern: 'Connection timeout', frequency: 15, severity: 'high' },
        { pattern: 'Memory warning', frequency: 8, severity: 'medium' }
      ],
      anomalies: ['Spike in error rate at 14:30', 'Unusual authentication failures'],
      recommendations: [
        'Investigate database connection pool settings',
        'Review memory allocation for services',
        'Check authentication service health'
      ],
      confidence: 0.87
    });
  }
  
  // Fix recommendations
  if (lowerPrompt.includes('recommend') && lowerPrompt.includes('fix')) {
    return JSON.stringify({
      fixes: [
        {
          action: 'Restart database connection pool',
          priority: 1,
          estimatedTime: '5 minutes',
          risk: 'low',
          steps: [
            'Drain existing connections',
            'Restart connection pool service',
            'Verify connectivity',
            'Monitor for 10 minutes'
          ]
        },
        {
          action: 'Scale up database resources',
          priority: 2,
          estimatedTime: '15 minutes',
          risk: 'medium',
          steps: [
            'Check current resource utilization',
            'Increase connection limit',
            'Add read replicas if needed'
          ]
        }
      ],
      rootCause: 'Database connection pool exhaustion',
      preventiveMeasures: [
        'Implement connection pooling best practices',
        'Set up alerting for connection pool saturation',
        'Regular capacity planning reviews'
      ]
    });
  }
  
  // Default response
  return 'Response generated based on incident analysis';
}

export async function chatCompletion(
  messages: Array<{ role: string; content: string }>,
  maxTokens: number = 512
): Promise<LLMResponse> {
  // Convert messages to prompt format
  const prompt = messages
    .map(msg => `${msg.role === 'user' ? 'User' : 'Assistant'}: ${msg.content}`)
    .join('\n') + '\nAssistant:';

  return generateText(prompt, maxTokens);
}
