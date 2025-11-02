/**
 * Mastra AI Demo Script
 * 
 * This demonstrates how the Mastra agent works:
 * 1. Automatic tool selection
 * 2. Type-safe parameters
 * 3. Structured outputs
 * 4. Built-in tracing
 * 
 * Run: npx tsx backend/src/mastra/demo.ts
 */

import dotenv from 'dotenv';
import { analyzeIncidentWithMastra } from './incident-agent';

dotenv.config();

// Check if API key is set
if (!process.env.OPENAI_API_KEY) {
  console.error('❌ OPENAI_API_KEY not found in .env file');
  process.exit(1);
}

console.log('🚀 Mastra AI Demo - OpsPilot Incident Classification\n');

async function runDemo() {
  const incidents = [
    {
      title: 'Database connection timeout',
      description: 'PostgreSQL connection pool exhausted. Users getting 500 errors. Started after traffic spike at 2PM.',
      source: 'api-gateway'
    },
    {
      title: 'Suspicious login attempts',
      description: 'Multiple failed login attempts from IP 192.168.1.100. Possible brute force attack.',
      source: 'auth-service'
    },
    {
      title: 'API response slow',
      description: 'Average API response time increased from 200ms to 3 seconds. CPU usage at 90%.',
      source: 'monitoring'
    }
  ];

  for (let i = 0; i < incidents.length; i++) {
    const incident = incidents[i];
    console.log(`\n${'='.repeat(80)}`);
    console.log(`📋 INCIDENT ${i + 1}/${incidents.length}: ${incident.title}`);
    console.log(`${'='.repeat(80)}\n`);
    
    console.log(`Description: ${incident.description}`);
    console.log(`Source: ${incident.source}\n`);

    try {
      const startTime = Date.now();
      const result = await analyzeIncidentWithMastra(incident);
      const duration = Date.now() - startTime;

      console.log(`✅ Analysis completed in ${duration}ms\n`);

      // Show tool calls (this is the magic!)
      if (result.steps && result.steps.length > 0) {
        console.log('🔧 Tools Used:');
        result.steps.forEach((step: any, stepIdx: number) => {
          step.toolCalls?.forEach((call: any, idx: number) => {
            console.log(`  ${stepIdx + 1}.${idx + 1} ${call.toolName || 'unknown'}`);
            if (call.args) {
              const argsStr = JSON.stringify(call.args, null, 2);
              console.log(`      Args:`, argsStr.split('\n').join('\n      '));
            }
          });
        });
        console.log('');
      }

      // Show classification
      if (result.classification) {
        console.log('📊 Classification:');
        console.log(`  Category: ${result.classification.category}`);
        console.log(`  Severity: ${result.classification.severity}`);
        console.log(`  Confidence: ${(result.classification.confidence * 100).toFixed(1)}%`);
        console.log(`  Reasoning: ${result.classification.reasoning}`);
      }

      // Show token usage
      if (result.usage) {
        console.log(`\n💰 Token Usage:`);
        console.log(`  Input: ${result.usage.inputTokens || 0}`);
        console.log(`  Output: ${result.usage.outputTokens || 0}`);
        console.log(`  Total: ${result.usage.totalTokens || 0}`);
        
        // Calculate cost (GPT-4o-mini pricing)
        const inputTokens = result.usage.inputTokens || 0;
        const outputTokens = result.usage.outputTokens || 0;
        const inputCost = (inputTokens / 1_000_000) * 0.15;
        const outputCost = (outputTokens / 1_000_000) * 0.60;
        const totalCost = inputCost + outputCost;
        console.log(`  Cost: $${totalCost.toFixed(6)}`);
      }

    } catch (error) {
      console.error('❌ Error analyzing incident:', error);
    }

    // Wait a bit between requests
    if (i < incidents.length - 1) {
      console.log('\n⏳ Waiting 2 seconds before next incident...');
      await new Promise(resolve => setTimeout(resolve, 2000));
    }
  }

  console.log(`\n${'='.repeat(80)}`);
  console.log('✨ Demo Complete!');
  console.log(`${'='.repeat(80)}\n`);

  console.log('🎯 Key Takeaways:');
  console.log('1. The agent AUTOMATICALLY decides when to use tools');
  console.log('2. Parameters are TYPE-SAFE (Zod validation)');
  console.log('3. All tool calls are LOGGED and TRACEABLE');
  console.log('4. Token usage is AUTOMATICALLY tracked');
  console.log('5. No manual prompt engineering needed!\n');
}

// Run the demo
runDemo().catch(error => {
  console.error('Fatal error:', error);
  process.exit(1);
});
