# Mastra AI Integration 🤖

This directory contains the Mastra AI integration for OpsPilot, demonstrating modern TypeScript-first AI agent orchestration.

## 🎯 What is This?

Mastra is a TypeScript framework for building AI agents and workflows. This integration shows how to:

1. **Automatic tool calling** - Agent decides when to use tools
2. **Type-safe parameters** - Zod schema validation
3. **Structured outputs** - Reliable JSON responses
4. **Built-in observability** - Automatic tracing and metrics

---

## 📁 Files

```
mastra/
├── incident-agent.ts  # Main agent definition with tools
├── routes.ts          # Express routes for API
├── demo.ts            # Standalone demo script
└── README.md          # This file
```

---

## 🚀 Quick Start

### 1. Run the Demo

```bash
npx tsx backend/src/mastra/demo.ts
```

This will analyze 3 sample incidents and show you:
- ✅ Which tools the agent used
- ✅ Classification results
- ✅ Token usage and cost
- ✅ Execution time

### 2. Use the API

Start the server:
```bash
npm run dev
```

Test the endpoint:
```bash
curl -X POST http://localhost:3001/api/mastra/analyze-incident \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Database connection timeout",
    "description": "PostgreSQL connection pool exhausted after traffic spike",
    "source": "api-gateway"
  }'
```

Health check:
```bash
curl http://localhost:3001/api/mastra/health
```

---

## 🔧 How It Works

### The Agent

```typescript
const incidentAgent = new Agent({
  name: 'OpsPilot Incident Classifier',
  model: openai('gpt-4o-mini'),
  tools: [searchKBTool, classifyIncidentTool],
  instructions: `You are an expert IT operations assistant...`
});
```

The agent has access to 2 tools:

#### 1. `search_knowledge_base`
Searches for relevant solutions in the knowledge base.

**Parameters:**
- `query: string` - Search query
- `category?: enum` - Optional category filter

**Example:**
```json
{
  "query": "database connection timeout",
  "category": "Database"
}
```

#### 2. `classify_incident`
Classifies an incident with confidence score.

**Parameters:**
- `title: string`
- `description: string`
- `category: enum` - One of 6 categories
- `severity: enum` - low | medium | high | critical
- `confidence: number` - 0 to 1
- `reasoning: string` - Explanation

**Example:**
```json
{
  "category": "Database",
  "severity": "high",
  "confidence": 0.89,
  "reasoning": "Clear database connection pool exhaustion pattern"
}
```

---

## 🎭 Agent Decision Making

Here's what happens when you call the agent:

### Example: Simple Incident

**Input:**
```
Title: "API timeout"
Description: "Users getting 504 errors"
```

**Agent thinks:** 
"This is vague, I need more context"

**Tool calls:**
1. `search_knowledge_base({ query: "API timeout 504", category: "Application" })`
2. `classify_incident({ category: "Application", severity: "high", ... })`

**Result:** 2 tool calls, more accurate classification

---

### Example: Clear Incident

**Input:**
```
Title: "Database connection timeout"
Description: "PostgreSQL connection pool exhausted. 
             Users getting 500 errors. 
             Started after traffic spike at 2PM."
```

**Agent thinks:**
"This is very clear, I can classify directly"

**Tool calls:**
1. `classify_incident({ category: "Database", severity: "high", ... })`

**Result:** 1 tool call, faster response

---

## 🔍 Type Safety

All tool parameters are validated with Zod:

```typescript
const searchKBTool = {
  name: 'search_knowledge_base',
  parameters: z.object({
    query: z.string(),
    category: z.enum(['Database', 'Network', ...]).optional()
  }),
  execute: async ({ query, category }) => {
    // ✅ query and category are typed!
    // ✅ TypeScript knows their types
    // ✅ Zod validates at runtime
  }
};
```

**Benefits:**
- ❌ No more `JSON.parse()` errors
- ❌ No more type mismatches
- ❌ No more missing parameters
- ✅ Full TypeScript autocomplete
- ✅ Runtime validation
- ✅ Clear error messages

---

## 📊 Observability

Every agent run is automatically traced:

```typescript
const result = await incidentAgent.generate(prompt);

console.log(result.usage);
// {
//   inputTokens: 150,
//   outputTokens: 50,
//   totalTokens: 200
// }

console.log(result.toolCalls);
// [
//   {
//     toolName: 'search_knowledge_base',
//     args: { query: '...', category: 'Database' }
//   },
//   {
//     toolName: 'classify_incident',
//     args: { category: 'Database', severity: 'high', ... }
//   }
// ]
```

**What you get:**
- ✅ Token usage (for cost tracking)
- ✅ All tool calls with arguments
- ✅ Execution time
- ✅ Error details if something fails

---

## 💰 Cost Tracking

GPT-4o-mini pricing:
- Input: $0.15 per 1M tokens
- Output: $0.60 per 1M tokens

Example incident analysis:
```
Input tokens: 150
Output tokens: 50
Total tokens: 200

Cost: $0.000042 (0.0042 cents)
```

For 1,000 incidents per day:
- Daily cost: ~$0.04
- Monthly cost: ~$1.20

**Much cheaper than GPT-4!**

---

## 🔄 vs Traditional Approach

### Old Way (Manual)

```typescript
// 1. Craft prompt manually
const prompt = `Classify this incident: ${title} - ${description}`;

// 2. Call LLM
const response = await chatCompletion(prompt);

// 3. Parse response (hope it's JSON!)
const result = JSON.parse(response); // 🚨 Can fail!

// 4. Validate manually
if (!result.category || !['Database', ...].includes(result.category)) {
  throw new Error('Invalid category');
}

// 5. Log everything manually
logger.info('Classification', result);
```

**Problems:**
- ❌ Manual prompt engineering
- ❌ Fragile JSON parsing
- ❌ No type safety
- ❌ Manual validation
- ❌ Manual logging

---

### New Way (Mastra)

```typescript
const result = await incidentAgent.generate(
  `Analyze: ${title} - ${description}`
);
```

**Benefits:**
- ✅ Agent handles prompting
- ✅ Structured outputs (reliable JSON)
- ✅ Type-safe tool calls
- ✅ Automatic validation (Zod)
- ✅ Automatic logging & tracing

---

## 🎓 Key Concepts

### 1. **Agents**
Autonomous AI that can use tools to solve tasks.

```typescript
const agent = new Agent({
  name: 'My Agent',
  model: openai('gpt-4o-mini'),
  tools: [tool1, tool2],
  instructions: 'You are...'
});
```

### 2. **Tools**
Functions the agent can call.

```typescript
const myTool = {
  name: 'my_tool',
  description: 'What this tool does',
  parameters: z.object({ /* schema */ }),
  execute: async (params) => { /* implementation */ }
};
```

### 3. **Structured Outputs**
Guaranteed JSON format.

```typescript
model: openai('gpt-4o-mini', {
  response_format: { type: 'json_object' }
})
```

### 4. **Observability**
Every run is automatically traced.

```typescript
result.usage       // Token counts
result.toolCalls   // Tools used
result.text        // Final response
```

---

## 🚀 Next Steps

This is just the **beginning**! Mastra can do much more:

1. **Workflows** - Multi-step processes with branching
2. **Memory** - Conversation history & semantic recall
3. **Human-in-the-loop** - Suspend/resume for approvals
4. **RAG** - Advanced vector search with reranking
5. **Evals** - Automated testing and benchmarking

See the main project README for the full integration plan.

---

## 📚 Resources

- [Mastra Documentation](https://mastra.ai/docs)
- [Mastra GitHub](https://github.com/mastra-ai/mastra)
- [Tool Calling Guide](https://mastra.ai/docs/agents/tool-calling)
- [Observability](https://mastra.ai/docs/observability/overview)

---

**Built with ❤️ using Mastra AI**
