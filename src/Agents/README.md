# Agent Framework

## Overview

The Agent Framework enables AI agents to intelligently use tools to solve complex problems. The framework supports multiple reasoning patterns and provides production-ready tools for real-world applications.

## Agent Types

The framework includes four specialized agent types:

### 1. **Agent<T>** (ReAct Pattern)
The foundational agent that alternates between reasoning and acting:
- **Reasoning (Thought)**: Thinking about what to do next
- **Acting (Action)**: Using a tool to perform an operation
- **Observing (Observation)**: Seeing the result of the action
- **Best for**: Dynamic tasks requiring iterative tool use

### 2. **ChainOfThoughtAgent<T>** (Explicit Reasoning)
Breaks down complex problems into explicit step-by-step reasoning:
- Shows detailed logical progression
- Supports pure reasoning mode (no tools) or tool-augmented mode
- **Best for**: Mathematical problems, logical deduction, transparent reasoning
- Based on "Chain-of-Thought Prompting" research (Wei et al., 2022)

### 3. **PlanAndExecuteAgent<T>** (Plan-First Strategy)
Creates a complete plan before execution:
- Plans all steps upfront
- Executes sequentially with optional plan revision
- **Best for**: Multi-step coordinated tasks, workflows with dependencies
- Based on "Least-to-Most Prompting" techniques

### 4. **RAGAgent<T>** (Retrieval-Augmented Generation)
Specialized for knowledge-intensive tasks:
- Retrieves relevant documents from knowledge base
- Optionally reranks for better relevance
- Generates grounded answers with citations
- **Best for**: Q&A over documents, knowledge base queries, factual tasks
- Based on RAG research (Lewis et al., 2020)

## Core Components

### Interfaces

- **`ILanguageModel<T>`**: Unified base interface for all language models
- **`IChatModel<T>`**: Chat model interface extending ILanguageModel
- **`ITool`**: Tool interface for agent actions
- **`IAgent<T>`**: Agent interface for reasoning and execution
- **`IRetriever<T>`**: Document retrieval interface (for RAG)
- **`IReranker<T>`**: Document reranking interface (for RAG)
- **`IGenerator<T>`**: Text generation interface (for RAG)

### Base Classes

- **`AgentBase<T>`**: Abstract base providing common agent functionality
- **`ChatModelBase<T>`**: Abstract base with retry logic and rate limiting
- **`Agent<T>`**: Concrete ReAct agent implementation
- **`ChainOfThoughtAgent<T>`**: Explicit step-by-step reasoning agent
- **`PlanAndExecuteAgent<T>`**: Plan-first execution agent
- **`RAGAgent<T>`**: Knowledge-intensive Q&A agent

### Language Model Providers

Production-ready implementations for major LLM providers:

- **`OpenAIChatModel<T>`**: OpenAI GPT models (GPT-3.5, GPT-4, GPT-4-turbo, GPT-4o)
- **`AnthropicChatModel<T>`**: Anthropic Claude models (Claude 2, Claude 3 family)
- **`AzureOpenAIChatModel<T>`**: Azure-hosted OpenAI models with enterprise features

All models include:
- Retry logic with exponential backoff
- Rate limiting and error handling
- Full parameter support (temperature, top_p, penalties, etc.)
- Comprehensive XML documentation

### Production Tools

- **`CalculatorTool`**: Mathematical expression evaluation
- **`VectorSearchTool<T>`**: Semantic search using IRetriever (dense/hybrid/BM25)
- **`RAGTool<T>`**: Full RAG pipeline (retrieve → rerank → generate)
- **`WebSearchTool`**: Real web search (Bing Search API, SerpAPI)
- **`PredictionModelTool<T,TInput,TOutput>`**: ML model inference integration
- **`SearchTool`**: Mock search for testing and examples

## Quick Start

### Example 1: Simple Agent with Calculator

```csharp
using AiDotNet.Agents;
using AiDotNet.Tools;
using AiDotNet.Interfaces;

// Create a chat model (you'll need to implement IChatModel<T> for your LLM provider)
IChatModel<double> chatModel = new YourChatModel<double>();

// Create tools
var tools = new List<ITool>
{
    new CalculatorTool()
};

// Create the agent
var agent = new Agent<double>(chatModel, tools);

// Run the agent
string result = await agent.RunAsync("What is (25 * 4) + 10?");
Console.WriteLine(result); // Output: 110

// View the reasoning process
Console.WriteLine("\nReasoning Process:");
Console.WriteLine(agent.Scratchpad);
```

### Example 2: Agent with Multiple Tools

```csharp
using AiDotNet.Agents;
using AiDotNet.Tools;
using AiDotNet.Interfaces;

// Create multiple tools
var tools = new List<ITool>
{
    new CalculatorTool(),
    new SearchTool()
};

// Create agent with multiple tools
var agent = new Agent<double>(chatModel, tools);

// The agent can now use both calculator and search
string result = await agent.RunAsync(
    "What is the capital of France and how many letters are in its name?");

Console.WriteLine(result);
Console.WriteLine("\n--- Scratchpad ---");
Console.WriteLine(agent.Scratchpad);
```

### Example 3: Custom Tool Implementation

```csharp
using AiDotNet.Interfaces;

public class WeatherTool : ITool
{
    public string Name => "Weather";

    public string Description =>
        "Gets current weather for a city. " +
        "Input should be a city name. " +
        "Returns temperature and conditions.";

    public string Execute(string input)
    {
        if (string.IsNullOrWhiteSpace(input))
        {
            return "Error: City name cannot be empty.";
        }

        // In a real implementation, you would call a weather API
        // This is a mock for demonstration
        return $"Weather in {input}: Sunny, 72°F";
    }
}

// Use your custom tool
var tools = new List<ITool>
{
    new CalculatorTool(),
    new WeatherTool()
};

var agent = new Agent<double>(chatModel, tools);
string result = await agent.RunAsync("What's the weather in Paris?");
```

### Example 4: Using Production LLM Providers

The framework includes production-ready implementations:

```csharp
using AiDotNet.LanguageModels;

// OpenAI GPT
var openai = new OpenAIChatModel<double>(
    apiKey: "your-openai-api-key",
    modelName: "gpt-4",
    temperature: 0.7
);

// Anthropic Claude
var claude = new AnthropicChatModel<double>(
    apiKey: "your-anthropic-api-key",
    modelName: "claude-3-sonnet-20240229",
    temperature: 0.7
);

// Azure OpenAI
var azureOpenai = new AzureOpenAIChatModel<double>(
    endpoint: "https://your-resource.openai.azure.com",
    apiKey: "your-azure-api-key",
    deploymentName: "gpt-4-deployment"
);

// Use any model with agents
var agent = new Agent<double>(openai, tools);
```

### Example 5: Chain of Thought Agent

For problems requiring explicit step-by-step reasoning:

```csharp
using AiDotNet.Agents;
using AiDotNet.LanguageModels;

var chatModel = new OpenAIChatModel<double>("your-api-key");
var calculator = new CalculatorTool();

// Pure reasoning mode (no tools)
var pureCoT = new ChainOfThoughtAgent<double>(chatModel, allowTools: false);
var result = await pureCoT.RunAsync("If I have 3 apples and buy 2 more, then give 1 away, how many do I have?");

// With tools for complex calculations
var cotWithTools = new ChainOfThoughtAgent<double>(chatModel, new[] { calculator });
var result2 = await cotWithTools.RunAsync("What is (125 * 8) + (64 / 4)?");

Console.WriteLine(cotWithTools.Scratchpad); // See step-by-step reasoning
```

### Example 6: Plan and Execute Agent

For multi-step tasks requiring coordination:

```csharp
using AiDotNet.Agents;

var tools = new ITool[] {
    new CalculatorTool(),
    new WebSearchTool(apiKey: "bing-api-key")
};

// Agent creates plan first, then executes
var planAgent = new PlanAndExecuteAgent<double>(chatModel, tools);

var result = await planAgent.RunAsync(
    "Find the current population of Tokyo and calculate what 5% of that would be");

// See the plan and execution
Console.WriteLine(planAgent.Scratchpad);
/*
=== PLANNING PHASE ===
Plan created:
  1. Search for current population of Tokyo
  2. Calculate 5% of the population
  3. Provide the final answer

=== EXECUTION PHASE ===
Step 1/3: Search for current population of Tokyo
Result: Tokyo population is approximately 14 million

Step 2/3: Calculate 5% of 14 million
Result: 700000

Step 3/3: Provide the final answer
*/
```

### Example 7: RAG Agent for Knowledge-Intensive Q&A

For answering questions based on a knowledge base:

```csharp
using AiDotNet.Agents;
using AiDotNet.RetrievalAugmentedGeneration;

// Set up RAG components (retriever, reranker, generator)
var retriever = new HybridRetriever<double>(vectorStore, bm25Index);
var reranker = new CrossEncoderReranker<double>();
var generator = new OpenAIGenerator<double>("your-api-key");
var chatModel = new OpenAIChatModel<double>("your-api-key");

// Create RAG agent
var ragAgent = new RAGAgent<double>(
    chatModel: chatModel,
    retriever: retriever,
    generator: generator,
    reranker: reranker,
    retrievalTopK: 10,      // Get 10 candidates
    rerankTopK: 5,          // Keep top 5
    includeCitations: true  // Show sources
);

// Ask questions about your knowledge base
var result = await ragAgent.RunAsync(
    "What are the system requirements for the Enterprise plan?");

Console.WriteLine(result);
/*
The Enterprise plan requires:
- 16GB RAM minimum [1]
- 100GB disk space [2]
- Network bandwidth of at least 100Mbps [1]

Sources:
  [1] Enterprise Setup Guide (section 2.1)
  [2] Infrastructure Requirements Document

Confidence: 95%
*/

// See the RAG pipeline execution
Console.WriteLine("\n--- Pipeline Details ---");
Console.WriteLine(ragAgent.Scratchpad);
/*
Query: What are the system requirements for the Enterprise plan?

=== RETRIEVAL PHASE ===
Searching knowledge base for top 10 documents...
Retrieved 10 documents.

=== RERANKING PHASE ===
Reranking documents for better relevance...
Kept top 5 documents after reranking.

=== GENERATION PHASE ===
Generating answer from 5 context documents...
Answer generated successfully.
*/
```

### Example 8: Using Production Tools

The framework includes several production-ready tools:

```csharp
using AiDotNet.Tools;

// Vector search over knowledge base
var vectorSearch = new VectorSearchTool<double>(retriever, topK: 5);

// Full RAG pipeline as a tool
var ragTool = new RAGTool<double>(retriever, reranker, generator);

// Real web search
var webSearch = new WebSearchTool(
    apiKey: "your-bing-api-key",
    provider: WebSearchTool.SearchProvider.Bing
);

// ML model inference
var predictionTool = PredictionModelTool<double, Vector<double>, Vector<double>>
    .CreateVectorInputTool(
        trainedModel,
        "HousePricePredictor",
        "Predicts house prices. Input: JSON array [sqft, bedrooms, bathrooms, age]"
    );

// Use all tools with an agent
var tools = new ITool[] {
    new CalculatorTool(),
    vectorSearch,
    ragTool,
    webSearch,
    predictionTool
};

var powerAgent = new Agent<double>(chatModel, tools);
var result = await powerAgent.RunAsync(
    "Search the web for current Seattle house prices, then predict a price for a 2000 sqft, 3 bed, 2 bath house built in 2010");
```

## How the ReAct Loop Works

When you call `agent.RunAsync("query")`, the agent:

1. **Iteration 1**:
   - Sends a prompt to the LLM with the query and available tools
   - LLM responds with a thought and action (or final answer)
   - If action specified, executes the tool and records the observation

2. **Iteration 2**:
   - Sends updated prompt including previous thoughts, actions, and observations
   - LLM decides next step based on accumulated context
   - Continues until final answer or max iterations reached

3. **Returns**: Final answer or partial results if max iterations reached

## Advanced Configuration

### Setting Max Iterations

```csharp
// Allow up to 10 reasoning steps
string result = await agent.RunAsync("Complex query", maxIterations: 10);
```

### Accessing the Scratchpad

The scratchpad contains the complete reasoning trace:

```csharp
string result = await agent.RunAsync("Calculate 5 + 3");
Console.WriteLine("Final Answer: " + result);
Console.WriteLine("\nReasoning Trace:");
Console.WriteLine(agent.Scratchpad);

/* Output:
Query: Calculate 5 + 3

=== Iteration 1 ===
Thought: I need to use the calculator
Action: Calculator
Action Input: 5 + 3
Observation: 8

=== Iteration 2 ===
Thought: I have the answer
Final Answer: 8
*/
```

## Testing

The framework includes comprehensive unit tests. See:
- `tests/UnitTests/Agents/AgentTests.cs`
- `tests/UnitTests/Tools/CalculatorToolTests.cs`
- `tests/UnitTests/Tools/SearchToolTests.cs`

### Mock Chat Model for Testing

```csharp
using AiDotNetTests.UnitTests.Agents;

var mockModel = new MockChatModel<double>(
    "{\"thought\": \"I'll calculate\", \"action\": \"Calculator\", \"action_input\": \"2 + 2\"}",
    "{\"thought\": \"Got it\", \"final_answer\": \"4\"}"
);

var agent = new Agent<double>(mockModel, tools);
var result = await agent.RunAsync("What is 2 + 2?");

Assert.Equal("4", result);
```

## LLM Response Format

The agent expects the LLM to respond in JSON format:

```json
{
  "thought": "I need to search for information about X",
  "action": "Search",
  "action_input": "query about X",
  "final_answer": ""
}
```

Or when done:

```json
{
  "thought": "I have all the information I need",
  "action": "",
  "action_input": "",
  "final_answer": "The answer is Y"
}
```

The agent also has fallback regex parsing if the LLM doesn't return valid JSON.

## Best Practices

1. **Tool Design**: Keep tools focused on a single responsibility
2. **Descriptions**: Write clear, detailed tool descriptions
3. **Error Handling**: Tools should return error messages, not throw exceptions
4. **Iterations**: Start with 5-10 max iterations; adjust based on task complexity
5. **Testing**: Use MockChatModel to test agent behavior without calling real APIs

## Integration with AiModelBuilder

While the agent framework is standalone, you can integrate it with the broader AiDotNet ecosystem:

```csharp
// Example: Use agent to help with feature selection
var featureAgent = new Agent<double>(chatModel, featureTools);
var recommendation = await featureAgent.RunAsync(
    "Which features should I select for predicting house prices?");
```

## Future Extensions

Potential areas for extension:

- **Memory**: Add long-term memory for agents
- **Multi-agent Systems**: Enable multiple agents to collaborate
- **Tool Learning**: Allow agents to learn new tools dynamically
- **Hierarchical Agents**: Parent agents that delegate to child agents
- **Evaluation Metrics**: Track success rates and reasoning quality
