# Language Models

Production-ready language model implementations for the AiDotNet agent framework.

## Overview

This directory contains concrete implementations of `ILanguageModel<T>` and `IChatModel<T>` for various LLM providers. These models can be used with agents or standalone for text generation tasks.

## Available Models

### OpenAIChatModel<T>
OpenAI's GPT models (GPT-3.5-turbo, GPT-4, GPT-4-turbo).

**Features:**
- Automatic retry logic with exponential backoff
- Rate limiting support
- Configurable temperature, top_p, penalties
- Token limit enforcement
- Comprehensive error handling

**Example:**
```csharp
using AiDotNet.LanguageModels;
using AiDotNet.Agents;
using AiDotNet.Tools;

// Create the language model
var chatModel = new OpenAIChatModel<double>(
    apiKey: "your-openai-api-key",
    modelName: "gpt-4",
    temperature: 0.7,
    maxTokens: 2048
);

// Use with agents
var tools = new List<ITool> { new CalculatorTool() };
var agent = new Agent<double>(chatModel, tools);
string result = await agent.RunAsync("What is 15 * 23 + 100?");

// Or use directly
string response = await chatModel.GenerateAsync("Explain quantum computing");
```

## ChatModelBase<T>

Abstract base class providing common functionality for all chat models:

- **HTTP Client Management**: Reusable HTTP client with proper configuration
- **Retry Logic**: Exponential backoff for transient failures (network issues, rate limits)
- **Error Handling**: Comprehensive exception handling and logging
- **Token Estimation**: Rough token counting for prompt validation
- **Synchronous Wrapper**: `Generate()` method wraps async for simple scripts

## Configuration

### OpenAI Models

```csharp
var model = new OpenAIChatModel<double>(
    apiKey: "sk-...",                    // Required: Your OpenAI API key
    modelName: "gpt-4",                  // Optional: Default = "gpt-3.5-turbo"
    temperature: 0.7,                    // Optional: 0.0-2.0, Default = 0.7
    maxTokens: 2048,                     // Optional: Default = 2048
    topP: 1.0,                          // Optional: 0.0-1.0, Default = 1.0
    frequencyPenalty: 0.0,              // Optional: -2.0 to 2.0, Default = 0.0
    presencePenalty: 0.0,               // Optional: -2.0 to 2.0, Default = 0.0
    httpClient: customHttpClient,        // Optional: For advanced scenarios
    endpoint: "https://..."              // Optional: For Azure OpenAI or proxies
);
```

### Model Selection Guide

**gpt-3.5-turbo** ($0.0005/1K tokens)
- Best for: Simple tasks, high-volume, cost-sensitive applications
- Context: 4K tokens (3 pages)
- Speed: Very fast
- Use cases: Basic Q&A, simple classification, formatting

**gpt-3.5-turbo-16k** ($0.001/1K tokens)
- Best for: Longer documents, more context
- Context: 16K tokens (12 pages)
- Speed: Fast
- Use cases: Document summarization, multi-turn conversations

**gpt-4** ($0.03/1K tokens)
- Best for: Complex reasoning, accuracy-critical tasks
- Context: 8K tokens (6 pages)
- Speed: Moderate
- Use cases: Code generation, complex analysis, creative writing

**gpt-4-turbo** ($0.01/1K tokens)
- Best for: Balance of capability and cost
- Context: 128K tokens (96 pages)
- Speed: Fast
- Use cases: Long document analysis, comprehensive research

**gpt-4o** ($0.005/1K tokens)
- Best for: Optimized performance at lower cost
- Context: 128K tokens
- Speed: Very fast
- Use cases: General purpose, multimodal tasks

### Temperature Guide

```csharp
temperature: 0.0   // Deterministic, consistent, factual
temperature: 0.3   // Slightly creative, mostly consistent
temperature: 0.7   // Balanced (default)
temperature: 1.0   // Creative, varied
temperature: 1.5+  // Very creative, potentially less coherent
```

## Error Handling

All models include automatic retry logic:

```csharp
var model = new OpenAIChatModel<double>("api-key");
model.MaxRetries = 3;                    // Default: 3
model.InitialRetryDelayMs = 1000;        // Default: 1000ms
model.TimeoutMs = 120000;                // Default: 2 minutes
model.EnableDetailedLogging = true;      // Default: false
```

Retryable errors:
- Network failures
- Rate limits (429)
- Server errors (5xx)
- Timeouts (408)

Non-retryable errors:
- Invalid API key (401)
- Invalid request (400)
- Not found (404)

## Best Practices

### 1. Use Environment Variables for API Keys

```csharp
var apiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY");
var model = new OpenAIChatModel<double>(apiKey);
```

### 2. Reuse HttpClient

```csharp
// Good: Single HttpClient for all models
private static readonly HttpClient SharedHttpClient = new HttpClient();

var model1 = new OpenAIChatModel<double>("key1", httpClient: SharedHttpClient);
var model2 = new OpenAIChatModel<double>("key2", httpClient: SharedHttpClient);
```

### 3. Set Appropriate Timeouts

```csharp
var model = new OpenAIChatModel<double>("api-key");
model.TimeoutMs = 30000;  // 30 seconds for quick responses
```

### 4. Monitor Token Usage

```csharp
int estimatedTokens = model.EstimateTokenCount(prompt);
if (estimatedTokens > model.MaxContextTokens * 0.8)
{
    // Prompt is getting large, consider summarizing
}
```

### 5. Handle Errors Gracefully

```csharp
try
{
    var response = await model.GenerateAsync(prompt);
}
catch (HttpRequestException ex) when (ex.StatusCode == System.Net.HttpStatusCode.Unauthorized)
{
    logger.LogError("Invalid API key");
}
catch (TaskCanceledException)
{
    logger.LogError("Request timed out");
}
catch (InvalidOperationException ex)
{
    logger.LogError($"API error: {ex.Message}");
}
```

## Integration with Agents

```csharp
using AiDotNet.LanguageModels;
using AiDotNet.Agents;
using AiDotNet.Tools;

// Create language model
var llm = new OpenAIChatModel<double>(
    apiKey: Environment.GetEnvironmentVariable("OPENAI_API_KEY"),
    modelName: "gpt-4",
    temperature: 0.2  // Lower temperature for more focused reasoning
);

// Create tools
var tools = new List<ITool>
{
    new CalculatorTool(),
    new SearchTool()
};

// Create agent
var agent = new Agent<double>(llm, tools);

// Run agent
var result = await agent.RunAsync(
    "What is the capital of France and what is 100 + 50?",
    maxIterations: 10
);

Console.WriteLine(result);
Console.WriteLine("\n--- Reasoning Process ---");
Console.WriteLine(agent.Scratchpad);
```

## Cost Optimization

### 1. Choose the Right Model

```csharp
// For simple tasks
var cheapModel = new OpenAIChatModel<double>(apiKey, modelName: "gpt-3.5-turbo");

// For complex reasoning
var powerfulModel = new OpenAIChatModel<double>(apiKey, modelName: "gpt-4");
```

### 2. Limit Response Length

```csharp
var model = new OpenAIChatModel<double>(
    apiKey: apiKey,
    maxTokens: 500  // Shorter responses = lower cost
);
```

### 3. Use Caching

```csharp
private readonly Dictionary<string, string> _cache = new();

public async Task<string> GetCachedResponseAsync(string prompt)
{
    if (_cache.TryGetValue(prompt, out var cached))
        return cached;

    var response = await model.GenerateAsync(prompt);
    _cache[prompt] = response;
    return response;
}
```

## Testing

Use MockChatModel for testing without API calls:

```csharp
using AiDotNetTests.UnitTests.Agents;

var mockModel = new MockChatModel<double>(
    "First response",
    "Second response",
    "Third response"
);

var agent = new Agent<double>(mockModel, tools);
var result = await agent.RunAsync("Test query");

// Verify calls
Assert.Single(mockModel.ReceivedPrompts);
```

## Future Models

Planned implementations:
- **AnthropicChatModel**: Claude 2, Claude 3 (Opus, Sonnet, Haiku)
- **AzureOpenAIChatModel**: Azure-hosted OpenAI models
- **OllamaChatModel**: Local model support (Llama 2, Mixtral, etc.)
- **HuggingFaceChatModel**: Hugging Face Inference API
