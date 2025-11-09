# Advanced Reasoning and Chain-of-Thought Implementation Guide

This guide covers the advanced reasoning capabilities implemented in AiDotNet, including Chain-of-Thought, Tree-of-Thoughts, Verified Reasoning, Multi-Step Reasoning, and Tool-Augmented Reasoning.

## Table of Contents

1. [Overview](#overview)
2. [Chain-of-Thought (CoT)](#chain-of-thought-cot)
3. [Tree-of-Thoughts (ToT)](#tree-of-thoughts-tot)
4. [Verified Reasoning](#verified-reasoning)
5. [Multi-Step Reasoning](#multi-step-reasoning)
6. [Tool-Augmented Reasoning](#tool-augmented-reasoning)
7. [Best Practices](#best-practices)
8. [Performance Considerations](#performance-considerations)

## Overview

The advanced reasoning system provides several sophisticated patterns for improving retrieval and reasoning quality:

- **Chain-of-Thought (CoT)**: Breaks down complex queries into sequential reasoning steps
- **Tree-of-Thoughts (ToT)**: Explores multiple reasoning paths in a tree structure
- **Verified Reasoning**: Validates each reasoning step with critic models
- **Multi-Step Reasoning**: Adapts reasoning based on previous step findings
- **Tool-Augmented Reasoning**: Incorporates external tools (calculators, code execution) into reasoning

## Chain-of-Thought (CoT)

### Basic Usage

```csharp
using AiDotNet.RetrievalAugmentedGeneration.AdvancedPatterns;
using AiDotNet.RetrievalAugmentedGeneration.Generators;
using AiDotNet.RetrievalAugmentedGeneration.Retrievers;

// Setup
var generator = new StubGenerator<double>(); // Replace with real LLM in production
var baseRetriever = new DenseRetriever<double>(embeddingModel, documentStore);

// Create Chain-of-Thought retriever
var cotRetriever = new ChainOfThoughtRetriever<double>(
    generator,
    baseRetriever
);

// Retrieve with reasoning
var documents = cotRetriever.Retrieve(
    "What are the economic impacts of renewable energy adoption?",
    topK: 10
);
```

### Self-Consistency Mode

For improved robustness, generate multiple reasoning paths:

```csharp
var documents = cotRetriever.RetrieveWithSelfConsistency(
    "What are the economic impacts of renewable energy adoption?",
    topK: 10,
    numPaths: 3  // Generate 3 different reasoning paths
);
```

### Few-Shot Examples

Provide examples to guide reasoning quality:

```csharp
var fewShotExamples = new List<string>
{
    @"Question: How does photosynthesis affect climate?
    Step 1: Understand photosynthesis process
    Step 2: Identify CO2 absorption mechanism
    Step 3: Connect to climate change impact",

    @"Question: What are the benefits of exercise?
    Step 1: Identify physical health benefits
    Step 2: Explore mental health aspects
    Step 3: Consider long-term wellness outcomes"
};

var cotRetriever = new ChainOfThoughtRetriever<double>(
    generator,
    baseRetriever,
    fewShotExamples
);
```

## Tree-of-Thoughts (ToT)

### Basic Usage

```csharp
using AiDotNet.RetrievalAugmentedGeneration.AdvancedPatterns;

var totRetriever = new TreeOfThoughtsRetriever<double>(
    generator,
    baseRetriever,
    maxDepth: 3,           // Explore 3 levels deep
    branchingFactor: 3     // Generate 3 alternatives at each level
);

var documents = totRetriever.Retrieve(
    "What are the applications of quantum computing?",
    topK: 15,
    TreeOfThoughtsRetriever<double>.TreeSearchStrategy.BestFirst
);
```

### Search Strategies

**Breadth-First Search**: Explores all nodes at each level before going deeper
```csharp
var documents = totRetriever.Retrieve(
    query,
    topK: 15,
    TreeOfThoughtsRetriever<double>.TreeSearchStrategy.BreadthFirst
);
```

**Depth-First Search**: Explores one branch fully before backtracking
```csharp
var documents = totRetriever.Retrieve(
    query,
    topK: 15,
    TreeOfThoughtsRetriever<double>.TreeSearchStrategy.DepthFirst
);
```

**Best-First Search**: Always explores the highest-scored node next (recommended)
```csharp
var documents = totRetriever.Retrieve(
    query,
    topK: 15,
    TreeOfThoughtsRetriever<double>.TreeSearchStrategy.BestFirst
);
```

### When to Use ToT

- Multiple valid reasoning approaches exist
- Comprehensive topic coverage needed
- Exploring alternative solution paths
- Complex problem spaces requiring systematic exploration

## Verified Reasoning

### Basic Usage

```csharp
var verifiedRetriever = new VerifiedReasoningRetriever<double>(
    generator,
    baseRetriever,
    verificationThreshold: 0.7,  // Minimum score to accept a step
    maxRefinementAttempts: 2      // Try to refine weak steps up to 2 times
);

var result = verifiedRetriever.RetrieveWithVerification(
    "What are the safety considerations for gene therapy?",
    topK: 10
);

// Access results
Console.WriteLine($"Average Verification Score: {result.AverageVerificationScore}");
Console.WriteLine($"Steps Refined: {result.RefinedStepsCount}");

foreach (var step in result.VerifiedSteps)
{
    Console.WriteLine($"Step: {step.Statement}");
    Console.WriteLine($"Verified: {step.IsVerified}");
    Console.WriteLine($"Score: {step.VerificationScore}");
    Console.WriteLine($"Feedback: {step.CritiqueFeedback}");
    Console.WriteLine();
}
```

### Configuration

**High Precision Mode** (strict verification):
```csharp
var verifiedRetriever = new VerifiedReasoningRetriever<double>(
    generator,
    baseRetriever,
    verificationThreshold: 0.9,   // Very strict
    maxRefinementAttempts: 3       // More refinement attempts
);
```

**Balanced Mode** (recommended):
```csharp
var verifiedRetriever = new VerifiedReasoningRetriever<double>(
    generator,
    baseRetriever,
    verificationThreshold: 0.7,   // Balanced
    maxRefinementAttempts: 2       // Reasonable refinement
);
```

### When to Use Verified Reasoning

- Accuracy is critical (medical, legal, scientific domains)
- Need to avoid hallucinations
- Require transparent, auditable reasoning
- Working with high-stakes decisions

## Multi-Step Reasoning

### Basic Usage

```csharp
var multiStepRetriever = new MultiStepReasoningRetriever<double>(
    generator,
    baseRetriever,
    maxSteps: 5  // Allow up to 5 reasoning steps
);

var result = multiStepRetriever.RetrieveMultiStep(
    "What are the environmental and economic impacts of solar energy adoption?",
    topK: 15
);

// Access results
Console.WriteLine($"Total Steps: {result.TotalSteps}");
Console.WriteLine($"Converged: {result.Converged}");
Console.WriteLine("\nReasoning Trace:");
Console.WriteLine(result.ReasoningTrace);

foreach (var step in result.StepResults)
{
    Console.WriteLine($"\nStep {step.StepNumber}: {step.StepQuery}");
    Console.WriteLine($"Summary: {step.StepSummary}");
    Console.WriteLine($"Documents Found: {step.Documents.Count}");
}
```

### Adaptive Reasoning

Multi-step reasoning adapts based on what it learns:

```csharp
// The retriever will:
// 1. Start with the initial query
// 2. Based on findings, determine the next step
// 3. Each step builds on previous knowledge
// 4. Stop when sufficient information is gathered or max steps reached
```

### When to Use Multi-Step Reasoning

- Answers require building knowledge progressively
- Later steps depend on earlier findings
- Need to adapt search strategy based on discoveries
- Complex research-style queries

## Tool-Augmented Reasoning

### Basic Usage

```csharp
var toolRetriever = new ToolAugmentedReasoningRetriever<double>(
    generator,
    baseRetriever
);

// Default tools: calculator, text_analyzer

var result = toolRetriever.RetrieveWithTools(
    "What is the compound annual growth rate of solar installations from 2015 to 2023?",
    topK: 10
);

// Access results
Console.WriteLine("Tool Invocations:");
foreach (var invocation in result.ToolInvocations)
{
    Console.WriteLine($"Tool: {invocation.ToolName}");
    Console.WriteLine($"Input: {invocation.Input}");
    Console.WriteLine($"Output: {invocation.Output}");
    Console.WriteLine($"Success: {invocation.Success}");
}
```

### Custom Tools

Register custom tools for domain-specific tasks:

```csharp
// Register a unit converter tool
toolRetriever.RegisterTool("unit_converter", input =>
{
    // Parse input like "100 celsius to fahrenheit"
    // Return conversion result
    return "212 fahrenheit";
});

// Register a data lookup tool
toolRetriever.RegisterTool("database_lookup", input =>
{
    // Query database based on input
    var result = database.Query(input);
    return result.ToString();
});

// Register a code execution tool
toolRetriever.RegisterTool("python_executor", input =>
{
    // Execute Python code safely
    var output = pythonEngine.Execute(input);
    return output;
});
```

### When to Use Tool-Augmented Reasoning

- Queries require calculations
- Need to execute code or scripts
- Require data transformations
- Need to access external APIs or databases

## Best Practices

### 1. Choose the Right Pattern

| Pattern | Best For | Complexity | Cost |
|---------|----------|------------|------|
| Chain-of-Thought | General complex queries | Low | Low |
| CoT + Self-Consistency | Robustness needed | Medium | Medium |
| Tree-of-Thoughts | Comprehensive exploration | High | High |
| Verified Reasoning | High-stakes accuracy | Medium | Medium |
| Multi-Step | Progressive knowledge building | Medium | Medium |
| Tool-Augmented | Computational tasks | Low-Medium | Low-Medium |

### 2. Combine Patterns

Patterns can be combined for enhanced capabilities:

```csharp
// Use Tree-of-Thoughts with Verified Reasoning
// 1. Explore multiple paths with ToT
// 2. Verify each path with VerifiedReasoningRetriever
// This requires custom integration but provides maximum quality
```

### 3. Optimize for Production

**Use Real LLM Generators**:
```csharp
// Replace StubGenerator with real LLM
var generator = new OpenAIGenerator<double>(apiKey, model: "gpt-4");
// Note: OpenAIGenerator and AnthropicGenerator are placeholder examples
// Use actual LLM integration libraries like Azure.AI.OpenAI or Anthropic SDK
// or
var generator = new AnthropicGenerator<double>(apiKey, model: "claude-3-opus");
```

**Configure Caching**:
```csharp
// Implement caching for repeated queries
var cache = new RedisReasoningCache<double>(redisConnection);
// Note: RedisReasoningCache is a placeholder example
// Use actual caching libraries like StackExchange.Redis or Microsoft.Extensions.Caching
// Check cache before expensive reasoning operations
```

**Monitor Costs**:
```csharp
// Track LLM API calls
// Chain-of-Thought: ~2-5 calls per query
// Tree-of-Thoughts: ~10-30 calls per query (depth * branching factor)
// Verified Reasoning: ~5-15 calls per query (depends on refinements)
```

### 4. Error Handling

```csharp
try
{
    var result = cotRetriever.Retrieve(query, topK: 10);
}
catch (ArgumentException ex)
{
    // Handle invalid input
    Console.WriteLine($"Invalid query: {ex.Message}");
}
catch (Exception ex)
{
    // Handle LLM failures, network issues, etc.
    Console.WriteLine($"Reasoning failed: {ex.Message}");
    // Fallback to simple retrieval
    var fallbackResults = baseRetriever.Retrieve(query, topK: 10);
}
```

## Performance Considerations

### Latency

| Pattern | Typical Latency | LLM Calls |
|---------|----------------|-----------|
| Chain-of-Thought | 2-5 seconds | 2-5 |
| CoT + Self-Consistency (3 paths) | 5-10 seconds | 6-15 |
| Tree-of-Thoughts (depth=3, branch=3) | 10-30 seconds | 10-30 |
| Verified Reasoning | 5-15 seconds | 5-15 |
| Multi-Step (max 5 steps) | 5-20 seconds | 5-20 |
| Tool-Augmented | 3-8 seconds | 3-8 |

### Cost Optimization

**Reduce Tree Depth/Branching**:
```csharp
// Instead of maxDepth=5, branchingFactor=5
var totRetriever = new TreeOfThoughtsRetriever<double>(
    generator,
    baseRetriever,
    maxDepth: 2,           // Reduced from 5
    branchingFactor: 2     // Reduced from 5
);
```

**Lower Verification Threshold**:
```csharp
// Accept more steps without refinement
var verifiedRetriever = new VerifiedReasoningRetriever<double>(
    generator,
    baseRetriever,
    verificationThreshold: 0.5,  // Lower threshold
    maxRefinementAttempts: 1      // Fewer attempts
);
```

**Use Smaller Models for Sub-Tasks**:
```csharp
// Use GPT-3.5 for reasoning, GPT-4 only for verification
var reasoningGenerator = new OpenAIGenerator<double>(apiKey, "gpt-3.5-turbo");
var verificationGenerator = new OpenAIGenerator<double>(apiKey, "gpt-4");
```

## Example: Complete Workflow

```csharp
using AiDotNet.RetrievalAugmentedGeneration.AdvancedPatterns;
using AiDotNet.RetrievalAugmentedGeneration.Generators;
using AiDotNet.RetrievalAugmentedGeneration.Retrievers;

// Setup components
var generator = new StubGenerator<double>(); // Use real LLM in production
var embeddingModel = new SentenceTransformerEmbedding<double>();
var documentStore = new InMemoryDocumentStore<double>();
var baseRetriever = new DenseRetriever<double>(embeddingModel, documentStore);

// Example 1: Simple Chain-of-Thought
var cotRetriever = new ChainOfThoughtRetriever<double>(generator, baseRetriever);
var cotResults = cotRetriever.Retrieve(
    "How does climate change affect biodiversity?",
    topK: 10
);

// Example 2: Tree-of-Thoughts for comprehensive exploration
var totRetriever = new TreeOfThoughtsRetriever<double>(
    generator,
    baseRetriever,
    maxDepth: 3,
    branchingFactor: 3
);
var totResults = totRetriever.Retrieve(
    "What are all the potential applications of CRISPR gene editing?",
    topK: 20,
    TreeOfThoughtsRetriever<double>.TreeSearchStrategy.BestFirst
);

// Example 3: Verified Reasoning for high-stakes query
var verifiedRetriever = new VerifiedReasoningRetriever<double>(
    generator,
    baseRetriever,
    verificationThreshold: 0.8,
    maxRefinementAttempts: 2
);
var verifiedResult = verifiedRetriever.RetrieveWithVerification(
    "What are the approved treatments for stage 4 melanoma?",
    topK: 10
);

// Example 4: Multi-Step for research query
var multiStepRetriever = new MultiStepReasoningRetriever<double>(
    generator,
    baseRetriever,
    maxSteps: 5
);
var multiStepResult = multiStepRetriever.RetrieveMultiStep(
    "Trace the historical development of quantum mechanics from 1900 to 1930",
    topK: 15
);

// Example 5: Tool-Augmented for computational query
var toolRetriever = new ToolAugmentedReasoningRetriever<double>(generator, baseRetriever);
toolRetriever.RegisterTool("statistics", input =>
{
    // Custom statistical calculations
    return "Mean: 42.5, Std Dev: 12.3";
});

var toolResult = toolRetriever.RetrieveWithTools(
    "Calculate the average and standard deviation of solar panel efficiency from 2010 to 2020",
    topK: 10
);
```

## Conclusion

The advanced reasoning system provides powerful tools for improving retrieval quality in complex scenarios. Choose the appropriate pattern based on your requirements for accuracy, comprehensiveness, and computational cost.

For more information, see:
- [Chain-of-Thought paper](https://arxiv.org/abs/2201.11903)
- [Tree-of-Thoughts paper](https://arxiv.org/abs/2305.10601)
- [Self-Consistency paper](https://arxiv.org/abs/2203.11171)
