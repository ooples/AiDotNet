# Prompt Engineering and LLM Tools Framework

## Overview

The AiDotNet Prompt Engineering framework provides comprehensive infrastructure for working with Large Language Models (LLMs), including prompt templates, tool/function calling, in-context learning, prompt optimization, and chain construction.

This framework enables:
- **LLM application development** with production-ready components
- **Automated prompt optimization** for better performance
- **Multi-step complex workflows** using chains
- **Tool-augmented LLM systems** with function calling
- **Few-shot learning** with intelligent example selection

## Table of Contents

1. [Prompt Templates](#prompt-templates)
2. [Tool/Function Calling](#tool-function-calling)
3. [Few-Shot Learning](#few-shot-learning)
4. [Prompt Optimization](#prompt-optimization)
5. [Chain Construction](#chain-construction)
6. [Quick Start Examples](#quick-start-examples)

---

## Prompt Templates

Prompt templates provide structured ways to create prompts with variable substitution, examples, and formatting.

### Template Types

#### 1. Simple Template
Basic variable substitution for straightforward prompts.

```csharp
using AiDotNet.PromptEngineering.Templates;

var template = new SimplePromptTemplate("Translate {text} to {language}");

var prompt = template.Format(new Dictionary<string, string>
{
    ["text"] = "Hello world",
    ["language"] = "Spanish"
});
// Result: "Translate Hello world to Spanish"
```

#### 2. Few-Shot Template
Include examples to guide model behavior.

```csharp
using AiDotNet.PromptEngineering.Templates;
using AiDotNet.PromptEngineering.FewShot;

// Create example selector
var selector = new RandomExampleSelector<double>();
selector.AddExample(new FewShotExample
{
    Input = "I loved this product!",
    Output = "Positive"
});
selector.AddExample(new FewShotExample
{
    Input = "Terrible experience.",
    Output = "Negative"
});

// Create few-shot template
var template = new FewShotPromptTemplate<double>(
    template: "Classify sentiment:\n\n{examples}\n\nText: {text}\nSentiment:",
    exampleSelector: selector,
    exampleCount: 2
);

var prompt = template.Format(new Dictionary<string, string>
{
    ["text"] = "This is amazing!"
});
```

#### 3. Chat Template
Structure conversations with role-based messages.

```csharp
var template = new ChatPromptTemplate()
    .AddSystemMessage("You are a helpful math tutor")
    .AddUserMessage("What is 2 + 2?")
    .AddAssistantMessage("2 + 2 equals 4")
    .AddUserMessage("What about 5 + 3?");

var prompt = template.Format(new Dictionary<string, string>());
```

### Using the Factory

```csharp
using AiDotNet.Factories;
using AiDotNet.Enums;

var template = PromptTemplateFactory.Create(
    PromptTemplateType.Simple,
    "Summarize {document}"
);
```

For Few-Shot templates, use the generic factory overload so your selector type matches:

```csharp
using AiDotNet.Factories;
using AiDotNet.Enums;
using AiDotNet.PromptEngineering.FewShot;

var selector = new RandomExampleSelector<double>();
selector.AddExample(new FewShotExample { Input = "Hello", Output = "Hola" });
selector.AddExample(new FewShotExample { Input = "Goodbye", Output = "Adios" });

var template = PromptTemplateFactory.Create<double>(
    PromptTemplateType.FewShot,
    "Translate English to Spanish.\n\n{examples}\n\nNow: {query}",
    selector,
    exampleCount: 2
);
```

---

## Tool/Function Calling

Enable LLMs to use external functions and APIs.

### Creating a Custom Tool

```csharp
using AiDotNet.PromptEngineering.Tools;
using System.Text.Json;

public class CalculatorTool : FunctionToolBase
{
    public CalculatorTool() : base(
        name: "calculator",
        description: "Performs basic arithmetic operations",
        parameterSchema: JsonDocument.Parse("""
        {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"]
                },
                "a": { "type": "number" },
                "b": { "type": "number" }
            },
            "required": ["operation", "a", "b"]
        }
        """))
    {
    }

    protected override string ExecuteCore(JsonDocument arguments)
    {
        var op = arguments.RootElement.GetProperty("operation").GetString();
        var a = arguments.RootElement.GetProperty("a").GetDouble();
        var b = arguments.RootElement.GetProperty("b").GetDouble();

        return op switch
        {
            "add" => (a + b).ToString(),
            "subtract" => (a - b).ToString(),
            "multiply" => (a * b).ToString(),
            "divide" => (a / b).ToString(),
            _ => "Unknown operation"
        };
    }
}
```

### Using the Tool Registry

```csharp
using AiDotNet.PromptEngineering.Tools;

var registry = new ToolRegistry();
registry.RegisterTool(new CalculatorTool());

// Execute a tool
var args = JsonDocument.Parse("""
{
    "operation": "add",
    "a": 15,
    "b": 27
}
""");

var result = registry.ExecuteTool("calculator", args);
// Result: "42"

// Get tools description for LLM prompt
var toolsDescription = registry.GenerateToolsDescription();
```

---

## Few-Shot Learning

Intelligently select examples to include in prompts.

### Selection Strategies

#### Random Selection
```csharp
var selector = new RandomExampleSelector<double>(seed: 42);
```

#### Fixed Order Selection
```csharp
var selector = new FixedExampleSelector<double>();
```

### Adding Examples

```csharp
selector.AddExample(new FewShotExample
{
    Input = "What is the capital of France?",
    Output = "The capital of France is Paris.",
    Metadata = new Dictionary<string, string>
    {
        ["category"] = "geography",
        ["difficulty"] = "easy"
    }
});

// Select examples for a query
var examples = selector.SelectExamples("What is the capital of Spain?", count: 3);
```

---

## Prompt Optimization

Automatically improve prompts for better performance.

### Discrete Search Optimizer

```csharp
using AiDotNet.PromptEngineering.Optimization;

var optimizer = new DiscreteSearchOptimizer<double>();

// Define evaluation function (returns accuracy score)
double EvaluatePrompt(string prompt)
{
    // Test prompt on validation set
    int correct = 0;
    foreach (var testCase in validationSet)
    {
        var result = model.Generate(prompt + testCase.Input);
        if (result.Trim() == testCase.Expected)
            correct++;
    }
    return correct / (double)validationSet.Count;
}

// Optimize
var optimizedTemplate = optimizer.Optimize(
    initialPrompt: "Classify the sentiment:",
    evaluationFunction: EvaluatePrompt,
    maxIterations: 50
);

// View optimization history
var history = optimizer.GetOptimizationHistory();
foreach (var entry in history)
{
    Console.WriteLine($"Iteration {entry.Iteration}: Score = {entry.Score}");
}
```

### Async Optimization

```csharp
async Task<double> EvaluatePromptAsync(string prompt)
{
    // Async evaluation (e.g., API calls)
    var results = await TestWithAPIAsync(prompt);
    return CalculateAccuracy(results);
}

var optimized = await optimizer.OptimizeAsync(
    initialPrompt: "Classify sentiment:",
    evaluationFunction: EvaluatePromptAsync,
    maxIterations: 50
);
```

---

## Chain Construction

Compose multiple LLM operations into workflows.

### Sequential Chain

```csharp
using AiDotNet.PromptEngineering.Chains;

var chain = new SequentialChain<string, string>(
    "DocumentProcessing",
    "Processes documents through multiple steps"
);

// Add steps
chain.AddStep("ExtractKeywords", text =>
{
    // Extract keywords from text
    return ExtractKeywords(text);
});

chain.AddStep("Summarize", keywords =>
{
    // Generate summary from keywords
    return GenerateSummary(keywords);
});

chain.AddStep("Translate", summary =>
{
    // Translate to target language
    return Translate(summary, "es");
});

// Run chain
var result = chain.Run("Long article text...");
```

### Async Chain

```csharp
chain.AddStepAsync("FetchData", async (input, ct) =>
{
    var data = await FetchFromAPIAsync(input.ToString(), ct);
    return data;
});

var result = await chain.RunAsync("input", CancellationToken.None);
```

---

## Quick Start Examples

### Example 1: Sentiment Classification with Few-Shot Learning

```csharp
using AiDotNet.PromptEngineering.Templates;
using AiDotNet.PromptEngineering.FewShot;

// Create example selector
var selector = new RandomExampleSelector<double>();
selector.AddExample(new FewShotExample
{
    Input = "I absolutely loved this product! Best purchase ever!",
    Output = "Positive"
});
selector.AddExample(new FewShotExample
{
    Input = "Terrible quality, complete waste of money.",
    Output = "Negative"
});
selector.AddExample(new FewShotExample
{
    Input = "It's okay, nothing special but it works.",
    Output = "Neutral"
});

// Create few-shot template
var template = new FewShotPromptTemplate<double>(
    template: "Classify the sentiment of customer reviews.\n\n{examples}\n\nReview: {review}\nSentiment:",
    exampleSelector: selector,
    exampleCount: 2
);

// Use template
var prompt = template.Format(new Dictionary<string, string>
{
    ["review"] = "This exceeded my expectations!"
});

// Send prompt to your LLM...
```

### Example 2: Tool-Augmented Research Assistant

```csharp
using AiDotNet.PromptEngineering.Tools;
using AiDotNet.PromptEngineering.Chains;

// Create tools
var registry = new ToolRegistry();
registry.RegisterTool(new SearchTool());
registry.RegisterTool(new CalculatorTool());
registry.RegisterTool(new WikipediaTool());

// Create chain
var researchChain = new SequentialChain<string, string>(
    "ResearchChain",
    "Researches topics using multiple tools"
);

researchChain.AddStep("Search", query =>
{
    var searchArgs = CreateSearchArgs(query);
    return registry.ExecuteTool("search", searchArgs);
});

researchChain.AddStep("Analyze", searchResults =>
{
    // Analyze search results and extract key info
    return AnalyzeResults(searchResults);
});

researchChain.AddStep("Synthesize", analysis =>
{
    // Create final answer
    return SynthesizeAnswer(analysis);
});

// Use chain
var answer = researchChain.Run("What is the population of Tokyo?");
```

### Example 3: Prompt Optimization for Question Answering

```csharp
using AiDotNet.PromptEngineering.Optimization;

var optimizer = new DiscreteSearchOptimizer<double>();

// Add custom variations
optimizer.AddInstructionVariations(
    "Answer the question concisely: ",
    "Provide a detailed answer to: ",
    "Based on the context, answer: "
);

// Evaluation function
double Evaluate(string promptTemplate)
{
    double totalScore = 0;
    foreach (var qa in testSet)
    {
        var prompt = promptTemplate + $"\n\nContext: {qa.Context}\nQuestion: {qa.Question}\nAnswer:";
        var answer = llm.Generate(prompt);
        totalScore += CalculateSimilarity(answer, qa.ExpectedAnswer);
    }
    return totalScore / testSet.Count;
}

// Optimize
var best = optimizer.Optimize(
    initialPrompt: "Answer the question:",
    evaluationFunction: Evaluate,
    maxIterations: 100
);

Console.WriteLine($"Best prompt: {best.Template}");
```

---

## Architecture

### Component Organization

```
src/
├── Enums/
│   ├── PromptTemplateType.cs
│   ├── ChainType.cs
│   ├── FewShotSelectionStrategy.cs
│   └── PromptOptimizationStrategy.cs
├── Interfaces/
│   ├── IPromptTemplate.cs
│   ├── IFunctionTool.cs
│   ├── IChain.cs
│   ├── IFewShotExampleSelector.cs
│   └── IPromptOptimizer.cs
├── PromptEngineering/
│   ├── Templates/
│   │   ├── PromptTemplateBase.cs
│   │   ├── SimplePromptTemplate.cs
│   │   ├── FewShotPromptTemplate.cs
│   │   └── ChatPromptTemplate.cs
│   ├── Tools/
│   │   ├── FunctionToolBase.cs
│   │   └── ToolRegistry.cs
│   ├── Chains/
│   │   ├── ChainBase.cs
│   │   └── SequentialChain.cs
│   ├── FewShot/
│   │   ├── FewShotExampleSelectorBase.cs
│   │   ├── RandomExampleSelector.cs
│   │   └── FixedExampleSelector.cs
│   └── Optimization/
│       ├── PromptOptimizerBase.cs
│       └── DiscreteSearchOptimizer.cs
└── Factories/
    └── PromptTemplateFactory.cs
```

---

## Best Practices

### 1. Prompt Template Design
- Keep templates focused on a single task
- Use descriptive variable names
- Validate inputs before formatting
- Include clear instructions in the prompt

### 2. Tool Implementation
- Provide comprehensive parameter schemas
- Handle errors gracefully
- Return informative error messages
- Validate arguments before execution

### 3. Few-Shot Examples
- Use diverse, high-quality examples
- Ensure examples match your use case
- Keep examples concise
- Update examples based on performance

### 4. Prompt Optimization
- Use representative validation sets
- Define clear evaluation metrics
- Start with reasonable initial prompts
- Monitor optimization progress

### 5. Chain Construction
- Keep individual steps focused
- Handle errors at each step
- Use async operations for I/O
- Validate inputs and outputs

---

## Performance Considerations

- **Template Caching**: Reuse template instances when possible
- **Example Selection**: Balance accuracy vs. speed (semantic similarity is slower than random)
- **Optimization**: More iterations = better results but higher cost
- **Chain Execution**: Use async operations for network I/O
- **Tool Calls**: Cache tool results when appropriate

---

## Troubleshooting

### Template Variables Not Replaced
- Ensure variable names match exactly (case-sensitive)
- Check that all required variables are provided
- Use `Validate()` before `Format()`

### Tool Execution Failures
- Verify parameter schema matches arguments
- Check that required fields are provided
- Review tool implementation for errors

### Poor Optimization Results
- Increase `maxIterations`
- Improve evaluation function
- Add more variation types
- Check validation set quality

---

## Future Enhancements

Planned additions to the framework:
- **Semantic Similarity Selector**: Using embeddings for example selection
- **MMR Selector**: Maximum marginal relevance for balanced selection
- **Conditional Chains**: Branching logic based on results
- **Parallel Chains**: Execute multiple paths simultaneously
- **Map-Reduce Chains**: Process collections efficiently
- **Gradient-Based Optimization**: Advanced prompt optimization
- **Ensemble Optimization**: Combine multiple prompts

---

## Contributing

When extending the framework:
1. Follow existing patterns (Base class + Interface)
2. Include comprehensive XML documentation
3. Add "For Beginners" sections to remarks
4. Write unit tests for new components
5. Update this documentation

---

## License

This framework is part of AiDotNet and follows the same license.
