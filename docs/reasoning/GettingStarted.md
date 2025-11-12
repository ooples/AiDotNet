# Getting Started with AiDotNet Reasoning Framework

Welcome to the AiDotNet Reasoning Framework - a cutting-edge system for advanced AI reasoning that rivals DeepSeek-R1 and ChatGPT o1/o3!

## Table of Contents
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Basic Concepts](#basic-concepts)
- [First Example](#first-example)
- [Next Steps](#next-steps)

## Quick Start

```csharp
using AiDotNet.Reasoning.Strategies;
using AiDotNet.Reasoning.Models;

// Initialize with your chat model
var chatModel = /* your IChatModel implementation */;
var strategy = new ChainOfThoughtStrategy<double>(chatModel);

// Solve a problem
var result = await strategy.ReasonAsync("What is 15 × 12?");

Console.WriteLine($"Answer: {result.FinalAnswer}");
Console.WriteLine($"Steps: {result.Chain.Steps.Count}");
```

## Installation

### Prerequisites
- .NET 6.0 or higher
- A chat model implementation (OpenAI, Anthropic, etc.)

### NuGet Package
```bash
dotnet add package AiDotNet
```

### From Source
```bash
git clone https://github.com/ooples/AiDotNet.git
cd AiDotNet
dotnet build
```

## Basic Concepts

### 1. Reasoning Strategies

The framework provides three main reasoning strategies:

#### **Chain-of-Thought (CoT)**
Linear step-by-step reasoning - best for straightforward problems.

```csharp
var cotStrategy = new ChainOfThoughtStrategy<double>(chatModel);
var result = await cotStrategy.ReasonAsync("Calculate the area of a circle with radius 5");
```

#### **Self-Consistency**
Generates multiple reasoning paths and aggregates results - best for problems with multiple valid approaches.

```csharp
var scStrategy = new SelfConsistencyStrategy<double>(chatModel);
var config = new ReasoningConfig { NumSamples = 5 };
var result = await scStrategy.ReasonAsync("What is the capital of France?", config);
```

#### **Tree-of-Thoughts (ToT)**
Explores multiple paths with backtracking - best for complex problems requiring exploration.

```csharp
var totStrategy = new TreeOfThoughtsStrategy<double>(chatModel);
var config = new ReasoningConfig { ExplorationDepth = 4, BranchingFactor = 3 };
var result = await totStrategy.ReasonAsync("Solve this logic puzzle: ...", config);
```

### 2. Configuration Presets

Choose the right preset for your use case:

```csharp
// Fast: Quick answers for simple problems
var fastConfig = ReasoningConfig.Fast;  // 3 steps, depth 2

// Default: Balanced for most problems
var defaultConfig = ReasoningConfig.Default;  // 10 steps, depth 3

// Thorough: Deep exploration for hard problems
var thoroughConfig = ReasoningConfig.Thorough;  // 20 steps, depth 5
```

### 3. Domain-Specific Reasoners

Use specialized reasoners for specific domains:

```csharp
// Mathematics
var mathReasoner = new MathematicalReasoner<double>(chatModel);
var result = await mathReasoner.SolveAsync("What is 347 + 892?");

// Code Generation
var codeReasoner = new CodeReasoner<double>(chatModel);
var result = await codeReasoner.GenerateCodeAsync(
    "Write a function to find the factorial of n",
    language: "python"
);

// Science
var scienceReasoner = new ScientificReasoner<double>(chatModel);
var result = await scienceReasoner.SolveAsync(
    "Calculate kinetic energy of 5kg object at 10m/s",
    domain: "physics"
);

// Logic
var logicReasoner = new LogicalReasoner<double>(chatModel);
var result = await logicReasoner.SolveAsync(
    "All A are B. All B are C. Therefore?",
    logicType: "deductive"
);
```

## First Example

Let's build a complete example that solves a math problem with verification:

```csharp
using AiDotNet.Reasoning.DomainSpecific;
using AiDotNet.Reasoning.Models;
using AiDotNet.Reasoning.Verification;

public class MathProblemSolver
{
    private readonly IChatModel _chatModel;
    private readonly MathematicalReasoner<double> _reasoner;
    private readonly CalculatorVerifier<double> _verifier;

    public MathProblemSolver(IChatModel chatModel)
    {
        _chatModel = chatModel;
        _reasoner = new MathematicalReasoner<double>(chatModel);
        _verifier = new CalculatorVerifier<double>();
    }

    public async Task<string> SolveWithVerificationAsync(string problem)
    {
        // Step 1: Solve the problem
        var result = await _reasoner.SolveAsync(
            problem,
            useVerification: true,
            useSelfConsistency: false  // Try setting to true for harder problems!
        );

        if (!result.Success)
        {
            return $"Failed to solve: {result.ErrorMessage}";
        }

        // Step 2: Verify the calculation
        var verification = await _verifier.VerifyAsync(result.Chain);

        // Step 3: Return results
        var output = new StringBuilder();
        output.AppendLine($"Problem: {problem}");
        output.AppendLine($"\nReasoning Steps:");

        foreach (var step in result.Chain.Steps)
        {
            output.AppendLine($"  {step.StepNumber}. {step.Content}");
        }

        output.AppendLine($"\nFinal Answer: {result.FinalAnswer}");
        output.AppendLine($"Verification: {(verification.IsValid ? "✓ Correct" : "✗ Incorrect")}");
        output.AppendLine($"Confidence: {result.ConfidenceScore:P0}");

        return output.ToString();
    }
}

// Usage
var chatModel = /* your chat model */;
var solver = new MathProblemSolver(chatModel);

var result = await solver.SolveWithVerificationAsync(
    "A store has 347 apples. They sell 129 in the morning and 85 in the afternoon. How many apples are left?"
);

Console.WriteLine(result);
```

**Output:**
```
Problem: A store has 347 apples...

Reasoning Steps:
  1. Start with initial amount: 347 apples
  2. Calculate morning sales: 347 - 129 = 218
  3. Calculate afternoon sales: 218 - 85 = 133

Final Answer: 133 apples
Verification: ✓ Correct
Confidence: 95%
```

## Next Steps

### Learn More
- [API Documentation](./ApiReference.md) - Complete API reference
- [Tutorials](./Tutorials.md) - Step-by-step guides
- [Best Practices](./BestPractices.md) - Tips and patterns
- [Benchmarks](./Benchmarks.md) - Evaluation guide

### Try These Examples
1. **Solve GSM8K Math Problems**: See `examples/GSM8KExample.cs`
2. **Generate Code with HumanEval**: See `examples/CodeGenerationExample.cs`
3. **Train with Reinforcement Learning**: See `examples/RLTrainingExample.cs`
4. **Build a Custom Reasoner**: See `examples/CustomReasonerExample.cs`

### Key Features to Explore

#### 1. Verification System
```csharp
// Critic-based verification
var criticModel = new CriticModel<double>(chatModel);
var critique = await criticModel.CritiqueStepAsync(step, context);

// Self-refinement
var refinementEngine = new SelfRefinementEngine<double>(chatModel);
var refined = await refinementEngine.RefineStepAsync(step, critique, context);
```

#### 2. Reward Models for RL
```csharp
// Process Reward Model (step-by-step scoring)
var prm = new ProcessRewardModel<double>(chatModel);

// Outcome Reward Model (final answer scoring)
var orm = new OutcomeRewardModel<double>(chatModel);

// Hybrid (best of both)
var hybrid = new HybridRewardModel<double>(prm, orm, 0.5, 0.5);
```

#### 3. Search Algorithms
```csharp
// Monte Carlo Tree Search
var mcts = new MonteCarloTreeSearch<double>(
    explorationConstant: 1.414,
    simulationCount: 100
);

// Best-First Search
var bestFirst = new BestFirstSearch<double>();

// Depth-First Search
var dfs = new DepthFirstSearch<double>();
```

#### 4. Benchmarking
```csharp
// Evaluate on GSM8K
var benchmark = new GSM8KBenchmark<double>();
var results = await benchmark.EvaluateAsync(
    async (problem) => {
        var result = await reasoner.SolveAsync(problem);
        return result.FinalAnswer;
    },
    sampleSize: 100
);

Console.WriteLine($"Accuracy: {results.Accuracy:P2}");
```

#### 5. Training with RL
```csharp
var rewardModel = new HybridRewardModel<double>(prm, orm);
var learner = new ReinforcementLearner<double>(chatModel, rewardModel);

var trainingData = await LoadTrainingDataAsync();
var validationData = await LoadValidationDataAsync();

var results = await learner.TrainAsync(trainingData, validationData);
Console.WriteLine($"Best Accuracy: {results.BestAccuracy:P2}");
```

## Common Patterns

### Pattern 1: Progressive Refinement
```csharp
var result = await strategy.ReasonAsync(problem);

while (result.ConfidenceScore < 0.9 && iterations < maxIterations)
{
    var critique = await critic.CritiqueChainAsync(result.Chain);
    result = await refinement.RefineAsync(result, critique);
    iterations++;
}
```

### Pattern 2: Ensemble Reasoning
```csharp
var strategies = new IReasoningStrategy<double>[]
{
    new ChainOfThoughtStrategy<double>(chatModel),
    new SelfConsistencyStrategy<double>(chatModel),
    new TreeOfThoughtsStrategy<double>(chatModel)
};

var results = await Task.WhenAll(
    strategies.Select(s => s.ReasonAsync(problem))
);

var bestResult = results.OrderByDescending(r => r.ConfidenceScore).First();
```

### Pattern 3: Adaptive Compute
```csharp
var scaler = new AdaptiveComputeScaler<double>();
var difficulty = scaler.EstimateDifficulty(problem);
var config = scaler.ScaleConfig(problem, difficulty);

var result = await strategy.ReasonAsync(problem, config);
```

## Troubleshooting

### Issue: Low Confidence Scores
**Solution**: Use Self-Consistency or enable verification:
```csharp
var config = new ReasoningConfig { NumSamples = 5 };
var result = await scStrategy.ReasonAsync(problem, config);
```

### Issue: Incomplete Reasoning
**Solution**: Increase max steps or use thorough config:
```csharp
var config = ReasoningConfig.Thorough;  // 20 steps instead of 10
var result = await strategy.ReasonAsync(problem, config);
```

### Issue: Wrong Answers
**Solution**: Add verification and refinement:
```csharp
var result = await mathReasoner.SolveAsync(problem, useVerification: true);
```

## Community & Support

- **Documentation**: https://docs.aidotnet.com
- **GitHub**: https://github.com/ooples/AiDotNet
- **Issues**: https://github.com/ooples/AiDotNet/issues
- **Discussions**: https://github.com/ooples/AiDotNet/discussions

## What's Next?

You're now ready to build advanced reasoning systems! Here are some ideas:

1. **Build a Math Tutor**: Use MathematicalReasoner with step-by-step explanations
2. **Create a Code Assistant**: Use CodeReasoner for code generation and debugging
3. **Build a Logic Puzzle Solver**: Use LogicalReasoner with ToT strategy
4. **Train Your Own Model**: Use the RL infrastructure to improve reasoning

Happy reasoning! 🚀
