# Issue #417: Junior Developer Implementation Guide
## Chain-of-Thought and Advanced Reasoning

---

## Table of Contents
1. [Understanding Chain-of-Thought](#understanding-chain-of-thought)
2. [Understanding Tree-of-Thoughts](#understanding-tree-of-thoughts)
3. [Understanding Self-Consistency](#understanding-self-consistency)
4. [Understanding Reasoning Verification](#understanding-reasoning-verification)
5. [Architecture Overview](#architecture-overview)
6. [Phase 1: Core Reasoning Framework](#phase-1-core-reasoning-framework)
7. [Phase 2: Chain-of-Thought](#phase-2-chain-of-thought)
8. [Phase 3: Tree-of-Thoughts](#phase-3-tree-of-thoughts)
9. [Phase 4: Self-Consistency](#phase-4-self-consistency)
10. [Phase 5: Reasoning Verification](#phase-5-reasoning-verification)
11. [Testing Strategy](#testing-strategy)
12. [Common Pitfalls](#common-pitfalls)

---

## Understanding Chain-of-Thought

### What Is Chain-of-Thought (CoT)?

Chain-of-Thought is a prompting technique that **instructs language models to explain their reasoning step-by-step** before providing a final answer.

### Why Chain-of-Thought Works

**Problem**: Models sometimes make reasoning errors when answering complex questions directly.

**Example (Without CoT)**:
```
Question: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?

Model: 10 tennis balls (WRONG!)
```

**Solution (With CoT)**:
```
Question: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now? Let's think step by step:

Model:
Step 1: Roger starts with 5 tennis balls.
Step 2: He buys 2 cans, and each can has 3 balls.
Step 3: 2 cans × 3 balls/can = 6 balls.
Step 4: Total = 5 + 6 = 11 balls.
Answer: 11 tennis balls (CORRECT!)
```

### Key Insight

By forcing the model to **"show its work"**, we:
1. **Reduce errors** in multi-step reasoning
2. **Make reasoning interpretable** (we can see where mistakes occur)
3. **Enable debugging** of model reasoning

### CoT Prompting Strategies

#### 1. Zero-Shot CoT
Add "Let's think step by step" to the prompt:
```csharp
string prompt = question + " Let's think step by step:";
string response = model.Generate(prompt);
```

#### 2. Few-Shot CoT
Provide examples with reasoning chains:
```csharp
string fewShotPrompt = @"
Q: Roger has 5 balls. He buys 2 cans of 3 balls each. How many balls now?
A: Let's think step by step:
1. Roger starts with 5 balls.
2. He buys 2 cans × 3 balls = 6 balls.
3. Total: 5 + 6 = 11 balls.
Answer: 11 balls.

Q: " + question;
```

#### 3. Manual CoT
Explicitly guide the model through each step:
```csharp
var steps = new List<string>
{
    "First, identify the initial quantity.",
    "Second, calculate the additional quantity.",
    "Third, sum the quantities.",
    "Finally, state the answer."
};

foreach (var step in steps)
{
    string stepPrompt = question + " " + step;
    string stepResponse = model.Generate(stepPrompt);
    reasoning.Add(stepResponse);
}
```

---

## Understanding Tree-of-Thoughts

### What Is Tree-of-Thoughts (ToT)?

Tree-of-Thoughts is an extension of CoT that **explores multiple reasoning paths** in a tree structure, evaluating and pruning paths to find the best solution.

### Why ToT Beats CoT

**Problem**: CoT follows a single reasoning chain, which may lead to dead ends.

**Example (Game of 24)**:
```
Input: 4 numbers {4, 9, 10, 13}
Goal: Use +, -, ×, ÷ to make 24

CoT (Single path):
Step 1: 4 + 9 = 13
Step 2: 13 + 10 = 23
Step 3: 23 + 13 = 36 (WRONG! Can't reach 24)

ToT (Multiple paths):
Path 1: (4 + 9) = 13, (13 + 10) = 23, ... FAIL
Path 2: (9 - 4) = 5, (5 × 10) = 50, ... FAIL
Path 3: (10 - 4) = 6, (6 × 9) = 54, (54 - 13) ... FAIL
Path 4: (13 - 9) = 4, (4 × 10) = 40, (40 - 4) ... FAIL
Path 5: (13 - 4) = 9, (9 + 10) = 19, (19 + 9) ... FAIL
Path 6: (10 - 9) = 1, (13 - 4) = 9, (9 × 1) ... FAIL
Path 7: (13 - 9) = 4, (10 - 4) = 6, (6 × 4) = 24 SUCCESS!
```

### ToT Algorithm

```
1. Generate multiple reasoning paths (breadth-first or best-first)
2. Evaluate each path's promise (using heuristic or model evaluation)
3. Prune unpromising paths
4. Expand promising paths
5. Repeat until solution found or budget exhausted
```

### ToT Components

#### 1. Thought Decomposition
Break problem into intermediate steps:
```csharp
public class Thought
{
    public string State { get; set; }        // Current state (e.g., "4 + 9 = 13")
    public string Action { get; set; }       // Action taken (e.g., "Add 4 and 9")
    public double Score { get; set; }        // Promise score
    public Thought Parent { get; set; }      // Parent thought
    public List<Thought> Children { get; set; }  // Child thoughts
}
```

#### 2. Thought Generator
Generate k candidate next thoughts:
```csharp
public List<Thought> GenerateThoughts(Thought current, int k)
{
    var prompt = $"Given state: {current.State}, generate {k} possible next steps.";
    var response = model.Generate(prompt);
    return ParseThoughts(response);
}
```

#### 3. State Evaluator
Score how promising a thought is:
```csharp
public double EvaluateThought(Thought thought)
{
    var prompt = $"Rate the promise of this reasoning step (0-10): {thought.State}";
    var response = model.Generate(prompt);
    return ParseScore(response);
}
```

#### 4. Search Algorithm
Explore the thought tree:
```csharp
public Thought BreadthFirstSearch(string problem, int maxDepth, int branchingFactor)
{
    var root = new Thought { State = problem };
    var queue = new Queue<Thought>();
    queue.Enqueue(root);

    while (queue.Count > 0)
    {
        var current = queue.Dequeue();

        if (IsSolution(current))
            return current;

        if (current.Depth < maxDepth)
        {
            var children = GenerateThoughts(current, branchingFactor);
            foreach (var child in children)
            {
                child.Score = EvaluateThought(child);
                if (child.Score > threshold)  // Prune low-scoring thoughts
                    queue.Enqueue(child);
            }
        }
    }

    return null;
}
```

### ToT Search Strategies

#### 1. Breadth-First Search (BFS)
Explore all thoughts at depth d before depth d+1:
```
Depth 0:        [Problem]
                /    |    \
Depth 1:      [T1] [T2] [T3]
              / |    | \
Depth 2:   [T4][T5][T6][T7]
```

#### 2. Depth-First Search (DFS)
Explore one path fully before backtracking:
```
[Problem] → [T1] → [T4] → ... (dead end)
                → [T5] → ... (solution!)
```

#### 3. Best-First Search
Always expand most promising thought:
```python
priority_queue = [(score, thought)]
while priority_queue:
    score, current = priority_queue.pop()
    if is_solution(current):
        return current
    for child in generate_children(current):
        priority_queue.push((evaluate(child), child))
```

---

## Understanding Self-Consistency

### What Is Self-Consistency?

Self-consistency is a technique that **samples multiple reasoning paths and takes the majority vote** to improve answer reliability.

### Why Self-Consistency Works

**Problem**: A single CoT reasoning path may contain errors.

**Solution**: Generate multiple paths and aggregate:

```
Question: If John has 3 apples and buys 2 more, how many does he have?

Sample 1 (CoT):
  "John starts with 3. He buys 2. 3 + 2 = 5. Answer: 5"

Sample 2 (CoT):
  "John has 3 apples. Buying 2 more gives 3 + 2 = 5. Answer: 5"

Sample 3 (CoT):
  "Start: 3 apples. Buys: 2. Total: 3 + 2 = 5. Answer: 5"

Majority Vote: 5 (3/3 agree) → High confidence!
```

### Self-Consistency Algorithm

```csharp
public string SelfConsistency(string question, int numSamples, double temperature)
{
    var answers = new Dictionary<string, int>();

    for (int i = 0; i < numSamples; i++)
    {
        // Generate reasoning path with randomness (temperature > 0)
        var reasoning = GenerateCoT(question, temperature);
        var answer = ExtractAnswer(reasoning);

        if (!answers.ContainsKey(answer))
            answers[answer] = 0;
        answers[answer]++;
    }

    // Return most frequent answer
    return answers.OrderByDescending(kv => kv.Value).First().Key;
}
```

### Key Parameters

1. **Temperature**: Controls randomness of sampling
   - Low (0.1): Similar paths, less diversity
   - High (0.8): Diverse paths, more exploration

2. **Number of Samples**: More samples = higher confidence, but more compute
   - Typical: 5-20 samples

3. **Voting Strategy**:
   - **Majority Vote**: Most common answer
   - **Weighted Vote**: Weight by reasoning quality score
   - **Confidence Threshold**: Only return answer if majority exceeds threshold

---

## Understanding Reasoning Verification

### What Is Reasoning Verification?

Reasoning verification is the process of **checking whether a reasoning chain is logically valid and factually correct**.

### Why Verification Matters

**Problem**: Models can generate fluent but incorrect reasoning.

**Example**:
```
Question: What is the capital of France?

Bad Reasoning:
"France is in Europe. Europe has many capitals. The largest city in France is Paris. Therefore, the capital is London."
(Fluent but factually wrong!)

Verified Reasoning:
"France is a country in Europe. The capital of France is defined as the seat of government. Paris is the seat of the French government. Therefore, the capital is Paris."
(Correct!)
```

### Verification Techniques

#### 1. Logical Consistency Checking
Verify each step follows logically from the previous:
```csharp
public bool IsLogicallyConsistent(List<string> reasoningSteps)
{
    for (int i = 1; i < reasoningSteps.Count; i++)
    {
        if (!StepFollowsFrom(reasoningSteps[i], reasoningSteps[i-1]))
            return false;
    }
    return true;
}
```

#### 2. Factual Verification
Check claims against knowledge base:
```csharp
public bool VerifyFact(string claim, IKnowledgeBase kb)
{
    var evidence = kb.Search(claim);
    return evidence.Confidence > 0.8;
}
```

#### 3. Mathematical Verification
Verify arithmetic steps:
```csharp
public bool VerifyCalculation(string step)
{
    // Parse "3 + 2 = 5"
    var match = Regex.Match(step, @"(\d+)\s*([+\-*/])\s*(\d+)\s*=\s*(\d+)");
    if (!match.Success) return true;  // Not a calculation

    int a = int.Parse(match.Groups[1].Value);
    int b = int.Parse(match.Groups[3].Value);
    int result = int.Parse(match.Groups[4].Value);
    string op = match.Groups[2].Value;

    return op switch
    {
        "+" => (a + b) == result,
        "-" => (a - b) == result,
        "*" => (a * b) == result,
        "/" => (a / b) == result,
        _ => true
    };
}
```

#### 4. Contradiction Detection
Check for internal contradictions:
```csharp
public bool HasContradiction(List<string> reasoningSteps)
{
    var claims = ExtractClaims(reasoningSteps);

    for (int i = 0; i < claims.Count; i++)
    {
        for (int j = i + 1; j < claims.Count; j++)
        {
            if (AreContradictory(claims[i], claims[j]))
                return true;
        }
    }

    return false;
}
```

#### 5. External Tool Verification
Use external tools (calculator, search engine):
```csharp
public bool VerifyWithCalculator(string expression)
{
    var calculator = new Calculator();
    var computed = calculator.Evaluate(expression);
    var claimed = ExtractResult(expression);
    return Math.Abs(computed - claimed) < 1e-6;
}
```

---

## Architecture Overview

### Component Relationships

```
┌─────────────────────────────────────────────────────────┐
│                   User Application                      │
└─────────────────────────────────────────────────────────┘
                          │
                          ↓
┌─────────────────────────────────────────────────────────┐
│              Reasoning Engine                           │
│  - Execute reasoning strategies                         │
│  - Coordinate components                                │
└─────────────────────────────────────────────────────────┘
         │              │              │              │
         ↓              ↓              ↓              ↓
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Chain-of-    │ │ Tree-of-     │ │ Self-        │ │ Reasoning    │
│ Thought      │ │ Thoughts     │ │ Consistency  │ │ Verification │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
         │              │              │              │
         ↓              ↓              ↓              ↓
┌─────────────────────────────────────────────────────────┐
│                Language Model Interface                 │
│  - Generate text                                        │
│  - Parse responses                                      │
└─────────────────────────────────────────────────────────┘
```

### Directory Structure

```
src/Reasoning/
├── Core/
│   ├── IReasoningStrategy.cs      # Strategy interface
│   ├── ReasoningEngine.cs         # Main orchestrator
│   ├── ReasoningConfig.cs         # Configuration
│   └── Models/
│       ├── ReasoningStep.cs       # Single reasoning step
│       ├── ReasoningChain.cs      # Sequence of steps
│       └── ReasoningResult.cs     # Final result
│
├── ChainOfThought/
│   ├── CoTStrategy.cs             # CoT implementation
│   ├── ZeroShotCoT.cs             # Zero-shot prompting
│   ├── FewShotCoT.cs              # Few-shot prompting
│   └── ManualCoT.cs               # Guided step-by-step
│
├── TreeOfThoughts/
│   ├── ToTStrategy.cs             # ToT implementation
│   ├── ThoughtNode.cs             # Tree node
│   ├── ThoughtGenerator.cs       # Generate child thoughts
│   ├── ThoughtEvaluator.cs       # Score thoughts
│   └── Search/
│       ├── BreadthFirstSearch.cs
│       ├── DepthFirstSearch.cs
│       └── BestFirstSearch.cs
│
├── SelfConsistency/
│   ├── SelfConsistencyStrategy.cs # Self-consistency impl
│   ├── AnswerAggregator.cs        # Vote aggregation
│   └── DiversitySampler.cs        # Generate diverse paths
│
├── Verification/
│   ├── IVerifier.cs               # Verifier interface
│   ├── LogicalVerifier.cs         # Logical consistency
│   ├── FactualVerifier.cs         # Factual checking
│   ├── MathematicalVerifier.cs    # Arithmetic verification
│   ├── ContradictionDetector.cs   # Find contradictions
│   └── ExternalToolVerifier.cs    # Use external tools
│
└── Utils/
    ├── PromptBuilder.cs           # Build prompts
    ├── ResponseParser.cs          # Parse model outputs
    └── Calculator.cs              # Simple calculator
```

---

## Phase 1: Core Reasoning Framework

### Step 1: Define Reasoning Models

**File**: `src/Reasoning/Core/Models/ReasoningStep.cs`

```csharp
namespace AiDotNet.Reasoning.Core.Models;

/// <summary>
/// Represents a single step in a reasoning chain.
/// </summary>
public class ReasoningStep
{
    public int StepNumber { get; set; }
    public string Description { get; set; } = string.Empty;
    public string Content { get; set; } = string.Empty;
    public double Confidence { get; set; } = 1.0;
    public Dictionary<string, object> Metadata { get; set; } = new();
    public bool IsVerified { get; set; } = false;
    public List<string> VerificationErrors { get; set; } = new();
}
```

**File**: `src/Reasoning/Core/Models/ReasoningChain.cs`

```csharp
namespace AiDotNet.Reasoning.Core.Models;

/// <summary>
/// Represents a sequence of reasoning steps.
/// </summary>
public class ReasoningChain
{
    public string ChainId { get; set; } = Guid.NewGuid().ToString();
    public string Question { get; set; } = string.Empty;
    public List<ReasoningStep> Steps { get; set; } = new();
    public string Answer { get; set; } = string.Empty;
    public double OverallConfidence { get; set; }
    public bool IsComplete { get; set; } = false;
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Get the full reasoning as a single string.
    /// </summary>
    public string GetFullReasoning()
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine($"Question: {Question}");
        sb.AppendLine("\nReasoning:");

        foreach (var step in Steps)
        {
            sb.AppendLine($"Step {step.StepNumber}: {step.Content}");
        }

        sb.AppendLine($"\nAnswer: {Answer}");
        return sb.ToString();
    }
}
```

**File**: `src/Reasoning/Core/Models/ReasoningResult.cs`

```csharp
namespace AiDotNet.Reasoning.Core.Models;

/// <summary>
/// Result of a reasoning process.
/// </summary>
public class ReasoningResult
{
    public string Question { get; set; } = string.Empty;
    public string Answer { get; set; } = string.Empty;
    public List<ReasoningChain> Chains { get; set; } = new();
    public double Confidence { get; set; }
    public ReasoningStrategy Strategy { get; set; }
    public TimeSpan ExecutionTime { get; set; }
    public Dictionary<string, object> Metadata { get; set; } = new();

    /// <summary>
    /// Get the best reasoning chain (highest confidence).
    /// </summary>
    public ReasoningChain GetBestChain()
    {
        return Chains.OrderByDescending(c => c.OverallConfidence).FirstOrDefault();
    }
}

public enum ReasoningStrategy
{
    ChainOfThought,
    TreeOfThoughts,
    SelfConsistency
}
```

### Step 2: Define Strategy Interface

**File**: `src/Reasoning/Core/IReasoningStrategy.cs`

```csharp
namespace AiDotNet.Reasoning.Core;

using AiDotNet.Reasoning.Core.Models;

/// <summary>
/// Interface for reasoning strategies.
/// </summary>
public interface IReasoningStrategy
{
    /// <summary>
    /// Execute the reasoning strategy on a question.
    /// </summary>
    ReasoningResult Reason(string question, ReasoningConfig config);

    /// <summary>
    /// Name of the strategy.
    /// </summary>
    string StrategyName { get; }
}
```

### Step 3: Define Configuration

**File**: `src/Reasoning/Core/ReasoningConfig.cs`

```csharp
namespace AiDotNet.Reasoning.Core;

/// <summary>
/// Configuration for reasoning strategies.
/// </summary>
public class ReasoningConfig
{
    // General settings
    public int MaxSteps { get; set; } = 10;
    public double Temperature { get; set; } = 0.7;
    public int MaxTokens { get; set; } = 500;

    // Chain-of-Thought settings
    public bool UseZeroShot { get; set; } = true;
    public List<string> FewShotExamples { get; set; } = new();

    // Tree-of-Thoughts settings
    public int BranchingFactor { get; set; } = 3;
    public int MaxDepth { get; set; } = 5;
    public double PruningThreshold { get; set; } = 0.3;
    public ToTSearchType SearchType { get; set; } = ToTSearchType.BreadthFirst;

    // Self-Consistency settings
    public int NumSamples { get; set; } = 5;
    public VotingStrategy VotingStrategy { get; set; } = VotingStrategy.Majority;

    // Verification settings
    public bool EnableVerification { get; set; } = true;
    public List<VerificationType> VerificationTypes { get; set; } = new()
    {
        VerificationType.Logical,
        VerificationType.Mathematical
    };
}

public enum ToTSearchType
{
    BreadthFirst,
    DepthFirst,
    BestFirst
}

public enum VotingStrategy
{
    Majority,
    Weighted,
    ConfidenceThreshold
}

public enum VerificationType
{
    Logical,
    Factual,
    Mathematical,
    Contradiction
}
```

### Step 4: Implement Reasoning Engine

**File**: `src/Reasoning/Core/ReasoningEngine.cs`

```csharp
namespace AiDotNet.Reasoning.Core;

using AiDotNet.Reasoning.Core.Models;

/// <summary>
/// Main orchestrator for reasoning tasks.
/// </summary>
public class ReasoningEngine
{
    private readonly ILanguageModel _model;
    private readonly Dictionary<ReasoningStrategy, IReasoningStrategy> _strategies;

    public ReasoningEngine(ILanguageModel model)
    {
        _model = model;
        _strategies = new Dictionary<ReasoningStrategy, IReasoningStrategy>();
    }

    /// <summary>
    /// Register a reasoning strategy.
    /// </summary>
    public void RegisterStrategy(ReasoningStrategy type, IReasoningStrategy strategy)
    {
        _strategies[type] = strategy;
    }

    /// <summary>
    /// Reason about a question using specified strategy.
    /// </summary>
    public ReasoningResult Reason(string question, ReasoningStrategy strategy, ReasoningConfig config = null)
    {
        config = config ?? new ReasoningConfig();

        if (!_strategies.ContainsKey(strategy))
            throw new ArgumentException($"Strategy {strategy} not registered");

        var startTime = DateTime.UtcNow;
        var result = _strategies[strategy].Reason(question, config);
        result.ExecutionTime = DateTime.UtcNow - startTime;
        result.Strategy = strategy;

        return result;
    }

    /// <summary>
    /// Reason using multiple strategies and compare results.
    /// </summary>
    public Dictionary<ReasoningStrategy, ReasoningResult> ReasonMultiStrategy(
        string question,
        List<ReasoningStrategy> strategies,
        ReasoningConfig config = null)
    {
        var results = new Dictionary<ReasoningStrategy, ReasoningResult>();

        foreach (var strategy in strategies)
        {
            results[strategy] = Reason(question, strategy, config);
        }

        return results;
    }
}
```

---

## Phase 2: Chain-of-Thought

### Step 1: Implement Zero-Shot CoT

**File**: `src/Reasoning/ChainOfThought/ZeroShotCoT.cs`

```csharp
namespace AiDotNet.Reasoning.ChainOfThought;

using AiDotNet.Reasoning.Core;
using AiDotNet.Reasoning.Core.Models;

/// <summary>
/// Zero-shot Chain-of-Thought reasoning.
/// Adds "Let's think step by step" to prompt.
/// </summary>
public class ZeroShotCoT : IReasoningStrategy
{
    private readonly ILanguageModel _model;

    public string StrategyName => "Zero-Shot CoT";

    public ZeroShotCoT(ILanguageModel model)
    {
        _model = model;
    }

    public ReasoningResult Reason(string question, ReasoningConfig config)
    {
        // Build prompt with zero-shot instruction
        string prompt = BuildPrompt(question);

        // Generate reasoning
        var response = _model.Generate(prompt, new GenerationConfig
        {
            Temperature = config.Temperature,
            MaxTokens = config.MaxTokens
        });

        // Parse response into reasoning chain
        var chain = ParseReasoningChain(question, response);

        return new ReasoningResult
        {
            Question = question,
            Answer = chain.Answer,
            Chains = new List<ReasoningChain> { chain },
            Confidence = chain.OverallConfidence
        };
    }

    private string BuildPrompt(string question)
    {
        return $"{question}\n\nLet's think step by step:";
    }

    private ReasoningChain ParseReasoningChain(string question, string response)
    {
        var chain = new ReasoningChain { Question = question };

        // Split response into steps
        var lines = response.Split('\n', StringSplitOptions.RemoveEmptyEntries);
        int stepNumber = 1;

        foreach (var line in lines)
        {
            var trimmed = line.Trim();

            // Check if this is an answer line
            if (trimmed.StartsWith("Answer:", StringComparison.OrdinalIgnoreCase))
            {
                chain.Answer = trimmed.Substring("Answer:".Length).Trim();
                chain.IsComplete = true;
                continue;
            }

            // Otherwise, treat as a reasoning step
            if (!string.IsNullOrWhiteSpace(trimmed))
            {
                chain.Steps.Add(new ReasoningStep
                {
                    StepNumber = stepNumber++,
                    Content = trimmed
                });
            }
        }

        // If no explicit answer, use last line
        if (string.IsNullOrEmpty(chain.Answer) && chain.Steps.Count > 0)
        {
            chain.Answer = chain.Steps.Last().Content;
        }

        chain.OverallConfidence = 0.7;  // Default confidence for zero-shot
        return chain;
    }
}
```

### Step 2: Implement Few-Shot CoT

**File**: `src/Reasoning/ChainOfThought/FewShotCoT.cs`

```csharp
namespace AiDotNet.Reasoning.ChainOfThought;

using AiDotNet.Reasoning.Core;
using AiDotNet.Reasoning.Core.Models;

/// <summary>
/// Few-shot Chain-of-Thought reasoning.
/// Provides examples with reasoning chains.
/// </summary>
public class FewShotCoT : IReasoningStrategy
{
    private readonly ILanguageModel _model;
    private readonly List<CoTExample> _examples;

    public string StrategyName => "Few-Shot CoT";

    public FewShotCoT(ILanguageModel model, List<CoTExample> examples)
    {
        _model = model;
        _examples = examples;
    }

    public ReasoningResult Reason(string question, ReasoningConfig config)
    {
        // Build prompt with examples
        string prompt = BuildPrompt(question, _examples);

        // Generate reasoning
        var response = _model.Generate(prompt, new GenerationConfig
        {
            Temperature = config.Temperature,
            MaxTokens = config.MaxTokens
        });

        // Parse response
        var chain = ParseReasoningChain(question, response);

        return new ReasoningResult
        {
            Question = question,
            Answer = chain.Answer,
            Chains = new List<ReasoningChain> { chain },
            Confidence = chain.OverallConfidence
        };
    }

    private string BuildPrompt(string question, List<CoTExample> examples)
    {
        var sb = new System.Text.StringBuilder();

        // Add examples
        foreach (var example in examples)
        {
            sb.AppendLine($"Q: {example.Question}");
            sb.AppendLine("A: Let's think step by step:");

            foreach (var step in example.ReasoningSteps)
            {
                sb.AppendLine(step);
            }

            sb.AppendLine($"Answer: {example.Answer}");
            sb.AppendLine();
        }

        // Add actual question
        sb.AppendLine($"Q: {question}");
        sb.Append("A: Let's think step by step:");

        return sb.ToString();
    }

    private ReasoningChain ParseReasoningChain(string question, string response)
    {
        // Similar to ZeroShotCoT parsing
        var chain = new ReasoningChain { Question = question };
        var lines = response.Split('\n', StringSplitOptions.RemoveEmptyEntries);
        int stepNumber = 1;

        foreach (var line in lines)
        {
            var trimmed = line.Trim();

            if (trimmed.StartsWith("Answer:", StringComparison.OrdinalIgnoreCase))
            {
                chain.Answer = trimmed.Substring("Answer:".Length).Trim();
                chain.IsComplete = true;
            }
            else if (!string.IsNullOrWhiteSpace(trimmed))
            {
                chain.Steps.Add(new ReasoningStep
                {
                    StepNumber = stepNumber++,
                    Content = trimmed
                });
            }
        }

        chain.OverallConfidence = 0.8;  // Higher confidence with examples
        return chain;
    }
}

/// <summary>
/// Example for few-shot learning.
/// </summary>
public class CoTExample
{
    public string Question { get; set; } = string.Empty;
    public List<string> ReasoningSteps { get; set; } = new();
    public string Answer { get; set; } = string.Empty;
}
```

### Testing Chain-of-Thought

**File**: `tests/UnitTests/Reasoning/ChainOfThoughtTests.cs`

```csharp
namespace AiDotNet.Tests.Reasoning;

using Xunit;
using AiDotNet.Reasoning.ChainOfThought;
using AiDotNet.Reasoning.Core;

public class ChainOfThoughtTests
{
    [Fact]
    public void ZeroShotCoT_MathProblem_GeneratesReasoning()
    {
        // Arrange
        var model = new MockLanguageModel();
        var strategy = new ZeroShotCoT(model);
        var config = new ReasoningConfig();

        string question = "If a train travels 60 miles in 1 hour, how far does it travel in 2.5 hours?";

        // Act
        var result = strategy.Reason(question, config);

        // Assert
        Assert.NotNull(result);
        Assert.NotEmpty(result.Chains);
        Assert.True(result.Chains[0].Steps.Count > 0);
        Assert.NotEmpty(result.Answer);
    }

    [Fact]
    public void FewShotCoT_WithExamples_UsesExamples()
    {
        // Arrange
        var model = new MockLanguageModel();
        var examples = new List<CoTExample>
        {
            new CoTExample
            {
                Question = "2 + 2 = ?",
                ReasoningSteps = new List<string> { "2 + 2 = 4" },
                Answer = "4"
            }
        };

        var strategy = new FewShotCoT(model, examples);
        var config = new ReasoningConfig();

        string question = "3 + 3 = ?";

        // Act
        var result = strategy.Reason(question, config);

        // Assert
        Assert.NotNull(result);
        Assert.Contains("6", result.Answer);
    }
}
```

---

## Phase 3: Tree-of-Thoughts

### Step 1: Implement Thought Node

**File**: `src/Reasoning/TreeOfThoughts/ThoughtNode.cs`

```csharp
namespace AiDotNet.Reasoning.TreeOfThoughts;

/// <summary>
/// Node in the Tree-of-Thoughts.
/// </summary>
public class ThoughtNode
{
    public string NodeId { get; set; } = Guid.NewGuid().ToString();
    public string State { get; set; } = string.Empty;
    public string Action { get; set; } = string.Empty;
    public double Score { get; set; } = 0.0;
    public int Depth { get; set; } = 0;
    public ThoughtNode Parent { get; set; }
    public List<ThoughtNode> Children { get; set; } = new();
    public bool IsLeaf => Children.Count == 0;
    public bool IsSolution { get; set; } = false;

    /// <summary>
    /// Get the path from root to this node.
    /// </summary>
    public List<ThoughtNode> GetPath()
    {
        var path = new List<ThoughtNode>();
        var current = this;

        while (current != null)
        {
            path.Insert(0, current);
            current = current.Parent;
        }

        return path;
    }

    /// <summary>
    /// Get the full reasoning as a string.
    /// </summary>
    public string GetReasoning()
    {
        var path = GetPath();
        var sb = new System.Text.StringBuilder();

        foreach (var node in path)
        {
            if (!string.IsNullOrEmpty(node.Action))
                sb.AppendLine($"Step {node.Depth}: {node.Action}");
            sb.AppendLine($"State: {node.State}");
        }

        return sb.ToString();
    }
}
```

### Step 2: Implement Thought Generator

**File**: `src/Reasoning/TreeOfThoughts/ThoughtGenerator.cs`

```csharp
namespace AiDotNet.Reasoning.TreeOfThoughts;

using AiDotNet.Reasoning.Core;

/// <summary>
/// Generates candidate next thoughts.
/// </summary>
public class ThoughtGenerator
{
    private readonly ILanguageModel _model;

    public ThoughtGenerator(ILanguageModel model)
    {
        _model = model;
    }

    /// <summary>
    /// Generate k candidate next thoughts.
    /// </summary>
    public List<ThoughtNode> GenerateThoughts(ThoughtNode current, int k, string problem)
    {
        var prompt = BuildPrompt(current, k, problem);
        var response = _model.Generate(prompt);

        return ParseThoughts(response, current);
    }

    private string BuildPrompt(ThoughtNode current, int k, string problem)
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine($"Problem: {problem}");
        sb.AppendLine($"\nCurrent state: {current.State}");
        sb.AppendLine($"\nGenerate {k} possible next steps to solve this problem.");
        sb.AppendLine("Format each step as:");
        sb.AppendLine("Action: [what to do]");
        sb.AppendLine("State: [resulting state]");

        return sb.ToString();
    }

    private List<ThoughtNode> ParseThoughts(string response, ThoughtNode parent)
    {
        var thoughts = new List<ThoughtNode>();
        var lines = response.Split('\n', StringSplitOptions.RemoveEmptyEntries);

        string currentAction = null;
        string currentState = null;

        foreach (var line in lines)
        {
            var trimmed = line.Trim();

            if (trimmed.StartsWith("Action:", StringComparison.OrdinalIgnoreCase))
            {
                currentAction = trimmed.Substring("Action:".Length).Trim();
            }
            else if (trimmed.StartsWith("State:", StringComparison.OrdinalIgnoreCase))
            {
                currentState = trimmed.Substring("State:".Length).Trim();

                // Create thought node
                if (!string.IsNullOrEmpty(currentAction) && !string.IsNullOrEmpty(currentState))
                {
                    thoughts.Add(new ThoughtNode
                    {
                        Action = currentAction,
                        State = currentState,
                        Parent = parent,
                        Depth = parent.Depth + 1
                    });

                    currentAction = null;
                    currentState = null;
                }
            }
        }

        return thoughts;
    }
}
```

### Step 3: Implement Thought Evaluator

**File**: `src/Reasoning/TreeOfThoughts/ThoughtEvaluator.cs`

```csharp
namespace AiDotNet.Reasoning.TreeOfThoughts;

using AiDotNet.Reasoning.Core;

/// <summary>
/// Evaluates the promise of a thought.
/// </summary>
public class ThoughtEvaluator
{
    private readonly ILanguageModel _model;

    public ThoughtEvaluator(ILanguageModel model)
    {
        _model = model;
    }

    /// <summary>
    /// Score how promising a thought is (0-10).
    /// </summary>
    public double EvaluateThought(ThoughtNode thought, string problem)
    {
        var prompt = BuildPrompt(thought, problem);
        var response = _model.Generate(prompt);

        return ParseScore(response);
    }

    /// <summary>
    /// Check if a thought is a solution.
    /// </summary>
    public bool IsSolution(ThoughtNode thought, string problem)
    {
        var prompt = $"Problem: {problem}\n\nState: {thought.State}\n\nIs this a valid solution? Answer yes or no.";
        var response = _model.Generate(prompt);

        return response.Trim().StartsWith("yes", StringComparison.OrdinalIgnoreCase);
    }

    private string BuildPrompt(ThoughtNode thought, string problem)
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine($"Problem: {problem}");
        sb.AppendLine($"\nCurrent reasoning path:");

        var path = thought.GetPath();
        foreach (var node in path)
        {
            if (!string.IsNullOrEmpty(node.Action))
                sb.AppendLine($"  {node.Action}");
        }

        sb.AppendLine($"\nRate how promising this path is for solving the problem (0-10):");
        sb.AppendLine("0 = Dead end, will never lead to solution");
        sb.AppendLine("10 = Very promising, likely to lead to solution");
        sb.AppendLine("\nScore:");

        return sb.ToString();
    }

    private double ParseScore(string response)
    {
        // Extract numeric score
        var match = System.Text.RegularExpressions.Regex.Match(response, @"(\d+(?:\.\d+)?)");
        if (match.Success && double.TryParse(match.Groups[1].Value, out double score))
        {
            return Math.Clamp(score / 10.0, 0.0, 1.0);  // Normalize to [0, 1]
        }

        return 0.5;  // Default neutral score
    }
}
```

### Step 4: Implement Breadth-First Search

**File**: `src/Reasoning/TreeOfThoughts/Search/BreadthFirstSearch.cs`

```csharp
namespace AiDotNet.Reasoning.TreeOfThoughts.Search;

using AiDotNet.Reasoning.Core;

/// <summary>
/// Breadth-first search for Tree-of-Thoughts.
/// </summary>
public class BreadthFirstSearch
{
    private readonly ThoughtGenerator _generator;
    private readonly ThoughtEvaluator _evaluator;

    public BreadthFirstSearch(ThoughtGenerator generator, ThoughtEvaluator evaluator)
    {
        _generator = generator;
        _evaluator = evaluator;
    }

    /// <summary>
    /// Search for a solution using BFS.
    /// </summary>
    public ThoughtNode Search(
        string problem,
        int maxDepth,
        int branchingFactor,
        double pruningThreshold)
    {
        var root = new ThoughtNode { State = problem, Depth = 0 };
        var queue = new Queue<ThoughtNode>();
        queue.Enqueue(root);

        int nodesExplored = 0;
        int maxNodes = 1000;  // Prevent infinite search

        while (queue.Count > 0 && nodesExplored < maxNodes)
        {
            var current = queue.Dequeue();
            nodesExplored++;

            // Check if current node is a solution
            if (_evaluator.IsSolution(current, problem))
            {
                current.IsSolution = true;
                return current;
            }

            // Don't expand beyond max depth
            if (current.Depth >= maxDepth)
                continue;

            // Generate child thoughts
            var children = _generator.GenerateThoughts(current, branchingFactor, problem);

            // Evaluate and prune children
            foreach (var child in children)
            {
                child.Score = _evaluator.EvaluateThought(child, problem);

                // Prune low-scoring thoughts
                if (child.Score >= pruningThreshold)
                {
                    current.Children.Add(child);
                    queue.Enqueue(child);
                }
            }
        }

        // No solution found, return best node
        return FindBestNode(root);
    }

    private ThoughtNode FindBestNode(ThoughtNode root)
    {
        // BFS to find highest-scoring node
        var queue = new Queue<ThoughtNode>();
        queue.Enqueue(root);

        ThoughtNode best = root;

        while (queue.Count > 0)
        {
            var current = queue.Dequeue();

            if (current.Score > best.Score)
                best = current;

            foreach (var child in current.Children)
                queue.Enqueue(child);
        }

        return best;
    }
}
```

### Step 5: Implement ToT Strategy

**File**: `src/Reasoning/TreeOfThoughts/ToTStrategy.cs`

```csharp
namespace AiDotNet.Reasoning.TreeOfThoughts;

using AiDotNet.Reasoning.Core;
using AiDotNet.Reasoning.Core.Models;
using AiDotNet.Reasoning.TreeOfThoughts.Search;

/// <summary>
/// Tree-of-Thoughts reasoning strategy.
/// </summary>
public class ToTStrategy : IReasoningStrategy
{
    private readonly ILanguageModel _model;
    private readonly ThoughtGenerator _generator;
    private readonly ThoughtEvaluator _evaluator;

    public string StrategyName => "Tree-of-Thoughts";

    public ToTStrategy(ILanguageModel model)
    {
        _model = model;
        _generator = new ThoughtGenerator(model);
        _evaluator = new ThoughtEvaluator(model);
    }

    public ReasoningResult Reason(string question, ReasoningConfig config)
    {
        // Select search algorithm
        ThoughtNode solution = config.SearchType switch
        {
            ToTSearchType.BreadthFirst => BreadthFirstSearch(question, config),
            ToTSearchType.DepthFirst => DepthFirstSearch(question, config),
            ToTSearchType.BestFirst => BestFirstSearch(question, config),
            _ => BreadthFirstSearch(question, config)
        };

        // Convert thought tree to reasoning chain
        var chain = ConvertToReasoningChain(solution, question);

        return new ReasoningResult
        {
            Question = question,
            Answer = chain.Answer,
            Chains = new List<ReasoningChain> { chain },
            Confidence = solution.Score,
            Metadata = new Dictionary<string, object>
            {
                { "nodes_explored", CountNodes(solution) },
                { "max_depth", solution.Depth }
            }
        };
    }

    private ThoughtNode BreadthFirstSearch(string problem, ReasoningConfig config)
    {
        var search = new BreadthFirstSearch(_generator, _evaluator);
        return search.Search(
            problem,
            config.MaxDepth,
            config.BranchingFactor,
            config.PruningThreshold);
    }

    private ThoughtNode DepthFirstSearch(string problem, ReasoningConfig config)
    {
        // Similar to BFS but use stack instead of queue
        throw new NotImplementedException();
    }

    private ThoughtNode BestFirstSearch(string problem, ReasoningConfig config)
    {
        // Use priority queue based on scores
        throw new NotImplementedException();
    }

    private ReasoningChain ConvertToReasoningChain(ThoughtNode solution, string question)
    {
        var chain = new ReasoningChain
        {
            Question = question,
            OverallConfidence = solution.Score,
            IsComplete = solution.IsSolution
        };

        var path = solution.GetPath();
        int stepNumber = 1;

        foreach (var node in path)
        {
            if (!string.IsNullOrEmpty(node.Action))
            {
                chain.Steps.Add(new ReasoningStep
                {
                    StepNumber = stepNumber++,
                    Content = node.Action,
                    Confidence = node.Score
                });
            }
        }

        chain.Answer = solution.State;
        return chain;
    }

    private int CountNodes(ThoughtNode root)
    {
        int count = 1;
        foreach (var child in root.Children)
            count += CountNodes(child);
        return count;
    }
}
```

---

## Phase 4: Self-Consistency

### Step 1: Implement Self-Consistency Strategy

**File**: `src/Reasoning/SelfConsistency/SelfConsistencyStrategy.cs`

```csharp
namespace AiDotNet.Reasoning.SelfConsistency;

using AiDotNet.Reasoning.Core;
using AiDotNet.Reasoning.Core.Models;
using AiDotNet.Reasoning.ChainOfThought;

/// <summary>
/// Self-consistency reasoning strategy.
/// Samples multiple reasoning paths and aggregates answers.
/// </summary>
public class SelfConsistencyStrategy : IReasoningStrategy
{
    private readonly ILanguageModel _model;
    private readonly ZeroShotCoT _cotStrategy;

    public string StrategyName => "Self-Consistency";

    public SelfConsistencyStrategy(ILanguageModel model)
    {
        _model = model;
        _cotStrategy = new ZeroShotCoT(model);
    }

    public ReasoningResult Reason(string question, ReasoningConfig config)
    {
        var chains = new List<ReasoningChain>();

        // Generate multiple reasoning paths with different temperatures
        for (int i = 0; i < config.NumSamples; i++)
        {
            var sampleConfig = new ReasoningConfig
            {
                Temperature = config.Temperature + (i * 0.1),  // Vary temperature
                MaxTokens = config.MaxTokens
            };

            var result = _cotStrategy.Reason(question, sampleConfig);
            chains.Add(result.Chains[0]);
        }

        // Aggregate answers
        var aggregator = new AnswerAggregator();
        var finalAnswer = aggregator.Aggregate(chains, config.VotingStrategy);

        // Compute confidence based on agreement
        double confidence = aggregator.ComputeAgreement(chains);

        return new ReasoningResult
        {
            Question = question,
            Answer = finalAnswer,
            Chains = chains,
            Confidence = confidence,
            Metadata = new Dictionary<string, object>
            {
                { "num_samples", chains.Count },
                { "unique_answers", aggregator.GetUniqueAnswers(chains).Count }
            }
        };
    }
}
```

### Step 2: Implement Answer Aggregator

**File**: `src/Reasoning/SelfConsistency/AnswerAggregator.cs`

```csharp
namespace AiDotNet.Reasoning.SelfConsistency;

using AiDotNet.Reasoning.Core;
using AiDotNet.Reasoning.Core.Models;

/// <summary>
/// Aggregates answers from multiple reasoning chains.
/// </summary>
public class AnswerAggregator
{
    /// <summary>
    /// Aggregate answers using specified voting strategy.
    /// </summary>
    public string Aggregate(List<ReasoningChain> chains, VotingStrategy strategy)
    {
        return strategy switch
        {
            VotingStrategy.Majority => MajorityVote(chains),
            VotingStrategy.Weighted => WeightedVote(chains),
            VotingStrategy.ConfidenceThreshold => ConfidenceThresholdVote(chains),
            _ => MajorityVote(chains)
        };
    }

    /// <summary>
    /// Simple majority vote.
    /// </summary>
    private string MajorityVote(List<ReasoningChain> chains)
    {
        var answerCounts = new Dictionary<string, int>();

        foreach (var chain in chains)
        {
            string normalizedAnswer = NormalizeAnswer(chain.Answer);

            if (!answerCounts.ContainsKey(normalizedAnswer))
                answerCounts[normalizedAnswer] = 0;

            answerCounts[normalizedAnswer]++;
        }

        return answerCounts.OrderByDescending(kv => kv.Value).First().Key;
    }

    /// <summary>
    /// Weighted vote by confidence scores.
    /// </summary>
    private string WeightedVote(List<ReasoningChain> chains)
    {
        var answerScores = new Dictionary<string, double>();

        foreach (var chain in chains)
        {
            string normalizedAnswer = NormalizeAnswer(chain.Answer);

            if (!answerScores.ContainsKey(normalizedAnswer))
                answerScores[normalizedAnswer] = 0.0;

            answerScores[normalizedAnswer] += chain.OverallConfidence;
        }

        return answerScores.OrderByDescending(kv => kv.Value).First().Key;
    }

    /// <summary>
    /// Only return answer if confidence threshold met.
    /// </summary>
    private string ConfidenceThresholdVote(List<ReasoningChain> chains, double threshold = 0.5)
    {
        var majority = MajorityVote(chains);
        double agreement = ComputeAgreement(chains);

        if (agreement >= threshold)
            return majority;

        return "Insufficient confidence";
    }

    /// <summary>
    /// Compute agreement score (0-1).
    /// </summary>
    public double ComputeAgreement(List<ReasoningChain> chains)
    {
        if (chains.Count == 0)
            return 0.0;

        var answerCounts = new Dictionary<string, int>();

        foreach (var chain in chains)
        {
            string normalizedAnswer = NormalizeAnswer(chain.Answer);

            if (!answerCounts.ContainsKey(normalizedAnswer))
                answerCounts[normalizedAnswer] = 0;

            answerCounts[normalizedAnswer]++;
        }

        int maxCount = answerCounts.Values.Max();
        return (double)maxCount / chains.Count;
    }

    /// <summary>
    /// Get unique answers across all chains.
    /// </summary>
    public List<string> GetUniqueAnswers(List<ReasoningChain> chains)
    {
        return chains.Select(c => NormalizeAnswer(c.Answer)).Distinct().ToList();
    }

    /// <summary>
    /// Normalize answer for comparison (lowercase, trim, remove punctuation).
    /// </summary>
    private string NormalizeAnswer(string answer)
    {
        if (string.IsNullOrEmpty(answer))
            return string.Empty;

        // Remove punctuation, convert to lowercase, trim
        var normalized = new string(answer.Where(c => !char.IsPunctuation(c)).ToArray());
        return normalized.ToLower().Trim();
    }
}
```

---

## Phase 5: Reasoning Verification

### Step 1: Implement Mathematical Verifier

**File**: `src/Reasoning/Verification/MathematicalVerifier.cs`

```csharp
namespace AiDotNet.Reasoning.Verification;

using System.Text.RegularExpressions;
using AiDotNet.Reasoning.Core.Models;

/// <summary>
/// Verifies mathematical calculations in reasoning steps.
/// </summary>
public class MathematicalVerifier : IVerifier
{
    public string VerifierName => "Mathematical";

    public bool Verify(ReasoningChain chain)
    {
        bool allValid = true;

        foreach (var step in chain.Steps)
        {
            if (!VerifyStep(step))
            {
                allValid = false;
                step.IsVerified = false;
                step.VerificationErrors.Add("Mathematical calculation error");
            }
            else
            {
                step.IsVerified = true;
            }
        }

        return allValid;
    }

    private bool VerifyStep(ReasoningStep step)
    {
        // Look for arithmetic expressions like "3 + 2 = 5"
        var matches = Regex.Matches(step.Content, @"(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)");

        foreach (Match match in matches)
        {
            double a = double.Parse(match.Groups[1].Value);
            string op = match.Groups[2].Value;
            double b = double.Parse(match.Groups[3].Value);
            double claimed = double.Parse(match.Groups[4].Value);

            double computed = op switch
            {
                "+" => a + b,
                "-" => a - b,
                "*" => a * b,
                "/" => b != 0 ? a / b : double.NaN,
                _ => double.NaN
            };

            // Check if computed matches claimed (with tolerance for floating point)
            if (double.IsNaN(computed) || Math.Abs(computed - claimed) > 1e-6)
            {
                return false;
            }
        }

        return true;
    }
}
```

### Step 2: Implement Logical Verifier

**File**: `src/Reasoning/Verification/LogicalVerifier.cs`

```csharp
namespace AiDotNet.Reasoning.Verification;

using AiDotNet.Reasoning.Core;
using AiDotNet.Reasoning.Core.Models;

/// <summary>
/// Verifies logical consistency of reasoning steps.
/// </summary>
public class LogicalVerifier : IVerifier
{
    private readonly ILanguageModel _model;

    public string VerifierName => "Logical";

    public LogicalVerifier(ILanguageModel model)
    {
        _model = model;
    }

    public bool Verify(ReasoningChain chain)
    {
        for (int i = 1; i < chain.Steps.Count; i++)
        {
            bool followsLogically = CheckLogicalConnection(
                chain.Steps[i - 1],
                chain.Steps[i]);

            if (!followsLogically)
            {
                chain.Steps[i].IsVerified = false;
                chain.Steps[i].VerificationErrors.Add("Does not follow logically from previous step");
                return false;
            }
            else
            {
                chain.Steps[i].IsVerified = true;
            }
        }

        return true;
    }

    private bool CheckLogicalConnection(ReasoningStep previous, ReasoningStep current)
    {
        var prompt = $@"
Previous step: {previous.Content}
Current step: {current.Content}

Does the current step follow logically from the previous step?
Answer 'yes' or 'no' and briefly explain why.";

        var response = _model.Generate(prompt);

        return response.Trim().StartsWith("yes", StringComparison.OrdinalIgnoreCase);
    }
}
```

### Step 3: Implement Verifier Interface

**File**: `src/Reasoning/Verification/IVerifier.cs`

```csharp
namespace AiDotNet.Reasoning.Verification;

using AiDotNet.Reasoning.Core.Models;

/// <summary>
/// Interface for reasoning verifiers.
/// </summary>
public interface IVerifier
{
    /// <summary>
    /// Name of the verifier.
    /// </summary>
    string VerifierName { get; }

    /// <summary>
    /// Verify a reasoning chain.
    /// </summary>
    /// <returns>True if verification passes, false otherwise</returns>
    bool Verify(ReasoningChain chain);
}
```

---

## Testing Strategy

### Unit Tests

**File**: `tests/UnitTests/Reasoning/VerificationTests.cs`

```csharp
namespace AiDotNet.Tests.Reasoning;

using Xunit;
using AiDotNet.Reasoning.Verification;
using AiDotNet.Reasoning.Core.Models;

public class VerificationTests
{
    [Fact]
    public void MathematicalVerifier_CorrectCalculation_Passes()
    {
        // Arrange
        var verifier = new MathematicalVerifier();
        var chain = new ReasoningChain
        {
            Steps = new List<ReasoningStep>
            {
                new ReasoningStep { Content = "First, 3 + 2 = 5" },
                new ReasoningStep { Content = "Then, 5 * 2 = 10" }
            }
        };

        // Act
        bool valid = verifier.Verify(chain);

        // Assert
        Assert.True(valid);
        Assert.All(chain.Steps, step => Assert.True(step.IsVerified));
    }

    [Fact]
    public void MathematicalVerifier_IncorrectCalculation_Fails()
    {
        // Arrange
        var verifier = new MathematicalVerifier();
        var chain = new ReasoningChain
        {
            Steps = new List<ReasoningStep>
            {
                new ReasoningStep { Content = "3 + 2 = 6" }  // WRONG!
            }
        };

        // Act
        bool valid = verifier.Verify(chain);

        // Assert
        Assert.False(valid);
        Assert.False(chain.Steps[0].IsVerified);
        Assert.NotEmpty(chain.Steps[0].VerificationErrors);
    }
}
```

---

## Common Pitfalls

### 1. Parsing Errors

**Problem**: Model output doesn't match expected format

**Solution**: Use robust parsing with fallbacks
```csharp
var match = Regex.Match(response, @"Answer:\s*(.+)");
string answer = match.Success ? match.Groups[1].Value : response.Trim();
```

### 2. Infinite Loops in ToT

**Problem**: Tree search never terminates

**Solution**: Enforce node budget
```csharp
int nodesExplored = 0;
int maxNodes = 1000;

while (queue.Count > 0 && nodesExplored < maxNodes)
{
    // ...
    nodesExplored++;
}
```

### 3. Low Agreement in Self-Consistency

**Problem**: All samples give different answers

**Solution**: Increase temperature diversity or return "uncertain"
```csharp
if (agreement < 0.3)
    return "Insufficient agreement to determine answer";
```

### 4. False Positives in Verification

**Problem**: Verifier incorrectly marks valid reasoning as invalid

**Solution**: Use multiple verifiers and require majority
```csharp
var verifiers = new List<IVerifier> { mathVerifier, logicalVerifier };
int passedCount = verifiers.Count(v => v.Verify(chain));
bool overallValid = passedCount >= verifiers.Count / 2;
```

---

## Summary

This guide covered:

1. **Chain-of-Thought**: Step-by-step reasoning with zero-shot and few-shot prompting
2. **Tree-of-Thoughts**: Exploring multiple reasoning paths with search algorithms
3. **Self-Consistency**: Sampling multiple paths and aggregating via majority vote
4. **Reasoning Verification**: Checking logical, mathematical, and factual correctness

**Key Takeaways**:
- CoT improves reasoning by forcing models to show their work
- ToT explores multiple paths to find the best solution
- Self-consistency increases reliability through aggregation
- Verification ensures reasoning is logically sound and factually correct

**Next Steps**:
- Implement more advanced verifiers (factual, contradiction)
- Add reinforcement learning to improve reasoning strategies
- Integrate with external tools (calculators, search engines)
- Build reasoning benchmarks (GSM8K, MATH, etc.)
