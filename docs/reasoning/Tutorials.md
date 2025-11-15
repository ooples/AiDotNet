# AiDotNet Reasoning Framework - Tutorials

Comprehensive tutorials for building advanced reasoning applications.

## Table of Contents
1. [Tutorial 1: Building a Math Problem Solver](#tutorial-1-building-a-math-problem-solver)
2. [Tutorial 2: Code Generation Assistant](#tutorial-2-code-generation-assistant)
3. [Tutorial 3: Logic Puzzle Solver](#tutorial-3-logic-puzzle-solver)
4. [Tutorial 4: Training with Reinforcement Learning](#tutorial-4-training-with-reinforcement-learning)
5. [Tutorial 5: Custom Benchmark Evaluation](#tutorial-5-custom-benchmark-evaluation)

---

## Tutorial 1: Building a Math Problem Solver

Build a complete system that solves math problems with verification and explanation.

### Step 1: Setup
```csharp
using AiDotNet.Reasoning.DomainSpecific;
using AiDotNet.Reasoning.Models;
using AiDotNet.Reasoning.Verification;

public class MathSolver
{
    private readonly MathematicalReasoner<double> _reasoner;
    private readonly CalculatorVerifier<double> _calculator;
    private readonly CriticModel<double> _critic;
    private readonly SelfRefinementEngine<double> _refinement;

    public MathSolver(IChatModel chatModel)
    {
        _reasoner = new MathematicalReasoner<double>(chatModel);
        _calculator = new CalculatorVerifier<double>();
        _critic = new CriticModel<double>(chatModel);
        _refinement = new SelfRefinementEngine<double>(chatModel);
    }
}
```

### Step 2: Solve with Verification
```csharp
public async Task<SolutionResult> SolveAsync(string problem)
{
    // Initial solution
    var result = await _reasoner.SolveAsync(
        problem,
        useVerification: true,
        useSelfConsistency: false
    );

    if (!result.Success)
    {
        return new SolutionResult { Success = false, Error = result.ErrorMessage };
    }

    // Verify calculations
    var verification = await _calculator.VerifyAsync(result.Chain, result.FinalAnswer);

    if (!verification.IsValid)
    {
        // Try to refine
        result = await RefineAsync(result, problem);
        verification = await _calculator.VerifyAsync(result.Chain, result.FinalAnswer);
    }

    return new SolutionResult
    {
        Success = true,
        Answer = result.FinalAnswer,
        Steps = result.Chain.Steps.Select(s => s.Content).ToList(),
        IsVerified = verification.IsValid,
        Confidence = result.ConfidenceScore
    };
}
```

### Step 3: Add Refinement
```csharp
private async Task<ReasoningResult<double>> RefineAsync(
    ReasoningResult<double> result,
    string problem)
{
    var context = new ReasoningContext
    {
        OriginalQuery = problem,
        Requirements = new List<string>
        {
            "Correct mathematical calculations",
            "Clear step-by-step reasoning",
            "Proper use of formulas"
        }
    };

    foreach (var step in result.Chain.Steps)
    {
        var critique = await _critic.CritiqueStepAsync(step, context);

        if (Convert.ToDouble(critique.OverallScore) < 0.7)
        {
            var refinedStep = await _refinement.RefineStepAsync(step, critique, context);
            // Update step content
            step.Content = refinedStep.Content;
            step.RefinementCount++;
        }
    }

    return result;
}
```

### Step 4: Test It
```csharp
var chatModel = /* your chat model */;
var solver = new MathSolver(chatModel);

var problems = new[]
{
    "What is 347 + 892?",
    "Calculate 15 × 24",
    "If a train travels at 60 km/h for 2.5 hours, how far does it go?"
};

foreach (var problem in problems)
{
    var result = await solver.SolveAsync(problem);
    Console.WriteLine($"\nProblem: {problem}");
    Console.WriteLine($"Answer: {result.Answer}");
    Console.WriteLine($"Verified: {(result.IsVerified ? "✓" : "✗")}");
    Console.WriteLine($"Confidence: {result.Confidence:P0}");

    Console.WriteLine("\nSteps:");
    for (int i = 0; i < result.Steps.Count; i++)
    {
        Console.WriteLine($"  {i + 1}. {result.Steps[i]}");
    }
}
```

---

## Tutorial 2: Code Generation Assistant

Build an AI coding assistant that generates, explains, and debugs code.

### Step 1: Setup
```csharp
using AiDotNet.Reasoning.DomainSpecific;
using AiDotNet.Reasoning.Verification;

public class CodeAssistant
{
    private readonly CodeReasoner<double> _reasoner;
    private readonly CodeExecutionVerifier<double> _executor;

    public CodeAssistant(IChatModel chatModel)
    {
        _reasoner = new CodeReasoner<double>(chatModel);
        _executor = new CodeExecutionVerifier<double>(timeoutMilliseconds: 5000);
    }
}
```

### Step 2: Generate Code
```csharp
public async Task<CodeGenerationResult> GenerateAsync(
    string specification,
    string language = "python",
    string[] testCases = null)
{
    // Generate code
    var result = await _reasoner.GenerateCodeAsync(specification, language);

    if (!result.Success)
    {
        return new CodeGenerationResult { Success = false, Error = result.ErrorMessage };
    }

    // Extract code from result
    string code = ExtractCode(result.FinalAnswer, language);

    // If test cases provided, execute and verify
    CodeExecutionResult<double> executionResult = null;
    if (testCases != null && testCases.Length > 0)
    {
        executionResult = await _executor.VerifyCodeAsync(code, testCases, language);
    }

    return new CodeGenerationResult
    {
        Success = true,
        Code = code,
        Explanation = GetExplanation(result.Chain),
        TestsPassed = executionResult?.AllTestsPassed ?? true,
        ExecutionSummary = executionResult?.GetSummary()
    };
}
```

### Step 3: Debug Code
```csharp
public async Task<DebugResult> DebugAsync(string buggyCode, string errorMessage)
{
    var debugPrompt = $@"Debug this code:
```
{buggyCode}
```

Error: {errorMessage}

Find the bug and provide:
1. Explanation of the bug
2. Corrected code
3. Test cases to prevent regression";

    var result = await _reasoner.SolveAsync(debugPrompt);

    return new DebugResult
    {
        BugExplanation = ExtractExplanation(result),
        FixedCode = ExtractCode(result.FinalAnswer, "python"),
        TestCases = ExtractTestCases(result)
    };
}
```

### Step 4: Explain Code
```csharp
public async Task<string> ExplainAsync(string code)
{
    var result = await _reasoner.ExplainCodeAsync(code);
    return result.FinalAnswer;
}
```

### Step 5: Test the Assistant
```csharp
var chatModel = /* your chat model */;
var assistant = new CodeAssistant(chatModel);

// Test 1: Generate code
var genResult = await assistant.GenerateAsync(
    "Write a function to check if a number is prime",
    "python",
    testCases: new[]
    {
        "assert is_prime(2) == True",
        "assert is_prime(4) == False",
        "assert is_prime(17) == True"
    }
);

Console.WriteLine($"Code generated: {genResult.Success}");
Console.WriteLine($"Tests passed: {genResult.TestsPassed}");
Console.WriteLine($"\nCode:\n{genResult.Code}");

// Test 2: Debug code
string buggyCode = @"
def factorial(n):
    if n = 1:  # Bug: should be ==
        return 1
    return n * factorial(n - 1)
";

var debugResult = await assistant.DebugAsync(buggyCode, "SyntaxError: invalid syntax");
Console.WriteLine($"\nBug: {debugResult.BugExplanation}");
Console.WriteLine($"\nFixed:\n{debugResult.FixedCode}");
```

---

## Tutorial 3: Logic Puzzle Solver

Build a system that solves complex logic puzzles using Tree-of-Thoughts.

### Step 1: Setup
```csharp
using AiDotNet.Reasoning.DomainSpecific;
using AiDotNet.Reasoning.Strategies;
using AiDotNet.Reasoning.Verification;

public class LogicPuzzleSolver
{
    private readonly LogicalReasoner<double> _reasoner;
    private readonly ContradictionDetector<double> _contradictionDetector;

    public LogicPuzzleSolver(IChatModel chatModel)
    {
        _reasoner = new LogicalReasoner<double>(chatModel);
        _contradictionDetector = new ContradictionDetector<double>(chatModel);
    }
}
```

### Step 2: Solve Puzzle
```csharp
public async Task<PuzzleSolution> SolveAsync(string puzzle)
{
    // Use tree search for complex puzzles
    var result = await _reasoner.SolvePuzzleAsync(puzzle);

    if (!result.Success)
    {
        return new PuzzleSolution { Success = false, Error = result.ErrorMessage };
    }

    // Check for contradictions
    var contradictions = await _contradictionDetector.DetectContradictionsAsync(result.Chain);

    if (contradictions.Count > 0)
    {
        // Try again with different approach
        var config = new ReasoningConfig
        {
            ExplorationDepth = 6,  // Deeper search
            BranchingFactor = 4     // More branches
        };
        result = await _reasoner.SolvePuzzleAsync(puzzle, config);
    }

    return new PuzzleSolution
    {
        Success = true,
        Solution = result.FinalAnswer,
        ReasoningPath = result.Chain.Steps.Select(s => s.Content).ToList(),
        HasContradictions = contradictions.Count > 0,
        Confidence = result.ConfidenceScore
    };
}
```

### Step 3: Test with Classic Puzzles
```csharp
var chatModel = /* your chat model */;
var solver = new LogicPuzzleSolver(chatModel);

// Classic: Knights and Knaves
string puzzle1 = @"
On an island, there are two types of people:
- Knights always tell the truth
- Knaves always lie

You meet two people, A and B.
A says: 'We are both knaves.'
B says nothing.

What are A and B?
";

var result = await solver.SolveAsync(puzzle1);
Console.WriteLine($"Solution: {result.Solution}");
Console.WriteLine($"\nReasoning:");
foreach (var step in result.ReasoningPath)
{
    Console.WriteLine($"  • {step}");
}

// Classic: Einstein's Riddle (simplified)
string puzzle2 = @"
There are 3 houses in different colors.
- The red house is to the left of the blue house
- The green house is to the right of the blue house
- Coffee is drunk in the green house

What is drunk in the blue house?
";

result = await solver.SolveAsync(puzzle2);
Console.WriteLine($"\nSolution: {result.Solution}");
```

---

## Tutorial 4: Training with Reinforcement Learning

Train a reasoning model using reinforcement learning.

### Step 1: Prepare Training Data
```csharp
using AiDotNet.Reasoning.Training;
using AiDotNet.Reasoning.Benchmarks;

public class ModelTrainer
{
    private readonly ReinforcementLearner<double> _learner;
    private readonly TrainingDataCollector<double> _dataCollector;

    public ModelTrainer(IChatModel chatModel, IRewardModel<double> rewardModel)
    {
        _learner = new ReinforcementLearner<double>(chatModel, rewardModel);
        _dataCollector = new TrainingDataCollector<double>();
    }

    public async Task<List<(string, string)>> LoadTrainingDataAsync()
    {
        // Load from GSM8K benchmark
        var benchmark = new GSM8KBenchmark<double>();
        var problems = await benchmark.LoadProblemsAsync(count: 1000);

        return problems.Select(p => (p.Problem, p.CorrectAnswer)).ToList();
    }
}
```

### Step 2: Configure Training
```csharp
public async Task TrainModelAsync()
{
    var trainingData = await LoadTrainingDataAsync();
    var validationData = await LoadValidationDataAsync();

    var config = new RLConfig
    {
        Epochs = 10,
        BatchSize = 32,
        LearningRate = 0.0001,
        ValidationFrequency = 1,
        EarlyStoppingPatience = 3,
        SaveCheckpoints = true
    };

    // Monitor progress
    _learner.OnEpochComplete += (sender, metrics) =>
    {
        Console.WriteLine($"\nEpoch {metrics.Epoch} Complete:");
        Console.WriteLine($"  Accuracy: {metrics.Accuracy:P2}");
        Console.WriteLine($"  Avg Reward: {Convert.ToDouble(metrics.AverageReward):F3}");
        Console.WriteLine($"  Avg Loss: {Convert.ToDouble(metrics.AverageLoss):F4}");
    };

    _learner.OnBatchComplete += (sender, progress) =>
    {
        if (progress.BatchNumber % 10 == 0)
        {
            Console.WriteLine($"  Batch {progress.BatchNumber}/{progress.TotalBatches}");
        }
    };

    // Train
    var results = await _learner.TrainAsync(trainingData, validationData);

    Console.WriteLine($"\n=== Training Complete ===");
    Console.WriteLine($"Best Accuracy: {results.BestAccuracy:P2}");
    Console.WriteLine($"Best Epoch: {results.BestEpoch}");
    Console.WriteLine($"Total Epochs: {results.EpochsTrained}");
}
```

### Step 3: Use STaR Training
```csharp
public async Task TrainWithSTaRAsync()
{
    var trainingData = await LoadTrainingDataAsync();
    var validationData = await LoadValidationDataAsync();

    Console.WriteLine("Starting Self-Taught Reasoner (STaR) training...");

    var results = await _learner.TrainSTaRAsync(
        trainingData,
        validationData,
        samplesPerProblem: 5  // Generate 5 attempts per problem
    );

    Console.WriteLine($"STaR Training Complete!");
    Console.WriteLine($"Best Accuracy: {results.BestAccuracy:P2}");
}
```

---

## Tutorial 5: Custom Benchmark Evaluation

Evaluate your model on multiple benchmarks.

### Step 1: Setup Evaluation
```csharp
using AiDotNet.Reasoning.Benchmarks;

public class BenchmarkRunner
{
    private readonly IChatModel _chatModel;
    private readonly IReasoningStrategy<double> _strategy;

    public BenchmarkRunner(IChatModel chatModel)
    {
        _chatModel = chatModel;
        _strategy = new ChainOfThoughtStrategy<double>(chatModel);
    }
}
```

### Step 2: Run Multiple Benchmarks
```csharp
public async Task<Dictionary<string, double>> EvaluateAllAsync(int sampleSize = 100)
{
    var results = new Dictionary<string, double>();

    // GSM8K
    Console.WriteLine("\nEvaluating GSM8K...");
    var gsm8k = new GSM8KBenchmark<double>();
    var gsm8kResult = await gsm8k.EvaluateAsync(SolveAsync, sampleSize);
    results["GSM8K"] = Convert.ToDouble(gsm8kResult.Accuracy);
    Console.WriteLine($"GSM8K Accuracy: {gsm8kResult.Accuracy:P2}");

    // HumanEval
    Console.WriteLine("\nEvaluating HumanEval...");
    var humaneval = new HumanEvalBenchmark<double>();
    var humanevalResult = await humaneval.EvaluateAsync(SolveAsync, sampleSize);
    results["HumanEval"] = Convert.ToDouble(humanevalResult.Accuracy);
    Console.WriteLine($"HumanEval Accuracy: {humanevalResult.Accuracy:P2}");

    // MMLU
    Console.WriteLine("\nEvaluating MMLU...");
    var mmlu = new MMLUBenchmark<double>();
    var mmluResult = await mmlu.EvaluateAsync(SolveAsync, sampleSize);
    results["MMLU"] = Convert.ToDouble(mmluResult.Accuracy);
    Console.WriteLine($"MMLU Accuracy: {mmluResult.Accuracy:P2}");

    return results;
}

private async Task<string> SolveAsync(string problem)
{
    var result = await _strategy.ReasonAsync(problem);
    return result.FinalAnswer;
}
```

### Step 3: Compare Results
```csharp
public async Task CompareStrategiesAsync()
{
    var strategies = new Dictionary<string, IReasoningStrategy<double>>
    {
        ["Chain-of-Thought"] = new ChainOfThoughtStrategy<double>(_chatModel),
        ["Self-Consistency"] = new SelfConsistencyStrategy<double>(_chatModel),
        ["Tree-of-Thoughts"] = new TreeOfThoughtsStrategy<double>(_chatModel)
    };

    var benchmark = new GSM8KBenchmark<double>();
    var results = new Dictionary<string, double>();

    foreach (var (name, strategy) in strategies)
    {
        Console.WriteLine($"\nTesting {name}...");

        var result = await benchmark.EvaluateAsync(
            async (problem) => {
                var r = await strategy.ReasonAsync(problem);
                return r.FinalAnswer;
            },
            sampleSize: 50
        );

        results[name] = Convert.ToDouble(result.Accuracy);
        Console.WriteLine($"{name}: {result.Accuracy:P2}");
    }

    // Show best strategy
    var best = results.OrderByDescending(kvp => kvp.Value).First();
    Console.WriteLine($"\nBest Strategy: {best.Key} with {best.Value:P2} accuracy");
}
```

---

## Next Steps

- **Advanced Topics**: See [Advanced Guide](./AdvancedGuide.md)
- **Best Practices**: See [Best Practices](./BestPractices.md)
- **API Reference**: See [API Documentation](./ApiReference.md)
- **Examples**: Browse `examples/` directory for more code samples

## Common Issues & Solutions

### Issue: Training takes too long
**Solution**: Reduce batch size or use fewer samples
```csharp
var config = new RLConfig
{
    BatchSize = 16,  // Smaller batches
    Epochs = 5       // Fewer epochs
};
```

### Issue: Low benchmark accuracy
**Solution**: Try different strategies or increase exploration
```csharp
var config = new ReasoningConfig
{
    ExplorationDepth = 5,
    BranchingFactor = 4,
    NumSamples = 10  // For Self-Consistency
};
```

### Issue: Out of memory errors
**Solution**: Use BeamSearch instead of BFS for memory efficiency
```csharp
var beamSearch = new BeamSearch<double>(beamWidth: 3);
// Use in TreeOfThoughtsStrategy
```

Happy building! 🎉
