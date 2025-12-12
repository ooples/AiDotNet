# Best Practices for AiDotNet Reasoning Framework

Essential patterns, tips, and best practices for building production-ready reasoning systems.

## Table of Contents
1. [Strategy Selection](#strategy-selection)
2. [Configuration Tuning](#configuration-tuning)
3. [Performance Optimization](#performance-optimization)
4. [Error Handling](#error-handling)
5. [Testing & Validation](#testing--validation)
6. [Production Deployment](#production-deployment)
7. [Common Pitfalls](#common-pitfalls)

---

## Strategy Selection

### When to Use Each Strategy

#### **Chain-of-Thought (CoT)**
✅ **Use when:**
- Problem has a clear, linear solution path
- Speed is important
- Single correct answer expected
- Simple to moderate complexity

❌ **Avoid when:**
- Multiple valid approaches exist
- Problem is highly ambiguous
- Need exploration of alternatives

**Example:**
```csharp
// Good: Straightforward calculation
await cotStrategy.ReasonAsync("What is 347 + 892?");

// Not ideal: Requires exploration
await cotStrategy.ReasonAsync("Find all possible solutions to this riddle");
```

#### **Self-Consistency**
✅ **Use when:**
- Multiple reasoning paths are valid
- Want to reduce variance
- Need confidence estimation
- Accuracy > speed

❌ **Avoid when:**
- Only one correct path exists
- Tight latency requirements
- Resource constraints

**Example:**
```csharp
// Good: Benefits from multiple perspectives
var config = new ReasoningConfig { NumSamples = 5 };
await scStrategy.ReasonAsync(
    "What are the possible causes of global warming?",
    config
);

// Overkill: Simple arithmetic
await scStrategy.ReasonAsync("What is 2 + 2?", config);  // Don't do this!
```

#### **Tree-of-Thoughts (ToT)**
✅ **Use when:**
- Complex problem requiring exploration
- Multiple decision points
- Backtracking may be needed
- Quality > speed

❌ **Avoid when:**
- Simple problems
- Strict latency requirements
- Limited compute budget

**Example:**
```csharp
// Good: Complex logic puzzle
var config = new ReasoningConfig
{
    ExplorationDepth = 5,
    BranchingFactor = 3
};
await totStrategy.ReasonAsync("Solve this sudoku puzzle: ...", config);

// Wasteful: Simple question
await totStrategy.ReasonAsync("What is the capital of France?", config);
```

### Strategy Combination Pattern
```csharp
// Adaptive strategy selection
public async Task<ReasoningResult<T>> SolveAdaptiveAsync(string problem)
{
    var difficulty = EstimateDifficulty(problem);

    return difficulty switch
    {
        < 0.3 => await _cotStrategy.ReasonAsync(problem),
        < 0.7 => await _scStrategy.ReasonAsync(problem, NumSamples: 3),
        _ => await _totStrategy.ReasonAsync(problem, ExplorationDepth: 5)
    };
}
```

---

## Configuration Tuning

### Start with Presets
```csharp
// Fast: 3 steps, depth 2, 1 sample
var fastConfig = ReasoningConfig.Fast;  // ~200ms

// Default: 10 steps, depth 3, 3 samples
var defaultConfig = ReasoningConfig.Default;  // ~2s

// Thorough: 20 steps, depth 5, 5 samples
var thoroughConfig = ReasoningConfig.Thorough;  // ~10s
```

### Progressive Refinement
```csharp
// Start fast, refine if needed
var result = await strategy.ReasonAsync(problem, ReasoningConfig.Fast);

if (result.ConfidenceScore < 0.8)
{
    // Try again with more compute
    result = await strategy.ReasonAsync(problem, ReasoningConfig.Default);
}

if (result.ConfidenceScore < 0.9)
{
    // Last resort: thorough analysis
    result = await strategy.ReasonAsync(problem, ReasoningConfig.Thorough);
}
```

### Custom Configuration Guidelines

#### MaxSteps
```csharp
// Too few: May not complete reasoning
MaxSteps = 3  // Only for very simple problems

// Recommended ranges:
MaxSteps = 5-10   // Most problems
MaxSteps = 10-20  // Complex problems
MaxSteps = 20-30  // Very complex problems

// Too many: Diminishing returns + cost
MaxSteps = 50  // Usually unnecessary
```

#### ExplorationDepth (ToT only)
```csharp
// Shallow: May miss optimal solution
ExplorationDepth = 2  // Quick but limited

// Recommended:
ExplorationDepth = 3-4  // Good balance
ExplorationDepth = 5-7  // Complex problems

// Deep: Exponential cost
ExplorationDepth = 10  // 3^10 = 59K nodes!
```

#### NumSamples (Self-Consistency only)
```csharp
// Minimum: No diversity
NumSamples = 1  // Just use CoT instead

// Recommended:
NumSamples = 3-5   // Good variance reduction
NumSamples = 5-10  // Important decisions

// Diminishing returns:
NumSamples = 20  // Rarely worth the cost
```

### Temperature Tuning
```csharp
// Creative/exploratory tasks
config.Temperature = 0.8  // More diverse outputs

// Factual/deterministic tasks
config.Temperature = 0.2  // More focused outputs

// Default (balanced)
config.Temperature = 0.7
```

---

## Performance Optimization

### 1. Caching Strategies
```csharp
public class CachedReasoner<T>
{
    private readonly IReasoningStrategy<T> _strategy;
    private readonly IMemoryCache _cache;

    public async Task<ReasoningResult<T>> ReasonAsync(string query)
    {
        // Check cache first
        if (_cache.TryGetValue(query, out ReasoningResult<T> cached))
        {
            return cached;
        }

        // Compute and cache
        var result = await _strategy.ReasonAsync(query);
        _cache.Set(query, result, TimeSpan.FromHours(1));
        return result;
    }
}
```

### 2. Parallel Processing
```csharp
// Good: Independent problems
var problems = LoadProblems();
var results = await Task.WhenAll(
    problems.Select(p => strategy.ReasonAsync(p))
);

// Bad: Dependencies between problems
var result1 = await strategy.ReasonAsync(problem1);
var problem2 = BuildUpon(result1);  // Depends on result1
var result2 = await strategy.ReasonAsync(problem2);
```

### 3. Batch Optimization
```csharp
// Inefficient: One at a time
foreach (var problem in problems)
{
    await strategy.ReasonAsync(problem);
}

// Better: Batched processing
const int batchSize = 10;
for (int i = 0; i < problems.Count; i += batchSize)
{
    var batch = problems.Skip(i).Take(batchSize);
    await Task.WhenAll(batch.Select(p => strategy.ReasonAsync(p)));
}
```

### 4. Early Stopping
```csharp
public async Task<ReasoningResult<T>> ReasonWithEarlyStopAsync(
    string problem,
    double confidenceThreshold = 0.95)
{
    var config = ReasoningConfig.Fast;

    while (true)
    {
        var result = await strategy.ReasonAsync(problem, config);

        if (result.ConfidenceScore >= confidenceThreshold)
        {
            return result;  // Good enough!
        }

        if (config.MaxSteps >= 20)
        {
            return result;  // Stop at max compute
        }

        // Increase compute
        config.MaxSteps = Math.Min(config.MaxSteps * 2, 20);
    }
}
```

---

## Error Handling

### Robust Error Handling Pattern
```csharp
public async Task<ReasoningResult<T>> SafeReasonAsync(
    string problem,
    int maxRetries = 3)
{
    for (int attempt = 0; attempt < maxRetries; attempt++)
    {
        try
        {
            var result = await strategy.ReasonAsync(problem);

            if (result.Success)
            {
                return result;
            }

            // Log failure
            _logger.LogWarning($"Reasoning failed: {result.ErrorMessage}");
        }
        catch (OperationCanceledException)
        {
            throw;  // Don't retry cancellations
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Reasoning attempt {attempt + 1} failed");

            if (attempt == maxRetries - 1)
            {
                throw;
            }

            // Exponential backoff
            await Task.Delay(TimeSpan.FromSeconds(Math.Pow(2, attempt)));
        }
    }

    throw new Exception("All reasoning attempts failed");
}
```

### Timeout Protection
```csharp
public async Task<ReasoningResult<T>> ReasonWithTimeoutAsync(
    string problem,
    TimeSpan timeout)
{
    using var cts = new CancellationTokenSource(timeout);

    try
    {
        return await strategy.ReasonAsync(problem, cancellationToken: cts.Token);
    }
    catch (OperationCanceledException)
    {
        _logger.LogWarning($"Reasoning timed out after {timeout}");
        return new ReasoningResult<T>
        {
            Success = false,
            ErrorMessage = "Reasoning exceeded time limit"
        };
    }
}
```

### Graceful Degradation
```csharp
public async Task<string> GetAnswerAsync(string question)
{
    // Try sophisticated reasoning first
    try
    {
        var result = await totStrategy.ReasonAsync(question, thorough);
        if (result.Success) return result.FinalAnswer;
    }
    catch (Exception ex)
    {
        _logger.LogWarning(ex, "ToT failed, falling back to SC");
    }

    // Fall back to Self-Consistency
    try
    {
        var result = await scStrategy.ReasonAsync(question, standard);
        if (result.Success) return result.FinalAnswer;
    }
    catch (Exception ex)
    {
        _logger.LogWarning(ex, "SC failed, falling back to CoT");
    }

    // Last resort: Simple CoT
    var finalResult = await cotStrategy.ReasonAsync(question, fast);
    return finalResult.FinalAnswer ?? "Unable to determine answer";
}
```

---

## Testing & Validation

### Unit Testing Best Practices
```csharp
[Fact]
public async Task Reasoning_WithValidInput_Succeeds()
{
    // Arrange
    var mockModel = new Mock<IChatModel>();
    mockModel.Setup(m => m.GenerateResponseAsync(It.IsAny<string>(), It.IsAny<CancellationToken>()))
             .ReturnsAsync("Step 1: Calculate\nAnswer: 42");

    var strategy = new ChainOfThoughtStrategy<double>(mockModel.Object);

    // Act
    var result = await strategy.ReasonAsync("What is 6 × 7?");

    // Assert
    Assert.True(result.Success);
    Assert.Contains("42", result.FinalAnswer);
    mockModel.Verify(m => m.GenerateResponseAsync(
        It.IsAny<string>(),
        It.IsAny<CancellationToken>()), Times.Once);
}
```

### Integration Testing
```csharp
[Fact]
public async Task EndToEnd_MathProblem_WithVerification()
{
    // Use real components
    var chatModel = GetTestChatModel();
    var reasoner = new MathematicalReasoner<double>(chatModel);
    var verifier = new CalculatorVerifier<double>();

    // Solve and verify
    var result = await reasoner.SolveAsync("What is 15 × 12?");
    var verification = await verifier.VerifyAsync(result.Chain);

    // Assert
    Assert.True(result.Success);
    Assert.True(verification.IsValid);
    Assert.Equal("180", result.FinalAnswer);
}
```

### Benchmark Testing
```csharp
[Theory]
[InlineData(0.70, "CoT should achieve 70% on GSM8K")]
[InlineData(0.80, "With verification should reach 80%")]
public async Task BenchmarkAccuracy_MeetsThreshold(
    double threshold,
    string reason)
{
    var benchmark = new GSM8KBenchmark<double>();
    var result = await benchmark.EvaluateAsync(SolveAsync, sampleSize: 100);

    Assert.True(
        Convert.ToDouble(result.Accuracy) >= threshold,
        $"Failed: {reason}. Got {result.Accuracy:P2}"
    );
}
```

---

## Production Deployment

### Monitoring & Logging
```csharp
public class MonitoredReasoner<T>
{
    private readonly IReasoningStrategy<T> _strategy;
    private readonly ILogger _logger;
    private readonly IMetrics _metrics;

    public async Task<ReasoningResult<T>> ReasonAsync(string problem)
    {
        var stopwatch = Stopwatch.StartNew();

        try
        {
            _logger.LogInformation($"Starting reasoning: {problem.Substring(0, 50)}...");

            var result = await _strategy.ReasonAsync(problem);

            _metrics.RecordLatency("reasoning.duration", stopwatch.ElapsedMilliseconds);
            _metrics.RecordCounter("reasoning.success", result.Success ? 1 : 0);
            _metrics.RecordGauge("reasoning.confidence", result.ConfidenceScore);
            _metrics.RecordGauge("reasoning.steps", result.Chain?.Steps.Count ?? 0);

            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Reasoning failed");
            _metrics.RecordCounter("reasoning.errors", 1);
            throw;
        }
    }
}
```

### Resource Management
```csharp
public class ResourceAwareReasoner<T>
{
    private readonly SemaphoreSlim _semaphore;
    private readonly IReasoningStrategy<T> _strategy;

    public ResourceAwareReasoner(IReasoningStrategy<T> strategy, int maxConcurrent = 5)
    {
        _strategy = strategy;
        _semaphore = new SemaphoreSlim(maxConcurrent);
    }

    public async Task<ReasoningResult<T>> ReasonAsync(string problem)
    {
        await _semaphore.WaitAsync();
        try
        {
            return await _strategy.ReasonAsync(problem);
        }
        finally
        {
            _semaphore.Release();
        }
    }
}
```

### Rate Limiting
```csharp
public class RateLimitedReasoner<T>
{
    private readonly IReasoningStrategy<T> _strategy;
    private readonly RateLimiter _rateLimiter;

    public async Task<ReasoningResult<T>> ReasonAsync(string problem)
    {
        using var lease = await _rateLimiter.AcquireAsync();

        if (!lease.IsAcquired)
        {
            throw new RateLimitExceededException("Too many requests");
        }

        return await _strategy.ReasonAsync(problem);
    }
}
```

---

## Common Pitfalls

### ❌ Pitfall 1: Using Wrong Strategy
```csharp
// BAD: Using ToT for simple arithmetic
var result = await totStrategy.ReasonAsync(
    "What is 2 + 2?",
    new ReasoningConfig { ExplorationDepth = 5 }
);
// Cost: 10x more expensive, same result

// GOOD: Use appropriate strategy
var result = await cotStrategy.ReasonAsync("What is 2 + 2?");
```

### ❌ Pitfall 2: Not Using Verification
```csharp
// BAD: Trust blindly
var result = await reasoner.SolveAsync("Calculate 15 × 24");
return result.FinalAnswer;  // Might be wrong!

// GOOD: Verify calculations
var result = await reasoner.SolveAsync("Calculate 15 × 24");
var verification = await calculator.VerifyAsync(result.Chain);

if (!verification.IsValid)
{
    // Retry or refine
    result = await RefineAsync(result);
}
```

### ❌ Pitfall 3: Ignoring Confidence Scores
```csharp
// BAD: Ignore confidence
var result = await strategy.ReasonAsync(problem);
return result.FinalAnswer;  // Even if confidence is 0.2!

// GOOD: Check confidence
var result = await strategy.ReasonAsync(problem);

if (result.ConfidenceScore < 0.7)
{
    // Try with more compute or ask for clarification
    return await HandleLowConfidenceAsync(result, problem);
}
```

### ❌ Pitfall 4: No Timeout Protection
```csharp
// BAD: Can hang indefinitely
var result = await strategy.ReasonAsync(complexProblem);

// GOOD: Always use timeouts
using var cts = new CancellationTokenSource(TimeSpan.FromMinutes(5));
var result = await strategy.ReasonAsync(complexProblem, cancellationToken: cts.Token);
```

### ❌ Pitfall 5: Overusing Self-Consistency
```csharp
// BAD: Excessive sampling
var config = new ReasoningConfig { NumSamples = 20 };  // Overkill!
var result = await scStrategy.ReasonAsync("Simple question", config);

// GOOD: Appropriate sampling
var config = new ReasoningConfig { NumSamples = 3-5 };  // Enough
var result = await scStrategy.ReasonAsync("Complex question", config);
```

---

## Performance Checklist

Before deploying to production:

- ✅ Chosen appropriate strategy for problem types
- ✅ Tuned configuration parameters
- ✅ Added verification for critical tasks
- ✅ Implemented error handling and retries
- ✅ Set up monitoring and logging
- ✅ Added timeout protection
- ✅ Implemented caching where appropriate
- ✅ Rate limiting in place
- ✅ Resource limits configured
- ✅ Tested on representative benchmark
- ✅ Measured and optimized latency
- ✅ Graceful degradation implemented

---

## Summary

**Key Takeaways:**
1. **Choose the right tool**: CoT for speed, SC for accuracy, ToT for complexity
2. **Start simple**: Use presets, scale up only if needed
3. **Verify important answers**: Don't trust blindly
4. **Handle errors gracefully**: Timeouts, retries, fallbacks
5. **Monitor in production**: Latency, success rate, confidence
6. **Optimize iteratively**: Measure, identify bottlenecks, improve

For more information:
- [Getting Started Guide](./GettingStarted.md)
- [Tutorials](./Tutorials.md)
- [API Reference](./ApiReference.md)

Happy reasoning! 🚀
