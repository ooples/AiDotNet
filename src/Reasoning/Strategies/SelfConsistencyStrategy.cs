using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Reasoning.Models;

namespace AiDotNet.Reasoning.Strategies;

/// <summary>
/// Implements Self-Consistency reasoning by sampling multiple reasoning paths and using majority voting.
/// </summary>
/// <typeparam name="T">The numeric type used for scoring (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Self-Consistency is like solving a problem multiple times independently
/// and then comparing answers. If you get the same answer most of the time, you can be confident it's correct.
///
/// **How it works:**
/// 1. Generate multiple independent reasoning chains for the same problem
/// 2. Each chain uses slightly different wording/approach (controlled by temperature)
/// 3. Extract the final answer from each chain
/// 4. Use majority voting to pick the most common answer
///
/// **Example:**
/// Problem: "What is 15% of 240?"
///
/// Attempt 1: 15% = 0.15, then 0.15 × 240 = 36 ✓
/// Attempt 2: 240 ÷ 100 × 15 = 36 ✓
/// Attempt 3: 240 × 15/100 = 36 ✓
/// Attempt 4: Convert to fraction: 240 × 3/20 = 36 ✓
/// Attempt 5: (Error) 240 × 1.5 = 360 ✗
///
/// Majority vote: "36" appears 4 times, "360" appears 1 time → Answer: "36"
///
/// **Why it works:**
/// - Random errors don't repeat consistently
/// - Correct reasoning tends to reach the same answer
/// - Filters out hallucinations and calculation mistakes
///
/// **Research basis:**
/// "Self-Consistency Improves Chain of Thought Reasoning in Language Models" (Wang et al., 2022)
/// showed significant improvements over standard CoT, especially on reasoning benchmarks.
///
/// **When to use:**
/// - Important decisions where accuracy matters
/// - Mathematical or logical reasoning
/// - When you can afford multiple LLM calls
/// - Problems where errors are common
/// </para>
/// <para><b>Example Usage:</b>
/// <code>
/// var chatModel = new OpenAIChatModel&lt;double&gt;("gpt-4");
/// var strategy = new SelfConsistencyStrategy&lt;double&gt;(chatModel);
///
/// var config = new ReasoningConfig
/// {
///     NumSamples = 10,  // Try 10 different reasoning paths
///     Temperature = 0.7  // Moderate diversity in approaches
/// };
///
/// var result = await strategy.ReasonAsync(
///     "If a train travels 60 mph for 2.5 hours, how far does it go?",
///     config
/// );
///
/// // Result includes all alternative chains for transparency
/// Console.WriteLine($"Final answer: {result.FinalAnswer}");
/// Console.WriteLine($"Based on {result.AlternativeChains.Count} samples");
/// </code>
/// </para>
/// </remarks>
public class SelfConsistencyStrategy<T> : ReasoningStrategyBase<T>
{
    private readonly ChainOfThoughtStrategy<T> _cotStrategy;
    private readonly IAnswerAggregator<T> _aggregator;

    /// <summary>
    /// Initializes a new instance of the <see cref="SelfConsistencyStrategy{T}"/> class.
    /// </summary>
    /// <param name="chatModel">The chat model used for reasoning.</param>
    /// <param name="tools">Optional tools available during reasoning.</param>
    /// <param name="aggregator">The aggregation method (defaults to majority voting).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a Self-Consistency strategy that uses Chain-of-Thought
    /// reasoning multiple times and aggregates the results. You can choose how to aggregate:
    /// - Majority voting (default): Most common answer wins
    /// - Weighted voting: Higher confidence answers count more
    /// </para>
    /// </remarks>
    public SelfConsistencyStrategy(
        IChatModel<T> chatModel,
        IEnumerable<ITool>? tools = null,
        IAnswerAggregator<T>? aggregator = null)
        : base(chatModel, tools)
    {
        _cotStrategy = new ChainOfThoughtStrategy<T>(chatModel, tools);
        _aggregator = aggregator ?? new Aggregation.MajorityVotingAggregator<T>();
    }

    /// <inheritdoc/>
    public override string StrategyName => "Self-Consistency";

    /// <inheritdoc/>
    public override string Description =>
        "Samples multiple diverse reasoning paths and aggregates answers via majority voting. " +
        "More reliable than single-path reasoning, especially for problems prone to errors. " +
        "Based on 'Self-Consistency Improves Chain of Thought Reasoning' (Wang et al., 2022).";

    /// <inheritdoc/>
    protected override async Task<ReasoningResult<T>> ReasonCoreAsync(
        string query,
        ReasoningConfig config,
        CancellationToken cancellationToken)
    {
        ValidateConfig(config);

        if (config.NumSamples < 1)
        {
            throw new ArgumentException("NumSamples must be at least 1 for Self-Consistency", nameof(config));
        }

        var result = new ReasoningResult<T>
        {
            StrategyUsed = StrategyName
        };

        AppendTrace($"Starting Self-Consistency reasoning with {config.NumSamples} samples");
        AppendTrace($"Aggregation method: {_aggregator.MethodName}");
        AppendTrace($"Temperature: {config.Temperature}");

        // Step 1: Generate multiple reasoning chains
        var chains = new List<ReasoningChain<T>>();
        var answers = new List<string>();
        var confidenceScores = new List<T>();

        var startTime = DateTime.UtcNow;

        // Generate samples in parallel for efficiency (up to 5 at a time to avoid rate limits)
        var tasks = new List<Task<ReasoningResult<T>>>();
        var semaphore = new SemaphoreSlim(5); // Limit concurrent requests

        for (int i = 0; i < config.NumSamples; i++)
        {
            await semaphore.WaitAsync(cancellationToken);

            int sampleNum = i + 1;
            var task = Task.Run(async () =>
            {
                try
                {
                    AppendTrace($"Generating sample {sampleNum}/{config.NumSamples}...");
                    return await _cotStrategy.ReasonAsync(query, config, cancellationToken);
                }
                finally
                {
                    semaphore.Release();
                }
            }, cancellationToken);

            tasks.Add(task);
        }

        // Wait for all samples to complete
        var sampleResults = await Task.WhenAll(tasks);

        // Step 2: Collect answers and scores from all samples
        for (int i = 0; i < sampleResults.Length; i++)
        {
            var sampleResult = sampleResults[i];

            if (sampleResult.Success && !string.IsNullOrWhiteSpace(sampleResult.FinalAnswer))
            {
                chains.Add(sampleResult.ReasoningChain);
                answers.Add(sampleResult.FinalAnswer);
                confidenceScores.Add(sampleResult.OverallConfidence);

                AppendTrace($"Sample {i + 1}: Answer = '{sampleResult.FinalAnswer}', Confidence = {sampleResult.OverallConfidence}");
            }
            else
            {
                AppendTrace($"Sample {i + 1}: FAILED - {sampleResult.ErrorMessage}");
            }
        }

        if (answers.Count == 0)
        {
            result.Success = false;
            result.ErrorMessage = "All reasoning samples failed";
            AppendTrace($"ERROR: {result.ErrorMessage}");
            return result;
        }

        // Step 3: Aggregate answers using the chosen method
        AppendTrace($"\nAggregating {answers.Count} answers using {_aggregator.MethodName}...");

        var confidenceVector = new Vector<T>(confidenceScores);
        string finalAnswer = _aggregator.Aggregate(answers, confidenceVector);

        AppendTrace($"Final answer selected: '{finalAnswer}'");

        // Step 4: Calculate consensus metrics
        int consensusCount = answers.Count(a =>
            a.Equals(finalAnswer, StringComparison.OrdinalIgnoreCase));
        double consensusRatio = (double)consensusCount / answers.Count;

        AppendTrace($"Consensus: {consensusCount}/{answers.Count} samples ({consensusRatio:P1}) agreed");

        // Step 5: Select the best reasoning chain that matches the final answer
        var bestChain = chains.FirstOrDefault(c =>
            c.FinalAnswer.Equals(finalAnswer, StringComparison.OrdinalIgnoreCase))
            ?? chains.First();

        // Step 6: Calculate overall confidence
        // Use average confidence of samples that agreed with the final answer
        var numOps = MathHelper.GetNumericOperations<T>();
        var agreedConfidences = new List<T>();
        for (int i = 0; i < answers.Count; i++)
        {
            if (answers[i].Equals(finalAnswer, StringComparison.OrdinalIgnoreCase))
            {
                agreedConfidences.Add(confidenceScores[i]);
            }
        }

        T overallConfidence;
        if (agreedConfidences.Count > 0)
        {
            var agreedVector = new Vector<T>(agreedConfidences);
            overallConfidence = agreedVector.Mean();
        }
        else
        {
            overallConfidence = numOps.FromDouble(0.5); // Fallback
        }

        // Boost confidence based on consensus ratio
        double confidenceMultiplier = 0.5 + (consensusRatio * 0.5); // Range: 0.5 to 1.0
        overallConfidence = numOps.Multiply(overallConfidence, numOps.FromDouble(confidenceMultiplier));

        // Step 7: Build final result
        result.FinalAnswer = finalAnswer;
        result.ReasoningChain = bestChain;
        result.AlternativeChains = chains.Where(c => c != bestChain).ToList();
        result.OverallConfidence = overallConfidence;
        result.ConfidenceScores = confidenceVector;
        result.Success = true;

        // Metrics
        result.Metrics["num_samples"] = config.NumSamples;
        result.Metrics["successful_samples"] = answers.Count;
        result.Metrics["consensus_count"] = consensusCount;
        result.Metrics["consensus_ratio"] = consensusRatio;
        result.Metrics["aggregation_method"] = _aggregator.MethodName;
        result.Metrics["total_steps"] = chains.Sum(c => c.Steps.Count);
        result.Metrics["avg_steps_per_sample"] = chains.Average(c => c.Steps.Count);

        var totalDuration = DateTime.UtcNow - startTime;
        AppendTrace($"\nSelf-Consistency complete in {totalDuration.TotalSeconds:F2}s");
        AppendTrace($"Final answer: '{finalAnswer}' (confidence: {overallConfidence})");

        return result;
    }
}
