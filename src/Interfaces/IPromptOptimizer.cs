namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for optimizing prompts to improve language model performance.
/// </summary>
/// <typeparam name="T">The type of numeric data used for scoring.</typeparam>
/// <remarks>
/// <para>
/// A prompt optimizer automatically refines prompts to achieve better performance on a specific task.
/// Optimization strategies include discrete search, gradient-based methods, ensemble approaches,
/// and evolutionary algorithms.
/// </para>
/// <para><b>For Beginners:</b> A prompt optimizer automatically improves your prompts.
///
/// Think of it like automatic recipe refinement:
/// - You start with a basic recipe
/// - The optimizer tries variations (more salt, less sugar, different temperature)
/// - It measures which variations taste better
/// - It keeps refining until it finds the best version
///
/// For prompts:
/// - You provide a basic prompt
/// - Optimizer generates variations
/// - Tests each variation's performance
/// - Returns the best-performing prompt
///
/// Example:
/// Initial prompt: "Classify this review"
/// After optimization: "Carefully analyze the sentiment and tone of the following product review.
///                     Classify it as positive, negative, or neutral based on the overall customer satisfaction."
/// Result: 15% accuracy improvement
///
/// Benefits:
/// - Better results without manual trial-and-error
/// - Discover optimal phrasings you wouldn't think of
/// - Systematic improvement process
/// - Measurable performance gains
/// </para>
/// </remarks>
public interface IPromptOptimizer<T>
{
    /// <summary>
    /// Optimizes a prompt for a specific task using the provided evaluation function.
    /// </summary>
    /// <param name="initialPrompt">The starting prompt to optimize.</param>
    /// <param name="evaluationFunction">Function that scores prompt performance (higher is better).</param>
    /// <param name="maxIterations">Maximum number of optimization iterations.</param>
    /// <returns>The optimized prompt template.</returns>
    /// <remarks>
    /// <para>
    /// Iteratively refines the initial prompt by generating variations, evaluating their performance
    /// using the provided function, and selecting better-performing candidates. The process continues
    /// until maxIterations is reached or convergence is achieved.
    /// </para>
    /// <para><b>For Beginners:</b> This automatically improves your prompt.
    ///
    /// How it works:
    /// 1. Start with your initial prompt
    /// 2. Generate variations (different wordings, structures, examples)
    /// 3. Test each variation using your evaluation function
    /// 4. Keep the best-performing versions
    /// 5. Generate new variations based on what worked
    /// 6. Repeat until you hit maxIterations or performance stops improving
    ///
    /// Example - Sentiment classification:
    /// ```csharp
    /// var optimizer = new DiscreteSearchOptimizer();
    ///
    /// // Your evaluation function tests accuracy
    /// Func<string, double> evaluate = (prompt) =>
    /// {
    ///     double correctCount = 0;
    ///     foreach (var testCase in testSet)
    ///     {
    ///         var result = model.Generate(prompt + testCase.Input);
    ///         if (result == testCase.ExpectedOutput)
    ///             correctCount++;
    ///     }
    ///     return correctCount / testSet.Count; // Returns accuracy 0.0 to 1.0
    /// };
    ///
    /// var optimized = optimizer.Optimize(
    ///     initialPrompt: "Classify sentiment:",
    ///     evaluationFunction: evaluate,
    ///     maxIterations: 50
    /// );
    ///
    /// // optimized might be: "Analyze the sentiment and emotional tone..."
    /// // with 20% better accuracy
    /// ```
    ///
    /// Parameters:
    /// - initialPrompt: Your starting point
    /// - evaluationFunction: How to measure "better" (returns a score)
    /// - maxIterations: How long to search (more = potentially better, but slower)
    ///
    /// Returns:
    /// - The best prompt found during optimization
    /// </para>
    /// </remarks>
    IPromptTemplate Optimize(
        string initialPrompt,
        Func<string, T> evaluationFunction,
        int maxIterations = 100);

    /// <summary>
    /// Optimizes a prompt asynchronously.
    /// </summary>
    /// <param name="initialPrompt">The starting prompt to optimize.</param>
    /// <param name="evaluationFunction">Async function that scores prompt performance (higher is better).</param>
    /// <param name="maxIterations">Maximum number of optimization iterations.</param>
    /// <param name="cancellationToken">Token to cancel the optimization process.</param>
    /// <returns>A task representing the asynchronous operation with the optimized prompt.</returns>
    /// <remarks>
    /// <para>
    /// Asynchronous version of Optimize() that doesn't block the calling thread. Essential for
    /// optimization involving API calls or long-running evaluations.
    /// </para>
    /// <para><b>For Beginners:</b> Same as Optimize(), but doesn't freeze your program.
    ///
    /// Optimization can take a while (testing many variations, calling APIs, etc.).
    /// OptimizeAsync lets your program keep working during optimization.
    ///
    /// Example:
    /// ```csharp
    /// // Evaluation function that calls an API
    /// async Task<double> evaluateAsync(string prompt)
    /// {
    ///     var results = await TestWithAPIAsync(prompt);
    ///     return CalculateAccuracy(results);
    /// }
    ///
    /// // Start optimization (doesn't block)
    /// var optimizationTask = optimizer.OptimizeAsync(
    ///     "Classify sentiment:",
    ///     evaluateAsync,
    ///     maxIterations: 50
    /// );
    ///
    /// // Do other work
    /// UpdateUI("Optimizing prompt...");
    ///
    /// // Wait for result
    /// var optimized = await optimizationTask;
    /// ```
    ///
    /// Benefits:
    /// - Doesn't freeze UI
    /// - Can cancel if needed
    /// - Handles API calls efficiently
    /// - Better resource utilization
    /// </para>
    /// </remarks>
    Task<IPromptTemplate> OptimizeAsync(
        string initialPrompt,
        Func<string, Task<T>> evaluationFunction,
        int maxIterations = 100,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Gets the optimization history showing performance over iterations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Returns a list of prompts and their scores from the optimization process,
    /// allowing analysis of how optimization progressed.
    /// </para>
    /// <para><b>For Beginners:</b> Shows how the prompt improved during optimization.
    ///
    /// Like a training log showing progress:
    /// - Iteration 1: "Classify sentiment" → Score: 0.65
    /// - Iteration 5: "Classify the sentiment of" → Score: 0.72
    /// - Iteration 10: "Analyze sentiment and classify as" → Score: 0.78
    /// - Iteration 20: "Carefully analyze the sentiment..." → Score: 0.85
    ///
    /// Use this to:
    /// - Visualize improvement over time
    /// - Understand what changes helped
    /// - Debug optimization issues
    /// - Decide if more iterations would help
    ///
    /// Example:
    /// ```csharp
    /// var history = optimizer.GetOptimizationHistory();
    ///
    /// foreach (var entry in history)
    /// {
    ///     Console.WriteLine($"Iteration {entry.Iteration}: Score {entry.Score}");
    ///     Console.WriteLine($"Prompt: {entry.Prompt}");
    /// }
    ///
    /// // Plot improvement curve
    /// PlotScores(history.Select(h => h.Score));
    /// ```
    /// </para>
    /// </remarks>
    IReadOnlyList<OptimizationHistoryEntry<T>> GetOptimizationHistory();
}

/// <summary>
/// Represents a single entry in the optimization history.
/// </summary>
/// <typeparam name="T">The type of the performance score.</typeparam>
public class OptimizationHistoryEntry<T>
{
    /// <summary>
    /// Gets or sets the iteration number.
    /// </summary>
    public int Iteration { get; set; }

    /// <summary>
    /// Gets or sets the prompt tested in this iteration.
    /// </summary>
    public string Prompt { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the performance score achieved.
    /// </summary>
    public T Score { get; set; } = default!;

    /// <summary>
    /// Gets or sets the timestamp of this iteration.
    /// </summary>
    public DateTime Timestamp { get; set; }

    /// <summary>
    /// Gets or sets optional metadata about this iteration.
    /// </summary>
    public Dictionary<string, string> Metadata { get; set; } = new();
}
