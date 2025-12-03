using AiDotNet.Interfaces;

namespace AiDotNet.PromptEngineering.Optimization;

/// <summary>
/// Base class for prompt optimizer implementations.
/// </summary>
/// <typeparam name="T">The type of numeric data used for scoring.</typeparam>
/// <remarks>
/// <para>
/// This base class provides common functionality for prompt optimizers including history tracking
/// and validation. Derived classes implement the specific optimization strategy.
/// </para>
/// <para><b>For Beginners:</b> This is the foundation for all prompt optimizers.
///
/// It handles:
/// - Tracking optimization history
/// - Validation
/// - Converting prompts to templates
///
/// Derived classes implement the optimization algorithm!
/// </para>
/// </remarks>
public abstract class PromptOptimizerBase<T> : IPromptOptimizer<T>
{
    /// <summary>
    /// History of optimization iterations.
    /// </summary>
    protected readonly List<OptimizationHistoryEntry<T>> History;

    /// <summary>
    /// Initializes a new instance of the PromptOptimizerBase class.
    /// </summary>
    protected PromptOptimizerBase()
    {
        History = new List<OptimizationHistoryEntry<T>>();
    }

    /// <summary>
    /// Optimizes a prompt for a specific task using the provided evaluation function.
    /// </summary>
    public IPromptTemplate Optimize(
        string initialPrompt,
        Func<string, T> evaluationFunction,
        int maxIterations = 100)
    {
        if (string.IsNullOrWhiteSpace(initialPrompt))
        {
            throw new ArgumentException("Initial prompt cannot be empty.", nameof(initialPrompt));
        }

        if (evaluationFunction == null)
        {
            throw new ArgumentNullException(nameof(evaluationFunction));
        }

        if (maxIterations <= 0)
        {
            throw new ArgumentException("Max iterations must be positive.", nameof(maxIterations));
        }

        History.Clear();
        return OptimizeCore(initialPrompt, evaluationFunction, maxIterations);
    }

    /// <summary>
    /// Optimizes a prompt asynchronously.
    /// </summary>
    public async Task<IPromptTemplate> OptimizeAsync(
        string initialPrompt,
        Func<string, Task<T>> evaluationFunction,
        int maxIterations = 100,
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(initialPrompt))
        {
            throw new ArgumentException("Initial prompt cannot be empty.", nameof(initialPrompt));
        }

        if (evaluationFunction == null)
        {
            throw new ArgumentNullException(nameof(evaluationFunction));
        }

        if (maxIterations <= 0)
        {
            throw new ArgumentException("Max iterations must be positive.", nameof(maxIterations));
        }

        History.Clear();
        return await OptimizeCoreAsync(initialPrompt, evaluationFunction, maxIterations, cancellationToken)
            .ConfigureAwait(false);
    }

    /// <summary>
    /// Gets the optimization history.
    /// </summary>
    public IReadOnlyList<OptimizationHistoryEntry<T>> GetOptimizationHistory()
    {
        return History.AsReadOnly();
    }

    /// <summary>
    /// Core optimization logic to be implemented by derived classes.
    /// </summary>
    protected abstract IPromptTemplate OptimizeCore(
        string initialPrompt,
        Func<string, T> evaluationFunction,
        int maxIterations);

    /// <summary>
    /// Core async optimization logic to be implemented by derived classes.
    /// </summary>
    protected abstract Task<IPromptTemplate> OptimizeCoreAsync(
        string initialPrompt,
        Func<string, Task<T>> evaluationFunction,
        int maxIterations,
        CancellationToken cancellationToken);

    /// <summary>
    /// Records an iteration in the history.
    /// </summary>
    protected void RecordIteration(int iteration, string prompt, T score)
    {
        History.Add(new OptimizationHistoryEntry<T>
        {
            Iteration = iteration,
            Prompt = prompt,
            Score = score,
            Timestamp = DateTime.UtcNow
        });
    }
}
