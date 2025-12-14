using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;

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
/// - Providing numeric operations for comparisons
///
/// Derived classes implement the optimization algorithm!
/// </para>
/// </remarks>
public abstract class PromptOptimizerBase<T> : IPromptOptimizer<T>
{
    /// <summary>
    /// Provides numeric operations for the specific type T.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property provides mathematical operations appropriate for the generic type T,
    /// allowing the algorithm to work consistently with different numeric types like
    /// float, double, or decimal.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This is a helper that knows how to do math (addition, multiplication, comparisons, etc.) with
    /// your specific number type, whether that's a regular double, a precise decimal value,
    /// or something else. It allows the optimizer to work with different types of numbers
    /// without changing its core logic.
    /// </para>
    /// </remarks>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// History of optimization iterations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Stores a record of each iteration during optimization, including the prompt tried,
    /// the score achieved, and when it happened.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This list keeps track of every prompt variation we've tried and how well it performed.
    /// You can use GetOptimizationHistory() to see this log after optimization completes.
    /// </para>
    /// </remarks>
    protected readonly List<OptimizationHistoryEntry<T>> History;

    /// <summary>
    /// Initializes a new instance of the PromptOptimizerBase class.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Creates a new optimizer with an empty optimization history.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This constructor sets up the basic infrastructure that all prompt optimizers need,
    /// like the history tracking list.
    /// </para>
    /// </remarks>
    protected PromptOptimizerBase()
    {
        History = new List<OptimizationHistoryEntry<T>>();
    }

    /// <summary>
    /// Optimizes a prompt for a specific task using the provided evaluation function.
    /// </summary>
    /// <param name="initialPrompt">The starting prompt to optimize.</param>
    /// <param name="evaluationFunction">A function that evaluates prompts and returns a score.</param>
    /// <param name="maxIterations">Maximum number of optimization iterations.</param>
    /// <returns>The best-performing prompt template found.</returns>
    /// <exception cref="ArgumentException">Thrown when initialPrompt is empty or maxIterations is not positive.</exception>
    /// <exception cref="ArgumentNullException">Thrown when evaluationFunction is null.</exception>
    /// <remarks>
    /// <para>
    /// This method validates inputs, clears any existing history, and delegates to OptimizeCore.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// Call this method to start optimizing your prompt. The optimizer will try different
    /// variations and use your evaluation function to score each one, returning the best.
    /// </para>
    /// </remarks>
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
    /// Optimizes a prompt asynchronously for a specific task using the provided evaluation function.
    /// </summary>
    /// <param name="initialPrompt">The starting prompt to optimize.</param>
    /// <param name="evaluationFunction">An async function that evaluates prompts and returns a score.</param>
    /// <param name="maxIterations">Maximum number of optimization iterations.</param>
    /// <param name="cancellationToken">Token to cancel the operation.</param>
    /// <returns>The best-performing prompt template found.</returns>
    /// <exception cref="ArgumentException">Thrown when initialPrompt is empty or maxIterations is not positive.</exception>
    /// <exception cref="ArgumentNullException">Thrown when evaluationFunction is null.</exception>
    /// <remarks>
    /// <para>
    /// This method validates inputs, clears any existing history, and delegates to OptimizeCoreAsync.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// Use this method when your evaluation function is asynchronous (e.g., calls an API).
    /// It works the same as Optimize but supports async operations and cancellation.
    /// </para>
    /// </remarks>
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
    /// Gets the optimization history from the most recent optimization run.
    /// </summary>
    /// <returns>A read-only list of history entries.</returns>
    /// <remarks>
    /// <para>
    /// Returns the history of all iterations from the most recent optimization,
    /// including scores and timestamps.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// Call this after optimization to see what prompts were tried and how they scored.
    /// Useful for understanding how the optimizer found the best prompt.
    /// </para>
    /// </remarks>
    public IReadOnlyList<OptimizationHistoryEntry<T>> GetOptimizationHistory()
    {
        return History.AsReadOnly();
    }

    /// <summary>
    /// Core optimization logic to be implemented by derived classes.
    /// </summary>
    /// <param name="initialPrompt">The starting prompt to optimize.</param>
    /// <param name="evaluationFunction">A function that evaluates prompts and returns a score.</param>
    /// <param name="maxIterations">Maximum number of optimization iterations.</param>
    /// <returns>The best-performing prompt template found.</returns>
    /// <remarks>
    /// <para>
    /// Derived classes implement this method with their specific optimization strategy.
    /// Use NumOps.GreaterThan for comparing scores.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// If you're creating a custom optimizer, this is where you put your optimization logic.
    /// Use NumOps for numeric comparisons (e.g., NumOps.GreaterThan(score1, score2)).
    /// </para>
    /// </remarks>
    protected abstract IPromptTemplate OptimizeCore(
        string initialPrompt,
        Func<string, T> evaluationFunction,
        int maxIterations);

    /// <summary>
    /// Core async optimization logic to be implemented by derived classes.
    /// </summary>
    /// <param name="initialPrompt">The starting prompt to optimize.</param>
    /// <param name="evaluationFunction">An async function that evaluates prompts and returns a score.</param>
    /// <param name="maxIterations">Maximum number of optimization iterations.</param>
    /// <param name="cancellationToken">Token to cancel the operation.</param>
    /// <returns>The best-performing prompt template found.</returns>
    /// <remarks>
    /// <para>
    /// Derived classes implement this method with their specific async optimization strategy.
    /// Use NumOps.GreaterThan for comparing scores.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// If you're creating a custom optimizer, this is where you put your async optimization logic.
    /// Check cancellationToken periodically and use ConfigureAwait(false) on awaits.
    /// </para>
    /// </remarks>
    protected abstract Task<IPromptTemplate> OptimizeCoreAsync(
        string initialPrompt,
        Func<string, Task<T>> evaluationFunction,
        int maxIterations,
        CancellationToken cancellationToken);

    /// <summary>
    /// Records an iteration in the history.
    /// </summary>
    /// <param name="iteration">The iteration number (0-based).</param>
    /// <param name="prompt">The prompt that was evaluated.</param>
    /// <param name="score">The score achieved by the prompt.</param>
    /// <remarks>
    /// <para>
    /// Call this method during optimization to record each prompt variation tried.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// Derived classes should call this for each prompt variation they evaluate.
    /// This creates the history that can be retrieved via GetOptimizationHistory().
    /// </para>
    /// </remarks>
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
