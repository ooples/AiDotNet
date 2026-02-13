namespace AiDotNet.PromptEngineering.Chains;

/// <summary>
/// Chain that repeatedly executes an operation until a termination condition is met.
/// </summary>
/// <typeparam name="TInput">The type of input accepted by the chain.</typeparam>
/// <typeparam name="TOutput">The type of output produced by the chain.</typeparam>
/// <remarks>
/// <para>
/// A loop chain executes an operation repeatedly, with each iteration's output becoming the next
/// iteration's input. The loop continues until a termination condition returns true or max iterations
/// is reached.
/// </para>
/// <para><b>For Beginners:</b> Keeps running until a condition is met.
///
/// Example:
/// <code>
/// var chain = new LoopChain&lt;string, string&gt;("RefineSummary", "Refines summary until it meets length requirements");
///
/// // Set the operation to repeat
/// chain.SetOperation(text => RefineText(text));
///
/// // Set when to stop
/// chain.SetTerminationCondition(text => text.Length &lt;= 100);
///
/// // Set safety limit
/// chain.MaxIterations = 5;
///
/// // Run chain - keeps refining until summary is under 100 chars or 5 iterations
/// var result = chain.Run("This is a very long summary that needs to be shortened...");
/// </code>
/// </para>
/// </remarks>
public class LoopChain<TInput, TOutput> : ChainBase<TInput, TOutput>
{
    private Func<object, object>? _syncOperation;
    private Func<object, CancellationToken, Task<object>>? _asyncOperation;
    private Func<object, bool>? _terminationCondition;
    private int _maxIterations;

    /// <summary>
    /// Initializes a new instance of the LoopChain class.
    /// </summary>
    /// <param name="name">The name of the chain.</param>
    /// <param name="description">The description of the chain.</param>
    /// <param name="maxIterations">Maximum number of iterations (safety limit).</param>
    public LoopChain(string name, string description = "", int maxIterations = 10)
        : base(name, description)
    {
        _maxIterations = maxIterations > 0 ? maxIterations : 10;
    }

    /// <summary>
    /// Gets or sets the maximum number of iterations.
    /// </summary>
    public int MaxIterations
    {
        get => _maxIterations;
        set => _maxIterations = value > 0 ? value : 10;
    }

    /// <summary>
    /// Gets the number of iterations executed in the last run.
    /// </summary>
    public int LastIterationCount { get; private set; }

    /// <summary>
    /// Sets the operation to repeat in each iteration.
    /// </summary>
    /// <param name="operation">The operation to execute.</param>
    /// <returns>This chain instance for method chaining.</returns>
    public LoopChain<TInput, TOutput> SetOperation(Func<object, object> operation)
    {
        Guard.NotNull(operation);
        _syncOperation = operation;
        return this;
    }

    /// <summary>
    /// Sets the async operation to repeat in each iteration.
    /// </summary>
    /// <param name="operation">The async operation to execute.</param>
    /// <returns>This chain instance for method chaining.</returns>
    public LoopChain<TInput, TOutput> SetOperationAsync(Func<object, CancellationToken, Task<object>> operation)
    {
        Guard.NotNull(operation);
        _asyncOperation = operation;
        return this;
    }

    /// <summary>
    /// Sets the condition that stops the loop when it returns true.
    /// </summary>
    /// <param name="condition">Function that returns true when the loop should stop.</param>
    /// <returns>This chain instance for method chaining.</returns>
    public LoopChain<TInput, TOutput> SetTerminationCondition(Func<object, bool> condition)
    {
        Guard.NotNull(condition);
        _terminationCondition = condition;
        return this;
    }

    /// <summary>
    /// Executes the loop until termination condition is met or max iterations reached.
    /// </summary>
    protected override TOutput RunCore(TInput input)
    {
        if (_syncOperation == null && _asyncOperation == null)
        {
            throw new InvalidOperationException("No operation set. Call SetOperation before running the chain.");
        }

        if (_terminationCondition == null)
        {
            throw new InvalidOperationException("No termination condition set. Call SetTerminationCondition before running the chain.");
        }

        if (_asyncOperation != null && _syncOperation == null)
        {
            throw new InvalidOperationException("Operation is async. Use RunAsync for chains with async operations.");
        }

        object current = input!;
        LastIterationCount = 0;

        for (int i = 0; i < _maxIterations; i++)
        {
            LastIterationCount = i + 1;

            // Check termination before executing
            if (_terminationCondition(current))
            {
                break;
            }

            current = _syncOperation!(current);
        }

        if (current is TOutput output)
        {
            return output;
        }

        throw new InvalidOperationException(
            $"Final output type mismatch. Expected {typeof(TOutput).Name}, got {current?.GetType().Name ?? "null"}");
    }

    /// <summary>
    /// Executes the loop asynchronously until termination condition is met or max iterations reached.
    /// </summary>
    protected override async Task<TOutput> RunCoreAsync(TInput input, CancellationToken cancellationToken)
    {
        if (_syncOperation == null && _asyncOperation == null)
        {
            throw new InvalidOperationException("No operation set. Call SetOperation before running the chain.");
        }

        if (_terminationCondition == null)
        {
            throw new InvalidOperationException("No termination condition set. Call SetTerminationCondition before running the chain.");
        }

        object current = input!;
        LastIterationCount = 0;

        for (int i = 0; i < _maxIterations; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            LastIterationCount = i + 1;

            // Check termination before executing
            if (_terminationCondition(current))
            {
                break;
            }

            if (_asyncOperation != null)
            {
                current = await _asyncOperation(current, cancellationToken).ConfigureAwait(false);
            }
            else if (_syncOperation != null)
            {
                current = _syncOperation(current);
            }
        }

        if (current is TOutput output)
        {
            return output;
        }

        throw new InvalidOperationException(
            $"Final output type mismatch. Expected {typeof(TOutput).Name}, got {current?.GetType().Name ?? "null"}");
    }
}
