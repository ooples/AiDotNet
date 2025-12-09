using AiDotNet.Interfaces;

namespace AiDotNet.PromptEngineering.Chains;

/// <summary>
/// Base class for chain implementations providing common functionality.
/// </summary>
/// <typeparam name="TInput">The type of input accepted by the chain.</typeparam>
/// <typeparam name="TOutput">The type of output produced by the chain.</typeparam>
/// <remarks>
/// <para>
/// This base class provides validation, error handling, and the template method pattern
/// for chain execution. Derived classes implement the core execution logic.
/// </para>
/// <para><b>For Beginners:</b> This is the foundation for all chain types.
///
/// It handles:
/// - Input validation
/// - Error handling
/// - Async/sync execution
/// - Logging and debugging
///
/// When you create a new chain type, inherit from this class!
/// </para>
/// </remarks>
public abstract class ChainBase<TInput, TOutput> : IChain<TInput, TOutput>
{
    /// <summary>
    /// Gets the name of this chain.
    /// </summary>
    public string Name { get; protected set; }

    /// <summary>
    /// Gets the description of what this chain does.
    /// </summary>
    public string Description { get; protected set; }

    /// <summary>
    /// Initializes a new instance of the ChainBase class.
    /// </summary>
    /// <param name="name">The name of the chain.</param>
    /// <param name="description">The description of the chain.</param>
    protected ChainBase(string name, string description)
    {
        if (string.IsNullOrWhiteSpace(name))
        {
            throw new ArgumentException("Chain name cannot be empty.", nameof(name));
        }

        Name = name;
        Description = description ?? string.Empty;
    }

    /// <summary>
    /// Executes the chain with the provided input.
    /// </summary>
    /// <param name="input">The input to process.</param>
    /// <returns>The chain's output result.</returns>
    public TOutput Run(TInput input)
    {
        if (!ValidateInput(input))
        {
            throw new ArgumentException("Input validation failed.", nameof(input));
        }

        return RunCore(input);
    }

    /// <summary>
    /// Executes the chain asynchronously with the provided input.
    /// </summary>
    /// <param name="input">The input to process.</param>
    /// <param name="cancellationToken">Token to cancel the operation.</param>
    /// <returns>A task representing the asynchronous operation with the chain's output.</returns>
    public async Task<TOutput> RunAsync(TInput input, CancellationToken cancellationToken = default)
    {
        if (!ValidateInput(input))
        {
            throw new ArgumentException("Input validation failed.", nameof(input));
        }

        return await RunCoreAsync(input, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Validates that the input is appropriate for this chain.
    /// </summary>
    /// <param name="input">The input to validate.</param>
    /// <returns>True if the input is valid; otherwise, false.</returns>
    public virtual bool ValidateInput(TInput input)
    {
        return input != null;
    }

    /// <summary>
    /// Core synchronous execution logic to be implemented by derived classes.
    /// </summary>
    /// <param name="input">The validated input.</param>
    /// <returns>The chain's output.</returns>
    protected abstract TOutput RunCore(TInput input);

    /// <summary>
    /// Core asynchronous execution logic to be implemented by derived classes.
    /// </summary>
    /// <param name="input">The validated input.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A task with the chain's output.</returns>
    protected abstract Task<TOutput> RunCoreAsync(TInput input, CancellationToken cancellationToken);
}
