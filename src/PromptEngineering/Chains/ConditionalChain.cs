using AiDotNet.Interfaces;

namespace AiDotNet.PromptEngineering.Chains;

/// <summary>
/// Chain that executes different branches based on conditions evaluated on the input.
/// </summary>
/// <typeparam name="TInput">The type of input accepted by the chain.</typeparam>
/// <typeparam name="TOutput">The type of output produced by the chain.</typeparam>
/// <remarks>
/// <para>
/// A conditional chain evaluates predicates on the input and routes execution to the appropriate branch.
/// Each branch can have its own processing logic, allowing different handling for different input types.
/// </para>
/// <para><b>For Beginners:</b> Routes input to different processing paths based on conditions.
///
/// Example:
/// <code>
/// var chain = new ConditionalChain&lt;string, string&gt;("SupportRouter", "Routes support tickets");
///
/// // Add branches with conditions
/// chain.AddBranch("complaint", text => text.Contains("problem") || text.Contains("issue"),
///     text => HandleComplaint(text));
/// chain.AddBranch("question", text => text.Contains("?") || text.Contains("how"),
///     text => HandleQuestion(text));
/// chain.SetDefaultBranch("general", text => HandleGeneral(text));
///
/// // Run chain - routes to appropriate handler
/// var result = chain.Run("I have a problem with my order");
/// // Routes to HandleComplaint
/// </code>
/// </para>
/// </remarks>
public class ConditionalChain<TInput, TOutput> : ChainBase<TInput, TOutput>
{
    private readonly List<ConditionalBranch> _branches;
    private ConditionalBranch? _defaultBranch;

    /// <summary>
    /// Initializes a new instance of the ConditionalChain class.
    /// </summary>
    /// <param name="name">The name of the chain.</param>
    /// <param name="description">The description of the chain.</param>
    public ConditionalChain(string name, string description = "")
        : base(name, description)
    {
        _branches = new List<ConditionalBranch>();
    }

    /// <summary>
    /// Adds a conditional branch to the chain.
    /// </summary>
    /// <param name="branchName">The name of the branch.</param>
    /// <param name="condition">The condition that must be true to take this branch.</param>
    /// <param name="handler">The function to execute when this branch is taken.</param>
    /// <returns>This chain instance for method chaining.</returns>
    public ConditionalChain<TInput, TOutput> AddBranch(
        string branchName,
        Func<TInput, bool> condition,
        Func<TInput, TOutput> handler)
    {
        if (string.IsNullOrWhiteSpace(branchName))
        {
            throw new ArgumentException("Branch name cannot be empty.", nameof(branchName));
        }

        _branches.Add(new ConditionalBranch(branchName, condition, handler, null));
        return this;
    }

    /// <summary>
    /// Adds an async conditional branch to the chain.
    /// </summary>
    /// <param name="branchName">The name of the branch.</param>
    /// <param name="condition">The condition that must be true to take this branch.</param>
    /// <param name="handler">The async function to execute when this branch is taken.</param>
    /// <returns>This chain instance for method chaining.</returns>
    public ConditionalChain<TInput, TOutput> AddBranchAsync(
        string branchName,
        Func<TInput, bool> condition,
        Func<TInput, CancellationToken, Task<TOutput>> handler)
    {
        if (string.IsNullOrWhiteSpace(branchName))
        {
            throw new ArgumentException("Branch name cannot be empty.", nameof(branchName));
        }

        _branches.Add(new ConditionalBranch(branchName, condition, null, handler));
        return this;
    }

    /// <summary>
    /// Sets the default branch to execute when no conditions match.
    /// </summary>
    /// <param name="branchName">The name of the default branch.</param>
    /// <param name="handler">The function to execute as default.</param>
    /// <returns>This chain instance for method chaining.</returns>
    public ConditionalChain<TInput, TOutput> SetDefaultBranch(
        string branchName,
        Func<TInput, TOutput> handler)
    {
        _defaultBranch = new ConditionalBranch(branchName, _ => true, handler, null);
        return this;
    }

    /// <summary>
    /// Sets the default async branch to execute when no conditions match.
    /// </summary>
    /// <param name="branchName">The name of the default branch.</param>
    /// <param name="handler">The async function to execute as default.</param>
    /// <returns>This chain instance for method chaining.</returns>
    public ConditionalChain<TInput, TOutput> SetDefaultBranchAsync(
        string branchName,
        Func<TInput, CancellationToken, Task<TOutput>> handler)
    {
        _defaultBranch = new ConditionalBranch(branchName, _ => true, null, handler);
        return this;
    }

    /// <summary>
    /// Gets the names of all branches.
    /// </summary>
    public IReadOnlyList<string> BranchNames => _branches.Select(b => b.Name).ToList().AsReadOnly();

    /// <summary>
    /// Executes the chain by evaluating conditions and routing to the appropriate branch.
    /// </summary>
    protected override TOutput RunCore(TInput input)
    {
        foreach (var branch in _branches)
        {
            if (branch.Condition(input))
            {
                if (branch.SyncHandler != null)
                {
                    return branch.SyncHandler(input);
                }

                throw new InvalidOperationException(
                    $"Branch '{branch.Name}' is async. Use RunAsync for chains with async branches.");
            }
        }

        if (_defaultBranch != null)
        {
            if (_defaultBranch.SyncHandler != null)
            {
                return _defaultBranch.SyncHandler(input);
            }

            throw new InvalidOperationException(
                "Default branch is async. Use RunAsync for chains with async branches.");
        }

        throw new InvalidOperationException("No matching condition found and no default branch set.");
    }

    /// <summary>
    /// Executes the chain asynchronously by evaluating conditions and routing to the appropriate branch.
    /// </summary>
    protected override async Task<TOutput> RunCoreAsync(TInput input, CancellationToken cancellationToken)
    {
        foreach (var branch in _branches)
        {
            cancellationToken.ThrowIfCancellationRequested();

            if (branch.Condition(input))
            {
                if (branch.AsyncHandler != null)
                {
                    return await branch.AsyncHandler(input, cancellationToken).ConfigureAwait(false);
                }

                if (branch.SyncHandler != null)
                {
                    return branch.SyncHandler(input);
                }
            }
        }

        if (_defaultBranch != null)
        {
            if (_defaultBranch.AsyncHandler != null)
            {
                return await _defaultBranch.AsyncHandler(input, cancellationToken).ConfigureAwait(false);
            }

            if (_defaultBranch.SyncHandler != null)
            {
                return _defaultBranch.SyncHandler(input);
            }
        }

        throw new InvalidOperationException("No matching condition found and no default branch set.");
    }

    private class ConditionalBranch
    {
        public string Name { get; }
        public Func<TInput, bool> Condition { get; }
        public Func<TInput, TOutput>? SyncHandler { get; }
        public Func<TInput, CancellationToken, Task<TOutput>>? AsyncHandler { get; }

        public ConditionalBranch(
            string name,
            Func<TInput, bool> condition,
            Func<TInput, TOutput>? syncHandler,
            Func<TInput, CancellationToken, Task<TOutput>>? asyncHandler)
        {
            Name = name;
            Condition = condition;
            SyncHandler = syncHandler;
            AsyncHandler = asyncHandler;
        }
    }
}
