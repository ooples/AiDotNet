using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.PromptEngineering.Chains;

/// <summary>
/// Chain that executes multiple operations in parallel and combines results.
/// </summary>
/// <typeparam name="TInput">The type of input accepted by the chain.</typeparam>
/// <typeparam name="TOutput">The type of output produced by the chain.</typeparam>
/// <remarks>
/// <para>
/// A parallel chain runs multiple independent operations simultaneously on the same input,
/// then combines their results using a reducer function.
/// </para>
/// <para><b>For Beginners:</b> Runs multiple operations at the same time for speed.
///
/// Example:
/// <code>
/// var chain = new ParallelChain&lt;string, AnalysisReport&gt;("ProductAnalysis", "Analyzes product from multiple angles");
///
/// // Add parallel branches
/// chain.AddBranch("sentiment", text => AnalyzeSentiment(text));
/// chain.AddBranch("keywords", text => ExtractKeywords(text));
/// chain.AddBranch("entities", text => ExtractEntities(text));
///
/// // Set how to combine results
/// chain.SetReducer((results) => new AnalysisReport
/// {
///     Sentiment = (string)results["sentiment"],
///     Keywords = (List&lt;string&gt;)results["keywords"],
///     Entities = (List&lt;string&gt;)results["entities"]
/// });
///
/// // Run chain - all branches execute simultaneously
/// var result = chain.Run("Great product! Love the fast delivery from Amazon.");
/// </code>
/// </para>
/// </remarks>
public class ParallelChain<TInput, TOutput> : ChainBase<TInput, TOutput>
{
    private readonly List<ParallelBranch> _branches;
    private Func<Dictionary<string, object>, TOutput>? _syncReducer;
    private Func<Dictionary<string, object>, CancellationToken, Task<TOutput>>? _asyncReducer;
    private int _maxDegreeOfParallelism;

    /// <summary>
    /// Initializes a new instance of the ParallelChain class.
    /// </summary>
    /// <param name="name">The name of the chain.</param>
    /// <param name="description">The description of the chain.</param>
    /// <param name="maxDegreeOfParallelism">Maximum concurrent operations. -1 for unlimited.</param>
    public ParallelChain(string name, string description = "", int maxDegreeOfParallelism = -1)
        : base(name, description)
    {
        _branches = new List<ParallelBranch>();
        _maxDegreeOfParallelism = maxDegreeOfParallelism > 0 ? maxDegreeOfParallelism : -1;
    }

    /// <summary>
    /// Gets or sets the maximum degree of parallelism.
    /// </summary>
    public int MaxDegreeOfParallelism
    {
        get => _maxDegreeOfParallelism;
        set => _maxDegreeOfParallelism = value > 0 ? value : -1;
    }

    /// <summary>
    /// Adds a parallel branch to the chain.
    /// </summary>
    /// <param name="branchName">The name of the branch.</param>
    /// <param name="handler">The function to execute for this branch.</param>
    /// <returns>This chain instance for method chaining.</returns>
    public ParallelChain<TInput, TOutput> AddBranch(string branchName, Func<TInput, object> handler)
    {
        if (string.IsNullOrWhiteSpace(branchName))
        {
            throw new ArgumentException("Branch name cannot be empty.", nameof(branchName));
        }

        _branches.Add(new ParallelBranch(branchName, handler, null));
        return this;
    }

    /// <summary>
    /// Adds an async parallel branch to the chain.
    /// </summary>
    /// <param name="branchName">The name of the branch.</param>
    /// <param name="handler">The async function to execute for this branch.</param>
    /// <returns>This chain instance for method chaining.</returns>
    public ParallelChain<TInput, TOutput> AddBranchAsync(
        string branchName,
        Func<TInput, CancellationToken, Task<object>> handler)
    {
        if (string.IsNullOrWhiteSpace(branchName))
        {
            throw new ArgumentException("Branch name cannot be empty.", nameof(branchName));
        }

        _branches.Add(new ParallelBranch(branchName, null, handler));
        return this;
    }

    /// <summary>
    /// Sets the reducer function that combines results from all branches.
    /// </summary>
    /// <param name="reducer">Function that takes branch results and produces final output.</param>
    /// <returns>This chain instance for method chaining.</returns>
    public ParallelChain<TInput, TOutput> SetReducer(Func<Dictionary<string, object>, TOutput> reducer)
    {
        Guard.NotNull(reducer);
        _syncReducer = reducer;
        return this;
    }

    /// <summary>
    /// Sets the async reducer function that combines results from all branches.
    /// </summary>
    /// <param name="reducer">Async function that takes branch results and produces final output.</param>
    /// <returns>This chain instance for method chaining.</returns>
    public ParallelChain<TInput, TOutput> SetReducerAsync(
        Func<Dictionary<string, object>, CancellationToken, Task<TOutput>> reducer)
    {
        Guard.NotNull(reducer);
        _asyncReducer = reducer;
        return this;
    }

    /// <summary>
    /// Gets the names of all branches.
    /// </summary>
    public IReadOnlyList<string> BranchNames => _branches.Select(b => b.Name).ToList().AsReadOnly();

    /// <summary>
    /// Executes all branches in parallel and combines results.
    /// </summary>
    protected override TOutput RunCore(TInput input)
    {
        if (_syncReducer == null && _asyncReducer == null)
        {
            throw new InvalidOperationException("No reducer set. Call SetReducer before running the chain.");
        }

        var results = new Dictionary<string, object>();
        var syncBranches = _branches.Where(b => b.SyncHandler != null).ToList();
        var asyncBranches = _branches.Where(b => b.AsyncHandler != null).ToList();

        if (asyncBranches.Count > 0)
        {
            throw new InvalidOperationException(
                "Chain contains async branches. Use RunAsync for chains with async branches.");
        }

        // Execute sync branches in parallel
        var parallelOptions = new ParallelOptions
        {
            MaxDegreeOfParallelism = _maxDegreeOfParallelism
        };

        var lockObj = new object();
        Parallel.ForEach(syncBranches, parallelOptions, branch =>
        {
            var result = branch.SyncHandler!(input);
            lock (lockObj)
            {
                results[branch.Name] = result;
            }
        });

        if (_syncReducer != null)
        {
            return _syncReducer(results);
        }

        throw new InvalidOperationException("Reducer is async. Use RunAsync for chains with async reducers.");
    }

    /// <summary>
    /// Executes all branches in parallel asynchronously and combines results.
    /// </summary>
    protected override async Task<TOutput> RunCoreAsync(TInput input, CancellationToken cancellationToken)
    {
        if (_syncReducer == null && _asyncReducer == null)
        {
            throw new InvalidOperationException("No reducer set. Call SetReducer before running the chain.");
        }

        var results = new Dictionary<string, object>();
        var tasks = new List<Task>();

        // Limit parallelism using semaphore if specified
        var semaphore = _maxDegreeOfParallelism > 0
            ? new SemaphoreSlim(_maxDegreeOfParallelism)
            : null;

        var lockObj = new object();

        foreach (var branch in _branches)
        {
            var branchCopy = branch;
            var task = Task.Run(async () =>
            {
                if (semaphore != null)
                {
                    await semaphore.WaitAsync(cancellationToken).ConfigureAwait(false);
                }

                try
                {
                    cancellationToken.ThrowIfCancellationRequested();

                    object result;
                    if (branchCopy.AsyncHandler != null)
                    {
                        result = await branchCopy.AsyncHandler(input, cancellationToken).ConfigureAwait(false);
                    }
                    else if (branchCopy.SyncHandler != null)
                    {
                        result = branchCopy.SyncHandler(input);
                    }
                    else
                    {
                        throw new InvalidOperationException($"Branch '{branchCopy.Name}' has no handler.");
                    }

                    lock (lockObj)
                    {
                        results[branchCopy.Name] = result;
                    }
                }
                finally
                {
                    semaphore?.Release();
                }
            }, cancellationToken);

            tasks.Add(task);
        }

        await Task.WhenAll(tasks).ConfigureAwait(false);
        semaphore?.Dispose();

        if (_asyncReducer != null)
        {
            return await _asyncReducer(results, cancellationToken).ConfigureAwait(false);
        }

        if (_syncReducer != null)
        {
            return _syncReducer(results);
        }

        throw new InvalidOperationException("No reducer found.");
    }

    private class ParallelBranch
    {
        public string Name { get; }
        public Func<TInput, object>? SyncHandler { get; }
        public Func<TInput, CancellationToken, Task<object>>? AsyncHandler { get; }

        public ParallelBranch(
            string name,
            Func<TInput, object>? syncHandler,
            Func<TInput, CancellationToken, Task<object>>? asyncHandler)
        {
            Name = name;
            SyncHandler = syncHandler;
            AsyncHandler = asyncHandler;
        }
    }
}
