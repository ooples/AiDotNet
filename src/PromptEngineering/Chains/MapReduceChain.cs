namespace AiDotNet.PromptEngineering.Chains;

/// <summary>
/// Chain that applies a map operation to each item in a collection, then reduces the results.
/// </summary>
/// <typeparam name="TItem">The type of items in the input collection.</typeparam>
/// <typeparam name="TMapResult">The type of result from the map operation.</typeparam>
/// <typeparam name="TOutput">The type of final output from the reduce operation.</typeparam>
/// <remarks>
/// <para>
/// A map-reduce chain processes collections by applying the same operation to each item (map phase),
/// then combining all results into a single output (reduce phase).
/// </para>
/// <para><b>For Beginners:</b> Process many items the same way, then combine results.
///
/// Example - Summarizing multiple documents:
/// <code>
/// var chain = new MapReduceChain&lt;Document, string, Summary&gt;("DocumentSummarizer", "Summarizes multiple documents");
///
/// // Map: Process each document
/// chain.SetMapper(doc => SummarizeDocument(doc));
///
/// // Reduce: Combine all summaries
/// chain.SetReducer(summaries => CombineSummaries(summaries));
///
/// // Run on collection
/// var documents = new[] { doc1, doc2, doc3, doc4, doc5 };
/// var result = chain.Run(documents);
/// // Each document is summarized individually, then all summaries are combined
/// </code>
/// </para>
/// </remarks>
public class MapReduceChain<TItem, TMapResult, TOutput> : ChainBase<IEnumerable<TItem>, TOutput>
{
    private Func<TItem, TMapResult>? _syncMapper;
    private Func<TItem, CancellationToken, Task<TMapResult>>? _asyncMapper;
    private Func<IEnumerable<TMapResult>, TOutput>? _syncReducer;
    private Func<IEnumerable<TMapResult>, CancellationToken, Task<TOutput>>? _asyncReducer;
    private int _maxDegreeOfParallelism;

    /// <summary>
    /// Initializes a new instance of the MapReduceChain class.
    /// </summary>
    /// <param name="name">The name of the chain.</param>
    /// <param name="description">The description of the chain.</param>
    /// <param name="maxDegreeOfParallelism">Maximum concurrent map operations. -1 for unlimited.</param>
    public MapReduceChain(string name, string description = "", int maxDegreeOfParallelism = -1)
        : base(name, description)
    {
        _maxDegreeOfParallelism = maxDegreeOfParallelism > 0 ? maxDegreeOfParallelism : -1;
    }

    /// <summary>
    /// Gets or sets the maximum degree of parallelism for the map phase.
    /// </summary>
    public int MaxDegreeOfParallelism
    {
        get => _maxDegreeOfParallelism;
        set => _maxDegreeOfParallelism = value > 0 ? value : -1;
    }

    /// <summary>
    /// Gets the number of items processed in the last run.
    /// </summary>
    public int LastItemCount { get; private set; }

    /// <summary>
    /// Sets the mapper function applied to each item.
    /// </summary>
    /// <param name="mapper">Function to apply to each item.</param>
    /// <returns>This chain instance for method chaining.</returns>
    public MapReduceChain<TItem, TMapResult, TOutput> SetMapper(Func<TItem, TMapResult> mapper)
    {
        Guard.NotNull(mapper);
        _syncMapper = mapper;
        return this;
    }

    /// <summary>
    /// Sets the async mapper function applied to each item.
    /// </summary>
    /// <param name="mapper">Async function to apply to each item.</param>
    /// <returns>This chain instance for method chaining.</returns>
    public MapReduceChain<TItem, TMapResult, TOutput> SetMapperAsync(
        Func<TItem, CancellationToken, Task<TMapResult>> mapper)
    {
        Guard.NotNull(mapper);
        _asyncMapper = mapper;
        return this;
    }

    /// <summary>
    /// Sets the reducer function that combines all map results.
    /// </summary>
    /// <param name="reducer">Function to combine map results.</param>
    /// <returns>This chain instance for method chaining.</returns>
    public MapReduceChain<TItem, TMapResult, TOutput> SetReducer(Func<IEnumerable<TMapResult>, TOutput> reducer)
    {
        Guard.NotNull(reducer);
        _syncReducer = reducer;
        return this;
    }

    /// <summary>
    /// Sets the async reducer function that combines all map results.
    /// </summary>
    /// <param name="reducer">Async function to combine map results.</param>
    /// <returns>This chain instance for method chaining.</returns>
    public MapReduceChain<TItem, TMapResult, TOutput> SetReducerAsync(
        Func<IEnumerable<TMapResult>, CancellationToken, Task<TOutput>> reducer)
    {
        Guard.NotNull(reducer);
        _asyncReducer = reducer;
        return this;
    }

    /// <summary>
    /// Executes the map-reduce operation.
    /// </summary>
    protected override TOutput RunCore(IEnumerable<TItem> input)
    {
        if (_syncMapper == null && _asyncMapper == null)
        {
            throw new InvalidOperationException("No mapper set. Call SetMapper before running the chain.");
        }

        if (_syncReducer == null && _asyncReducer == null)
        {
            throw new InvalidOperationException("No reducer set. Call SetReducer before running the chain.");
        }

        if (_asyncMapper != null && _syncMapper == null)
        {
            throw new InvalidOperationException("Mapper is async. Use RunAsync for chains with async mappers.");
        }

        if (_asyncReducer != null && _syncReducer == null)
        {
            throw new InvalidOperationException("Reducer is async. Use RunAsync for chains with async reducers.");
        }

        var items = input.ToList();
        LastItemCount = items.Count;

        // Map phase - parallel execution
        var mapResults = new TMapResult[items.Count];
        var parallelOptions = new ParallelOptions
        {
            MaxDegreeOfParallelism = _maxDegreeOfParallelism
        };

        Parallel.For(0, items.Count, parallelOptions, i =>
        {
            mapResults[i] = _syncMapper!(items[i]);
        });

        // Reduce phase
        return _syncReducer!(mapResults);
    }

    /// <summary>
    /// Executes the map-reduce operation asynchronously.
    /// </summary>
    protected override async Task<TOutput> RunCoreAsync(IEnumerable<TItem> input, CancellationToken cancellationToken)
    {
        if (_syncMapper == null && _asyncMapper == null)
        {
            throw new InvalidOperationException("No mapper set. Call SetMapper before running the chain.");
        }

        if (_syncReducer == null && _asyncReducer == null)
        {
            throw new InvalidOperationException("No reducer set. Call SetReducer before running the chain.");
        }

        var items = input.ToList();
        LastItemCount = items.Count;

        // Map phase - parallel execution with semaphore for throttling
        var semaphore = _maxDegreeOfParallelism > 0
            ? new SemaphoreSlim(_maxDegreeOfParallelism)
            : null;

        var mapTasks = items.Select(async item =>
        {
            if (semaphore != null)
            {
                await semaphore.WaitAsync(cancellationToken).ConfigureAwait(false);
            }

            try
            {
                cancellationToken.ThrowIfCancellationRequested();

                if (_asyncMapper != null)
                {
                    return await _asyncMapper(item, cancellationToken).ConfigureAwait(false);
                }

                return _syncMapper!(item);
            }
            finally
            {
                semaphore?.Release();
            }
        }).ToList();

        var mapResults = await Task.WhenAll(mapTasks).ConfigureAwait(false);
        semaphore?.Dispose();

        // Reduce phase
        if (_asyncReducer != null)
        {
            return await _asyncReducer(mapResults, cancellationToken).ConfigureAwait(false);
        }

        return _syncReducer!(mapResults);
    }
}
