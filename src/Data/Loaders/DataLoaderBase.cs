namespace AiDotNet.Data.Loaders;

/// <summary>
/// Abstract base class providing common functionality for all data loaders.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// DataLoaderBase implements shared functionality for all data loaders including:
/// - State management (loaded/unloaded)
/// - Iteration tracking (current index, progress)
/// - Reset functionality
/// - Thread-safe operations where needed
/// </para>
/// <para>
/// Domain-specific base classes (GraphDataLoaderBase, InputOutputDataLoaderBase)
/// extend this class with specialized functionality.
/// </para>
/// <para><b>For Beginners:</b> This class handles the "boring but important" stuff
/// that all data loaders need to do: tracking where you are in the data, resetting
/// to start over, and making sure data is loaded before you try to use it.
///
/// When you create a custom data loader, you extend one of the domain-specific base
/// classes (like GraphDataLoaderBase) which in turn extends this class, so you get
/// all this functionality for free.
/// </para>
/// </remarks>
public abstract class DataLoaderBase<T> : IDataLoader<T>
{
    private int _currentIndex;
    private int _currentBatchIndex;
    private bool _isLoaded;
    private int _batchSize;
    private readonly object _lock = new();

    /// <summary>
    /// Initializes a new instance of the DataLoaderBase class.
    /// </summary>
    /// <param name="batchSize">The batch size for iteration. Default is 32.</param>
    protected DataLoaderBase(int batchSize = 32)
    {
        _batchSize = batchSize;
    }

    /// <inheritdoc/>
    public abstract string Name { get; }

    /// <inheritdoc/>
    public virtual string Description => $"{Name} data loader";

    /// <inheritdoc/>
    public bool IsLoaded
    {
        get
        {
            lock (_lock)
            {
                return _isLoaded;
            }
        }
        protected set
        {
            lock (_lock)
            {
                _isLoaded = value;
            }
        }
    }

    /// <inheritdoc/>
    public abstract int TotalCount { get; }

    /// <inheritdoc/>
    public int CurrentIndex
    {
        get
        {
            lock (_lock)
            {
                return _currentIndex;
            }
        }
        protected set
        {
            lock (_lock)
            {
                _currentIndex = value;
            }
        }
    }

    /// <summary>
    /// Gets or sets the batch size for iteration.
    /// </summary>
    public virtual int BatchSize
    {
        get => _batchSize;
        set => _batchSize = value;
    }

    /// <inheritdoc/>
    public int BatchCount => TotalCount > 0 ? (int)Math.Ceiling((double)TotalCount / BatchSize) : 0;

    /// <inheritdoc/>
    public int CurrentBatchIndex
    {
        get
        {
            lock (_lock)
            {
                return _currentBatchIndex;
            }
        }
        protected set
        {
            lock (_lock)
            {
                _currentBatchIndex = value;
            }
        }
    }

    /// <inheritdoc/>
    public double Progress => TotalCount > 0 ? (double)CurrentIndex / TotalCount : 0.0;

    /// <inheritdoc/>
    public virtual void Reset()
    {
        lock (_lock)
        {
            _currentIndex = 0;
            _currentBatchIndex = 0;
        }
        OnReset();
    }

    /// <inheritdoc/>
    public async Task LoadAsync(CancellationToken cancellationToken = default)
    {
        if (IsLoaded)
        {
            return; // Idempotent - already loaded
        }

        await LoadDataCoreAsync(cancellationToken);
        IsLoaded = true;
        Reset();
    }

    /// <inheritdoc/>
    public virtual void Unload()
    {
        UnloadDataCore();
        IsLoaded = false;
        Reset();
    }

    /// <summary>
    /// Core data loading implementation to be provided by derived classes.
    /// </summary>
    /// <param name="cancellationToken">Cancellation token for async operation.</param>
    /// <returns>A task that completes when loading is finished.</returns>
    /// <remarks>
    /// <para>
    /// Derived classes must implement this to perform actual data loading:
    /// - Load from files, databases, or remote sources
    /// - Parse and validate data format
    /// - Store in appropriate internal structures
    /// </para>
    /// </remarks>
    protected abstract Task LoadDataCoreAsync(CancellationToken cancellationToken);

    /// <summary>
    /// Core data unloading implementation to be provided by derived classes.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Derived classes should implement this to release resources:
    /// - Clear internal data structures
    /// - Release file handles or connections
    /// - Allow garbage collection of loaded data
    /// </para>
    /// </remarks>
    protected abstract void UnloadDataCore();

    /// <summary>
    /// Called after Reset() to allow derived classes to perform additional reset operations.
    /// </summary>
    /// <remarks>
    /// Override this to reset any domain-specific state. The base indices are already reset
    /// when this is called.
    /// </remarks>
    protected virtual void OnReset()
    {
        // Default implementation does nothing - override in derived classes if needed
    }

    /// <summary>
    /// Ensures data is loaded before operations that require it.
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown when data is not loaded.</exception>
    protected void EnsureLoaded()
    {
        if (!IsLoaded)
        {
            throw new InvalidOperationException(
                $"Data has not been loaded. Call LoadAsync() before accessing data from {Name}.");
        }
    }

    /// <summary>
    /// Advances the current index by the specified amount.
    /// </summary>
    /// <param name="count">Number of samples to advance by.</param>
    protected void AdvanceIndex(int count)
    {
        lock (_lock)
        {
            _currentIndex = Math.Min(_currentIndex + count, TotalCount);
        }
    }

    /// <summary>
    /// Advances the batch index by one.
    /// </summary>
    protected void AdvanceBatchIndex()
    {
        lock (_lock)
        {
            _currentBatchIndex++;
        }
    }
}
