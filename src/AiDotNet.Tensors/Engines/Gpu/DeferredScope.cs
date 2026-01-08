using System.Diagnostics;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu.Graph;
using AiDotNet.Tensors.Engines.Gpu.Graph.Optimization;

namespace AiDotNet.Tensors.Engines.Gpu;

/// <summary>
/// Concrete implementation of IDeferredScope that records operations
/// and executes them as an optimized execution graph.
/// </summary>
public sealed class DeferredScope : IDeferredScope
{
    private readonly IAsyncGpuBackend _backend;
    private readonly RecordingGpuBackend _recordingBackend;
    private readonly GpuStreamPool? _streamPool;
    private readonly Stopwatch _totalTimer;
    private readonly List<DeferredDownload> _deferredDownloads = new();
    private ExecutionGraph? _compiledGraph;
    private OptimizationStatistics? _compiledStatistics;
    private DeferredScopeStatistics? _statistics;
    private bool _disposed;

    /// <inheritdoc/>
    public ExecutionGraphBuilder GraphBuilder { get; }

    /// <inheritdoc/>
    public IDirectGpuBackend RecordingBackend => _recordingBackend;

    /// <inheritdoc/>
    public bool IsRecording => _recordingBackend.IsRecording;

    /// <inheritdoc/>
    public GpuExecutionOptions Options { get; }

    /// <inheritdoc/>
    public bool IsExecuted { get; private set; }

    /// <inheritdoc/>
    public int OperationCount => GraphBuilder.NodeCount;

    /// <summary>
    /// Creates a new deferred scope.
    /// </summary>
    /// <param name="backend">The GPU backend for execution.</param>
    /// <param name="options">Execution options.</param>
    /// <param name="streamPool">Optional stream pool for multi-stream execution.</param>
    public DeferredScope(IAsyncGpuBackend backend, GpuExecutionOptions options, GpuStreamPool? streamPool = null)
    {
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        Options = options ?? throw new ArgumentNullException(nameof(options));
        _streamPool = streamPool;
        GraphBuilder = new ExecutionGraphBuilder();
        _totalTimer = Stopwatch.StartNew();

        // Create the recording backend wrapper and start recording
        _recordingBackend = new RecordingGpuBackend(backend);
        _recordingBackend.BeginRecording(GraphBuilder);
    }

    /// <summary>
    /// Creates a deferred download that will execute during graph execution.
    /// </summary>
    /// <param name="buffer">The GPU buffer to download.</param>
    /// <param name="size">The number of elements to download.</param>
    /// <returns>A deferred download handle to retrieve the data after execution.</returns>
    /// <exception cref="InvalidOperationException">Thrown when called after execution.</exception>
    public DeferredDownload DownloadDeferred(IGpuBuffer buffer, int size)
    {
        ThrowIfDisposed();
        ThrowIfExecuted();

        var download = _recordingBackend.DownloadBufferDeferred(buffer, size);
        _deferredDownloads.Add(download);
        return download;
    }

    /// <summary>
    /// Creates a typed deferred download that will execute during graph execution.
    /// </summary>
    /// <typeparam name="T">The target element type for the downloaded data.</typeparam>
    /// <param name="buffer">The GPU buffer to download.</param>
    /// <param name="size">The number of elements to download.</param>
    /// <returns>A typed deferred download handle to retrieve the data after execution.</returns>
    /// <exception cref="InvalidOperationException">Thrown when called after execution.</exception>
    public DeferredDownload<T> DownloadDeferred<T>(IGpuBuffer buffer, int size)
    {
        var download = DownloadDeferred(buffer, size);
        return new DeferredDownload<T>(download);
    }

    /// <inheritdoc/>
    public void Execute()
    {
        ThrowIfDisposed();
        ThrowIfExecuted();

        // Stop recording before compiling and executing
        _recordingBackend.EndRecording();

        var compilationTimer = Stopwatch.StartNew();
        var graph = CompileInternal();
        compilationTimer.Stop();

        var executionTimer = Stopwatch.StartNew();

        if (_streamPool != null)
        {
            graph.Execute(_backend, _streamPool);
        }
        else
        {
            // Execute without stream pool - use default stream
            foreach (var node in graph.TopologicalOrder)
            {
                node.Execute(_backend);
            }
        }

        executionTimer.Stop();
        _totalTimer.Stop();

        // Mark all deferred downloads as executed
        foreach (var download in _deferredDownloads)
        {
            download.MarkExecuted();
        }

        RecordStatistics(graph, compilationTimer.Elapsed, executionTimer.Elapsed);
        IsExecuted = true;
    }

    /// <inheritdoc/>
    public async Task ExecuteAsync(CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        ThrowIfExecuted();

        // Stop recording before compiling and executing
        _recordingBackend.EndRecording();

        var compilationTimer = Stopwatch.StartNew();
        var graph = CompileInternal();
        compilationTimer.Stop();

        var executionTimer = Stopwatch.StartNew();

        if (_streamPool != null)
        {
            await graph.ExecuteAsync(_backend, _streamPool, cancellationToken);
        }
        else
        {
            // Execute synchronously without stream pool
            await Task.Run(() =>
            {
                foreach (var node in graph.TopologicalOrder)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    node.Execute(_backend);
                }
            }, cancellationToken);
        }

        executionTimer.Stop();
        _totalTimer.Stop();

        // Mark all deferred downloads as executed
        foreach (var download in _deferredDownloads)
        {
            download.MarkExecuted();
        }

        RecordStatistics(graph, compilationTimer.Elapsed, executionTimer.Elapsed);
        IsExecuted = true;
    }

    /// <inheritdoc/>
    public ExecutionGraph Compile()
    {
        ThrowIfDisposed();
        // Stop recording before compiling to ensure all operations are captured
        if (_recordingBackend.IsRecording)
        {
            _recordingBackend.EndRecording();
        }
        return CompileInternal();
    }

    /// <inheritdoc/>
    public DeferredScopeStatistics? GetStatistics()
    {
        return _statistics;
    }

    private ExecutionGraph CompileInternal()
    {
        if (_compiledGraph != null)
        {
            return _compiledGraph;
        }

        var (graph, statistics) = GraphBuilder.BuildOptimizedWithStatistics(Options, _backend);
        _compiledGraph = graph;
        _compiledStatistics = statistics;
        return _compiledGraph;
    }

    private void RecordStatistics(ExecutionGraph graph, TimeSpan compilationTime, TimeSpan executionTime)
    {
        // Use the actual optimization statistics from compilation instead of creating a new empty context
        var stats = _compiledStatistics ?? new OptimizationStatistics();

        _statistics = new DeferredScopeStatistics
        {
            OperationsRecorded = GraphBuilder.NodeCount,
            NodesAfterCompilation = graph.TopologicalOrder.Count,
            FusedOperations = stats.NodesFused,
            EliminatedOperations = stats.NodesEliminated,
            CompilationTime = compilationTime,
            ExecutionTime = executionTime,
            TotalTime = _totalTimer.Elapsed
        };
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(DeferredScope));
        }
    }

    private void ThrowIfExecuted()
    {
        if (IsExecuted)
        {
            throw new InvalidOperationException(
                "This deferred scope has already been executed. Create a new scope for additional operations.");
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;

        // If not executed, execute on dispose
        if (!IsExecuted && GraphBuilder.NodeCount > 0)
        {
            try
            {
                Execute();
            }
            catch
            {
                // Suppress exceptions during dispose
            }
        }

        GraphBuilder.Dispose();
    }
}

/// <summary>
/// Extension methods for creating deferred scopes.
/// </summary>
public static class DeferredScopeExtensions
{
    /// <summary>
    /// Begins a deferred execution scope for recording and optimizing GPU operations.
    /// </summary>
    /// <param name="backend">The async GPU backend.</param>
    /// <param name="options">Optional execution options (uses defaults if null).</param>
    /// <param name="streamPool">Optional stream pool for multi-stream execution.</param>
    /// <returns>A deferred scope for recording operations.</returns>
    /// <remarks>
    /// <para><b>Example:</b></para>
    /// <code>
    /// using (var scope = backend.BeginDeferredScope())
    /// {
    ///     // Record operations here...
    ///     scope.Execute();
    /// }
    /// </code>
    /// </remarks>
    public static IDeferredScope BeginDeferredScope(
        this IAsyncGpuBackend backend,
        GpuExecutionOptions? options = null,
        GpuStreamPool? streamPool = null)
    {
        var effectiveOptions = options ?? GpuExecutionOptions.FromEnvironment();
        return new DeferredScope(backend, effectiveOptions, streamPool);
    }

    /// <summary>
    /// Executes an action within a deferred scope and returns the result.
    /// </summary>
    /// <typeparam name="TResult">The result type.</typeparam>
    /// <param name="backend">The async GPU backend.</param>
    /// <param name="action">The action to execute within the scope.</param>
    /// <param name="options">Optional execution options.</param>
    /// <returns>The result of the action.</returns>
    public static TResult ExecuteDeferred<TResult>(
        this IAsyncGpuBackend backend,
        Func<IDeferredScope, TResult> action,
        GpuExecutionOptions? options = null)
    {
        using var scope = backend.BeginDeferredScope(options);
        var result = action(scope);
        scope.Execute();
        return result;
    }

    /// <summary>
    /// Executes an action within a deferred scope asynchronously.
    /// </summary>
    /// <typeparam name="TResult">The result type.</typeparam>
    /// <param name="backend">The async GPU backend.</param>
    /// <param name="action">The action to execute within the scope.</param>
    /// <param name="options">Optional execution options.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A task representing the async operation with the result.</returns>
    public static async Task<TResult> ExecuteDeferredAsync<TResult>(
        this IAsyncGpuBackend backend,
        Func<IDeferredScope, TResult> action,
        GpuExecutionOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        using var scope = backend.BeginDeferredScope(options);
        var result = action(scope);
        await scope.ExecuteAsync(cancellationToken);
        return result;
    }

    /// <summary>
    /// Executes a void action within a deferred scope.
    /// </summary>
    /// <param name="backend">The async GPU backend.</param>
    /// <param name="action">The action to execute within the scope.</param>
    /// <param name="options">Optional execution options.</param>
    public static void ExecuteDeferred(
        this IAsyncGpuBackend backend,
        Action<IDeferredScope> action,
        GpuExecutionOptions? options = null)
    {
        using var scope = backend.BeginDeferredScope(options);
        action(scope);
        scope.Execute();
    }

    /// <summary>
    /// Executes a void action within a deferred scope asynchronously.
    /// </summary>
    /// <param name="backend">The async GPU backend.</param>
    /// <param name="action">The action to execute within the scope.</param>
    /// <param name="options">Optional execution options.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A task representing the async operation.</returns>
    public static async Task ExecuteDeferredAsync(
        this IAsyncGpuBackend backend,
        Action<IDeferredScope> action,
        GpuExecutionOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        using var scope = backend.BeginDeferredScope(options);
        action(scope);
        await scope.ExecuteAsync(cancellationToken);
    }
}
