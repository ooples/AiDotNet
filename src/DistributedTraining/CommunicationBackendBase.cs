using System.Collections.Concurrent;
using System.Diagnostics;
using System.Runtime.ExceptionServices;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.LinearAlgebra;


namespace AiDotNet.DistributedTraining;

/// <summary>
/// Provides base implementation for distributed communication backends.
/// </summary>
/// <remarks>
/// <para>
/// This abstract class implements common functionality for all communication backends,
/// including state management, validation, and helper methods for collective operations.
/// Derived classes implement the specific communication mechanisms (MPI, NCCL, shared-memory, in-memory, etc.).
/// </para>
/// <para><b>For Beginners:</b> This is the foundation that all communication systems build upon.
///
/// Think of this as a template that defines how any communication system should work.
/// It handles common tasks like:
/// - Keeping track of whether the system is initialized
/// - Validating inputs (checking for null values, correct sizes, etc.)
/// - Providing helper methods and background FIFO queuing for overlapping communication with computation
///
/// Specific communication backends (like MPI, Gloo, shared-memory, or in-memory) inherit from this and add
/// their own implementation details. This prevents code duplication and ensures
/// all backends work consistently.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for operations</typeparam>
public abstract class CommunicationBackendBase<T> : IAsyncCommunicationBackend<T>
{
    /// <summary>
    /// Provides numeric operations for type T.
    /// </summary>
    protected readonly INumericOperations<T> NumOps;

    private bool _isInitialized;
    private long _totalCollectiveStopwatchTicks;
    private long _collectiveCount;
    private readonly object _asyncWorkerLock = new object();
    private BlockingCollection<Action>? _asyncQueue;
    private Thread? _asyncWorkerThread;
    private int _asyncPendingCount;

    /// <inheritdoc/>
    public abstract int Rank { get; }

    /// <inheritdoc/>
    public abstract int WorldSize { get; }

    /// <inheritdoc/>
    public bool IsInitialized => _isInitialized;

    /// <summary>
    /// Initializes a new instance of the CommunicationBackendBase class.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This constructor sets up the numeric operations provider that will be used
    /// for all mathematical operations on type T.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor is called when creating any communication backend.
    ///
    /// It sets up the math helper that allows the backend to perform operations
    /// like addition, multiplication, and comparison on any numeric type (double, float, etc.)
    /// without knowing in advance which type will be used.
    /// </para>
    /// </remarks>
    protected CommunicationBackendBase()
    {
        NumOps = MathHelper.GetNumericOperations<T>();
        _isInitialized = false;
    }

    /// <inheritdoc/>
    public virtual void Initialize()
    {
        if (_isInitialized)
        {
            return;
        }

        OnInitialize();
        _isInitialized = true;
    }

    /// <inheritdoc/>
    public virtual void Shutdown()
    {
        if (!_isInitialized)
        {
            return;
        }

        StopAsyncWorker();
        OnShutdown();
        _isInitialized = false;
    }

    /// <inheritdoc/>
    public double TotalCollectiveMilliseconds
        => Interlocked.Read(ref _totalCollectiveStopwatchTicks) * 1000.0 / Stopwatch.Frequency;

    /// <inheritdoc/>
    public long CollectiveCount => Interlocked.Read(ref _collectiveCount);

    /// <inheritdoc/>
    public virtual void ResetDiagnostics()
    {
        Interlocked.Exchange(ref _totalCollectiveStopwatchTicks, 0L);
        Interlocked.Exchange(ref _collectiveCount, 0L);
    }

    /// <summary>
    /// Records raw <see cref="Stopwatch"/> ticks spent in a transport collective operation.
    /// </summary>
    /// <param name="elapsedStopwatchTicks">Raw elapsed ticks from <see cref="Stopwatch.GetTimestamp"/>.</param>
    protected void RecordCollectiveDurationTicks(long elapsedStopwatchTicks)
    {
        if (elapsedStopwatchTicks > 0)
        {
            Interlocked.Add(ref _totalCollectiveStopwatchTicks, elapsedStopwatchTicks);
        }

        Interlocked.Increment(ref _collectiveCount);
    }

    /// <inheritdoc/>
    public virtual IAsyncCollectiveHandle AllReduceAsync(Vector<T> data, ReductionOperation operation)
    {
        EnsureInitialized();
        ValidateData(data, nameof(data));

        if (WorldSize <= 1)
        {
            return CompletedCollectiveResultHandle.Completed(data);
        }

        return EnqueueAsyncCollective(() =>
        {
            AllReduce(data, operation);
            return data;
        });
    }

    /// <inheritdoc/>
    public virtual IAsyncCollectiveResultHandle<T> ReduceScatterAsync(Vector<T> data, ReductionOperation operation)
    {
        EnsureInitialized();
        ValidateData(data, nameof(data));

        if (WorldSize <= 1)
        {
            var copy = data.Clone();
            return CompletedCollectiveResultHandle.Completed(copy);
        }

        return EnqueueAsyncCollective(() => ReduceScatter(data, operation));
    }

    /// <inheritdoc/>
    public virtual IAsyncCollectiveResultHandle<T> AllGatherAsync(Vector<T> sendData)
    {
        EnsureInitialized();
        ValidateData(sendData, nameof(sendData));

        if (WorldSize <= 1)
        {
            var copy = sendData.Clone();
            return CompletedCollectiveResultHandle.Completed(copy);
        }

        return EnqueueAsyncCollective(() => AllGather(sendData));
    }

    private IAsyncCollectiveResultHandle<T> EnqueueAsyncCollective(Func<Vector<T>> work)
    {
        var handle = new QueuedCollectiveHandle(work);
        var queue = EnsureAsyncWorkerStarted();
        Interlocked.Increment(ref _asyncPendingCount);
        try
        {
            queue.Add(() =>
            {
                try
                {
                    handle.Execute();
                }
                finally
                {
                    Interlocked.Decrement(ref _asyncPendingCount);
                }
            });
        }
        catch
        {
            Interlocked.Decrement(ref _asyncPendingCount);
            throw;
        }

        return handle;
    }

    /// <summary>
    /// Ensures any previously enqueued asynchronous collective operations on this backend have completed before
    /// executing a synchronous collective on a caller thread. Re-entrant calls from the worker thread itself
    /// return immediately.
    /// </summary>
    protected void DrainAsyncCollectives()
    {
        if (Volatile.Read(ref _asyncPendingCount) == 0)
        {
            return;
        }

        BlockingCollection<Action>? queue;
        Thread? worker;
        lock (_asyncWorkerLock)
        {
            queue = _asyncQueue;
            worker = _asyncWorkerThread;
        }

        if (queue is null || worker is null || Thread.CurrentThread == worker)
        {
            return;
        }

        using var barrier = new ManualResetEventSlim(false);
        try
        {
            queue.Add(() => barrier.Set());
            barrier.Wait();
        }
        catch (InvalidOperationException)
        {
            // Queue completed or disposed during shutdown.
        }
    }

    private BlockingCollection<Action> EnsureAsyncWorkerStarted()
    {
        lock (_asyncWorkerLock)
        {
            if (_asyncQueue is not null && !_asyncQueue.IsAddingCompleted)
            {
                return _asyncQueue;
            }

            var queue = new BlockingCollection<Action>();
            var worker = new Thread(() => RunAsyncWorkerLoop(queue))
            {
                IsBackground = true,
                Name = $"AiDotNet-CollectiveWorker-Rank{Rank}",
            };
            _asyncQueue = queue;
            _asyncWorkerThread = worker;
            worker.Start();
            return queue;
        }
    }

    private static void RunAsyncWorkerLoop(BlockingCollection<Action> queue)
    {
        try
        {
            foreach (var item in queue.GetConsumingEnumerable())
            {
                item();
            }
        }
        catch (ObjectDisposedException)
        {
            // Normal teardown race when Dispose happens after GetConsumingEnumerable exits.
        }
    }

    private void StopAsyncWorker()
    {
        BlockingCollection<Action>? queue;
        Thread? worker;
        lock (_asyncWorkerLock)
        {
            queue = _asyncQueue;
            worker = _asyncWorkerThread;
            _asyncQueue = null;
            _asyncWorkerThread = null;
        }

        if (queue is null)
        {
            return;
        }

        try
        {
            queue.CompleteAdding();
        }
        catch (ObjectDisposedException)
        {
        }

        if (worker is not null && worker.IsAlive && Thread.CurrentThread != worker)
        {
            worker.Join(millisecondsTimeout: 10_000);
        }

        queue.Dispose();
    }

    private sealed class CompletedCollectiveResultHandle : IAsyncCollectiveResultHandle<T>
    {
        private readonly Vector<T> _result;

        private CompletedCollectiveResultHandle(Vector<T> result)
        {
            _result = result;
        }

        public static CompletedCollectiveResultHandle Completed(Vector<T> result) => new CompletedCollectiveResultHandle(result);

        public bool IsCompleted => true;

        public TimeSpan Elapsed => TimeSpan.Zero;

        public void Wait()
        {
        }

        public Task WaitAsync(CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            return Task.CompletedTask;
        }

        public Vector<T> GetResult() => _result;

        public void Dispose()
        {
        }
    }

    private sealed class QueuedCollectiveHandle : IAsyncCollectiveResultHandle<T>
    {
        private readonly Func<Vector<T>> _work;
        private readonly TaskCompletionSource<Vector<T>> _tcs =
            new TaskCompletionSource<Vector<T>>(TaskCreationOptions.RunContinuationsAsynchronously);
        private long _elapsedStopwatchTicks;

        public QueuedCollectiveHandle(Func<Vector<T>> work)
        {
            _work = work;
        }

        public bool IsCompleted => _tcs.Task.IsCompleted;

        public TimeSpan Elapsed
        {
            get
            {
                long ticks = Interlocked.Read(ref _elapsedStopwatchTicks);
                return TimeSpan.FromSeconds((double)ticks / Stopwatch.Frequency);
            }
        }

        public void Execute()
        {
            long start = Stopwatch.GetTimestamp();
            try
            {
                var result = _work();
                Interlocked.Exchange(ref _elapsedStopwatchTicks, Stopwatch.GetTimestamp() - start);
                _tcs.TrySetResult(result);
            }
            catch (Exception ex)
            {
                Interlocked.Exchange(ref _elapsedStopwatchTicks, Stopwatch.GetTimestamp() - start);
                _tcs.TrySetException(ex);
            }
        }

        public void Wait()
        {
            try
            {
                _tcs.Task.GetAwaiter().GetResult();
            }
            catch (Exception ex)
            {
                ExceptionDispatchInfo.Capture(ex).Throw();
            }
        }

        public async Task WaitAsync(CancellationToken cancellationToken = default)
        {
            if (cancellationToken.CanBeCanceled)
            {
                var cancelTcs = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
                using (cancellationToken.Register(static s => ((TaskCompletionSource<bool>)s!).TrySetResult(true), cancelTcs))
                {
                    var completed = await Task.WhenAny(_tcs.Task, cancelTcs.Task).ConfigureAwait(false);
                    if (completed == cancelTcs.Task)
                    {
                        cancellationToken.ThrowIfCancellationRequested();
                    }
                }
            }

            await _tcs.Task.ConfigureAwait(false);
        }

        public Vector<T> GetResult()
        {
            Wait();
            return _tcs.Task.GetAwaiter().GetResult();
        }

        public void Dispose()
        {
        }
    }

    /// <summary>
    /// Called during initialization to perform backend-specific setup.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Derived classes override this method to implement their specific initialization logic,
    /// such as connecting to MPI or setting up shared memory structures.
    /// </para>
    /// <para><b>For Beginners:</b> This is where each specific backend does its setup work.
    ///
    /// For example:
    /// - An MPI backend would connect to the MPI environment
    /// - An in-memory backend would create shared data structures
    /// - An NCCL backend would initialize GPU communication channels
    /// </para>
    /// </remarks>
    protected virtual void OnInitialize()
    {
        // Base implementation does nothing - derived classes override
    }

    /// <summary>
    /// Called during shutdown to perform backend-specific cleanup.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Derived classes override this method to implement their specific cleanup logic,
    /// such as disconnecting from MPI or releasing shared memory.
    /// </para>
    /// <para><b>For Beginners:</b> This is where each backend cleans up its resources.
    ///
    /// It's like turning off equipment when you're done - releasing memory,
    /// closing connections, and ensuring everything shuts down cleanly.
    /// </para>
    /// </remarks>
    protected virtual void OnShutdown()
    {
        // Base implementation does nothing - derived classes override
    }

    /// <inheritdoc/>
    public abstract void Barrier();

    /// <inheritdoc/>
    public abstract void AllReduce(Vector<T> data, ReductionOperation operation);

    /// <inheritdoc/>
    public abstract Vector<T> AllGather(Vector<T> sendData);

    /// <inheritdoc/>
    public abstract Vector<T> Broadcast(Vector<T> data, int root = 0);

    /// <inheritdoc/>
    public abstract Vector<T> Scatter(Vector<T> sendData, int root = 0);

    /// <inheritdoc/>
    public abstract Vector<T> ReduceScatter(Vector<T> data, ReductionOperation operation);

    /// <inheritdoc/>
    public abstract void Send(Vector<T> data, int destinationRank, int tag = 0);

    /// <inheritdoc/>
    public abstract Vector<T> Receive(int sourceRank, int count, int tag = 0);

    /// <summary>
    /// Ensures the backend is initialized before performing operations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method throws an exception if the backend has not been initialized.
    /// All communication operations should call this before proceeding.
    /// </para>
    /// <para><b>For Beginners:</b> This is a safety check.
    ///
    /// Before doing any communication, we make sure the system is ready.
    /// If someone tries to use the backend without initializing it first,
    /// this method will throw an error with a helpful message.
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">Thrown if the backend is not initialized</exception>
    protected void EnsureInitialized()
    {
        if (!_isInitialized)
        {
            throw new InvalidOperationException(
                "Communication backend is not initialized. Call Initialize() first.");
        }
    }

    /// <summary>
    /// Validates that a root rank is within valid bounds.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method ensures that the specified root rank is a valid process ID
    /// (between 0 and WorldSize - 1).
    /// </para>
    /// <para><b>For Beginners:</b> When one process acts as the "root" or "leader",
    /// we need to make sure that process actually exists.
    ///
    /// For example, if you have 4 processes (ranks 0-3), specifying rank 5 as the
    /// root would be an error. This method catches such mistakes.
    /// </para>
    /// </remarks>
    /// <param name="root">The root rank to validate</param>
    /// <exception cref="ArgumentException">Thrown if root is out of bounds</exception>
    protected void ValidateRoot(int root)
    {
        if (root < 0 || root >= WorldSize)
        {
            throw new ArgumentException(
                $"Invalid root {root}. Must be between 0 and {WorldSize - 1}.",
                nameof(root));
        }
    }

    /// <summary>
    /// Validates that a rank is within valid bounds.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method ensures that the specified rank is a valid process ID
    /// (between 0 and WorldSize - 1) and different from the current process rank.
    /// </para>
    /// <para><b>For Beginners:</b> When sending or receiving from another process,
    /// we need to make sure:
    /// 1. That process actually exists (valid rank number)
    /// 2. We're not trying to send/receive from ourselves (rank != Rank)
    ///
    /// For example, if you have 4 processes (ranks 0-3), rank 5 would be invalid.
    /// Also, a process shouldn't send to itself.
    /// </para>
    /// </remarks>
    /// <param name="rank">The rank to validate</param>
    /// <param name="paramName">The parameter name for error messages</param>
    /// <exception cref="ArgumentException">Thrown if rank is out of bounds or equals current rank</exception>
    protected void ValidateRank(int rank, string paramName)
    {
        if (rank < 0 || rank >= WorldSize)
        {
            throw new ArgumentException(
                $"Invalid rank {rank}. Must be between 0 and {WorldSize - 1}.",
                paramName);
        }

        if (rank == Rank)
        {
            throw new ArgumentException(
                $"Cannot send/receive to/from self (rank {rank}).",
                paramName);
        }
    }

    /// <summary>
    /// Validates that data is not null.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method ensures that the data vector is not null before attempting
    /// communication operations.
    /// </para>
    /// <para><b>For Beginners:</b> This is a basic safety check to make sure
    /// we're not trying to send or receive null data, which would cause errors.
    /// </para>
    /// </remarks>
    /// <param name="data">The data to validate</param>
    /// <param name="paramName">The parameter name for error messages</param>
    /// <exception cref="ArgumentNullException">Thrown if data is null</exception>
    protected void ValidateData(Vector<T>? data, string paramName)
    {
        if (data == null)
        {
            throw new ArgumentNullException(paramName, "Data cannot be null.");
        }
    }

    /// <summary>
    /// Applies a reduction operation to two values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This helper method performs the specified reduction operation (Sum, Product, Min, Max)
    /// on two values. It's used internally by AllReduce and ReduceScatter implementations.
    /// </para>
    /// <para><b>For Beginners:</b> This is a helper that knows how to combine two numbers
    /// in different ways.
    ///
    /// For example:
    /// - Sum operation: 3 + 5 = 8
    /// - Product operation: 3 * 5 = 15
    /// - Min operation: Min(3, 5) = 3
    /// - Max operation: Max(3, 5) = 5
    ///
    /// We use this when combining values from multiple processes.
    /// </para>
    /// </remarks>
    /// <param name="a">The first value</param>
    /// <param name="b">The second value</param>
    /// <param name="operation">The reduction operation to apply</param>
    /// <returns>The result of applying the operation</returns>
    protected T ApplyReductionOperation(T a, T b, ReductionOperation operation)
    {
        return operation switch
        {
            ReductionOperation.Sum or ReductionOperation.Average => NumOps.Add(a, b),
            ReductionOperation.Product => NumOps.Multiply(a, b),
            ReductionOperation.Min => NumOps.LessThan(a, b) ? a : b,
            ReductionOperation.Max => NumOps.GreaterThan(a, b) ? a : b,
            _ => throw new NotSupportedException($"Operation {operation} is not supported.")
        };
    }
}
