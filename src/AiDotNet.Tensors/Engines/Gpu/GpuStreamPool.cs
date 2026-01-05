using System.Collections.Concurrent;

namespace AiDotNet.Tensors.Engines.Gpu;

/// <summary>
/// Manages a pool of GPU streams for efficient compute/transfer overlap.
/// Streams are created lazily and reused to minimize creation overhead.
/// </summary>
/// <remarks>
/// <para><b>Usage Pattern:</b></para>
/// <code>
/// using var streamPool = new GpuStreamPool(backend, options);
///
/// // Acquire a compute stream
/// var computeStream = streamPool.AcquireStream(GpuStreamType.Compute);
/// try
/// {
///     backend.GemmAsync(a, b, c, m, n, k, 1f, 0f, computeStream);
/// }
/// finally
/// {
///     streamPool.ReleaseStream(computeStream);
/// }
/// </code>
/// </remarks>
public sealed class GpuStreamPool : IDisposable
{
    private readonly IAsyncGpuBackend _backend;
    private readonly GpuExecutionOptions _options;
    private readonly ConcurrentDictionary<GpuStreamType, ConcurrentBag<IGpuStream>> _availableStreams;
    private readonly ConcurrentDictionary<IGpuStream, GpuStreamType> _streamTypes;
    private readonly object _creationLock = new();
    private int _totalComputeStreams;
    private int _totalTransferStreams;
    private bool _disposed;

    /// <summary>
    /// Gets the default compute stream.
    /// </summary>
    public IGpuStream DefaultComputeStream { get; }

    /// <summary>
    /// Gets the default host-to-device transfer stream.
    /// </summary>
    public IGpuStream DefaultH2DStream { get; }

    /// <summary>
    /// Gets the default device-to-host transfer stream.
    /// </summary>
    public IGpuStream DefaultD2HStream { get; }

    /// <summary>
    /// Gets the total number of streams created.
    /// </summary>
    public int TotalStreamsCreated => _totalComputeStreams + _totalTransferStreams;

    /// <summary>
    /// Gets the backend associated with this pool.
    /// </summary>
    public IAsyncGpuBackend Backend => _backend;

    /// <summary>
    /// Creates a new stream pool for the given backend.
    /// </summary>
    /// <param name="backend">The async GPU backend.</param>
    /// <param name="options">Execution options controlling stream behavior.</param>
    public GpuStreamPool(IAsyncGpuBackend backend, GpuExecutionOptions? options = null)
    {
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        _options = options ?? new GpuExecutionOptions();

        _availableStreams = new ConcurrentDictionary<GpuStreamType, ConcurrentBag<IGpuStream>>();
        _streamTypes = new ConcurrentDictionary<IGpuStream, GpuStreamType>();

        // Initialize stream bags for each type
        foreach (GpuStreamType streamType in Enum.GetValues(typeof(GpuStreamType)))
        {
            _availableStreams[streamType] = new ConcurrentBag<IGpuStream>();
        }

        // Create default streams
        DefaultComputeStream = CreateStreamInternal(GpuStreamType.Compute, priority: 0);
        DefaultH2DStream = CreateStreamInternal(GpuStreamType.HostToDevice, priority: 0);
        DefaultD2HStream = CreateStreamInternal(GpuStreamType.DeviceToHost, priority: 0);
    }

    /// <summary>
    /// Acquires a stream of the specified type from the pool.
    /// Creates a new stream if none are available and limits haven't been reached.
    /// </summary>
    /// <param name="streamType">The type of stream to acquire.</param>
    /// <returns>A GPU stream ready for use.</returns>
    /// <exception cref="InvalidOperationException">Thrown if max streams reached and none available.</exception>
    public IGpuStream AcquireStream(GpuStreamType streamType)
    {
        ThrowIfDisposed();

        // Try to get an existing stream from the pool
        if (_availableStreams.TryGetValue(streamType, out var bag) && bag.TryTake(out var stream))
        {
            return stream;
        }

        // Check if we can create more streams
        lock (_creationLock)
        {
            int maxStreams = GetMaxStreamsForType(streamType);
            int currentCount = GetCurrentStreamCount(streamType);

            if (currentCount >= maxStreams)
            {
                // Try one more time - someone might have released a stream
                if (bag != null && bag.TryTake(out stream))
                {
                    return stream;
                }

                throw new InvalidOperationException(
                    $"Maximum number of {streamType} streams ({maxStreams}) reached. " +
                    "Increase MaxComputeStreams or TransferStreams in GpuExecutionOptions.");
            }

            // Create a new stream
            return CreateStreamInternal(streamType, priority: currentCount);
        }
    }

    /// <summary>
    /// Releases a stream back to the pool for reuse.
    /// </summary>
    /// <param name="stream">The stream to release.</param>
    public void ReleaseStream(IGpuStream stream)
    {
        if (_disposed || stream == null)
        {
            return;
        }

        // Don't release default streams back to the pool
        if (stream == DefaultComputeStream || stream == DefaultH2DStream || stream == DefaultD2HStream)
        {
            return;
        }

        if (_streamTypes.TryGetValue(stream, out var streamType))
        {
            if (_availableStreams.TryGetValue(streamType, out var bag))
            {
                bag.Add(stream);
            }
        }
    }

    /// <summary>
    /// Gets a stream appropriate for the given operation.
    /// For compute operations, returns the default compute stream.
    /// For transfers, returns the appropriate default transfer stream.
    /// </summary>
    /// <param name="isH2DTransfer">True for host-to-device transfer.</param>
    /// <param name="isD2HTransfer">True for device-to-host transfer.</param>
    /// <returns>An appropriate stream for the operation. Callers do NOT need to release these streams.</returns>
    /// <remarks>
    /// This method returns default streams which don't need to be released.
    /// For explicit stream acquisition with release semantics, use AcquireStream/ReleaseStream.
    /// </remarks>
    public IGpuStream GetStreamForOperation(bool isH2DTransfer = false, bool isD2HTransfer = false)
    {
        if (isH2DTransfer)
        {
            // Return default transfer stream - no release needed
            return _options.EnableComputeTransferOverlap
                ? DefaultH2DStream
                : DefaultComputeStream;
        }

        if (isD2HTransfer)
        {
            // Return default transfer stream - no release needed
            return _options.EnableComputeTransferOverlap
                ? DefaultD2HStream
                : DefaultComputeStream;
        }

        // Return default compute stream - no release needed
        return DefaultComputeStream;
    }

    /// <summary>
    /// Synchronizes all streams in the pool.
    /// </summary>
    public void SynchronizeAll()
    {
        ThrowIfDisposed();

        // Synchronize default streams
        _backend.SynchronizeStream(DefaultComputeStream);
        _backend.SynchronizeStream(DefaultH2DStream);
        _backend.SynchronizeStream(DefaultD2HStream);

        // Synchronize all pooled streams
        foreach (var bag in _availableStreams.Values)
        {
            foreach (var stream in bag)
            {
                _backend.SynchronizeStream(stream);
            }
        }
    }

    /// <summary>
    /// Waits for all streams to complete their current operations.
    /// </summary>
    public void WaitAllComplete()
    {
        SynchronizeAll();
    }

    private IGpuStream CreateStreamInternal(GpuStreamType streamType, int priority)
    {
        var stream = _backend.CreateStream(streamType, priority);
        _streamTypes[stream] = streamType;

        if (streamType == GpuStreamType.Compute)
        {
            Interlocked.Increment(ref _totalComputeStreams);
        }
        else
        {
            Interlocked.Increment(ref _totalTransferStreams);
        }

        return stream;
    }

    private int GetMaxStreamsForType(GpuStreamType streamType)
    {
        return streamType switch
        {
            GpuStreamType.Compute => _options.MaxComputeStreams,
            GpuStreamType.HostToDevice or GpuStreamType.DeviceToHost => _options.TransferStreams,
            GpuStreamType.DeviceToDevice => 2, // Usually just 2 for D2D
            _ => 1
        };
    }

    private int GetCurrentStreamCount(GpuStreamType streamType)
    {
        return streamType switch
        {
            GpuStreamType.Compute => _totalComputeStreams,
            _ => _totalTransferStreams
        };
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(GpuStreamPool));
        }
    }

    /// <summary>
    /// Disposes all streams in the pool.
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;

        // Synchronize before disposing
        try
        {
            SynchronizeAll();
        }
        catch
        {
            // Ignore synchronization errors during disposal
        }

        // Dispose all streams
        foreach (var bag in _availableStreams.Values)
        {
            while (bag.TryTake(out var stream))
            {
                stream.Dispose();
            }
        }

        // Dispose default streams
        DefaultComputeStream.Dispose();
        DefaultH2DStream.Dispose();
        DefaultD2HStream.Dispose();

        _availableStreams.Clear();
        _streamTypes.Clear();
    }
}

/// <summary>
/// Extension methods for stream pool usage patterns.
/// </summary>
public static class GpuStreamPoolExtensions
{
    /// <summary>
    /// Executes an action with an acquired stream, automatically releasing it afterward.
    /// </summary>
    /// <param name="pool">The stream pool.</param>
    /// <param name="streamType">The type of stream to acquire.</param>
    /// <param name="action">The action to execute with the stream.</param>
    public static void WithStream(this GpuStreamPool pool, GpuStreamType streamType, Action<IGpuStream> action)
    {
        var stream = pool.AcquireStream(streamType);
        try
        {
            action(stream);
        }
        finally
        {
            pool.ReleaseStream(stream);
        }
    }

    /// <summary>
    /// Executes a function with an acquired stream, automatically releasing it afterward.
    /// </summary>
    /// <typeparam name="T">The return type.</typeparam>
    /// <param name="pool">The stream pool.</param>
    /// <param name="streamType">The type of stream to acquire.</param>
    /// <param name="func">The function to execute with the stream.</param>
    /// <returns>The function result.</returns>
    public static T WithStream<T>(this GpuStreamPool pool, GpuStreamType streamType, Func<IGpuStream, T> func)
    {
        var stream = pool.AcquireStream(streamType);
        try
        {
            return func(stream);
        }
        finally
        {
            pool.ReleaseStream(stream);
        }
    }
}
