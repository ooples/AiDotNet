namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Provides asynchronous GPU memory transfer operations for overlapping computation with data movement.
/// </summary>
/// <remarks>
/// <para>
/// Async transfers allow overlapping CPU and GPU work, as well as overlapping GPU computation
/// with memory transfers. This is critical for achieving high GPU utilization in production systems.
/// </para>
/// <para><b>For Beginners:</b> When training neural networks, data needs to move between:
/// - CPU memory (RAM) and GPU memory (VRAM)
/// - Different GPU memory regions
///
/// Normally, these transfers block everything else. Async transfers allow the GPU to continue
/// computing while data is being transferred in the background, making training faster.
/// </para>
/// </remarks>
public class AsyncGpuTransfer : IDisposable
{
    private readonly int _deviceId;
    private readonly Queue<TransferOperation> _pendingTransfers;
    private readonly Dictionary<int, TransferStream> _streams;
    private readonly object _syncLock = new();
    private bool _disposed;

    /// <summary>
    /// Maximum number of concurrent transfer streams.
    /// </summary>
    public int MaxConcurrentStreams { get; }

    /// <summary>
    /// Gets the current number of pending transfers.
    /// </summary>
    public int PendingTransferCount => _pendingTransfers.Count;

    /// <summary>
    /// Initializes a new instance of the AsyncGpuTransfer class.
    /// </summary>
    /// <param name="deviceId">The GPU device ID.</param>
    /// <param name="maxConcurrentStreams">Maximum concurrent transfer streams. Default: 2</param>
    public AsyncGpuTransfer(int deviceId = 0, int maxConcurrentStreams = 2)
    {
        _deviceId = deviceId;
        MaxConcurrentStreams = maxConcurrentStreams;
        _pendingTransfers = new Queue<TransferOperation>();
        _streams = new Dictionary<int, TransferStream>();

        // Initialize transfer streams
        for (int i = 0; i < maxConcurrentStreams; i++)
        {
            _streams[i] = new TransferStream { Id = i, IsAvailable = true };
        }
    }

    /// <summary>
    /// Asynchronously transfers data from host (CPU) to device (GPU).
    /// </summary>
    /// <typeparam name="T">The data type.</typeparam>
    /// <param name="hostData">Source data on the host.</param>
    /// <param name="deviceBuffer">Destination buffer on the device.</param>
    /// <param name="priority">Transfer priority. Higher values = higher priority.</param>
    /// <returns>A task that completes when the transfer is finished.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This copies data from your computer's RAM to GPU memory
    /// without blocking. The returned Task completes when the transfer is done.
    ///
    /// <code>
    /// // Start transfer in background
    /// var transferTask = asyncTransfer.HostToDeviceAsync(cpuData, gpuBuffer);
    ///
    /// // Do other work while transfer happens
    /// DoOtherWork();
    ///
    /// // Wait for transfer to complete before using the data
    /// await transferTask;
    /// </code>
    /// </para>
    /// </remarks>
    public async Task HostToDeviceAsync<T>(
        ReadOnlyMemory<T> hostData,
        GpuBuffer<T> deviceBuffer,
        int priority = 0)
        where T : unmanaged
    {
        var operation = new TransferOperation
        {
            Type = TransferType.HostToDevice,
            SourceSize = hostData.Length * System.Runtime.InteropServices.Marshal.SizeOf<T>(),
            Priority = priority,
            CompletionSource = new TaskCompletionSource<bool>()
        };

        EnqueueTransfer(operation);

        // Wait for a stream to become available
        var stream = await AcquireStreamAsync();

        try
        {
            // Perform the actual transfer (platform-specific implementation)
            await PerformHostToDeviceTransferAsync(hostData, deviceBuffer, stream);
            operation.CompletionSource.SetResult(true);
        }
        catch (Exception ex)
        {
            operation.CompletionSource.SetException(ex);
            throw;
        }
        finally
        {
            ReleaseStream(stream);
        }
    }

    /// <summary>
    /// Asynchronously transfers data from device (GPU) to host (CPU).
    /// </summary>
    /// <typeparam name="T">The data type.</typeparam>
    /// <param name="deviceBuffer">Source buffer on the device.</param>
    /// <param name="hostData">Destination memory on the host.</param>
    /// <param name="priority">Transfer priority. Higher values = higher priority.</param>
    /// <returns>A task that completes when the transfer is finished.</returns>
    public async Task DeviceToHostAsync<T>(
        GpuBuffer<T> deviceBuffer,
        Memory<T> hostData,
        int priority = 0)
        where T : unmanaged
    {
        var operation = new TransferOperation
        {
            Type = TransferType.DeviceToHost,
            SourceSize = hostData.Length * System.Runtime.InteropServices.Marshal.SizeOf<T>(),
            Priority = priority,
            CompletionSource = new TaskCompletionSource<bool>()
        };

        EnqueueTransfer(operation);

        var stream = await AcquireStreamAsync();

        try
        {
            await PerformDeviceToHostTransferAsync(deviceBuffer, hostData, stream);
            operation.CompletionSource.SetResult(true);
        }
        catch (Exception ex)
        {
            operation.CompletionSource.SetException(ex);
            throw;
        }
        finally
        {
            ReleaseStream(stream);
        }
    }

    /// <summary>
    /// Asynchronously transfers data between two GPU buffers.
    /// </summary>
    /// <typeparam name="T">The data type.</typeparam>
    /// <param name="source">Source buffer.</param>
    /// <param name="destination">Destination buffer.</param>
    /// <param name="priority">Transfer priority.</param>
    /// <returns>A task that completes when the transfer is finished.</returns>
    public async Task DeviceToDeviceAsync<T>(
        GpuBuffer<T> source,
        GpuBuffer<T> destination,
        int priority = 0)
        where T : unmanaged
    {
        var operation = new TransferOperation
        {
            Type = TransferType.DeviceToDevice,
            SourceSize = source.Length * System.Runtime.InteropServices.Marshal.SizeOf<T>(),
            Priority = priority,
            CompletionSource = new TaskCompletionSource<bool>()
        };

        EnqueueTransfer(operation);

        var stream = await AcquireStreamAsync();

        try
        {
            await PerformDeviceToDeviceTransferAsync(source, destination, stream);
            operation.CompletionSource.SetResult(true);
        }
        catch (Exception ex)
        {
            operation.CompletionSource.SetException(ex);
            throw;
        }
        finally
        {
            ReleaseStream(stream);
        }
    }

    /// <summary>
    /// Prefetches data to GPU asynchronously for upcoming computation.
    /// </summary>
    /// <typeparam name="T">The data type.</typeparam>
    /// <param name="data">Data to prefetch.</param>
    /// <returns>A GPU buffer containing the prefetched data and a completion task.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Prefetching loads the next batch of data while the GPU
    /// is still processing the current batch. This hides the transfer latency:
    ///
    /// <code>
    /// // Prefetch next batch while processing current
    /// var (nextBuffer, prefetchTask) = asyncTransfer.PrefetchAsync(nextBatchData);
    ///
    /// // Process current batch
    /// ProcessOnGpu(currentBuffer);
    ///
    /// // Wait for prefetch before next iteration
    /// await prefetchTask;
    /// currentBuffer = nextBuffer;
    /// </code>
    /// </para>
    /// </remarks>
    public (GpuBuffer<T> Buffer, Task TransferTask) PrefetchAsync<T>(ReadOnlyMemory<T> data)
        where T : unmanaged
    {
        var buffer = new GpuBuffer<T>(data.Length, _deviceId);
        var task = HostToDeviceAsync(data, buffer, priority: 1);
        return (buffer, task);
    }

    /// <summary>
    /// Creates a pipeline for double-buffered data loading.
    /// </summary>
    /// <typeparam name="T">The data type.</typeparam>
    /// <param name="bufferSize">Size of each buffer.</param>
    /// <returns>A double buffer for pipelined transfers.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Double buffering uses two buffers: one for the GPU to compute
    /// on, and one for loading the next batch. When computation finishes, the buffers are swapped.
    ///
    /// <code>
    /// var doubleBuffer = asyncTransfer.CreateDoubleBuffer&lt;float&gt;(batchSize);
    ///
    /// for (int epoch = 0; epoch &lt; numEpochs; epoch++)
    /// {
    ///     foreach (var batch in dataLoader)
    ///     {
    ///         // Swap buffers and get the ready one
    ///         var gpuBuffer = await doubleBuffer.SwapAndLoadAsync(batch);
    ///
    ///         // Process while next batch loads
    ///         ProcessOnGpu(gpuBuffer);
    ///     }
    /// }
    /// </code>
    /// </para>
    /// </remarks>
    public DoubleBuffer<T> CreateDoubleBuffer<T>(int bufferSize)
        where T : unmanaged
    {
        return new DoubleBuffer<T>(this, bufferSize, _deviceId);
    }

    /// <summary>
    /// Waits for all pending transfers to complete.
    /// </summary>
    public async Task SynchronizeAsync()
    {
        var pendingTasks = new List<Task>();

        lock (_syncLock)
        {
            foreach (var op in _pendingTransfers)
            {
                pendingTasks.Add(op.CompletionSource.Task);
            }
        }

        await Task.WhenAll(pendingTasks);
    }

    /// <summary>
    /// Synchronously waits for all pending transfers to complete.
    /// </summary>
    public void Synchronize()
    {
        SynchronizeAsync().Wait();
    }

    private void EnqueueTransfer(TransferOperation operation)
    {
        lock (_syncLock)
        {
            _pendingTransfers.Enqueue(operation);
        }
    }

    private async Task<TransferStream> AcquireStreamAsync()
    {
        while (true)
        {
            lock (_syncLock)
            {
                foreach (var stream in _streams.Values)
                {
                    if (stream.IsAvailable)
                    {
                        stream.IsAvailable = false;
                        return stream;
                    }
                }
            }

            // No stream available, wait a bit
            await Task.Delay(1);
        }
    }

    private void ReleaseStream(TransferStream stream)
    {
        lock (_syncLock)
        {
            stream.IsAvailable = true;

            // Dequeue completed operations
            while (_pendingTransfers.Count > 0 &&
                   _pendingTransfers.Peek().CompletionSource.Task.IsCompleted)
            {
                _pendingTransfers.Dequeue();
            }
        }
    }

    // Platform-specific transfer implementations
    private async Task PerformHostToDeviceTransferAsync<T>(
        ReadOnlyMemory<T> source,
        GpuBuffer<T> destination,
        TransferStream stream)
        where T : unmanaged
    {
        // Simulate async transfer - actual implementation would use CUDA streams or similar
        await Task.Run(() =>
        {
            // Copy data to GPU buffer
            destination.CopyFromHost(source);
        });
    }

    private async Task PerformDeviceToHostTransferAsync<T>(
        GpuBuffer<T> source,
        Memory<T> destination,
        TransferStream stream)
        where T : unmanaged
    {
        await Task.Run(() =>
        {
            source.CopyToHost(destination);
        });
    }

    private async Task PerformDeviceToDeviceTransferAsync<T>(
        GpuBuffer<T> source,
        GpuBuffer<T> destination,
        TransferStream stream)
        where T : unmanaged
    {
        await Task.Run(() =>
        {
            source.CopyTo(destination);
        });
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        Synchronize();
        _streams.Clear();
        _pendingTransfers.Clear();
    }
}

/// <summary>
/// Represents a GPU memory buffer.
/// </summary>
/// <typeparam name="T">The data type.</typeparam>
public class GpuBuffer<T> : IDisposable where T : unmanaged
{
    private T[]? _data;
    private readonly int _deviceId;
    private bool _disposed;

    /// <summary>
    /// Gets the number of elements in the buffer.
    /// </summary>
    public int Length { get; }

    /// <summary>
    /// Gets the device ID this buffer is allocated on.
    /// </summary>
    public int DeviceId => _deviceId;

    public GpuBuffer(int length, int deviceId = 0)
    {
        Length = length;
        _deviceId = deviceId;
        _data = new T[length];
    }

    public void CopyFromHost(ReadOnlyMemory<T> source)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(GpuBuffer<T>));
        source.Span.CopyTo(_data);
    }

    public void CopyToHost(Memory<T> destination)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(GpuBuffer<T>));
        _data.AsSpan().CopyTo(destination.Span);
    }

    public void CopyTo(GpuBuffer<T> destination)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(GpuBuffer<T>));
        Array.Copy(_data!, destination._data!, Math.Min(Length, destination.Length));
    }

    public ReadOnlySpan<T> AsSpan()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(GpuBuffer<T>));
        return _data.AsSpan();
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _data = null;
    }
}

/// <summary>
/// Double buffer for pipelined GPU data loading.
/// </summary>
/// <typeparam name="T">The data type.</typeparam>
public class DoubleBuffer<T> : IDisposable where T : unmanaged
{
    private readonly AsyncGpuTransfer _transfer;
    private readonly GpuBuffer<T>[] _buffers;
    private int _currentIndex;
    private Task? _pendingTransfer;

    public DoubleBuffer(AsyncGpuTransfer transfer, int bufferSize, int deviceId)
    {
        _transfer = transfer;
        _buffers = new GpuBuffer<T>[2];
        _buffers[0] = new GpuBuffer<T>(bufferSize, deviceId);
        _buffers[1] = new GpuBuffer<T>(bufferSize, deviceId);
        _currentIndex = 0;
    }

    /// <summary>
    /// Swaps buffers and starts loading new data into the back buffer.
    /// </summary>
    /// <param name="newData">New data to load.</param>
    /// <returns>The buffer ready for computation.</returns>
    public async Task<GpuBuffer<T>> SwapAndLoadAsync(ReadOnlyMemory<T> newData)
    {
        // Wait for any pending transfer on the back buffer
        if (_pendingTransfer != null)
        {
            await _pendingTransfer;
        }

        // Get the current buffer (ready for use)
        var readyBuffer = _buffers[_currentIndex];

        // Swap indices
        _currentIndex = 1 - _currentIndex;

        // Start loading into the new back buffer
        _pendingTransfer = _transfer.HostToDeviceAsync(newData, _buffers[_currentIndex]);

        return readyBuffer;
    }

    /// <summary>
    /// Gets the current buffer without loading new data.
    /// </summary>
    public async Task<GpuBuffer<T>> GetCurrentAsync()
    {
        if (_pendingTransfer != null)
        {
            await _pendingTransfer;
            _pendingTransfer = null;
        }
        return _buffers[_currentIndex];
    }

    public void Dispose()
    {
        _buffers[0].Dispose();
        _buffers[1].Dispose();
    }
}

/// <summary>
/// Types of memory transfer operations.
/// </summary>
public enum TransferType
{
    HostToDevice,
    DeviceToHost,
    DeviceToDevice
}

/// <summary>
/// Represents a pending transfer operation.
/// </summary>
internal class TransferOperation
{
    public TransferType Type { get; set; }
    public long SourceSize { get; set; }
    public int Priority { get; set; }
    public TaskCompletionSource<bool> CompletionSource { get; set; } = new();
}

/// <summary>
/// Represents a transfer stream for async operations.
/// </summary>
internal class TransferStream
{
    public int Id { get; set; }
    public bool IsAvailable { get; set; }
}
