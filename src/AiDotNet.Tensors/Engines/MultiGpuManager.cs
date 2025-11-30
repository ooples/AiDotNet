using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Manages multiple GPU devices for parallel computation within a single process.
/// </summary>
/// <remarks>
/// <para>
/// MultiGpuManager enables using multiple GPUs for training or inference. It supports
/// data parallelism (same model on multiple GPUs, different data) and can coordinate
/// gradient synchronization across devices.
/// </para>
/// <para><b>For Beginners:</b> If you have multiple GPUs, this lets you use them all at once!
///
/// Benefits:
/// - Train faster by processing more data in parallel
/// - Handle larger models that don't fit on a single GPU
/// - Increase throughput for inference
///
/// Common patterns:
/// - Data Parallelism: Same model on each GPU, different data batches
/// - Model Parallelism: Different parts of model on different GPUs
/// </para>
/// </remarks>
public class MultiGpuManager : IDisposable
{
    private readonly List<GpuDevice> _devices;
    private readonly Dictionary<int, AsyncGpuTransfer> _transfers;
    private readonly object _syncLock = new();
    private bool _disposed;

    /// <summary>
    /// Gets the number of available GPU devices.
    /// </summary>
    public int DeviceCount => _devices.Count;

    /// <summary>
    /// Gets all available GPU devices.
    /// </summary>
    public IReadOnlyList<GpuDevice> Devices => _devices;

    /// <summary>
    /// Gets or sets the primary device used for aggregation.
    /// </summary>
    public int PrimaryDeviceId { get; set; } = 0;

    /// <summary>
    /// Initializes a new instance of the MultiGpuManager class.
    /// </summary>
    /// <param name="deviceIds">Specific device IDs to use. If null, uses all available devices.</param>
    public MultiGpuManager(int[]? deviceIds = null)
    {
        _devices = new List<GpuDevice>();
        _transfers = new Dictionary<int, AsyncGpuTransfer>();

        // Detect available GPUs
        var availableDevices = DetectGpuDevices();

        if (deviceIds != null)
        {
            foreach (var id in deviceIds)
            {
                var device = availableDevices.FirstOrDefault(d => d.Id == id);
                if (device != null)
                {
                    _devices.Add(device);
                    _transfers[id] = new AsyncGpuTransfer(id);
                }
            }
        }
        else
        {
            _devices.AddRange(availableDevices);
            foreach (var device in _devices)
            {
                _transfers[device.Id] = new AsyncGpuTransfer(device.Id);
            }
        }

        if (_devices.Count > 0)
        {
            PrimaryDeviceId = _devices[0].Id;
        }
    }

    /// <summary>
    /// Detects available GPU devices on the system.
    /// </summary>
    private static List<GpuDevice> DetectGpuDevices()
    {
        var devices = new List<GpuDevice>();

        // Simulate GPU detection - actual implementation would query CUDA/Metal/Vulkan
        // For now, we'll check for environment hints or use defaults
        var gpuCountEnv = Environment.GetEnvironmentVariable("CUDA_VISIBLE_DEVICES");
        int gpuCount = 1;

        if (!string.IsNullOrEmpty(gpuCountEnv))
        {
            var ids = gpuCountEnv.Split(',');
            gpuCount = ids.Length;
        }
        else
        {
            // Check for simulated GPU count
            var simCountEnv = Environment.GetEnvironmentVariable("AIDOTNET_GPU_COUNT");
            if (int.TryParse(simCountEnv, out var simCount))
            {
                gpuCount = simCount;
            }
        }

        for (int i = 0; i < gpuCount; i++)
        {
            devices.Add(new GpuDevice
            {
                Id = i,
                Name = $"GPU {i}",
                TotalMemory = 8L * 1024 * 1024 * 1024, // 8GB simulated
                ComputeCapability = "8.0"
            });
        }

        return devices;
    }

    /// <summary>
    /// Distributes data across all GPUs for data parallel training.
    /// </summary>
    /// <typeparam name="T">The data type.</typeparam>
    /// <param name="data">The data to distribute.</param>
    /// <returns>Dictionary mapping device ID to its data portion.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This splits your training batch across GPUs:
    ///
    /// <code>
    /// // If you have 128 samples and 4 GPUs:
    /// var distributed = manager.DistributeData(batchData);
    /// // GPU 0 gets samples 0-31
    /// // GPU 1 gets samples 32-63
    /// // GPU 2 gets samples 64-95
    /// // GPU 3 gets samples 96-127
    /// </code>
    /// </para>
    /// </remarks>
    public Dictionary<int, T[]> DistributeData<T>(T[] data)
    {
        var result = new Dictionary<int, T[]>();
        int chunkSize = data.Length / _devices.Count;
        int remainder = data.Length % _devices.Count;

        int offset = 0;
        for (int i = 0; i < _devices.Count; i++)
        {
            int size = chunkSize + (i < remainder ? 1 : 0);
            var chunk = new T[size];
            Array.Copy(data, offset, chunk, 0, size);
            result[_devices[i].Id] = chunk;
            offset += size;
        }

        return result;
    }

    /// <summary>
    /// Distributes tensor data across all GPUs.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="tensor">The tensor to distribute.</param>
    /// <returns>Dictionary mapping device ID to its tensor portion.</returns>
    public Dictionary<int, Tensor<T>> DistributeTensor<T>(Tensor<T> tensor)
    {
        var result = new Dictionary<int, Tensor<T>>();
        var data = tensor.AsSpan().ToArray();
        var batchSize = tensor.Shape[0];
        int chunkSize = batchSize / _devices.Count;

        int offset = 0;
        int elementsPerSample = data.Length / batchSize;

        for (int i = 0; i < _devices.Count; i++)
        {
            int samples = chunkSize + (i < batchSize % _devices.Count ? 1 : 0);
            var newShape = (int[])tensor.Shape.Clone();
            newShape[0] = samples;

            var chunk = new Tensor<T>(newShape);
            var chunkData = new T[samples * elementsPerSample];
            Array.Copy(data, offset, chunkData, 0, chunkData.Length);

            for (int j = 0; j < chunkData.Length; j++)
            {
                chunk[j] = chunkData[j];
            }

            result[_devices[i].Id] = chunk;
            offset += chunkData.Length;
        }

        return result;
    }

    /// <summary>
    /// Gathers gradients from all GPUs and averages them.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="gradients">Dictionary mapping device ID to its gradients.</param>
    /// <returns>Averaged gradients on the primary device.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> After each GPU computes gradients on its data portion,
    /// this combines them by averaging. The result is the same as if you had trained
    /// on all the data with a single GPU.
    ///
    /// <code>
    /// // Each GPU computes gradients
    /// var allGrads = new Dictionary&lt;int, Tensor&lt;float&gt;&gt;();
    /// foreach (var device in manager.Devices)
    /// {
    ///     allGrads[device.Id] = ComputeGradientsOnDevice(device.Id);
    /// }
    ///
    /// // Combine gradients
    /// var avgGradients = manager.AllReduceGradients(allGrads);
    ///
    /// // Update model with combined gradients
    /// optimizer.Step(avgGradients);
    /// </code>
    /// </para>
    /// </remarks>
    public Tensor<T> AllReduceGradients<T>(Dictionary<int, Tensor<T>> gradients)
    {
        if (gradients.Count == 0)
        {
            throw new ArgumentException("No gradients provided", nameof(gradients));
        }

        var first = gradients.Values.First();
        var result = new Tensor<T>(first.Shape);
        var numOps = Helpers.MathHelper.GetNumericOperations<T>();

        // Sum all gradients
        for (int i = 0; i < result.Length; i++)
        {
            T sum = numOps.Zero;
            foreach (var grad in gradients.Values)
            {
                sum = numOps.Add(sum, grad[i]);
            }
            // Average
            result[i] = numOps.Divide(sum, numOps.FromDouble(gradients.Count));
        }

        return result;
    }

    /// <summary>
    /// Broadcasts model parameters from primary device to all other devices.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="parameters">Parameters on the primary device.</param>
    /// <returns>Dictionary mapping device ID to replicated parameters.</returns>
    public async Task<Dictionary<int, Tensor<T>>> BroadcastParametersAsync<T>(Tensor<T> parameters)
        where T : unmanaged
    {
        var result = new Dictionary<int, Tensor<T>>();

        var data = new T[parameters.Length];
        for (int i = 0; i < parameters.Length; i++)
        {
            data[i] = parameters[i];
        }

        var tasks = new List<Task>();
        var buffers = new List<GpuBuffer<T>>();

        foreach (var device in _devices)
        {
            var deviceParams = new Tensor<T>(parameters.Shape);
            for (int i = 0; i < data.Length; i++)
            {
                deviceParams[i] = data[i];
            }
            result[device.Id] = deviceParams;

            // Simulate async transfer
            if (device.Id != PrimaryDeviceId && _transfers.ContainsKey(device.Id))
            {
                var buffer = new GpuBuffer<T>(data.Length, device.Id);
                buffers.Add(buffer);
                tasks.Add(_transfers[device.Id].HostToDeviceAsync(data.AsMemory(), buffer));
            }
        }

        await Task.WhenAll(tasks);

        // Dispose buffers after transfers complete
        foreach (var buffer in buffers)
        {
            buffer.Dispose();
        }

        return result;
    }

    /// <summary>
    /// Executes a function on all GPUs in parallel.
    /// </summary>
    /// <typeparam name="TInput">Input type.</typeparam>
    /// <typeparam name="TOutput">Output type.</typeparam>
    /// <param name="inputs">Dictionary of inputs per device.</param>
    /// <param name="function">Function to execute on each device.</param>
    /// <returns>Dictionary of outputs per device.</returns>
    public async Task<Dictionary<int, TOutput>> ExecuteOnAllDevicesAsync<TInput, TOutput>(
        Dictionary<int, TInput> inputs,
        Func<int, TInput, Task<TOutput>> function)
    {
        var tasks = inputs.Select(async kvp =>
        {
            var result = await function(kvp.Key, kvp.Value);
            return (kvp.Key, result);
        });

        var results = await Task.WhenAll(tasks);
        return results.ToDictionary(r => r.Key, r => r.result);
    }

    /// <summary>
    /// Executes a function on all GPUs in parallel (synchronous version).
    /// </summary>
    public Dictionary<int, TOutput> ExecuteOnAllDevices<TInput, TOutput>(
        Dictionary<int, TInput> inputs,
        Func<int, TInput, TOutput> function)
    {
        var results = new Dictionary<int, TOutput>();

        Parallel.ForEach(inputs, kvp =>
        {
            var result = function(kvp.Key, kvp.Value);
            lock (_syncLock)
            {
                results[kvp.Key] = result;
            }
        });

        return results;
    }

    /// <summary>
    /// Gets memory usage across all devices.
    /// </summary>
    /// <returns>Dictionary mapping device ID to memory usage info.</returns>
    public Dictionary<int, GpuMemoryInfo> GetMemoryUsage()
    {
        return _devices.ToDictionary(
            d => d.Id,
            d => new GpuMemoryInfo
            {
                DeviceId = d.Id,
                TotalMemory = d.TotalMemory,
                UsedMemory = d.TotalMemory / 4, // Simulated 25% usage
                FreeMemory = d.TotalMemory * 3 / 4
            });
    }

    /// <summary>
    /// Selects the best device based on available memory.
    /// </summary>
    public int SelectBestDevice()
    {
        var memoryInfo = GetMemoryUsage();
        return memoryInfo.OrderByDescending(kvp => kvp.Value.FreeMemory)
                        .First().Key;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        foreach (var transfer in _transfers.Values)
        {
            transfer.Dispose();
        }
        _transfers.Clear();
        _devices.Clear();
    }
}

/// <summary>
/// Represents information about a GPU device.
/// </summary>
public class GpuDevice
{
    /// <summary>
    /// Device ID.
    /// </summary>
    public int Id { get; set; }

    /// <summary>
    /// Device name.
    /// </summary>
    public string Name { get; set; } = "";

    /// <summary>
    /// Total memory in bytes.
    /// </summary>
    public long TotalMemory { get; set; }

    /// <summary>
    /// CUDA compute capability or equivalent.
    /// </summary>
    public string ComputeCapability { get; set; } = "";
}

/// <summary>
/// Memory usage information for a GPU.
/// </summary>
public class GpuMemoryInfo
{
    public int DeviceId { get; set; }
    public long TotalMemory { get; set; }
    public long UsedMemory { get; set; }
    public long FreeMemory { get; set; }
}

/// <summary>
/// Configuration for data parallel training.
/// </summary>
public class DataParallelConfig
{
    /// <summary>
    /// Whether to use synchronized batch normalization across GPUs.
    /// </summary>
    public bool SyncBatchNorm { get; set; } = true;

    /// <summary>
    /// Gradient reduction mode.
    /// </summary>
    public GradientReductionMode ReductionMode { get; set; } = GradientReductionMode.Mean;

    /// <summary>
    /// Whether to overlap gradient communication with backward pass.
    /// </summary>
    public bool OverlapCommunication { get; set; } = true;

    /// <summary>
    /// Bucket size for gradient bucketing (in MB).
    /// </summary>
    public int BucketSizeMb { get; set; } = 25;
}

/// <summary>
/// Gradient reduction modes for multi-GPU training.
/// </summary>
public enum GradientReductionMode
{
    /// <summary>Average gradients across devices.</summary>
    Mean,
    /// <summary>Sum gradients across devices.</summary>
    Sum
}
