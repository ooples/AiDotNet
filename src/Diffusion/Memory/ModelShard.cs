using AiDotNet.Interfaces;

namespace AiDotNet.Diffusion.Memory;

/// <summary>
/// Enables model sharding across multiple devices for large model inference.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Model sharding (also known as model parallelism or pipeline parallelism) splits
/// a large model across multiple devices (GPUs/CPUs) when it's too large to fit
/// on a single device.
/// </para>
/// <para>
/// <b>For Beginners:</b> Large AI models like Stable Diffusion XL may not fit in a single GPU.
///
/// Model sharding solves this by:
/// - Splitting the model layers across devices
/// - Running each device's layers in sequence
/// - Passing intermediate results between devices
///
/// Example: UNet with 24 layers on 4 GPUs:
/// - GPU 0: Layers 1-6
/// - GPU 1: Layers 7-12
/// - GPU 2: Layers 13-18
/// - GPU 3: Layers 19-24
///
/// Data flows: Input -> GPU0 -> GPU1 -> GPU2 -> GPU3 -> Output
///
/// Usage:
/// ```csharp
/// var shard = new ModelShard&lt;float&gt;(layers, numDevices: 4);
/// var output = shard.Forward(input);
/// ```
/// </para>
/// </remarks>
public class ModelShard<T>
{
    /// <summary>
    /// Provides numeric operations for the specific type T.
    /// </summary>
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Layers assigned to each device.
    /// </summary>
    private readonly Dictionary<int, List<ILayer<T>>> _deviceLayers;

    /// <summary>
    /// Number of devices/shards.
    /// </summary>
    private readonly int _numDevices;

    /// <summary>
    /// Device assignment for each layer.
    /// </summary>
    private readonly Dictionary<ILayer<T>, int> _layerDevices;

    /// <summary>
    /// Memory usage per device in bytes.
    /// </summary>
    private readonly long[] _deviceMemory;

    /// <summary>
    /// Whether to use pipeline parallelism (overlapping compute/transfer).
    /// </summary>
    private readonly bool _usePipelineParallelism;

    /// <summary>
    /// Sharding configuration.
    /// </summary>
    public ShardingConfig Config { get; }

    /// <summary>
    /// Initializes model sharding across specified number of devices.
    /// </summary>
    /// <param name="layers">Layers to shard.</param>
    /// <param name="numDevices">Number of devices to use.</param>
    /// <param name="config">Optional sharding configuration.</param>
    /// <remarks>
    /// <para>
    /// Layers are distributed evenly by default. Use ShardingConfig for
    /// custom distribution based on memory requirements or compute costs.
    /// </para>
    /// </remarks>
    public ModelShard(IEnumerable<ILayer<T>> layers, int numDevices, ShardingConfig? config = null)
    {
        if (numDevices <= 0)
            throw new ArgumentOutOfRangeException(nameof(numDevices), "Must have at least 1 device.");

        _numDevices = numDevices;
        Config = config ?? new ShardingConfig();
        _deviceLayers = new Dictionary<int, List<ILayer<T>>>();
        _layerDevices = new Dictionary<ILayer<T>, int>();
        _deviceMemory = new long[numDevices];
        _usePipelineParallelism = Config.UsePipelineParallelism;

        // Initialize device layer lists
        for (int i = 0; i < numDevices; i++)
        {
            _deviceLayers[i] = new List<ILayer<T>>();
        }

        // Distribute layers
        DistributeLayers(layers.ToList());
    }

    /// <summary>
    /// Distributes layers across devices using the configured strategy.
    /// </summary>
    private void DistributeLayers(List<ILayer<T>> layers)
    {
        if (layers.Count == 0)
            return;

        switch (Config.Strategy)
        {
            case ShardingStrategy.EvenSplit:
                DistributeEvenly(layers);
                break;

            case ShardingStrategy.MemoryBalanced:
                DistributeByMemory(layers);
                break;

            case ShardingStrategy.Custom:
                DistributeCustom(layers);
                break;

            default:
                DistributeEvenly(layers);
                break;
        }
    }

    /// <summary>
    /// Distributes layers evenly across devices.
    /// </summary>
    private void DistributeEvenly(List<ILayer<T>> layers)
    {
        int layersPerDevice = (layers.Count + _numDevices - 1) / _numDevices;

        for (int i = 0; i < layers.Count; i++)
        {
            int device = Math.Min(i / layersPerDevice, _numDevices - 1);
            AssignLayerToDevice(layers[i], device);
        }
    }

    /// <summary>
    /// Distributes layers to balance memory usage across devices.
    /// </summary>
    private void DistributeByMemory(List<ILayer<T>> layers)
    {
        // Estimate memory for each layer
        var layerMemory = layers.Select(l => EstimateLayerMemory(l)).ToList();
        long totalMemory = layerMemory.Sum();
        long targetPerDevice = totalMemory / _numDevices;

        int currentDevice = 0;
        long currentDeviceMemory = 0;

        for (int i = 0; i < layers.Count; i++)
        {
            // Move to next device if current is full (but not for last device)
            if (currentDeviceMemory >= targetPerDevice && currentDevice < _numDevices - 1)
            {
                currentDevice++;
                currentDeviceMemory = 0;
            }

            AssignLayerToDevice(layers[i], currentDevice);
            currentDeviceMemory += layerMemory[i];
        }
    }

    /// <summary>
    /// Distributes layers using custom device assignments from config.
    /// </summary>
    private void DistributeCustom(List<ILayer<T>> layers)
    {
        for (int i = 0; i < layers.Count; i++)
        {
            int device = 0;
            if (Config.CustomDeviceAssignments != null && i < Config.CustomDeviceAssignments.Length)
            {
                device = MathPolyfill.Clamp(Config.CustomDeviceAssignments[i], 0, _numDevices - 1);
            }
            else
            {
                device = i % _numDevices;
            }
            AssignLayerToDevice(layers[i], device);
        }
    }

    /// <summary>
    /// Assigns a layer to a specific device.
    /// </summary>
    private void AssignLayerToDevice(ILayer<T> layer, int device)
    {
        _deviceLayers[device].Add(layer);
        _layerDevices[layer] = device;
        _deviceMemory[device] += EstimateLayerMemory(layer);
    }

    /// <summary>
    /// Estimates memory usage for a layer in bytes.
    /// </summary>
    private static long EstimateLayerMemory(ILayer<T> layer)
    {
        try
        {
            var parameters = layer.GetParameters();
            // Assume 4 bytes per parameter (float)
            // Multiply by 3 for: parameters + gradients + optimizer state
            return parameters.Length * 4 * 3;
        }
        catch
        {
            // Default estimate if GetParameters fails
            return 1024 * 1024; // 1MB default
        }
    }

    /// <summary>
    /// Performs forward pass through all sharded layers.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Output tensor after all layers.</returns>
    /// <remarks>
    /// <para>
    /// Data flows sequentially through devices. Each device processes its
    /// assigned layers before passing results to the next device.
    /// </para>
    /// </remarks>
    public Tensor<T> Forward(Tensor<T> input)
    {
        var current = input;

        for (int device = 0; device < _numDevices; device++)
        {
            // Move tensor to device (conceptual - actual implementation depends on backend)
            current = MoveToDevice(current, device);

            // Process all layers on this device
            foreach (var layer in _deviceLayers[device])
            {
                current = layer.Forward(current);
            }
        }

        return current;
    }

    /// <summary>
    /// Performs forward pass with context (for conditional generation).
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <param name="context">Context tensor (e.g., timestep, conditioning).</param>
    /// <returns>Output tensor.</returns>
    public Tensor<T> Forward(Tensor<T> input, Tensor<T>? context)
    {
        var current = input;

        for (int device = 0; device < _numDevices; device++)
        {
            current = MoveToDevice(current, device);

            foreach (var layer in _deviceLayers[device])
            {
                // Try to call forward with context if supported
                if (layer is IContextualLayer<T> contextualLayer)
                {
                    current = contextualLayer.Forward(current, context);
                }
                else
                {
                    current = layer.Forward(current);
                }
            }
        }

        return current;
    }

    /// <summary>
    /// Performs backward pass through all sharded layers.
    /// </summary>
    /// <param name="outputGradient">Gradient from subsequent layer.</param>
    /// <returns>Gradient with respect to input.</returns>
    public Tensor<T> Backward(Tensor<T> outputGradient)
    {
        var current = outputGradient;

        // Backward pass goes in reverse order
        for (int device = _numDevices - 1; device >= 0; device--)
        {
            current = MoveToDevice(current, device);

            // Process layers in reverse order
            var layers = _deviceLayers[device];
            for (int i = layers.Count - 1; i >= 0; i--)
            {
                current = layers[i].Backward(current);
            }
        }

        return current;
    }

    /// <summary>
    /// Updates parameters on all devices.
    /// </summary>
    /// <param name="learningRate">Learning rate for update.</param>
    public void UpdateParameters(T learningRate)
    {
        for (int device = 0; device < _numDevices; device++)
        {
            foreach (var layer in _deviceLayers[device])
            {
                layer.UpdateParameters(learningRate);
            }
        }
    }

    /// <summary>
    /// Gets layers assigned to a specific device.
    /// </summary>
    public IReadOnlyList<ILayer<T>> GetDeviceLayers(int device)
    {
        if (device < 0 || device >= _numDevices)
            throw new ArgumentOutOfRangeException(nameof(device));

        return _deviceLayers[device].AsReadOnly();
    }

    /// <summary>
    /// Gets the device assignment for a layer.
    /// </summary>
    public int GetLayerDevice(ILayer<T> layer)
    {
        if (_layerDevices.TryGetValue(layer, out var device))
            return device;

        throw new ArgumentException("Layer not found in shard.", nameof(layer));
    }

    /// <summary>
    /// Gets memory usage per device.
    /// </summary>
    public IReadOnlyDictionary<int, long> GetDeviceMemoryUsage()
    {
        var usage = new Dictionary<int, long>();
        for (int i = 0; i < _numDevices; i++)
        {
            usage[i] = _deviceMemory[i];
        }
        return usage;
    }

    /// <summary>
    /// Moves a tensor to a specific device (placeholder for GPU implementation).
    /// </summary>
    /// <remarks>
    /// <para>
    /// In a real multi-GPU implementation, this would:
    /// - Check if tensor is already on target device
    /// - Perform async device-to-device copy if needed
    /// - Use CUDA streams for overlapping compute/transfer
    ///
    /// Current implementation is a pass-through for CPU-only execution.
    /// </para>
    /// </remarks>
    private Tensor<T> MoveToDevice(Tensor<T> tensor, int device)
    {
        // In CPU-only mode, this is a no-op
        // In GPU mode, this would be implemented to move data between devices
        return tensor;
    }

    /// <summary>
    /// Gets a summary of the sharding distribution.
    /// </summary>
    public override string ToString()
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine($"ModelShard: {_numDevices} devices, Strategy: {Config.Strategy}");

        for (int i = 0; i < _numDevices; i++)
        {
            var layers = _deviceLayers[i];
            var memory = _deviceMemory[i] / (1024.0 * 1024.0);
            sb.AppendLine($"  Device {i}: {layers.Count} layers, {memory:F1} MB");
        }

        return sb.ToString();
    }
}

/// <summary>
/// Configuration for model sharding.
/// </summary>
public class ShardingConfig
{
    /// <summary>
    /// Strategy for distributing layers across devices.
    /// </summary>
    public ShardingStrategy Strategy { get; set; } = ShardingStrategy.EvenSplit;

    /// <summary>
    /// Whether to use pipeline parallelism for overlapping compute and transfer.
    /// </summary>
    public bool UsePipelineParallelism { get; set; } = false;

    /// <summary>
    /// Number of micro-batches for pipeline parallelism.
    /// </summary>
    /// <remarks>
    /// Higher values improve GPU utilization but increase memory usage.
    /// Typical values: 4-8 for most models.
    /// </remarks>
    public int MicroBatchCount { get; set; } = 4;

    /// <summary>
    /// Custom device assignments when using ShardingStrategy.Custom.
    /// Index is layer index, value is device index.
    /// </summary>
    public int[]? CustomDeviceAssignments { get; set; }

    /// <summary>
    /// Memory limit per device in bytes (for memory-balanced sharding).
    /// </summary>
    public long? MaxMemoryPerDevice { get; set; }
}

/// <summary>
/// Strategy for distributing layers across devices.
/// </summary>
public enum ShardingStrategy
{
    /// <summary>
    /// Distribute layers evenly by count.
    /// </summary>
    EvenSplit,

    /// <summary>
    /// Balance memory usage across devices.
    /// </summary>
    MemoryBalanced,

    /// <summary>
    /// Use custom device assignments.
    /// </summary>
    Custom
}

/// <summary>
/// Interface for layers that accept context (conditioning).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
[AiDotNet.Configuration.YamlConfigurable("ContextualLayer")]
public interface IContextualLayer<T> : ILayer<T>
{
    /// <summary>
    /// Forward pass with context.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <param name="context">Context tensor.</param>
    /// <returns>Output tensor.</returns>
    Tensor<T> Forward(Tensor<T> input, Tensor<T>? context);
}
