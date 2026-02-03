using AiDotNet.Diffusion.Memory;
using AiDotNet.Interfaces;

namespace AiDotNet.Training.Memory;

/// <summary>
/// Manages memory optimization during neural network training including gradient checkpointing,
/// activation pooling, and model sharding.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This manager helps you train larger neural networks by:
///
/// 1. **Gradient Checkpointing**: Saves memory by recomputing activations during backward pass
/// 2. **Activation Pooling**: Reuses tensor memory to reduce garbage collection
/// 3. **Model Sharding**: Distributes layers across multiple GPUs
///
/// Example usage:
/// <code>
/// var config = TrainingMemoryConfig.MemoryEfficient();
/// var memoryManager = new TrainingMemoryManager&lt;float&gt;(config, network.Layers);
///
/// // During training
/// foreach (var layer in layers)
/// {
///     if (memoryManager.ShouldCheckpoint(layerIndex))
///     {
///         output = memoryManager.ForwardWithCheckpoint(layer, input);
///     }
///     else
///     {
///         output = layer.Forward(input);
///     }
/// }
/// </code>
/// </para>
/// </remarks>
public class TrainingMemoryManager<T> : IDisposable
{
    private readonly ActivationPool<T>? _activationPool;
    private readonly ModelShard<T>? _modelShard;
    private readonly Dictionary<int, CheckpointedActivation<T>> _checkpoints;
    private readonly HashSet<int> _checkpointIndices;
    private bool _disposed;

    /// <summary>
    /// Gets the memory configuration.
    /// </summary>
    public TrainingMemoryConfig Config { get; }

    /// <summary>
    /// Gets pool statistics if activation pooling is enabled.
    /// </summary>
    public ActivationPoolStats? PoolStats => _activationPool?.Stats;

    /// <summary>
    /// Gets whether gradient checkpointing is enabled.
    /// </summary>
    public bool IsCheckpointingEnabled => Config.UseGradientCheckpointing;

    /// <summary>
    /// Gets whether activation pooling is enabled.
    /// </summary>
    public bool IsPoolingEnabled => Config.UseActivationPooling && _activationPool is not null;

    /// <summary>
    /// Gets whether model sharding is enabled.
    /// </summary>
    public bool IsShardingEnabled => Config.NumDevices > 1 && _modelShard is not null;

    /// <summary>
    /// Initializes a new instance of the TrainingMemoryManager.
    /// </summary>
    /// <param name="config">Memory configuration.</param>
    /// <param name="layers">Optional layers for model sharding.</param>
    public TrainingMemoryManager(TrainingMemoryConfig? config = null, IEnumerable<ILayer<T>>? layers = null)
    {
        Config = config ?? new TrainingMemoryConfig();
        _checkpoints = new Dictionary<int, CheckpointedActivation<T>>();
        _checkpointIndices = new HashSet<int>();

        // Initialize activation pool
        if (Config.UseActivationPooling)
        {
            _activationPool = new ActivationPool<T>(Config.MaxPoolMemoryMB);
        }

        // Initialize model sharding
        if (Config.NumDevices > 1 && layers is not null)
        {
            var shardingConfig = new ShardingConfig
            {
                Strategy = Config.ShardingStrategy,
                UsePipelineParallelism = Config.UsePipelineParallelism,
                MicroBatchCount = Config.MicroBatchCount
            };
            _modelShard = new ModelShard<T>(layers, Config.NumDevices, shardingConfig);
        }
    }

    #region Checkpoint Index Management

    /// <summary>
    /// Determines which layers should be checkpointed based on configuration.
    /// </summary>
    /// <param name="totalLayers">Total number of layers in the network.</param>
    /// <param name="layerTypes">Optional list of layer type names for smart checkpointing.</param>
    public void ComputeCheckpointIndices(int totalLayers, IReadOnlyList<string>? layerTypes = null)
    {
        _checkpointIndices.Clear();

        if (!Config.UseGradientCheckpointing)
            return;

        for (int i = 0; i < totalLayers; i++)
        {
            bool shouldCheckpoint = false;

            // Check by layer interval
            if (i % Config.CheckpointEveryNLayers == 0)
            {
                shouldCheckpoint = true;
            }

            // Check by layer type if type info is provided
            if (layerTypes is not null && i < layerTypes.Count)
            {
                var typeName = layerTypes[i].ToLowerInvariant();

                if (Config.CheckpointAttentionLayers &&
                    (typeName.Contains("attention") || typeName.Contains("multihead")))
                {
                    shouldCheckpoint = true;
                }

                if (Config.CheckpointResidualBlocks &&
                    (typeName.Contains("residual") || typeName.Contains("resblock") || typeName.Contains("resnet")))
                {
                    shouldCheckpoint = true;
                }
            }

            if (shouldCheckpoint)
            {
                _checkpointIndices.Add(i);
            }
        }
    }

    /// <summary>
    /// Determines if a specific layer should be checkpointed.
    /// </summary>
    /// <param name="layerIndex">Index of the layer.</param>
    /// <returns>True if the layer should be checkpointed.</returns>
    public bool ShouldCheckpoint(int layerIndex)
    {
        if (!Config.UseGradientCheckpointing)
            return false;

        // If we've precomputed indices, use those
        if (_checkpointIndices.Count > 0)
            return _checkpointIndices.Contains(layerIndex);

        // Otherwise use simple interval-based checkpointing
        return layerIndex % Config.CheckpointEveryNLayers == 0;
    }

    #endregion

    #region Forward Pass with Checkpointing

    /// <summary>
    /// Performs a forward pass with checkpointing for a single layer.
    /// </summary>
    /// <param name="layer">The layer to execute.</param>
    /// <param name="input">Input tensor.</param>
    /// <param name="layerIndex">Index of this layer (for checkpoint storage).</param>
    /// <returns>Output tensor from the layer.</returns>
    public Tensor<T> ForwardWithCheckpoint(ILayer<T> layer, Tensor<T> input, int layerIndex)
    {
        if (!Config.UseGradientCheckpointing || !ShouldCheckpoint(layerIndex))
        {
            // Use precision-aware forward pass for mixed-precision support
            return layer.ForwardWithPrecisionCheck(input);
        }

        // Save checkpoint: store input and layer reference
        _checkpoints[layerIndex] = new CheckpointedActivation<T>
        {
            Input = CloneTensor(input),
            Layer = layer,
            LayerIndex = layerIndex
        };

        // Run forward pass with precision awareness
        return layer.ForwardWithPrecisionCheck(input);
    }

    /// <summary>
    /// Performs forward pass through multiple layers with checkpointing.
    /// </summary>
    /// <param name="layers">Sequence of layers to execute.</param>
    /// <param name="input">Initial input tensor.</param>
    /// <returns>Output tensor from the final layer.</returns>
    public Tensor<T> ForwardSequence(IEnumerable<ILayer<T>> layers, Tensor<T> input)
    {
        var current = input;
        int index = 0;

        foreach (var layer in layers)
        {
            current = ForwardWithCheckpoint(layer, current, index);
            index++;
        }

        return current;
    }

    #endregion

    #region Backward Pass with Recomputation

    /// <summary>
    /// Performs backward pass, recomputing activations from checkpoints.
    /// </summary>
    /// <param name="layer">The layer to backpropagate through.</param>
    /// <param name="outputGradient">Gradient from the next layer.</param>
    /// <param name="layerIndex">Index of this layer.</param>
    /// <returns>Gradient with respect to input.</returns>
    public Tensor<T> BackwardWithRecompute(ILayer<T> layer, Tensor<T> outputGradient, int layerIndex)
    {
        // If this is a checkpointed layer, we need to recompute forward first
        if (_checkpoints.TryGetValue(layerIndex, out var checkpoint))
        {
            // Recompute forward pass from checkpoint with precision awareness
            _ = layer.ForwardWithPrecisionCheck(checkpoint.Input);
        }

        // Now run backward pass
        return layer.Backward(outputGradient);
    }

    /// <summary>
    /// Performs backward pass through multiple layers with recomputation.
    /// </summary>
    /// <param name="layers">Layers in forward order.</param>
    /// <param name="outputGradient">Final output gradient.</param>
    /// <returns>Gradient with respect to initial input.</returns>
    public Tensor<T> BackwardSequence(IReadOnlyList<ILayer<T>> layers, Tensor<T> outputGradient)
    {
        var gradient = outputGradient;

        // Process layers in reverse order
        for (int i = layers.Count - 1; i >= 0; i--)
        {
            gradient = BackwardWithRecompute(layers[i], gradient, i);
        }

        return gradient;
    }

    /// <summary>
    /// Clears all stored checkpoints to free memory.
    /// </summary>
    public void ClearCheckpoints()
    {
        foreach (var checkpoint in _checkpoints.Values)
        {
            // Return pooled tensors if pooling is enabled
            if (_activationPool is not null && checkpoint.Input is not null)
            {
                _activationPool.Return(checkpoint.Input);
            }
        }
        _checkpoints.Clear();
    }

    #endregion

    #region Activation Pooling

    /// <summary>
    /// Rents a tensor from the activation pool.
    /// </summary>
    /// <param name="shape">Desired tensor shape.</param>
    /// <returns>A tensor (may contain uninitialized data).</returns>
    public Tensor<T> RentTensor(int[] shape)
    {
        if (_activationPool is not null)
        {
            return _activationPool.Rent(shape);
        }

        return new Tensor<T>(shape);
    }

    /// <summary>
    /// Returns a tensor to the activation pool.
    /// </summary>
    /// <param name="tensor">Tensor to return.</param>
    public void ReturnTensor(Tensor<T> tensor)
    {
        _activationPool?.Return(tensor);
    }

    /// <summary>
    /// Gets current memory usage from the activation pool.
    /// </summary>
    public long GetPoolMemoryUsage()
    {
        return _activationPool?.GetMemoryUsage() ?? 0;
    }

    #endregion

    #region Model Sharding

    /// <summary>
    /// Performs forward pass through sharded model.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Output tensor.</returns>
    public Tensor<T> ShardedForward(Tensor<T> input)
    {
        if (_modelShard is null)
            throw new InvalidOperationException("Model sharding is not enabled.");

        return _modelShard.Forward(input);
    }

    /// <summary>
    /// Performs backward pass through sharded model.
    /// </summary>
    /// <param name="outputGradient">Gradient from loss.</param>
    /// <returns>Gradient with respect to input.</returns>
    public Tensor<T> ShardedBackward(Tensor<T> outputGradient)
    {
        if (_modelShard is null)
            throw new InvalidOperationException("Model sharding is not enabled.");

        return _modelShard.Backward(outputGradient);
    }

    #endregion

    #region Memory Estimation

    /// <summary>
    /// Estimates memory savings from current configuration.
    /// </summary>
    /// <param name="modelParameters">Total number of model parameters.</param>
    /// <param name="batchSize">Training batch size.</param>
    /// <param name="sequenceLength">Sequence length (for transformers).</param>
    /// <returns>Estimated memory savings information.</returns>
    public MemorySavingsEstimate EstimateMemorySavings(
        long modelParameters,
        int batchSize,
        int sequenceLength = 512)
    {
        // Rough estimates based on typical memory breakdown:
        // - Model parameters: 4 bytes per parameter (float32)
        // - Gradients: Same as parameters
        // - Optimizer states: 2x parameters (Adam has momentum + variance)
        // - Activations: Varies widely, rough estimate

        long paramBytes = modelParameters * 4; // float32
        long gradientBytes = paramBytes;
        long optimizerBytes = paramBytes * 2; // Adam

        // Activation memory: very rough estimate
        // For transformers: O(batch * seq * hidden * layers)
        long activationBytesWithout = batchSize * sequenceLength * modelParameters / 100 * 4;

        // With checkpointing: sqrt reduction
        long activationBytesWith = (long)Math.Sqrt(activationBytesWithout) * Config.CheckpointEveryNLayers;

        long totalWithout = paramBytes + gradientBytes + optimizerBytes + activationBytesWithout;
        long totalWith = paramBytes + gradientBytes + optimizerBytes + activationBytesWith;

        return new MemorySavingsEstimate
        {
            WithoutOptimization = totalWithout,
            WithOptimization = totalWith,
            SavingsBytes = totalWithout - totalWith,
            SavingsPercentage = 100.0 * (totalWithout - totalWith) / totalWithout,
            CheckpointingEnabled = Config.UseGradientCheckpointing,
            PoolingEnabled = Config.UseActivationPooling,
            ShardingEnabled = Config.NumDevices > 1
        };
    }

    #endregion


    #region Helper Methods

    private Tensor<T> CloneTensor(Tensor<T> source)
    {
        // If pooling is enabled, get a tensor from the pool
        if (_activationPool is not null)
        {
            var clone = _activationPool.Rent(source.Shape);
            // Copy data
            var sourceSpan = source.AsSpan();
            var cloneSpan = clone.AsWritableSpan();
            for (int i = 0; i < sourceSpan.Length; i++)
            {
                cloneSpan[i] = sourceSpan[i];
            }
            return clone;
        }

        // Otherwise create a new tensor
        var newTensor = new Tensor<T>(source.Shape);
        var srcSpan = source.AsSpan();
        var dstSpan = newTensor.AsWritableSpan();
        for (int i = 0; i < srcSpan.Length; i++)
        {
            dstSpan[i] = srcSpan[i];
        }
        return newTensor;
    }

    #endregion

    #region IDisposable

    /// <summary>
    /// Disposes resources used by the memory manager.
    /// </summary>
    public void Dispose()
    {
        if (!_disposed)
        {
            ClearCheckpoints();
            _activationPool?.Dispose();
            _disposed = true;
        }
        GC.SuppressFinalize(this);
    }

    #endregion
}

/// <summary>
/// Stores information about a checkpointed activation.
/// </summary>
internal class CheckpointedActivation<T>
{
    /// <summary>
    /// The input tensor that was saved at this checkpoint.
    /// </summary>
    public Tensor<T> Input { get; set; } = null!;

    /// <summary>
    /// Reference to the layer for recomputation.
    /// </summary>
    public ILayer<T> Layer { get; set; } = null!;

    /// <summary>
    /// Index of this layer in the network.
    /// </summary>
    public int LayerIndex { get; set; }
}

/// <summary>
/// Memory savings estimate from optimization techniques.
/// </summary>
public class MemorySavingsEstimate
{
    /// <summary>
    /// Estimated memory usage without optimization (bytes).
    /// </summary>
    public long WithoutOptimization { get; set; }

    /// <summary>
    /// Estimated memory usage with optimization (bytes).
    /// </summary>
    public long WithOptimization { get; set; }

    /// <summary>
    /// Memory saved (bytes).
    /// </summary>
    public long SavingsBytes { get; set; }

    /// <summary>
    /// Percentage of memory saved.
    /// </summary>
    public double SavingsPercentage { get; set; }

    /// <summary>
    /// Whether gradient checkpointing is enabled.
    /// </summary>
    public bool CheckpointingEnabled { get; set; }

    /// <summary>
    /// Whether activation pooling is enabled.
    /// </summary>
    public bool PoolingEnabled { get; set; }

    /// <summary>
    /// Whether model sharding is enabled.
    /// </summary>
    public bool ShardingEnabled { get; set; }

    /// <inheritdoc />
    public override string ToString()
    {
        return $"Memory: {WithoutOptimization / 1024 / 1024}MB -> {WithOptimization / 1024 / 1024}MB " +
               $"(saved {SavingsPercentage:F1}%) " +
               $"[Checkpoint:{CheckpointingEnabled}, Pool:{PoolingEnabled}, Shard:{ShardingEnabled}]";
    }
}
