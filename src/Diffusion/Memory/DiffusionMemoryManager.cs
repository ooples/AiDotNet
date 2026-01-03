using AiDotNet.Autodiff;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Interfaces;

namespace AiDotNet.Diffusion.Memory;

/// <summary>
/// Memory management utilities for diffusion models including gradient checkpointing,
/// activation pooling, and model sharding integration.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This class provides memory-efficient training utilities specifically designed for
/// large diffusion models (UNet, VAE, etc.) that may not fit in GPU memory during training.
/// </para>
/// <para>
/// <b>For Beginners:</b> Training large models uses a lot of memory because we need to store:
/// 1. The model parameters
/// 2. Intermediate activations (outputs from each layer during forward pass)
/// 3. Gradients for each parameter
///
/// This class helps reduce memory usage through several techniques:
///
/// <b>Gradient Checkpointing:</b>
/// - Instead of storing all activations, only store "checkpoints"
/// - During backward pass, recompute activations between checkpoints
/// - Trades ~30% more compute time for ~50% less memory
///
/// <b>Activation Pooling:</b>
/// - Reuse tensor memory instead of allocating new tensors
/// - Reduces GC pressure and memory fragmentation
///
/// <b>Model Sharding:</b>
/// - Split large models across multiple GPUs
/// - Each GPU only holds part of the model
/// </para>
/// </remarks>
public class DiffusionMemoryManager<T>
{
    /// <summary>
    /// Provides numeric operations for the specific type T.
    /// </summary>
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Activation pool for tensor reuse.
    /// </summary>
    private readonly ActivationPool<T>? _activationPool;

    /// <summary>
    /// Model sharding configuration (if multi-GPU).
    /// </summary>
    private readonly ModelShard<T>? _modelShard;

    /// <summary>
    /// Memory configuration.
    /// </summary>
    public DiffusionMemoryConfig Config { get; }

    /// <summary>
    /// Whether gradient checkpointing is enabled.
    /// </summary>
    public bool CheckpointingEnabled => Config.UseGradientCheckpointing;

    /// <summary>
    /// Whether activation pooling is enabled.
    /// </summary>
    public bool PoolingEnabled => Config.UseActivationPooling;

    /// <summary>
    /// Whether model sharding is active.
    /// </summary>
    public bool ShardingEnabled => _modelShard != null;

    /// <summary>
    /// Initializes a new instance of the DiffusionMemoryManager class.
    /// </summary>
    /// <param name="config">Memory configuration options.</param>
    /// <param name="layers">Optional layers for model sharding.</param>
    public DiffusionMemoryManager(DiffusionMemoryConfig? config = null, IEnumerable<ILayer<T>>? layers = null)
    {
        Config = config ?? new DiffusionMemoryConfig();

        // Initialize activation pool if enabled
        if (Config.UseActivationPooling)
        {
            _activationPool = new ActivationPool<T>(Config.MaxPoolMemoryMB);
        }

        // Initialize model sharding if configured
        if (Config.NumDevices > 1 && layers != null)
        {
            var shardingConfig = new ShardingConfig
            {
                Strategy = Config.ShardingStrategy,
                UsePipelineParallelism = Config.UsePipelineParallelism
            };
            _modelShard = new ModelShard<T>(layers, Config.NumDevices, shardingConfig);
        }
    }

    #region Gradient Checkpointing Integration

    /// <summary>
    /// Wraps a function with gradient checkpointing for memory-efficient training.
    /// </summary>
    /// <param name="function">The function to execute with checkpointing.</param>
    /// <param name="inputs">The input computation nodes.</param>
    /// <returns>The checkpointed output node.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this to wrap expensive computations (like attention blocks).
    ///
    /// ```csharp
    /// // Without checkpointing (stores all activations):
    /// var output = attentionBlock.Forward(input);
    ///
    /// // With checkpointing (recomputes during backward):
    /// var output = memoryManager.Checkpoint(
    ///     () => attentionBlock.Forward(inputNode),
    ///     new[] { inputNode }
    /// );
    /// ```
    /// </para>
    /// </remarks>
    public ComputationNode<T> Checkpoint(
        Func<ComputationNode<T>> function,
        IEnumerable<ComputationNode<T>> inputs)
    {
        if (Config.UseGradientCheckpointing)
        {
            return GradientCheckpointing<T>.Checkpoint(function, inputs);
        }

        // If checkpointing disabled, just execute the function
        return function();
    }

    /// <summary>
    /// Applies checkpointing to a sequence of layer functions.
    /// </summary>
    /// <param name="layers">The sequence of layer forward functions.</param>
    /// <param name="input">The input node.</param>
    /// <returns>The output after all layers.</returns>
    public ComputationNode<T> CheckpointSequence(
        IReadOnlyList<Func<ComputationNode<T>, ComputationNode<T>>> layers,
        ComputationNode<T> input)
    {
        if (Config.UseGradientCheckpointing)
        {
            return GradientCheckpointing<T>.SequentialCheckpoint(
                layers, input, Config.CheckpointEveryNLayers);
        }

        // If checkpointing disabled, execute all layers directly
        var current = input;
        foreach (var layer in layers)
        {
            current = layer(current);
        }
        return current;
    }

    /// <summary>
    /// Executes a forward pass through layers with optional checkpointing.
    /// </summary>
    /// <param name="layers">The layers to execute.</param>
    /// <param name="input">Input tensor.</param>
    /// <returns>Output tensor after all layers.</returns>
    /// <remarks>
    /// <para>
    /// This is the tensor-based equivalent for layers that don't use the autodiff system.
    /// It provides checkpointing by storing only checkpoint activations and recomputing
    /// intermediate ones during backward pass.
    /// </para>
    /// </remarks>
    public (Tensor<T> Output, LayerCheckpointState<T> State) ForwardWithCheckpointing(
        IReadOnlyList<ILayer<T>> layers,
        Tensor<T> input)
    {
        var state = new LayerCheckpointState<T>();
        var current = input;

        for (int i = 0; i < layers.Count; i++)
        {
            // Store checkpoint at specified intervals
            if (Config.UseGradientCheckpointing && i % Config.CheckpointEveryNLayers == 0)
            {
                state.SaveCheckpoint(i, current.Clone());
            }

            current = layers[i].Forward(current);
        }

        state.FinalOutput = current;
        state.Layers = layers;
        return (current, state);
    }

    /// <summary>
    /// Performs backward pass with checkpointing, recomputing activations as needed.
    /// </summary>
    /// <param name="outputGradient">Gradient from subsequent layer.</param>
    /// <param name="state">Checkpoint state from forward pass.</param>
    /// <returns>Gradient with respect to input.</returns>
    public Tensor<T> BackwardWithCheckpointing(
        Tensor<T> outputGradient,
        LayerCheckpointState<T> state)
    {
        if (state.Layers == null)
            throw new InvalidOperationException("Checkpoint state is missing layer information.");

        var layers = state.Layers;
        var gradient = outputGradient;

        // Process backward in reverse order
        for (int i = layers.Count - 1; i >= 0; i--)
        {
            // If we need to recompute activations for this segment
            if (Config.UseGradientCheckpointing && ShouldRecompute(i, state))
            {
                RecomputeForBackward(layers, i, state);
            }

            gradient = layers[i].Backward(gradient);
        }

        return gradient;
    }

    private bool ShouldRecompute(int layerIndex, LayerCheckpointState<T> state)
    {
        // Check if this layer is at a checkpoint boundary
        int checkpointIndex = (layerIndex / Config.CheckpointEveryNLayers) * Config.CheckpointEveryNLayers;
        return !state.HasActivation(layerIndex) && state.HasCheckpoint(checkpointIndex);
    }

    private void RecomputeForBackward(IReadOnlyList<ILayer<T>> layers, int targetIndex, LayerCheckpointState<T> state)
    {
        // Find the nearest checkpoint before the target
        int checkpointIndex = (targetIndex / Config.CheckpointEveryNLayers) * Config.CheckpointEveryNLayers;
        var checkpoint = state.GetCheckpoint(checkpointIndex);

        if (checkpoint == null)
            return;

        // Recompute forward from checkpoint to target
        var current = checkpoint;
        for (int i = checkpointIndex; i < targetIndex; i++)
        {
            current = layers[i].Forward(current);
            state.SaveActivation(i + 1, current);
        }
    }

    #endregion

    #region Activation Pooling Integration

    /// <summary>
    /// Rents a tensor from the activation pool.
    /// </summary>
    /// <param name="shape">Desired tensor shape.</param>
    /// <returns>A tensor from the pool (or newly allocated if pool unavailable).</returns>
    public Tensor<T> RentTensor(int[] shape)
    {
        if (_activationPool != null)
        {
            return _activationPool.Rent(shape);
        }

        return new Tensor<T>(shape);
    }

    /// <summary>
    /// Returns a tensor to the activation pool for reuse.
    /// </summary>
    /// <param name="tensor">The tensor to return.</param>
    public void ReturnTensor(Tensor<T> tensor)
    {
        _activationPool?.Return(tensor);
    }

    /// <summary>
    /// Gets pooling statistics if available.
    /// </summary>
    public ActivationPoolStats? GetPoolStats() => _activationPool?.Stats;

    #endregion

    #region Model Sharding Integration

    /// <summary>
    /// Performs forward pass through sharded model.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Output tensor.</returns>
    public Tensor<T> ShardedForward(Tensor<T> input)
    {
        if (_modelShard == null)
            throw new InvalidOperationException("Model sharding is not configured.");

        return _modelShard.Forward(input);
    }

    /// <summary>
    /// Performs forward pass through sharded model with context.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <param name="context">Context tensor (e.g., timestep embedding).</param>
    /// <returns>Output tensor.</returns>
    public Tensor<T> ShardedForward(Tensor<T> input, Tensor<T>? context)
    {
        if (_modelShard == null)
            throw new InvalidOperationException("Model sharding is not configured.");

        return _modelShard.Forward(input, context);
    }

    /// <summary>
    /// Performs backward pass through sharded model.
    /// </summary>
    /// <param name="outputGradient">Gradient from subsequent layer.</param>
    /// <returns>Gradient with respect to input.</returns>
    public Tensor<T> ShardedBackward(Tensor<T> outputGradient)
    {
        if (_modelShard == null)
            throw new InvalidOperationException("Model sharding is not configured.");

        return _modelShard.Backward(outputGradient);
    }

    /// <summary>
    /// Updates parameters across all shards.
    /// </summary>
    /// <param name="learningRate">Learning rate.</param>
    public void ShardedUpdateParameters(T learningRate)
    {
        _modelShard?.UpdateParameters(learningRate);
    }

    /// <summary>
    /// Gets memory usage per device.
    /// </summary>
    public IReadOnlyDictionary<int, long>? GetDeviceMemoryUsage()
    {
        return _modelShard?.GetDeviceMemoryUsage();
    }

    #endregion

    #region Memory Estimation

    /// <summary>
    /// Estimates memory savings from current configuration.
    /// </summary>
    /// <param name="numLayers">Number of layers in the model.</param>
    /// <param name="activationSizeBytes">Size of activations per layer in bytes.</param>
    /// <returns>Estimated memory usage information.</returns>
    public MemoryEstimate EstimateMemory(int numLayers, long activationSizeBytes)
    {
        var estimate = new MemoryEstimate();

        // Without checkpointing: store all activations
        estimate.WithoutCheckpointing = numLayers * activationSizeBytes;

        // With checkpointing
        if (Config.UseGradientCheckpointing)
        {
            var (_, withCheckpoint, _) = GradientCheckpointing<T>.EstimateMemorySavings(
                numLayers, activationSizeBytes, Config.CheckpointEveryNLayers);
            estimate.WithCheckpointing = withCheckpoint;
        }
        else
        {
            estimate.WithCheckpointing = estimate.WithoutCheckpointing;
        }

        // With sharding
        if (_modelShard != null)
        {
            var memoryPerDevice = _modelShard.GetDeviceMemoryUsage();
            estimate.PerDeviceMemory = memoryPerDevice.Values.Max();
            estimate.TotalShardedMemory = memoryPerDevice.Values.Sum();
        }

        estimate.SavingsPercent = 100.0 * (1.0 - (double)estimate.WithCheckpointing / estimate.WithoutCheckpointing);

        return estimate;
    }

    #endregion
}

/// <summary>
/// Configuration for diffusion model memory management.
/// </summary>
public class DiffusionMemoryConfig
{
    /// <summary>
    /// Whether to use gradient checkpointing.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Enable this to reduce memory usage by about 50%
    /// at the cost of about 30% more computation time.
    /// </para>
    /// </remarks>
    public bool UseGradientCheckpointing { get; set; } = true;

    /// <summary>
    /// Number of layers between checkpoints.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Smaller values = more checkpoints = less recomputation but more memory.
    /// Larger values = fewer checkpoints = more recomputation but less memory.
    /// Default of 2 is a good balance for most models.
    /// </para>
    /// </remarks>
    public int CheckpointEveryNLayers { get; set; } = 2;

    /// <summary>
    /// Whether to use activation pooling.
    /// </summary>
    public bool UseActivationPooling { get; set; } = true;

    /// <summary>
    /// Maximum memory for activation pool in MB.
    /// </summary>
    public long MaxPoolMemoryMB { get; set; } = 2048;

    /// <summary>
    /// Maximum tensors per size class in the pool.
    /// </summary>
    public int MaxTensorsPerSize { get; set; } = 8;

    /// <summary>
    /// Number of devices for model sharding.
    /// </summary>
    /// <remarks>
    /// Set to 1 for single-GPU/CPU training.
    /// Set to higher values for multi-GPU training.
    /// </remarks>
    public int NumDevices { get; set; } = 1;

    /// <summary>
    /// Strategy for distributing layers across devices.
    /// </summary>
    public ShardingStrategy ShardingStrategy { get; set; } = ShardingStrategy.EvenSplit;

    /// <summary>
    /// Whether to use pipeline parallelism for overlapping compute and transfer.
    /// </summary>
    public bool UsePipelineParallelism { get; set; } = false;

    /// <summary>
    /// Creates a default configuration for single-GPU training.
    /// </summary>
    public static DiffusionMemoryConfig Default => new();

    /// <summary>
    /// Creates a memory-efficient configuration with aggressive checkpointing.
    /// </summary>
    public static DiffusionMemoryConfig MemoryEfficient => new()
    {
        UseGradientCheckpointing = true,
        CheckpointEveryNLayers = 1, // Checkpoint every layer
        UseActivationPooling = true,
        MaxPoolMemoryMB = 1024
    };

    /// <summary>
    /// Creates a speed-optimized configuration with less checkpointing.
    /// </summary>
    public static DiffusionMemoryConfig SpeedOptimized => new()
    {
        UseGradientCheckpointing = true,
        CheckpointEveryNLayers = 4, // Checkpoint every 4 layers
        UseActivationPooling = true,
        MaxPoolMemoryMB = 4096
    };

    /// <summary>
    /// Creates a multi-GPU configuration.
    /// </summary>
    /// <param name="numDevices">Number of GPUs.</param>
    public static DiffusionMemoryConfig MultiGpu(int numDevices) => new()
    {
        UseGradientCheckpointing = true,
        CheckpointEveryNLayers = 2,
        UseActivationPooling = true,
        NumDevices = numDevices,
        ShardingStrategy = ShardingStrategy.MemoryBalanced,
        UsePipelineParallelism = true
    };
}

/// <summary>
/// State for layer-based gradient checkpointing.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class LayerCheckpointState<T>
{
    /// <summary>
    /// Saved checkpoints at specific layer indices.
    /// </summary>
    private readonly Dictionary<int, Tensor<T>> _checkpoints = new();

    /// <summary>
    /// Recomputed activations during backward pass.
    /// </summary>
    private readonly Dictionary<int, Tensor<T>> _activations = new();

    /// <summary>
    /// The layers used in forward pass.
    /// </summary>
    public IReadOnlyList<ILayer<T>>? Layers { get; set; }

    /// <summary>
    /// Final output from forward pass.
    /// </summary>
    public Tensor<T>? FinalOutput { get; set; }

    /// <summary>
    /// Saves a checkpoint at the given layer index.
    /// </summary>
    public void SaveCheckpoint(int layerIndex, Tensor<T> activation)
    {
        _checkpoints[layerIndex] = activation;
    }

    /// <summary>
    /// Saves a recomputed activation.
    /// </summary>
    public void SaveActivation(int layerIndex, Tensor<T> activation)
    {
        _activations[layerIndex] = activation;
    }

    /// <summary>
    /// Gets a checkpoint if available.
    /// </summary>
    public Tensor<T>? GetCheckpoint(int layerIndex)
    {
        return _checkpoints.TryGetValue(layerIndex, out var checkpoint) ? checkpoint : null;
    }

    /// <summary>
    /// Gets a recomputed activation if available.
    /// </summary>
    public Tensor<T>? GetActivation(int layerIndex)
    {
        return _activations.TryGetValue(layerIndex, out var activation) ? activation : null;
    }

    /// <summary>
    /// Checks if a checkpoint exists.
    /// </summary>
    public bool HasCheckpoint(int layerIndex) => _checkpoints.ContainsKey(layerIndex);

    /// <summary>
    /// Checks if an activation exists.
    /// </summary>
    public bool HasActivation(int layerIndex) => _activations.ContainsKey(layerIndex);

    /// <summary>
    /// Clears all stored state to free memory.
    /// </summary>
    public void Clear()
    {
        _checkpoints.Clear();
        _activations.Clear();
        FinalOutput = null;
    }
}

/// <summary>
/// Memory usage estimate.
/// </summary>
public class MemoryEstimate
{
    /// <summary>
    /// Estimated memory usage without checkpointing (bytes).
    /// </summary>
    public long WithoutCheckpointing { get; set; }

    /// <summary>
    /// Estimated memory usage with checkpointing (bytes).
    /// </summary>
    public long WithCheckpointing { get; set; }

    /// <summary>
    /// Memory per device if sharded (bytes).
    /// </summary>
    public long? PerDeviceMemory { get; set; }

    /// <summary>
    /// Total memory across all devices if sharded (bytes).
    /// </summary>
    public long? TotalShardedMemory { get; set; }

    /// <summary>
    /// Percentage memory savings from checkpointing.
    /// </summary>
    public double SavingsPercent { get; set; }

    /// <summary>
    /// Gets a human-readable summary.
    /// </summary>
    public override string ToString()
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine($"Memory Estimate:");
        sb.AppendLine($"  Without checkpointing: {WithoutCheckpointing / (1024.0 * 1024.0):F1} MB");
        sb.AppendLine($"  With checkpointing:    {WithCheckpointing / (1024.0 * 1024.0):F1} MB");
        sb.AppendLine($"  Savings:               {SavingsPercent:F1}%");

        if (PerDeviceMemory.HasValue)
        {
            sb.AppendLine($"  Per device:            {PerDeviceMemory.Value / (1024.0 * 1024.0):F1} MB");
            sb.AppendLine($"  Total sharded:         {TotalShardedMemory!.Value / (1024.0 * 1024.0):F1} MB");
        }

        return sb.ToString();
    }
}
