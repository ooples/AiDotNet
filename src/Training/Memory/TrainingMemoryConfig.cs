using AiDotNet.Diffusion.Memory;
using AiDotNet.Initialization;
using AiDotNet.Memory;

namespace AiDotNet.Training.Memory;

/// <summary>
/// Configuration for training memory management including gradient checkpointing,
/// activation pooling, and model sharding.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Training neural networks requires a lot of memory:
///
/// 1. **Model weights**: The parameters being trained
/// 2. **Activations**: Intermediate results saved for backpropagation
/// 3. **Gradients**: Computed during backward pass
///
/// This configuration helps reduce memory usage through:
/// - **Gradient Checkpointing**: Trade compute for memory by recomputing activations
/// - **Activation Pooling**: Reuse memory buffers to reduce garbage collection
/// - **Model Sharding**: Split large models across multiple GPUs
/// </para>
/// <para>
/// Example usage with PredictionModelBuilder:
/// <code>
/// // Using a preset for common scenarios
/// var builder = new PredictionModelBuilder&lt;double, double[], double&gt;()
///     .ConfigureMemoryManagement(TrainingMemoryConfig.ForTransformers());
///
/// // Or with custom settings
/// var builder = new PredictionModelBuilder&lt;double, double[], double&gt;()
///     .ConfigureMemoryManagement(new TrainingMemoryConfig
///     {
///         UseGradientCheckpointing = true,
///         CheckpointEveryNLayers = 2,
///         UseActivationPooling = true,
///         MaxPoolMemoryMB = 2048
///     });
/// </code>
/// </para>
/// </remarks>
public class TrainingMemoryConfig
{
    #region Gradient Checkpointing

    /// <summary>
    /// Gets or sets whether to use gradient checkpointing.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Gradient checkpointing saves memory by not storing
    /// all intermediate activations. Instead, it saves checkpoints and recomputes
    /// activations during the backward pass. This trades ~30% extra compute time
    /// for ~40-50% memory savings.
    /// </para>
    /// </remarks>
    public bool UseGradientCheckpointing { get; set; } = false;

    /// <summary>
    /// Gets or sets how often to create checkpoints (every N layers).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Lower values = more memory savings but more recomputation.
    /// Higher values = less recomputation but more memory usage.
    /// Typical values: 1-4 for transformers, 2-3 for CNNs.
    /// </para>
    /// </remarks>
    public int CheckpointEveryNLayers { get; set; } = 2;

    /// <summary>
    /// Gets or sets whether to checkpoint attention layers specifically.
    /// </summary>
    /// <remarks>
    /// Attention layers are memory-intensive (O(n^2) for sequence length n).
    /// Checkpointing them can significantly reduce memory for transformer models.
    /// </remarks>
    public bool CheckpointAttentionLayers { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to checkpoint residual blocks.
    /// </summary>
    public bool CheckpointResidualBlocks { get; set; } = true;

    #endregion

    #region Activation Pooling

    /// <summary>
    /// Gets or sets whether to use activation pooling.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Activation pooling reuses tensor memory instead of
    /// allocating new memory for each operation. This reduces garbage collection
    /// pressure and can significantly improve training speed for large models.
    /// </para>
    /// </remarks>
    public bool UseActivationPooling { get; set; } = true;

    /// <summary>
    /// Gets or sets the maximum memory for the activation pool in megabytes.
    /// </summary>
    public long MaxPoolMemoryMB { get; set; } = 4096;

    #endregion

    #region Model Sharding

    /// <summary>
    /// Gets or sets the number of devices for model sharding.
    /// </summary>
    /// <remarks>
    /// Set to 1 for single-device training. Values > 1 enable model parallelism.
    /// </remarks>
    public int NumDevices { get; set; } = 1;

    /// <summary>
    /// Gets or sets the sharding strategy.
    /// </summary>
    public ShardingStrategy ShardingStrategy { get; set; } = ShardingStrategy.EvenSplit;

    /// <summary>
    /// Gets or sets whether to use pipeline parallelism.
    /// </summary>
    /// <remarks>
    /// Pipeline parallelism overlaps computation across shards for better GPU utilization.
    /// </remarks>
    public bool UsePipelineParallelism { get; set; } = false;

    /// <summary>
    /// Gets or sets the number of micro-batches for pipeline parallelism.
    /// </summary>
    public int MicroBatchCount { get; set; } = 4;

    #endregion

    #region Memory Estimation

    /// <summary>
    /// Gets or sets whether to track detailed memory statistics.
    /// </summary>
    public bool TrackMemoryStatistics { get; set; } = false;

    /// <summary>
    /// Gets or sets the warning threshold for memory usage (as fraction of max).
    /// </summary>
    public double MemoryWarningThreshold { get; set; } = 0.9;

    #endregion

    #region Tensor Pooling

    /// <summary>
    /// Gets or sets whether tensor pooling is enabled.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Tensor pooling reduces memory allocations by reusing
    /// tensor buffers. When enabled, tensors are borrowed from and returned to a pool
    /// instead of being allocated and garbage collected each time.
    /// </para>
    /// </remarks>
    public bool UseTensorPooling { get; set; } = true;

    /// <summary>
    /// Gets or sets the tensor pooling options.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Configure advanced pooling behavior including:
    /// - Maximum pool size in MB
    /// - Maximum elements per buffer to pool
    /// - Whether to use weak references (allows GC under memory pressure)
    /// </para>
    /// </remarks>
    public PoolingOptions? TensorPoolOptions { get; set; }

    /// <summary>
    /// Gets or sets whether to use a shared global tensor pool.
    /// </summary>
    /// <remarks>
    /// When true, uses TensorPoolManager.Shared for all pooling operations.
    /// When false, each training session creates its own pool.
    /// </remarks>
    public bool UseSharedTensorPool { get; set; } = true;

    #endregion

    #region Weight Initialization

    /// <summary>
    /// Gets or sets the default initialization strategy type.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This controls how layer weights are initialized:
    /// - <see cref="InitializationStrategyType.Eager"/>: Initialize immediately (traditional)
    /// - <see cref="InitializationStrategyType.Lazy"/>: Defer until first use (faster construction)
    /// - <see cref="InitializationStrategyType.Zero"/>: All zeros (for testing only)
    /// - <see cref="InitializationStrategyType.FromFile"/>: Load from a file (transfer learning)
    /// </para>
    /// </remarks>
    public InitializationStrategyType DefaultInitializationStrategy { get; set; } = InitializationStrategyType.Eager;

    /// <summary>
    /// Gets or sets the path to a weights file for FromFile initialization.
    /// </summary>
    /// <remarks>
    /// Only used when <see cref="DefaultInitializationStrategy"/> is set to FromFile.
    /// </remarks>
    public string? WeightsFilePath { get; set; }

    /// <summary>
    /// Gets or sets the format of the weights file.
    /// </summary>
    public WeightFileFormat WeightsFileFormat { get; set; } = WeightFileFormat.Auto;

    #endregion

    #region Factory Methods

    /// <summary>
    /// Creates a default configuration with minimal memory optimization.
    /// </summary>
    public static TrainingMemoryConfig Default() => new();

    /// <summary>
    /// Creates a memory-efficient configuration for large models.
    /// </summary>
    /// <remarks>
    /// Enables gradient checkpointing and activation pooling for maximum memory savings.
    /// Best for training models that don't fit in GPU memory otherwise.
    /// </remarks>
    public static TrainingMemoryConfig MemoryEfficient() => new()
    {
        UseGradientCheckpointing = true,
        CheckpointEveryNLayers = 1,
        CheckpointAttentionLayers = true,
        CheckpointResidualBlocks = true,
        UseActivationPooling = true,
        MaxPoolMemoryMB = 2048
    };

    /// <summary>
    /// Creates a speed-optimized configuration (less memory optimization).
    /// </summary>
    /// <remarks>
    /// Disables gradient checkpointing for maximum speed.
    /// Best when you have enough GPU memory for your model.
    /// </remarks>
    public static TrainingMemoryConfig SpeedOptimized() => new()
    {
        UseGradientCheckpointing = false,
        UseActivationPooling = true,
        MaxPoolMemoryMB = 8192
    };

    /// <summary>
    /// Creates a configuration for multi-GPU training.
    /// </summary>
    /// <param name="numDevices">Number of GPUs to use.</param>
    public static TrainingMemoryConfig MultiGpu(int numDevices) => new()
    {
        UseGradientCheckpointing = true,
        CheckpointEveryNLayers = 2,
        UseActivationPooling = true,
        MaxPoolMemoryMB = 4096,
        NumDevices = numDevices,
        ShardingStrategy = ShardingStrategy.MemoryBalanced,
        UsePipelineParallelism = true,
        MicroBatchCount = 4
    };

    /// <summary>
    /// Creates a configuration optimized for transformer models.
    /// </summary>
    public static TrainingMemoryConfig ForTransformers() => new()
    {
        UseGradientCheckpointing = true,
        CheckpointEveryNLayers = 1,
        CheckpointAttentionLayers = true,
        CheckpointResidualBlocks = true,
        UseActivationPooling = true,
        MaxPoolMemoryMB = 4096
    };

    /// <summary>
    /// Creates a configuration optimized for convolutional networks.
    /// </summary>
    public static TrainingMemoryConfig ForConvNets() => new()
    {
        UseGradientCheckpointing = true,
        CheckpointEveryNLayers = 3,
        CheckpointAttentionLayers = false,
        CheckpointResidualBlocks = true,
        UseActivationPooling = true,
        MaxPoolMemoryMB = 2048
    };

    /// <summary>
    /// Creates a configuration for transfer learning from a pre-trained model.
    /// </summary>
    /// <param name="weightsPath">Path to the pre-trained weights file.</param>
    /// <param name="format">Format of the weights file. Default is auto-detect.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Transfer learning lets you start with a model that was
    /// already trained on a similar task. This is much faster than training from scratch
    /// and often produces better results, especially with limited data.
    /// </para>
    /// </remarks>
    public static TrainingMemoryConfig ForTransferLearning(string weightsPath, WeightFileFormat format = WeightFileFormat.Auto) => new()
    {
        UseGradientCheckpointing = true,
        CheckpointEveryNLayers = 2,
        UseActivationPooling = true,
        MaxPoolMemoryMB = 4096,
        UseTensorPooling = true,
        DefaultInitializationStrategy = InitializationStrategyType.FromFile,
        WeightsFilePath = weightsPath,
        WeightsFileFormat = format
    };

    /// <summary>
    /// Creates a configuration for fast model construction (good for testing).
    /// </summary>
    /// <remarks>
    /// Uses lazy initialization to defer weight allocation, making network construction fast.
    /// Useful when you just want to inspect the network architecture or run tests.
    /// </remarks>
    public static TrainingMemoryConfig FastConstruction() => new()
    {
        UseGradientCheckpointing = false,
        UseActivationPooling = false,
        UseTensorPooling = false,
        DefaultInitializationStrategy = InitializationStrategyType.Lazy
    };

    /// <summary>
    /// Creates a configuration with aggressive memory pooling.
    /// </summary>
    /// <param name="maxPoolSizeMB">Maximum size of the tensor pool in MB.</param>
    /// <remarks>
    /// Maximizes tensor reuse to minimize garbage collection.
    /// Best for training loops where the same tensor shapes are used repeatedly.
    /// </remarks>
    public static TrainingMemoryConfig AggressivePooling(int maxPoolSizeMB = 512) => new()
    {
        UseActivationPooling = true,
        MaxPoolMemoryMB = maxPoolSizeMB,
        UseTensorPooling = true,
        UseSharedTensorPool = true,
        TensorPoolOptions = new PoolingOptions
        {
            MaxPoolSizeMB = maxPoolSizeMB,
            MaxItemsPerBucket = 20,
            UseWeakReferences = false
        }
    };

    #endregion
}
