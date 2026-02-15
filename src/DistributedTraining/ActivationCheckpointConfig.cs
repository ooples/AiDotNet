namespace AiDotNet.DistributedTraining;

/// <summary>
/// Configuration for activation checkpointing in pipeline parallel training.
/// </summary>
/// <remarks>
/// <para>
/// Activation checkpointing (also called gradient checkpointing) trades compute for memory
/// by only storing activations at checkpoint layers during the forward pass. Intermediate
/// activations are recomputed from the nearest checkpoint during the backward pass.
/// </para>
/// <para><b>For Beginners:</b> During training, the forward pass must save intermediate results
/// (activations) so the backward pass can compute gradients. For very deep models, storing all
/// these activations uses enormous amounts of memory.
///
/// Activation checkpointing is like taking notes at chapter boundaries instead of every page:
/// - Without checkpointing: Save every activation (lots of memory, no recomputation)
/// - With checkpointing: Save every Nth activation, recompute the rest (less memory, more compute)
///
/// Memory savings: O(L) → O(sqrt(L)) where L = number of layers.
/// For 100 layers, this reduces memory from 100 activations to ~10 activations.
///
/// The trade-off is ~33% more compute time, but this enables training models that otherwise
/// wouldn't fit in memory.
/// </para>
/// <para><b>Reference:</b> Chen et al., "Training Deep Nets with Sublinear Memory Cost", 2016.
/// https://arxiv.org/abs/1604.06174</para>
/// </remarks>
public class ActivationCheckpointConfig
{
    private int _checkpointEveryNLayers = 10;
    private int _maxActivationsInMemory;

    /// <summary>
    /// Gets or sets whether activation checkpointing is enabled.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Set this to true to enable memory savings. Default is false
    /// (no checkpointing, standard behavior).</para>
    /// </remarks>
    public bool Enabled { get; set; }

    /// <summary>
    /// Gets or sets how often to save a checkpoint (every N layers).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Lower values save more activations (more memory, less recomputation).
    /// Higher values save fewer (less memory, more recomputation).
    ///
    /// Optimal value is approximately sqrt(total_layers) for minimum total cost.
    /// For a 100-layer model, checkpointing every 10 layers is a good default.
    ///
    /// Default: 10 layers between checkpoints.</para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when value is less than 1.</exception>
    public int CheckpointEveryNLayers
    {
        get => _checkpointEveryNLayers;
        set
        {
            if (value < 1)
            {
                throw new ArgumentOutOfRangeException(nameof(value),
                    $"CheckpointEveryNLayers must be at least 1, but was {value}. " +
                    "A value of 0 would cause division-by-zero in interval-based checkpointing.");
            }
            _checkpointEveryNLayers = value;
        }
    }

    /// <summary>
    /// Gets or sets the recomputation strategy to use during the backward pass.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// - Selective: Only recompute activations that are needed and not checkpointed (recommended)
    /// - Full: Recompute all non-checkpointed activations from the previous checkpoint
    /// - None: Don't recompute, equivalent to no checkpointing (for testing/debugging)
    /// </para>
    /// </remarks>
    public RecomputeStrategy RecomputeStrategy { get; set; } = RecomputeStrategy.Selective;

    /// <summary>
    /// Gets or sets the maximum number of activations to keep in memory simultaneously.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This caps how many activations are stored at once.
    /// Set to 0 for no limit (uses CheckpointEveryNLayers to determine storage).
    /// A non-zero value overrides CheckpointEveryNLayers by dynamically adjusting
    /// the checkpoint frequency to stay within the memory budget.</para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when value is negative.</exception>
    public int MaxActivationsInMemory
    {
        get => _maxActivationsInMemory;
        set
        {
            if (value < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(value),
                    $"MaxActivationsInMemory must be non-negative, but was {value}. " +
                    "Use 0 for no limit.");
            }
            _maxActivationsInMemory = value;
        }
    }

    /// <summary>
    /// Gets or sets whether to checkpoint the very first layer's input.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The first layer's input is always needed for the backward pass.
    /// If true, it's saved as a checkpoint. If false, the caller must ensure the input is
    /// available during the backward pass (which is usually the case).</para>
    /// </remarks>
    public bool CheckpointFirstLayer { get; set; } = true;
}

/// <summary>
/// Strategy for recomputing activations during the backward pass.
/// </summary>
public enum RecomputeStrategy
{
    /// <summary>
    /// Only recompute activations that are needed for the current backward step.
    /// This is the most memory-efficient but requires careful bookkeeping.
    /// </summary>
    Selective,

    /// <summary>
    /// Recompute all activations between the two nearest checkpoints during backward.
    /// Simpler implementation but may do slightly more work than necessary.
    /// </summary>
    Full,

    /// <summary>
    /// No recomputation. Equivalent to disabled checkpointing. Useful for debugging.
    /// </summary>
    None
}
