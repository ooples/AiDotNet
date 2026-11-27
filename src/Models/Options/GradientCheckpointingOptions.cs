using System;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configures activation gradient checkpointing for <c>AiModelBuilder.ConfigureGradientCheckpointing</c>.
/// </summary>
/// <remarks>
/// <para>
/// Gradient checkpointing partitions a deep network into segments of <see cref="CheckpointEveryNLayers"/> layers,
/// storing only segment-boundary activations during the forward pass and recomputing interior activations on demand
/// during backward. When paired with streaming training, gradients at each segment boundary are flushed immediately
/// to the streaming optimizer via batched GPU updates.
/// </para>
/// <para><b>For Beginners:</b> Deep models normally keep every layer's intermediate output in memory until the
/// backward pass finishes. Gradient checkpointing saves memory by keeping only periodic checkpoints and recalculating
/// the layers in between, allowing deeper models or larger batch sizes on the same GPU.</para>
/// <example>
/// <code>
/// var options = new GradientCheckpointingOptions
/// {
///     Enabled = true,
///     CheckpointEveryNLayers = 4,
/// };
/// options.Validate();
/// </code>
/// </example>
/// </remarks>
public class GradientCheckpointingOptions : ModelOptions
{
    /// <summary>Initializes a new instance with default settings.</summary>
    public GradientCheckpointingOptions() { }

    /// <summary>Initializes a new instance with explicit enablement and segment length.</summary>
    /// <param name="enabled">Whether gradient checkpointing is enabled.</param>
    /// <param name="checkpointEveryNLayers">Layers per checkpointed segment, or <c>0</c> for automatic <c>sqrt(L)</c> sizing.</param>
    public GradientCheckpointingOptions(bool enabled, int checkpointEveryNLayers = 0)
    {
        Enabled = enabled;
        CheckpointEveryNLayers = checkpointEveryNLayers;
    }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException"><paramref name="other"/> is null.</exception>
    public GradientCheckpointingOptions(GradientCheckpointingOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        Enabled = other.Enabled;
        CheckpointEveryNLayers = other.CheckpointEveryNLayers;
    }

    /// <summary>Gets or sets whether activation gradient checkpointing is enabled.</summary>
    /// <value>Defaults to <c>true</c>.</value>
    public bool Enabled { get; set; } = true;

    /// <summary>
    /// Gets or sets the number of layers per checkpointed segment, or <c>0</c> to choose <c>sqrt(layerCount)</c> automatically.
    /// </summary>
    /// <value>Non-negative integer; defaults to <c>0</c> (automatic segment sizing).</value>
    public int CheckpointEveryNLayers { get; set; } = 0;

    /// <summary>Validates option values and throws if any setting is out of range.</summary>
    /// <exception cref="ArgumentOutOfRangeException"><see cref="CheckpointEveryNLayers"/> is negative.</exception>
    public void Validate()
    {
        if (CheckpointEveryNLayers < 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(CheckpointEveryNLayers),
                CheckpointEveryNLayers,
                "CheckpointEveryNLayers must be non-negative (0 selects sqrt(layerCount) automatically).");
        }
    }
}
