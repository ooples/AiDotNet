using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the MobileNetV2Network.
/// </summary>
public class MobileNetV2Options : ModelHyperparameterOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="MobileNetV2Options"/> class.
    /// </summary>
    public MobileNetV2Options()
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="MobileNetV2Options"/> class
    /// by copying an existing options instance.
    /// </summary>
    /// <param name="other">The options instance to copy.</param>
    public MobileNetV2Options(MobileNetV2Options other)
    {
        if (other is null)
            throw new ArgumentNullException(nameof(other));

        MaxGradNorm = other.MaxGradNorm;
        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
        DisableFusedOptimizerStep = other.DisableFusedOptimizerStep;
    }

    /// <summary>
    /// Gets or sets whether MobileNetV2 should bypass the generic fused
    /// compiled optimizer step and train through the eager tape path.
    /// </summary>
    /// <remarks>
    /// Defaults to <see langword="false"/>: the compiled step reproduces the eager tape's loss and update for the
    /// stem and the inverted-residual blocks MobileNetV2 builds (with and without expansion, stride 1 with the
    /// residual add, stride 2; #2382). It was <see langword="true"/>
    /// while the eager path froze the top-level batch-norm statistics and the compiled step did not, which read as a
    /// plan defect. Set it to train through the eager tape instead.
    /// </remarks>
    public bool DisableFusedOptimizerStep { get; set; } = false;
}
