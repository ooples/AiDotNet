using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the MobileNetV2Network.
/// </summary>
public class MobileNetV2Options : NeuralNetworkOptions
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

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
        DisableFusedOptimizerStep = other.DisableFusedOptimizerStep;
    }

    /// <summary>
    /// Gets or sets whether MobileNetV2 should bypass the generic fused
    /// compiled optimizer step and train through the eager tape path.
    /// </summary>
    /// <remarks>
    /// The inverted residual and linear bottleneck blocks are non-sequential
    /// composites with internally materialized depthwise/projection layers.
    /// The eager tape path mirrors PyTorch autograd for this topology and
    /// avoids replaying a flattened compiled plan that does not preserve the
    /// block-local tensor flow.
    /// </remarks>
    public bool DisableFusedOptimizerStep { get; set; } = true;
}
