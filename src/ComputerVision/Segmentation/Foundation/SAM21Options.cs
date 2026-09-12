using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Foundation;

/// <summary>
/// Configuration options for SAM 2.1 (Segment Anything Model 2.1).
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> SAM 2.1 is an improved version of SAM 2 with refined checkpoints
/// for better segmentation accuracy. Options inherit from NeuralNetworkOptions.
/// </para>
/// </remarks>
public class SAM21Options : SegmentationModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public SAM21Options()
    {
        NumClasses = 1;
        DropRate = 0.1;
        ModelSize = SAM21ModelSize.Large;
    }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public SAM21Options(SAM21Options other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
        MaxGradNorm = other.MaxGradNorm;
        NumClasses = other.NumClasses;
        DropRate = other.DropRate;
        ModelSize = other.ModelSize;
        MemoryBankSize = other.MemoryBankSize;
    }

    /// <summary>
    /// Maximum number of frames to keep in the memory bank for video segmentation.
    /// When null, defaults to 7 (as in the SAM 2 paper).
    /// </summary>
    public int? MemoryBankSize { get; set; }

    /// <summary>
    /// Gets or sets which published size variant of the model to build.
    /// Default: <c>SAM21ModelSize.Large</c>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Most of these models were published in several sizes that
    /// trade accuracy against speed and memory. Picking a variant selects the widths and
    /// depths the paper reports for it; it is not a hint, it changes the network that gets
    /// built.
    /// </para>
    /// <para>
    /// Declared here rather than on <see cref="SegmentationModelOptions"/> because each
    /// model names its own variants with its own enum, so there is no shared type to
    /// declare.
    /// </para>
    /// </remarks>
    public SAM21ModelSize ModelSize { get; set; }

    /// <summary>
    /// Throws when a value on this instance cannot produce a working model.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when <see cref="SegmentationModelOptions.NumClasses"/> is not positive, or when
    /// <see cref="SegmentationModelOptions.DropRate"/> is not a fraction in [0, 1).
    /// </exception>
    public void Validate() => ValidateSegmentationCore();
}
