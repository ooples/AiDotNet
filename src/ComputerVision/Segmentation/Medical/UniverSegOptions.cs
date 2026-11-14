using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Medical;

/// <summary>
/// Configuration options for UniverSeg few-shot medical segmentation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the UniverSeg model. Default values follow the original paper settings.</para>
/// </remarks>
public class UniverSegOptions : SegmentationModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public UniverSegOptions()
    {
        NumClasses = 1;
        DropRate = 0;
    }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public UniverSegOptions(UniverSegOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
        MaxGradNorm = other.MaxGradNorm;
        NumClasses = other.NumClasses;
        DropRate = other.DropRate;
        EncoderFeatures = other.EncoderFeatures?.ToArray() ?? Array.Empty<int>();
        DecoderFeatures = other.DecoderFeatures?.ToArray() ?? Array.Empty<int>();
    }

    /// <summary>
    /// Gets or sets the width of each encoder CrossBlock, one per resolution level. Default: 64 at four levels
    /// (Butoi et al. 2023).
    /// </summary>
    public int[] EncoderFeatures { get; set; } = [64, 64, 64, 64];

    /// <summary>
    /// Gets or sets the width of each decoder CrossBlock; one fewer than the encoder levels. Default: 64 at three
    /// levels (Butoi et al. 2023).
    /// </summary>
    public int[] DecoderFeatures { get; set; } = [64, 64, 64];

    /// <summary>
    /// Throws when a value on this instance cannot produce a working model.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when <see cref="SegmentationModelOptions.NumClasses"/> is not positive, or when
    /// <see cref="SegmentationModelOptions.DropRate"/> is not a fraction in [0, 1).
    /// </exception>
    public void Validate() => ValidateSegmentationCore();
}
