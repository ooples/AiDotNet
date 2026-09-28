using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Referring;

/// <summary>
/// Configuration options for GLaMM grounding language model segmentation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the GLaMM model. Default values follow the original paper settings.</para>
/// </remarks>
public class GLaMMOptions : SegmentationModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public GLaMMOptions()
    {
        NumClasses = 1;
        DropRate = 0;
    }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public GLaMMOptions(GLaMMOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
        MaxGradNorm = other.MaxGradNorm;
        NumClasses = other.NumClasses;
        DropRate = other.DropRate;
    }

    /// <summary>
    /// Throws when a value on this instance cannot produce a working model.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when <see cref="SegmentationModelOptions.NumClasses"/> is not positive, or when
    /// <see cref="SegmentationModelOptions.DropRate"/> is not a fraction in [0, 1).
    /// </exception>
    public void Validate() => ValidateSegmentationCore();
}
