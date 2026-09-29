using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Foundation;

/// <summary>
/// Configuration options for the U2Seg model.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> U2Seg is a unified unsupervised segmentation framework that performs
/// instance, semantic, and panoptic segmentation without requiring any human annotations.
/// Options inherit from NeuralNetworkOptions.
/// </para>
/// </remarks>
public class U2SegOptions : SegmentationModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public U2SegOptions()
    {
        NumClasses = 150;
        DropRate = 0.1;
    }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public U2SegOptions(U2SegOptions other)
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
