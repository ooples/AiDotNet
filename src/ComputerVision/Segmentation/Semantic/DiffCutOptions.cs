using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Semantic;

/// <summary>
/// Configuration options for the DiffCut semantic segmentation model.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> DiffCut options inherit from NeuralNetworkOptions, which provides
/// a Seed property for reproducibility. DiffCut uses diffusion model features combined with
/// Normalized Cut graph partitioning for zero-shot semantic segmentation — no training labels needed.
/// </para>
/// </remarks>
public class DiffCutOptions : SegmentationModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public DiffCutOptions()
    {
        NumClasses = 150;
        DropRate = 0.1;
    }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public DiffCutOptions(DiffCutOptions other)
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
