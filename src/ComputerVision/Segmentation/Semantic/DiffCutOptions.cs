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
public class DiffCutOptions : NeuralNetworkOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public DiffCutOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public DiffCutOptions(DiffCutOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
    }

}
