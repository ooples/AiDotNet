using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Foundation;

/// <summary>
/// Configuration options for the UNINEXT model.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> UNINEXT reformulates 10+ instance perception tasks as object discovery
/// and retrieval. It achieves SOTA on 20+ benchmarks. Options inherit from NeuralNetworkOptions.
/// </para>
/// </remarks>
public class UNINEXTOptions : PanopticSegmentationOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public UNINEXTOptions()
    {
        NumClasses = 80;   // COCO
        NumQueries = 300;
        DropRate = 0.1;
        ModelSize = UNINEXTModelSize.R50;
    }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public UNINEXTOptions(UNINEXTOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
        NumClasses = other.NumClasses;
        NumQueries = other.NumQueries;
        DropRate = other.DropRate;
        ModelSize = other.ModelSize;
    }

    /// <summary>
    /// Gets or sets the backbone size variant. Default: <see cref="UNINEXTModelSize.R50"/>.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Which size of backbone to use. A larger one is more
    /// accurate and slower.</para>
    /// </remarks>
    public UNINEXTModelSize ModelSize { get; set; }

    /// <summary>
    /// Throws if a value this model requires has been left unset or is not positive.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when a required dimension is zero or negative.
    /// </exception>
    public void Validate() => ValidateCore();
}
