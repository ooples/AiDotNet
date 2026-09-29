using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Foundation;

/// <summary>
/// Configuration options for the Mask DINO model.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Mask DINO unifies detection and segmentation in one framework.
/// Options inherit from NeuralNetworkOptions and provide defaults suitable for most use cases.
/// </para>
/// </remarks>
public class MaskDINOOptions : PanopticSegmentationOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public MaskDINOOptions()
    {
        NumClasses = 80;   // COCO
        NumQueries = 300;
        DropRate = 0.1;
        ModelSize = MaskDINOModelSize.R50;
    }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public MaskDINOOptions(MaskDINOOptions other)
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
    /// Gets or sets the backbone size variant. Default: <see cref="MaskDINOModelSize.R50"/>.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Which size of backbone to use. A larger one is more
    /// accurate and slower.</para>
    /// </remarks>
    public MaskDINOModelSize ModelSize { get; set; }

    /// <summary>
    /// Throws if a value this model requires has been left unset or is not positive.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when a required dimension is zero or negative.
    /// </exception>
    public void Validate() => ValidateCore();
}
