using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Foundation;

/// <summary>
/// Configuration options for the EoMT (Encoder-only Mask Transformer) model.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> EoMT removes the pixel and transformer decoders used by Mask2Former,
/// placing mask queries directly inside a plain ViT (DINOv2). This yields 4.4x faster inference.
/// The values here are the ones the paper publishes, so you can use the model without setting
/// anything.
/// </para>
/// </remarks>
public class EoMTOptions : PanopticSegmentationOptions
{
    /// <summary>
    /// Initializes a new instance carrying EoMT's published defaults.
    /// </summary>
    public EoMTOptions()
    {
        NumClasses = 150;   // ADE20K
        NumQueries = 100;
        DropRate = 0.1;
        ModelSize = EoMTModelSize.Base;
    }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public EoMTOptions(EoMTOptions other)
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
    /// Gets or sets the ViT encoder size variant. Default: <see cref="EoMTModelSize.Base"/>.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Which size of the underlying vision transformer to use. A
    /// larger one is more accurate and slower.</para>
    /// </remarks>
    public EoMTModelSize ModelSize { get; set; }

    /// <summary>
    /// Throws if a value this model requires has been left unset or is not positive.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when a required dimension is zero or negative.
    /// </exception>
    public void Validate() => ValidateCore();
}
