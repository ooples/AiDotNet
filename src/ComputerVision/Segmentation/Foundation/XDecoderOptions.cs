using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Foundation;

/// <summary>
/// Configuration options for the X-Decoder model.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> X-Decoder is a generalist vision decoder that handles referring
/// segmentation, open-vocabulary segmentation, and image captioning in one model.
/// Options inherit from NeuralNetworkOptions.
/// </para>
/// </remarks>
public class XDecoderOptions : PanopticSegmentationOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public XDecoderOptions()
    {
        NumClasses = 150;   // ADE20K
        NumQueries = 100;
        DropRate = 0.1;
        ModelSize = XDecoderModelSize.Tiny;
    }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public XDecoderOptions(XDecoderOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
        NumStuffClasses = other.NumStuffClasses;
        NumClasses = other.NumClasses;
        NumQueries = other.NumQueries;
        DropRate = other.DropRate;
        ModelSize = other.ModelSize;
    }

    /// <summary>
    /// Number of stuff (non-countable background) classes for panoptic segmentation.
    /// When null, defaults to numClasses / 3.
    /// </summary>
    public int? NumStuffClasses { get; set; }

    /// <summary>
    /// Gets or sets the backbone size variant. Default: <see cref="XDecoderModelSize.Tiny"/>.
    /// </summary>
    public XDecoderModelSize ModelSize { get; set; }

    /// <summary>
    /// Throws if a value this model requires has been left unset or is inconsistent.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when a required dimension is zero or negative.
    /// </exception>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when <see cref="NumStuffClasses"/> is set but does not leave at least one thing
    /// class, i.e. it is outside 1..NumClasses-1.
    /// </exception>
    /// <remarks>
    /// <para>
    /// The stuff/thing split check moved here from the constructor along with the values it
    /// compares: both NumStuffClasses and NumClasses now live on this class, so the constructor
    /// is no longer where they can be checked against each other.
    /// </para>
    /// </remarks>
    public void Validate()
    {
        ValidateCore();

        if (NumStuffClasses is int stuff && (stuff <= 0 || stuff >= NumClasses))
        {
            throw new ArgumentOutOfRangeException(
                nameof(NumStuffClasses), "NumStuffClasses must be between 1 and NumClasses-1.");
        }
    }
}
