using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Foundation;

/// <summary>
/// Configuration options for the OMG-Seg model.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> OMG-Seg handles 10+ segmentation tasks with one model using only 70M
/// trainable parameters. Options inherit from NeuralNetworkOptions.
/// </para>
/// </remarks>
public class OMGSegOptions : PanopticSegmentationOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public OMGSegOptions()
    {
        NumClasses = 150;   // ADE20K
        NumQueries = 200;
        DropRate = 0.1;
        ModelSize = OMGSegModelSize.Base;
    }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public OMGSegOptions(OMGSegOptions other)
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
    /// Gets or sets the backbone size variant. Default: <see cref="OMGSegModelSize.Base"/>.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Which size of backbone to use. A larger one is more
    /// accurate and slower.</para>
    /// </remarks>
    public OMGSegModelSize ModelSize { get; set; }

    /// <summary>
    /// Throws if a value this model requires has been left unset or is not positive.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when a required dimension is zero or negative.
    /// </exception>
    public void Validate() => ValidateCore();
}
