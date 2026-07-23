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
public class XDecoderOptions : NeuralNetworkOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public XDecoderOptions() { }

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
    }

    /// <summary>
    /// Number of stuff (non-countable background) classes for panoptic segmentation.
    /// When null, defaults to numClasses / 3.
    /// </summary>
    public int? NumStuffClasses { get; set; }
}
