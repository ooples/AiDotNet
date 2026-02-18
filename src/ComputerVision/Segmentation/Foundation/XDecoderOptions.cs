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
    /// <summary>
    /// Number of stuff (non-countable background) classes for panoptic segmentation.
    /// When null, defaults to numClasses / 3.
    /// </summary>
    public int? NumStuffClasses { get; set; }
}
