using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Video;

/// <summary>
/// Configuration options for DEVA decoupled video segmentation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the DEVA model. Default values follow the original paper settings.</para>
/// </remarks>
public class DEVAOptions : NeuralNetworkOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public DEVAOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public DEVAOptions(DEVAOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
        ChannelDimensions = other.ChannelDimensions?.ToArray();
        StageDepths = other.StageDepths?.ToArray();
        DecoderDimension = other.DecoderDimension;
        UseGroupNormalization = other.UseGroupNormalization;
    }

    /// <summary>
    /// Gets or sets the four encoder-stage channel dimensions. A null value uses
    /// the selected <see cref="AiDotNet.Enums.DEVAModelSize"/> research configuration.
    /// </summary>
    public int[]? ChannelDimensions { get; set; }

    /// <summary>
    /// Gets or sets the number of blocks in each of the four encoder stages. A
    /// null value uses the selected model-size research configuration.
    /// </summary>
    public int[]? StageDepths { get; set; }

    /// <summary>
    /// Gets or sets the decoder feature dimension. A null value uses the selected
    /// model-size research configuration.
    /// </summary>
    public int? DecoderDimension { get; set; }

    /// <summary>
    /// Gets or sets whether the native encoder uses group normalization instead
    /// of the research-default batch normalization. This is useful for genuine
    /// single-frame training batches; the default remains <see langword="false"/>.
    /// </summary>
    public bool UseGroupNormalization { get; set; }
}
