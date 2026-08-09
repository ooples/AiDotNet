using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Referring;

/// <summary>
/// Configuration options for VideoLISA video reasoning segmentation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the VideoLISA model. Default values follow the original paper settings.</para>
/// </remarks>
public class VideoLISAOptions : NeuralNetworkOptions
{
    /// <summary>Gets or sets the output channel width of each visual-encoder stage.</summary>
    public int[] ChannelDimensions { get; set; } = [64, 128, 320, 768];

    /// <summary>Gets or sets the number of convolutional blocks in each visual-encoder stage.</summary>
    public int[] EncoderDepths { get; set; } = [2, 2, 4, 12];

    /// <summary>Gets or sets the hidden channel width of the promptable mask decoder.</summary>
    public int DecoderDimension { get; set; } = 256;

    /// <summary>Gets or sets the AdamW learning rate used by native training.</summary>
    public double LearningRate { get; set; } = 1e-4;

    /// <summary>Gets or sets the AdamW weight-decay coefficient used by native training.</summary>
    public double WeightDecay { get; set; } = 1e-4;

    /// <summary>Initializes a new instance with default values.</summary>
    public VideoLISAOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public VideoLISAOptions(VideoLISAOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
        ChannelDimensions = (int[])other.ChannelDimensions.Clone();
        EncoderDepths = (int[])other.EncoderDepths.Clone();
        DecoderDimension = other.DecoderDimension;
        LearningRate = other.LearningRate;
        WeightDecay = other.WeightDecay;
    }

}
