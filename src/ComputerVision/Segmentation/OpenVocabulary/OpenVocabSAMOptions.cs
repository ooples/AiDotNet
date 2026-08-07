using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.OpenVocabulary;

/// <summary>
/// Configuration options for Open-Vocabulary SAM interactive recognition.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the OpenVocabSAM model. Default values follow the original paper settings.</para>
/// </remarks>
public class OpenVocabSAMOptions : NeuralNetworkOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public OpenVocabSAMOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public OpenVocabSAMOptions(OpenVocabSAMOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
        ChannelDimensions = (int[])other.ChannelDimensions.Clone();
        StageDepths = (int[])other.StageDepths.Clone();
        NeckEmbeddingDimension = other.NeckEmbeddingDimension;
        DecoderDimension = other.DecoderDimension;
        LearningRate = other.LearningRate;
        WeightDecay = other.WeightDecay;
    }

    /// <summary>
    /// Gets or sets the four CLIP RN50x16 feature-stage channel dimensions.
    /// </summary>
    public int[] ChannelDimensions { get; set; } = [384, 768, 1536, 3072];

    /// <summary>
    /// Gets or sets the four CLIP RN50x16 residual-stage depths.
    /// </summary>
    public int[] StageDepths { get; set; } = [6, 8, 18, 8];

    /// <summary>
    /// Gets or sets the SAM2CLIP transformer-neck embedding dimension.
    /// </summary>
    public int NeckEmbeddingDimension { get; set; } = 1280;

    /// <summary>
    /// Gets or sets the CLIP2SAM FPN and mask-decoder channel dimension.
    /// </summary>
    public int DecoderDimension { get; set; } = 256;

    /// <summary>Gets or sets the AdamW learning rate.</summary>
    public double LearningRate { get; set; } = 1e-4;

    /// <summary>Gets or sets the AdamW weight decay.</summary>
    public double WeightDecay { get; set; } = 0.05;
}
