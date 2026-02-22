namespace AiDotNet.VisionLanguage.Encoders;

/// <summary>
/// Configuration options for SigLIP 2 (Multilingual Vision-Language Encoders with
/// Improved Semantic Understanding).
/// </summary>
/// <remarks>
/// <para>
/// SigLIP 2 (Tschannen et al., 2025) improves upon SigLIP by combining multiple training
/// objectives: sigmoid contrastive loss, autoregressive captioning loss, and self-supervised
/// masked image modeling. This multi-objective approach produces encoders with better semantic
/// understanding while maintaining efficient scaling. The model also supports 32+ languages
/// through a multilingual text encoder.
/// </para>
/// <para>
/// <b>Key Differences from SigLIP:</b>
/// <list type="bullet">
/// <item><b>Multi-objective training</b>: Contrastive + captioning + self-supervised losses</item>
/// <item><b>Captioning decoder</b>: Lightweight autoregressive decoder for generating image descriptions</item>
/// <item><b>Masked image modeling</b>: Self-supervised patch prediction for better spatial understanding</item>
/// <item><b>Multilingual</b>: Extended vocabulary with 32+ language support via mPaLM tokenizer</item>
/// <item><b>Online data curation</b>: Dynamic mixing of data sources during training</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> SigLIP 2 is an improved version of SigLIP that learns from three tasks
/// simultaneously: (1) matching images with text, (2) generating image descriptions, and
/// (3) predicting hidden parts of images. This multi-task learning produces better representations
/// and supports many languages.
/// </para>
/// </remarks>
public class SigLIP2Options : ContrastiveEncoderOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public SigLIP2Options(SigLIP2Options other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        ImageSize = other.ImageSize;
        VisionEmbeddingDim = other.VisionEmbeddingDim;
        VisionEncoderVariant = other.VisionEncoderVariant;
        PatchSize = other.PatchSize;
        NumVisionLayers = other.NumVisionLayers;
        NumVisionHeads = other.NumVisionHeads;
        VisionFfnMultiplier = other.VisionFfnMultiplier;
        TextEmbeddingDim = other.TextEmbeddingDim;
        TextEncoderVariant = other.TextEncoderVariant;
        MaxSequenceLength = other.MaxSequenceLength;
        VocabSize = other.VocabSize;
        NumTextLayers = other.NumTextLayers;
        NumTextHeads = other.NumTextHeads;
        ProjectionDim = other.ProjectionDim;
        Temperature = other.Temperature;
        DropoutRate = other.DropoutRate;
        ImageMean = other.ImageMean;
        ImageStd = other.ImageStd;
        ImageEncoderModelPath = other.ImageEncoderModelPath;
        TextEncoderModelPath = other.TextEncoderModelPath;
        OnnxOptions = other.OnnxOptions;
        LearningRate = other.LearningRate;
        WeightDecay = other.WeightDecay;
        WarmUpSteps = other.WarmUpSteps;
        LabelSmoothing = other.LabelSmoothing;
        LossType = other.LossType;
        SigmoidBias = other.SigmoidBias;
        CaptioningLossWeight = other.CaptioningLossWeight;
        SelfSupervisedLossWeight = other.SelfSupervisedLossWeight;
        MimMaskRatio = other.MimMaskRatio;
        NumCaptioningDecoderLayers = other.NumCaptioningDecoderLayers;
        NumCaptioningDecoderHeads = other.NumCaptioningDecoderHeads;
        CaptioningDecoderDim = other.CaptioningDecoderDim;
        MaxCaptionLength = other.MaxCaptionLength;
        Multilingual = other.Multilingual;
        MimDecoderDim = other.MimDecoderDim;
        NumMimDecoderLayers = other.NumMimDecoderLayers;
        IncludeCaptioningDecoder = other.IncludeCaptioningDecoder;
    }

    /// <summary>
    /// Gets or sets the contrastive loss type (default: Sigmoid for SigLIP family).
    /// </summary>
    public ContrastiveLossType LossType { get; set; } = ContrastiveLossType.Sigmoid;

    /// <summary>
    /// Gets or sets the bias term for sigmoid contrastive loss.
    /// </summary>
    /// <remarks>
    /// <para>The sigmoid loss computes: -log(sigmoid(z * (sim/t + b))) where z=+1 for positive
    /// and z=-1 for negative pairs. The bias b is learnable.</para>
    /// </remarks>
    public double SigmoidBias { get; set; } = -10.0;

    /// <summary>
    /// Gets or sets the weight for the captioning loss in multi-objective training.
    /// </summary>
    /// <remarks>
    /// <para>SigLIP 2 combines sigmoid contrastive loss with an autoregressive captioning loss.
    /// The captioning loss encourages the vision encoder to capture detailed semantics.
    /// The paper uses a weight of 1.0 for contrastive and 1.0 for captioning in the final model.</para>
    /// </remarks>
    public double CaptioningLossWeight { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the weight for the self-supervised masked image modeling loss.
    /// </summary>
    /// <remarks>
    /// <para>The MIM objective masks random patches and predicts their features, similar to MAE.
    /// This improves spatial understanding and localization capabilities.</para>
    /// </remarks>
    public double SelfSupervisedLossWeight { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the mask ratio for masked image modeling (fraction of patches masked).
    /// </summary>
    public double MimMaskRatio { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the number of captioning decoder layers.
    /// </summary>
    /// <remarks>
    /// <para>SigLIP 2 uses a lightweight autoregressive decoder (4 layers by default) for the
    /// captioning objective. This decoder is used during training but can be detached for
    /// inference when only contrastive embeddings are needed.</para>
    /// </remarks>
    public int NumCaptioningDecoderLayers { get; set; } = 4;

    /// <summary>
    /// Gets or sets the number of attention heads in the captioning decoder.
    /// </summary>
    public int NumCaptioningDecoderHeads { get; set; } = 12;

    /// <summary>
    /// Gets or sets the hidden dimension for the captioning decoder.
    /// </summary>
    public int CaptioningDecoderDim { get; set; } = 768;

    /// <summary>
    /// Gets or sets the maximum caption length for the captioning decoder.
    /// </summary>
    public int MaxCaptionLength { get; set; } = 64;

    /// <summary>
    /// Gets or sets whether multilingual text encoding is enabled.
    /// </summary>
    /// <remarks>
    /// <para>When enabled, SigLIP 2 uses an extended vocabulary (250K+ tokens) with
    /// SentencePiece tokenization supporting 32+ languages including CJK, Arabic,
    /// Devanagari, and more.</para>
    /// </remarks>
    public bool Multilingual { get; set; } = true;

    /// <summary>
    /// Gets or sets the MIM decoder dimension for predicting masked patch features.
    /// </summary>
    public int MimDecoderDim { get; set; } = 512;

    /// <summary>
    /// Gets or sets the number of MIM decoder layers.
    /// </summary>
    public int NumMimDecoderLayers { get; set; } = 2;

    /// <summary>
    /// Gets or sets whether to include the captioning decoder in inference mode.
    /// </summary>
    /// <remarks>
    /// <para>If false, the captioning decoder layers are excluded during inference,
    /// reducing memory and compute. The contrastive embeddings are still available.</para>
    /// </remarks>
    public bool IncludeCaptioningDecoder { get; set; } = true;

    /// <summary>
    /// Initializes default SigLIP 2 options.
    /// </summary>
    public SigLIP2Options()
    {
        // SigLIP 2 defaults: ViT-B/16 at 256px, larger vocab for multilingual
        VisionEncoderVariant = ViTVariant.ViTB16;
        ImageSize = 256;
        VisionEmbeddingDim = 768;
        TextEmbeddingDim = 768;
        ProjectionDim = 768;
        PatchSize = 16;
        Temperature = 1.0; // SigLIP family uses higher temperature with sigmoid loss
        VocabSize = 250000; // Multilingual SentencePiece vocabulary
        MaxSequenceLength = 64; // Longer for multilingual text
        NumVisionLayers = 12;
        NumTextLayers = 12;
        NumVisionHeads = 12;
        NumTextHeads = 12;
    }
}
