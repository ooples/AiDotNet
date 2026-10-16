namespace AiDotNet.VisionLanguage.Grounding;

/// <summary>
/// Configuration options for Grounding DINO (Liu et al. 2023): open-set detection combining DINO with grounded
/// pre-training.
/// </summary>
/// <remarks>
/// <para>The defaults are Grounding DINO-T as released (the official GroundingDINO_SwinT_OGC configuration): a Swin-T image
/// backbone, a BERT-base-uncased text backbone, a 256-wide feature enhancer and decoder of six layers each with eight
/// heads, four feature levels, four sampling points, 900 queries and a 256-token text head. The shared vision-language
/// options map as follows: <see cref="GenerativeVLMOptions.VisionDim"/> is the Swin embedding width,
/// <see cref="GenerativeVLMOptions.DecoderDim"/> the model width, <see cref="GenerativeVLMOptions.NumVisionLayers"/> the
/// feature-enhancer layers, <see cref="GenerativeVLMOptions.NumHeads"/> the encoder and decoder heads,
/// <see cref="GenerativeVLMOptions.MaxSequenceLength"/> the scored text tokens, <see cref="GenerativeVLMOptions.VocabSize"/>
/// the BERT vocabulary and <see cref="GroundingVLMOptions.ConfidenceThreshold"/> the box threshold.</para>
/// <para><b>For Beginners:</b> These options configure the GroundingDINO model. Default values follow the released
/// model.</para>
/// </remarks>
public class GroundingDINOOptions : GroundingVLMOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public GroundingDINOOptions(GroundingDINOOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        ImageSize = other.ImageSize;
        VisionDim = other.VisionDim;
        DecoderDim = other.DecoderDim;
        NumVisionLayers = other.NumVisionLayers;
        NumDecoderLayers = other.NumDecoderLayers;
        NumHeads = other.NumHeads;
        VocabSize = other.VocabSize;
        MaxSequenceLength = other.MaxSequenceLength;
        MaxGenerationLength = other.MaxGenerationLength;
        DropoutRate = other.DropoutRate;
        ArchitectureType = other.ArchitectureType;
        ImageMean = (double[])other.ImageMean.Clone();
        ImageStd = (double[])other.ImageStd.Clone();
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
        LearningRate = other.LearningRate;
        WeightDecay = other.WeightDecay;
        MaxDetections = other.MaxDetections;
        ConfidenceThreshold = other.ConfidenceThreshold;
        NmsThreshold = other.NmsThreshold;
        BoxDimension = other.BoxDimension;
        NumQueryPositions = other.NumQueryPositions;
        TextThreshold = other.TextThreshold;
        SwinDepths = (int[])other.SwinDepths.Clone();
        SwinHeads = (int[])other.SwinHeads.Clone();
        SwinWindowSize = other.SwinWindowSize;
        SwinDropPathRate = other.SwinDropPathRate;
        TextHiddenSize = other.TextHiddenSize;
        TextLayers = other.TextLayers;
        TextHeads = other.TextHeads;
        TextIntermediateSize = other.TextIntermediateSize;
        FeedForwardDim = other.FeedForwardDim;
        NumFeatureLevels = other.NumFeatureLevels;
        NumSamplingPoints = other.NumSamplingPoints;
        FusionDropPathRate = other.FusionDropPathRate;
    }

    public GroundingDINOOptions()
    {
        VisionDim = 96;
        DecoderDim = 256;
        NumVisionLayers = 6;
        NumDecoderLayers = 6;
        NumHeads = 8;
        VocabSize = 30522;
        MaxSequenceLength = 256;
        DropoutRate = 0.0;
        ImageSize = 800;
        ImageMean = [0.485, 0.456, 0.406];
        ImageStd = [0.229, 0.224, 0.225];
        MaxDetections = 900;
        ConfidenceThreshold = 0.35;
        WeightDecay = 1e-4;
    }

    /// <summary>Gets or sets the number of object queries the decoder refines (900).</summary>
    public int NumQueryPositions { get; set; } = 900;

    /// <summary>Gets or sets the token probability above which a token joins a detection's phrase (0.25, the official
    /// demo's text threshold).</summary>
    public double TextThreshold { get; set; } = 0.25;

    /// <summary>Gets or sets the Swin blocks per stage (Swin-T: 2, 2, 6, 2).</summary>
    public int[] SwinDepths { get; set; } = [2, 2, 6, 2];

    /// <summary>Gets or sets the Swin attention heads per stage (Swin-T: 3, 6, 12, 24).</summary>
    public int[] SwinHeads { get; set; } = [3, 6, 12, 24];

    /// <summary>Gets or sets the Swin attention window (7).</summary>
    public int SwinWindowSize { get; set; } = 7;

    /// <summary>Gets or sets the Swin stochastic depth rate (0.2, the official backbone's).</summary>
    public double SwinDropPathRate { get; set; } = 0.2;

    /// <summary>Gets or sets the BERT width (768).</summary>
    public int TextHiddenSize { get; set; } = 768;

    /// <summary>Gets or sets the BERT layers (12).</summary>
    public int TextLayers { get; set; } = 12;

    /// <summary>Gets or sets the BERT attention heads (12).</summary>
    public int TextHeads { get; set; } = 12;

    /// <summary>Gets or sets the BERT feed-forward width (3072).</summary>
    public int TextIntermediateSize { get; set; } = 3072;

    /// <summary>Gets or sets the feed-forward width of the feature enhancer and decoder layers (2048).</summary>
    public int FeedForwardDim { get; set; } = 2048;

    /// <summary>Gets or sets the feature levels: the backbone's three plus one stride-2 convolution of the last (4).</summary>
    public int NumFeatureLevels { get; set; } = 4;

    /// <summary>Gets or sets the deformable-attention sampling points per head and level (4).</summary>
    public int NumSamplingPoints { get; set; } = 4;

    /// <summary>Gets or sets the stochastic depth rate of the vision-text fusion layers (0.1).</summary>
    public double FusionDropPathRate { get; set; } = 0.1;
}
