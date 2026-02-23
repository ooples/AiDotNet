namespace AiDotNet.VisionLanguage.Encoders;

/// <summary>
/// Configuration options for Meta's Perception Encoder vision foundation model.
/// </summary>
/// <remarks>
/// <para>Perception Encoder (Meta, 2025) is a vision encoder designed for multimodal alignment tasks.
/// It combines contrastive learning with dense prediction objectives to produce features suitable
/// for both global understanding and local spatial reasoning.</para>
/// </remarks>
public class PerceptionEncoderOptions : VisionEncoderOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public PerceptionEncoderOptions(PerceptionEncoderOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        ImageSize = other.ImageSize;
        EmbeddingDim = other.EmbeddingDim;
        PatchSize = other.PatchSize;
        NumLayers = other.NumLayers;
        NumHeads = other.NumHeads;
        FfnMultiplier = other.FfnMultiplier;
        DropoutRate = other.DropoutRate;
        ImageMean = other.ImageMean;
        ImageStd = other.ImageStd;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
        LearningRate = other.LearningRate;
        WeightDecay = other.WeightDecay;
        UseDenseFeatures = other.UseDenseFeatures;
        DenseFeatureStride = other.DenseFeatureStride;
        GlobalPooling = other.GlobalPooling;
        AlignmentProjectionDim = other.AlignmentProjectionDim;
    }

    /// <summary>
    /// Gets or sets whether to use dense prediction features alongside global features.
    /// </summary>
    public bool UseDenseFeatures { get; set; } = true;

    /// <summary>
    /// Gets or sets the dense feature output stride.
    /// </summary>
    public int DenseFeatureStride { get; set; } = 16;

    /// <summary>
    /// Gets or sets the global feature pooling strategy.
    /// </summary>
    public PoolingStrategy GlobalPooling { get; set; } = PoolingStrategy.ClsToken;

    /// <summary>
    /// Gets or sets the alignment projection dimension.
    /// </summary>
    public int AlignmentProjectionDim { get; set; } = 768;

    public PerceptionEncoderOptions()
    {
        ImageSize = 384;
        EmbeddingDim = 1024;
        PatchSize = 14;
        NumLayers = 24;
        NumHeads = 16;
    }
}
