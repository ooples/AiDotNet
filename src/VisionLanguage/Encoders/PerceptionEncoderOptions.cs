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
