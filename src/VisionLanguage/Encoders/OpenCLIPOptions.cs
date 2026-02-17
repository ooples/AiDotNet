namespace AiDotNet.VisionLanguage.Encoders;

/// <summary>
/// Configuration options for the OpenCLIP model.
/// </summary>
/// <remarks>
/// <para>
/// OpenCLIP (Ilharco et al., 2021) is an open-source reproduction of CLIP trained on the LAION-2B
/// and LAION-5B datasets. It supports a wide range of vision encoder architectures (ViT-B/32 through
/// ViT-bigG/14) and achieves comparable or better performance than OpenAI's original CLIP.
/// </para>
/// <para>
/// <b>For Beginners:</b> OpenCLIP is essentially the same as CLIP but trained on publicly available
/// data instead of proprietary data. This makes it more transparent and reproducible while achieving
/// similar or better performance. It supports more model sizes and configurations.
/// </para>
/// </remarks>
public class OpenCLIPOptions : ContrastiveEncoderOptions
{
    /// <summary>
    /// Gets or sets the pre-training dataset.
    /// </summary>
    public PretrainingDataset Dataset { get; set; } = PretrainingDataset.LAION2B;

    /// <summary>
    /// Gets or sets whether to use the CoCa (Contrastive Captioners) variant.
    /// </summary>
    /// <remarks>
    /// <para>When enabled, the model uses a dual-loss approach combining contrastive loss
    /// with captioning loss, improving both retrieval and generation capabilities.</para>
    /// </remarks>
    public bool UseCoCaVariant { get; set; }

    /// <summary>
    /// Gets or sets the precision mode for inference.
    /// </summary>
    public ModelPrecision Precision { get; set; } = ModelPrecision.Float32;

    /// <summary>
    /// Gets or sets the contrastive loss type.
    /// </summary>
    public ContrastiveLossType LossType { get; set; } = ContrastiveLossType.SymmetricCrossEntropy;
}
