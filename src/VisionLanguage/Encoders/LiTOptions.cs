namespace AiDotNet.VisionLanguage.Encoders;

/// <summary>
/// Configuration options for the LiT (Locked-image Tuning) model.
/// </summary>
/// <remarks>
/// <para>
/// LiT (Zhai et al., CVPR 2022) freezes a pre-trained image encoder and only trains the text encoder
/// and projection layers. This "locked-image tuning" approach achieves strong zero-shot performance
/// while being much cheaper to train than full contrastive learning from scratch.
/// </para>
/// <para>
/// <b>For Beginners:</b> LiT takes a shortcut: instead of training both image and text encoders from
/// scratch (which is expensive), it takes an already-trained image model and only teaches the text model
/// to align with it. This is much faster and often works just as well.
/// </para>
/// </remarks>
public class LiTOptions : ContrastiveEncoderOptions
{
    /// <summary>
    /// Gets or sets the contrastive loss type.
    /// </summary>
    public ContrastiveLossType LossType { get; set; } = ContrastiveLossType.InfoNCE;

    /// <summary>
    /// Gets or sets whether to freeze the vision encoder during training.
    /// </summary>
    /// <remarks>
    /// <para>The core LiT strategy: freeze the pre-trained image encoder and only train
    /// the text encoder. Set to false for ablation studies or fine-tuning both encoders.</para>
    /// </remarks>
    public bool FreezeVisionEncoder { get; set; } = true;

    /// <summary>
    /// Gets or sets the path to the pre-trained vision encoder weights.
    /// </summary>
    public string? PretrainedVisionWeightsPath { get; set; }

    /// <summary>
    /// Gets or sets whether to use a pre-trained text encoder as initialization.
    /// </summary>
    public bool InitializeTextFromPretrained { get; set; }

    /// <summary>
    /// Initializes default LiT options.
    /// </summary>
    public LiTOptions()
    {
        VisionEncoderVariant = ViTVariant.ViTL14;
        ImageSize = 224;
        VisionEmbeddingDim = 1024;
        TextEmbeddingDim = 768;
        ProjectionDim = 768;
        NumVisionLayers = 24;
        NumVisionHeads = 16;
        Temperature = 0.07;
    }
}
