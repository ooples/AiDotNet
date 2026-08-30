using AiDotNet.Models.Options;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the SAM2 segmentation model.
/// </summary>
public class SAM2Options : NeuralNetworkOptions
{
    /// <summary>
    /// Optional Hiera stem width. Null selects the published width for the requested model size
    /// (Tiny/Small 96, Base+ 112, Large 144).
    /// </summary>
    public int? HieraEmbeddingDimension { get; set; }

    /// <summary>
    /// Optional four-stage Hiera depths. Null selects the published model-size preset.
    /// This is a capacity control; all four hierarchical stages remain present.
    /// </summary>
    public int[]? HieraStageDepths { get; set; }

    /// <summary>Optional initial Hiera head count. Null selects the published preset.</summary>
    public int? HieraInitialHeadCount { get; set; }

    /// <summary>Optional four-stage window sizes. Null selects the published preset.</summary>
    public int[]? HieraWindowSizes { get; set; }

    /// <summary>
    /// Optional zero-based Hiera block indexes that use global attention instead of window attention.
    /// Null selects the published indexes for the requested model size.
    /// </summary>
    public int[]? HieraGlobalAttentionBlockIndexes { get; set; }

    /// <summary>Gets or sets SAM 2's image/prompt/mask-decoder embedding width.</summary>
    public int ModelDimension { get; set; } = 256;

    /// <summary>Gets or sets the compressed spatial-memory width.</summary>
    public int MemoryDimension { get; set; } = 64;

    /// <summary>Gets or sets the mask decoder's attention head count.</summary>
    public int DecoderHeadCount { get; set; } = 8;

    /// <summary>Gets or sets the number of memory-attention transformer layers.</summary>
    public int MemoryAttentionLayerCount { get; set; } = 4;

    /// <summary>Gets or sets the two-way mask decoder depth.</summary>
    public int MaskDecoderDepth { get; set; } = 2;

    /// <summary>Gets or sets the two-way decoder MLP width.</summary>
    public int MaskDecoderMlpDimension { get; set; } = 2048;

    /// <summary>Gets or sets the axial rotary-position base.</summary>
    public double RopeTheta { get; set; } = 10000.0;

    /// <summary>Gets or sets the sigmoid probability scale used before memory encoding.</summary>
    public double MemoryMaskScale { get; set; } = 20.0;

    /// <summary>Gets or sets the sigmoid probability bias used before memory encoding.</summary>
    public double MemoryMaskBias { get; set; } = -10.0;

    /// <summary>
    /// Gets or sets the focal-loss weight in the mask objective. The default, 20, is the focal:dice
    /// ratio of 20:1 that SAM 2 inherits from SAM (Ravi et al. 2024, §D Training details; Kirillov
    /// et al. 2023, §3).
    /// </summary>
    public double MaskFocalWeight { get; set; } = 20.0;

    /// <summary>
    /// Gets or sets the dice-loss weight in the mask objective. The default, 1, is the paper's
    /// focal:dice ratio of 20:1.
    /// </summary>
    public double MaskDiceWeight { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets focal loss's focusing parameter gamma. The default, 2, is the RetinaNet value
    /// the SAM/SAM 2 mask objective uses.
    /// </summary>
    public double FocalGamma { get; set; } = 2.0;

    /// <summary>
    /// Gets or sets focal loss's class-balance parameter alpha. The default, 0.25, is the RetinaNet
    /// value the SAM/SAM 2 mask objective uses.
    /// </summary>
    public double FocalAlpha { get; set; } = 0.25;

    /// <summary>Gets or sets the auxiliary IoU-regression loss weight.</summary>
    public double IouLossWeight { get; set; } = 1.0;

    /// <summary>Gets or sets the object-presence classification loss weight.</summary>
    public double ObjectPresenceLossWeight { get; set; } = 1.0;
}
