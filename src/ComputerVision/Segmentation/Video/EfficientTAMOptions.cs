using AiDotNet.ComputerVision.Segmentation.Foundation;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Video;

/// <summary>
/// Configuration options for <see cref="EfficientTAM{T}"/>, Efficient Track Anything (Xiong et al. 2024): SAM 2's tracker on
/// a plain ViT image encoder, optionally with the efficient (2×2 pooled) memory cross-attention.
/// </summary>
/// <remarks>
/// <para>
/// The defaults are EfficientTAM-S (ViT-Small, SAM 2's memory block) at 1024 pixels and the paper's full training (Table
/// in section 4): AdamW (0.9, 0.999), 6e-5 for the image encoder and 3e-4 for the rest with layer-wise decay 0.8, a
/// 15k-step linear warmup into cosine decay over 300k steps, weight decay 0.1, no drop path, and the focal (20), dice (1),
/// ℓ1 IoU (1) and occlusion (1) losses. Where the paper is silent (the gradient clip, the decay exemptions, the prompt
/// simulation) SAM 2's recipe, which EfficientTAM follows, fills in. <see cref="MemoryCrossAttention"/> selects the "/2"
/// models' efficient cross-attention.
/// </para>
/// <para><b>For Beginners:</b> Choose <see cref="ModelSize"/> and, for the faster "/2" models,
/// <see cref="MemoryCrossAttention"/>; leave the rest at its defaults.</para>
/// </remarks>
public class EfficientTAMOptions : SegmentationModelOptions, ISam2Settings
{
    /// <summary>Initializes a new instance with default values.</summary>
    public EfficientTAMOptions()
    {
        NumClasses = 1;
    }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public EfficientTAMOptions(EfficientTAMOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));
        foreach (var property in typeof(EfficientTAMOptions).GetProperties())
            if (property.CanRead && property.CanWrite) property.SetValue(this, property.GetValue(other));
    }

    /// <summary>The image encoder: ViT-Tiny or ViT-Small (default: Small).</summary>
    public EfficientTAMModelSize ModelSize { get; set; } = EfficientTAMModelSize.Small;

    /// <summary>How the memory attention reads the spatial memories: SAM 2's full attention (EfficientTAM-Ti and -S, the
    /// default), or the efficient 2×2 pooled cross-attention of the "/2" models (eq. 5, or eq. 6's key form).</summary>
    public MemoryCrossAttention MemoryCrossAttention { get; set; } = MemoryCrossAttention.Full;

    /// <summary>The square size frames are resized to for the image encoder (null: the architecture's larger side; the
    /// released models use 1024). Must be a multiple of 16.</summary>
    public int? ImageSize { get; set; }

    /// <summary>The memory bank: the prompted frames and up to <c>MemoryBankSize − 1</c> recent frames (default: 7).</summary>
    public int MemoryBankSize { get; set; } = 7;

    /// <summary>The ViT's width (null: Tiny 192, Small 384).</summary>
    public int? VitEmbeddingDimension { get; set; }

    /// <summary>The ViT's blocks (null: 12).</summary>
    public int? VitDepth { get; set; }

    /// <summary>The ViT's attention heads (null: Tiny 3, Small 6).</summary>
    public int? VitHeadCount { get; set; }

    /// <summary>The ViT's attention window (null: 14).</summary>
    public int? VitWindowSize { get; set; }

    /// <summary>The ViT's global-attention blocks (null: 2, 5, 8, 11).</summary>
    public int[]? VitGlobalAttentionBlockIndexes { get; set; }

    /// <summary>The prompt, mask-decoder and memory-attention width (null: 256).</summary>
    public int? ModelDimension { get; set; }

    /// <summary>The memory width (null: 64).</summary>
    public int? MemoryDimension { get; set; }

    /// <summary>The mask decoder's attention heads (null: 8).</summary>
    public int? DecoderHeadCount { get; set; }

    /// <summary>The mask decoder's MLP width (null: 2048).</summary>
    public int? MaskDecoderMlpDimension { get; set; }

    /// <summary>The memory attention layers (null: 4).</summary>
    public int? MemoryAttentionLayerCount { get; set; }

    /// <summary>The memory attention's feed-forward width (null: 2048).</summary>
    public int? MemoryAttentionFeedForwardDimension { get; set; }

    /// <summary>The memory fuser's hidden width (null: 1024).</summary>
    public int? MemoryFuserDimension { get; set; }

    /// <summary>The learning rate of everything but the image encoder (paper: 3e-4).</summary>
    public double LearningRate { get; set; } = 3e-4;

    /// <summary>The image encoder's learning rate (paper: 6e-5).</summary>
    public double ImageEncoderLearningRate { get; set; } = 6e-5;

    /// <summary>The per-block layer-wise decay of the ViT's learning rates (paper: 0.8).</summary>
    public double? LayerDecay { get; set; } = 0.8;

    /// <summary>AdamW's decoupled weight decay (paper: 0.1).</summary>
    public double WeightDecay { get; set; } = 0.1;

    /// <summary>The gradient-norm clip (SAM 2's 0.1; the paper gives none).</summary>
    public double MaxGradientNorm { get; set; } = 0.1;

    /// <summary>The training steps the cosine schedule spans (paper: 300k).</summary>
    public int TrainingSteps { get; set; } = 300_000;

    /// <summary>The linear warmup steps (paper: 15k).</summary>
    public int WarmupSteps { get; set; } = 15_000;

    /// <summary>The largest stochastic-depth rate (paper: none).</summary>
    public double? DropPath { get; set; } = 0.0;

    /// <summary>The focal-loss weight (paper: 20).</summary>
    public double MaskFocalWeight { get; set; } = 20.0;

    /// <summary>The dice-loss weight (paper: 1).</summary>
    public double MaskDiceWeight { get; set; } = 1.0;

    /// <summary>The ℓ1 IoU-loss weight (paper: 1).</summary>
    public double IouLossWeight { get; set; } = 1.0;

    /// <summary>The occlusion cross-entropy weight (paper: 1).</summary>
    public double OcclusionLossWeight { get; set; } = 1.0;

    /// <summary>Focal loss's γ (2).</summary>
    public double FocalGamma { get; set; } = 2.0;

    /// <summary>Focal loss's α (0.25).</summary>
    public double FocalAlpha { get; set; } = 0.25;

    /// <summary>Corrective clicks per corrected training frame (SAM 2's 7).</summary>
    public int CorrectionPoints { get; set; } = 7;

    // Hiera's capacity settings do not apply to the plain ViT encoder.
    int[]? ISam2Settings.HieraStageDepths => null;
    int? ISam2Settings.HieraEmbeddingDimension => null;
    int? ISam2Settings.HieraInitialHeadCount => null;
    int[]? ISam2Settings.HieraWindowSizes => null;
    int[]? ISam2Settings.HieraGlobalAttentionBlockIndexes => null;

    /// <summary>Throws if a setting is out of range. EfficientTAM segments one prompted object per tracked mask, so
    /// <c>NumClasses</c> is 1, and <c>DropRate</c> stays 0 (its regularizer would be <see cref="DropPath"/>).</summary>
    public void Validate()
    {
        if (NumClasses != 1) throw new ArgumentException("EfficientTAM segments one prompted object per mask: NumClasses must be 1.", nameof(NumClasses));
        if (DropRate != 0) throw new ArgumentException("EfficientTAM's regularizer is stochastic depth: set DropPath, not DropRate.", nameof(DropRate));
        foreach (var (value, name) in new[]
        {
            (VitEmbeddingDimension, nameof(VitEmbeddingDimension)), (VitDepth, nameof(VitDepth)), (VitHeadCount, nameof(VitHeadCount)),
            (VitWindowSize, nameof(VitWindowSize)),
        })
            if (value is <= 0) throw new ArgumentOutOfRangeException(name);
        if (VitGlobalAttentionBlockIndexes is { } globals && globals.Any(g => g < 0))
            throw new ArgumentException("The global-attention blocks must be non-negative.", nameof(VitGlobalAttentionBlockIndexes));
        Sam2SettingsValidation.Validate(this, imageMultiple: 16);
    }
}
