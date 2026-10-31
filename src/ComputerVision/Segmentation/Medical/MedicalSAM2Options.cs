using AiDotNet.ComputerVision.Segmentation.Foundation;
using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Medical;

/// <summary>The SAM 2 sizes Medical SAM 2 releases configurations for.</summary>
public enum MedicalSAM2ModelSize
{
    /// <summary>SAM 2 Hiera-T.</summary>
    Tiny,

    /// <summary>SAM 2 Hiera-S, the official training scripts' choice.</summary>
    Small,
}

/// <summary>
/// Configuration options for <see cref="MedicalSAM2{T}"/>, Medical SAM 2 (Zhu et al. 2024): SAM 2 with a self-sorting memory
/// bank for sets of medical images, and SAM 2's tracking for volumes.
/// </summary>
/// <remarks>
/// <para>
/// The paper fixes the 256-pixel input, the bank's update (keep the K most mutually dissimilar confident memories) and its
/// resampling (draws in proportion to similarity), AdamW with a linear warmup into cosine decay, and SAM 2's prompt
/// simulation. Where it is silent the defaults follow SAM 2's recipe (Ravi et al. 2024, Table 12b), which it builds on, and
/// the official code (the Hiera-S model, a 16-memory bank). The code's own bank rule and resampling are available as
/// <see cref="MemoryBankRule"/> and <see cref="MemoryResampling"/>.
/// </para>
/// <para><b>For Beginners:</b> Leave these at their defaults; set <see cref="ConfidenceThreshold"/> only if you want low-quality
/// predictions kept out of the bank.</para>
/// </remarks>
public class MedicalSAM2Options : SegmentationModelOptions, ISam2Settings
{
    /// <summary>Initializes a new instance with default values.</summary>
    public MedicalSAM2Options()
    {
        NumClasses = 1;
    }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public MedicalSAM2Options(MedicalSAM2Options other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));
        foreach (var property in typeof(MedicalSAM2Options).GetProperties())
            if (property.CanRead && property.CanWrite) property.SetValue(this, property.GetValue(other));
    }

    /// <summary>The SAM 2 size (default: Small, as the official training scripts use).</summary>
    public MedicalSAM2ModelSize ModelSize { get; set; } = MedicalSAM2ModelSize.Small;

    /// <summary>K, the memories the self-sorting bank keeps and draws per image (the official code's 16; the paper gives no
    /// value).</summary>
    public int MemoryBankCapacity { get; set; } = 16;

    /// <summary>The predicted IoU a memory needs to be considered for the bank (the paper's c_thresh, for which it gives no
    /// value; null considers every memory).</summary>
    public double? ConfidenceThreshold { get; set; }

    /// <summary>How the bank decides what to keep (default: the paper's top-K total dissimilarity).</summary>
    public MemoryBankUpdateRule MemoryBankRule { get; set; } = MemoryBankUpdateRule.TopKDissimilarity;

    /// <summary>How the bank draws an image's memories (default: the paper's normalized similarity, eq. 4).</summary>
    public MemoryBankResampling MemoryResampling { get; set; } = MemoryBankResampling.NormalizedSimilarity;

    /// <summary>The square size images are resized to for the image encoder (null: the paper's 256). Must be a multiple of
    /// 32.</summary>
    public int? ImageSize { get; set; }

    /// <summary>The memory bank: the prompted frames and up to <c>MemoryBankSize − 1</c> recent frames (default: 7).</summary>
    public int MemoryBankSize { get; set; } = 7;

    /// <summary>The Hiera stem width (null: Tiny's and Small's 96); it doubles per stage.</summary>
    public int? HieraEmbeddingDimension { get; set; }

    /// <summary>The Hiera blocks per stage (four values; null: the size's).</summary>
    public int[]? HieraStageDepths { get; set; }

    /// <summary>The first stage's attention heads (null: Tiny's and Small's 1); they double per stage.</summary>
    public int? HieraInitialHeadCount { get; set; }

    /// <summary>The windowed-attention size per stage (four values; null: the size's).</summary>
    public int[]? HieraWindowSizes { get; set; }

    /// <summary>The blocks (0-based over all stages) with global attention (null: the size's).</summary>
    public int[]? HieraGlobalAttentionBlockIndexes { get; set; }

    /// <summary>The FPN, prompt and mask-decoder width (null: 256).</summary>
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

    /// <summary>The per-block layer-wise decay of the Hiera learning rates (null: the paper's 0.8 for Tiny and Small, 0.9
    /// for Base+, 0.925 for Large).</summary>
    public double? LayerDecay { get; set; }

    /// <summary>AdamW's decoupled weight decay (paper: 0.1).</summary>
    public double WeightDecay { get; set; } = 0.1;

    /// <summary>The gradient-norm clip (paper: 0.1).</summary>
    public double MaxGradientNorm { get; set; } = 0.1;

    /// <summary>The training steps the cosine schedule spans (paper: about 150k).</summary>
    public int TrainingSteps { get; set; } = 150_000;

    /// <summary>The linear warmup steps (paper: 7.5k).</summary>
    public int WarmupSteps { get; set; } = 7_500;

    /// <summary>The largest stochastic-depth rate of the Hiera blocks (null: the paper's 0.1 for Tiny and Small, 0.2 for
    /// Base+, 0.3 for Large).</summary>
    public double? DropPath { get; set; }

    /// <summary>The focal-loss weight (paper: 20).</summary>
    public double MaskFocalWeight { get; set; } = 20.0;

    /// <summary>The dice-loss weight (paper: 1).</summary>
    public double MaskDiceWeight { get; set; } = 1.0;

    /// <summary>The ℓ1 IoU-loss weight (paper: 1).</summary>
    public double IouLossWeight { get; set; } = 1.0;

    /// <summary>The occlusion cross-entropy weight (paper: 1).</summary>
    public double OcclusionLossWeight { get; set; } = 1.0;

    /// <summary>Focal loss's γ (2, the RetinaNet value SAM uses).</summary>
    public double FocalGamma { get; set; } = 2.0;

    /// <summary>Focal loss's α (0.25, the RetinaNet value SAM uses).</summary>
    public double FocalAlpha { get; set; } = 0.25;

    /// <summary>Corrective clicks per corrected training frame (paper: 7).</summary>
    public int CorrectionPoints { get; set; } = 7;

    /// <summary>Throws if a setting is out of range. Medical SAM 2 segments one prompted structure at a time, so
    /// <c>NumClasses</c> is 1, and <c>DropRate</c> stays 0 (its regularizer would be the stochastic depth
    /// <see cref="DropPath"/>).</summary>
    public void Validate()
    {
        if (NumClasses != 1) throw new ArgumentException("Medical SAM 2 segments one prompted structure at a time: NumClasses must be 1.", nameof(NumClasses));
        if (DropRate != 0) throw new ArgumentException("Medical SAM 2 regularizes with stochastic depth: set DropPath, not DropRate.", nameof(DropRate));
        if (MemoryBankCapacity < 1) throw new ArgumentOutOfRangeException(nameof(MemoryBankCapacity));
        if (ConfidenceThreshold is { } c && (double.IsNaN(c) || c < 0 || c > 1)) throw new ArgumentOutOfRangeException(nameof(ConfidenceThreshold));
        Sam2SettingsValidation.Validate(this);
    }
}
