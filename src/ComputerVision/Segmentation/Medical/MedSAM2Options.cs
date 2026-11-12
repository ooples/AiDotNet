using AiDotNet.ComputerVision.Segmentation.Foundation;
using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Medical;

/// <summary>
/// Configuration options for <see cref="MedSAM2{T}"/>, MedSAM2 (Ma et al. 2025): SAM2.1-Tiny at 512 pixels fine-tuned for
/// 3D medical images and videos.
/// </summary>
/// <remarks>
/// <para>
/// The defaults are the paper's: SAM2.1-Tiny (Hiera blocks 1-2-7-2, global attention at blocks 5, 7 and 9) at 512 pixels,
/// AdamW (0.9, 0.999) with weight decay 0.01, learning rates 3e-5 for the image encoder and 5e-5 for the rest, and the
/// focal and dice losses at 20:1. The official config fills what the paper leaves out: a cosine schedule to a tenth with no
/// warmup, layer-wise decay 0.9, gradient clipping at 0.1, the IoU and occlusion losses at 1, and no weight decay on biases
/// and LayerNorms. The capacity settings default to SAM2.1-Tiny, so the model can load the released checkpoint.
/// </para>
/// <para><b>For Beginners:</b> Leave these at their defaults; they reproduce the published model.</para>
/// </remarks>
public class MedSAM2Options : SegmentationModelOptions, ISam2Settings
{
    /// <summary>Initializes a new instance with default values.</summary>
    public MedSAM2Options()
    {
        NumClasses = 1;
    }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public MedSAM2Options(MedSAM2Options other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));
        foreach (var property in typeof(MedSAM2Options).GetProperties())
            if (property.CanRead && property.CanWrite) property.SetValue(this, property.GetValue(other));
    }

    /// <summary>The square size frames are resized to for the image encoder (null: the architecture's larger side; the
    /// MedSAM2 uses 512). Must be a multiple of 32.</summary>
    public int? ImageSize { get; set; }

    /// <summary>The memory bank: the prompted frames and up to <c>MemoryBankSize − 1</c> recent frames (default: 7).</summary>
    public int MemoryBankSize { get; set; } = 7;

    /// <summary>The Hiera stem width (null: SAM2.1-Tiny's 96); it doubles per stage.</summary>
    public int? HieraEmbeddingDimension { get; set; }

    /// <summary>The Hiera blocks per stage (four values; null: the size's).</summary>
    public int[]? HieraStageDepths { get; set; }

    /// <summary>The first stage's attention heads (null: SAM2.1-Tiny's 1); they double per stage.</summary>
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

    /// <summary>The learning rate of everything but the image encoder (MedSAM2 paper: 5e-5).</summary>
    public double LearningRate { get; set; } = 5e-5;

    /// <summary>The image encoder's learning rate (MedSAM2 paper: 3e-5).</summary>
    public double ImageEncoderLearningRate { get; set; } = 3e-5;

    /// <summary>The per-block layer-wise decay of the Hiera learning rates (the MedSAM2 config's 0.9).</summary>
    public double? LayerDecay { get; set; } = 0.9;

    /// <summary>AdamW's decoupled weight decay (MedSAM2 paper: 0.01).</summary>
    public double WeightDecay { get; set; } = 0.01;

    /// <summary>The gradient-norm clip (paper: 0.1).</summary>
    public double MaxGradientNorm { get; set; } = 0.1;

    /// <summary>The training steps the cosine schedule spans (MedSAM2 trains 70 epochs: set it to 70 × the steps per
    /// epoch of your data).</summary>
    public int TrainingSteps { get; set; } = 150_000;

    /// <summary>The linear warmup steps (the MedSAM2 config has none).</summary>
    public int WarmupSteps { get; set; } = 0;

    /// <summary>The largest stochastic-depth rate of the Hiera blocks (the MedSAM2 config uses none).</summary>
    public double? DropPath { get; set; } = 0.0;

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

    /// <summary>Throws if a setting is out of range. MedSAM2 segments one prompted structure at a time, so <c>NumClasses</c>
    /// is 1, and <c>DropRate</c> stays 0 (its regularizer would be the stochastic depth <see cref="DropPath"/>).</summary>
    public void Validate()
    {
        if (NumClasses != 1) throw new ArgumentException("MedSAM2 segments one prompted structure at a time: NumClasses must be 1.", nameof(NumClasses));
        if (DropRate != 0) throw new ArgumentException("MedSAM2 regularizes with stochastic depth: set DropPath, not DropRate.", nameof(DropRate));
        Sam2SettingsValidation.Validate(this);
    }
}
