using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Foundation;

/// <summary>
/// Configuration options for <see cref="SAM21{T}"/>, SAM 2.1 (Meta's refreshed SAM 2 checkpoints, with the temporal
/// encoding of the object pointers and the occlusion embedding of the memories).
/// </summary>
/// <remarks>
/// <para>
/// The defaults are the released SAM 2.1 Hiera-L model at 1024 pixels and SAM 2's full-training recipe (Ravi et al. 2024,
/// Table 12b): AdamW with weight decay 0.1 (not on biases and LayerNorms), learning rates 3e-4 and 6e-5 for the image
/// encoder with layer-wise decay, a 7.5k-step linear warmup into cosine decay, clipping at 0.1, and the focal (20), dice
/// (1), l1 IoU (1) and occlusion (1) losses. The <c>Hiera…</c>, <c>ModelDimension</c> and memory settings are capacity
/// overrides: left null, each follows <see cref="ModelSize"/>, so the model can load that size's checkpoint.
/// </para>
/// <para><b>For Beginners:</b> You only need <see cref="ModelSize"/>; leave the rest at its defaults unless you are
/// building a smaller model to experiment with.</para>
/// </remarks>
public class SAM21Options : SegmentationModelOptions, ISam2Settings
{
    /// <summary>Initializes a new instance with default values.</summary>
    public SAM21Options()
    {
        NumClasses = 1;
    }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public SAM21Options(SAM21Options other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));
        foreach (var property in typeof(SAM21Options).GetProperties())
            if (property.CanRead && property.CanWrite) property.SetValue(this, property.GetValue(other));
    }

    /// <summary>The released size (default: Large).</summary>
    public SAM21ModelSize ModelSize { get; set; } = SAM21ModelSize.Large;

    /// <summary>The square size frames are resized to for the image encoder (null: the architecture's larger side; the
    /// released models use 1024). Must be a multiple of 32.</summary>
    public int? ImageSize { get; set; }

    /// <summary>The memory bank: the prompted frames and up to <c>MemoryBankSize − 1</c> recent frames (default: 7).</summary>
    public int MemoryBankSize { get; set; } = 7;

    /// <summary>The Hiera stem width (null: Tiny and Small 96, Base+ 112, Large 144); it doubles per stage.</summary>
    public int? HieraEmbeddingDimension { get; set; }

    /// <summary>The Hiera blocks per stage (four values; null: the size's).</summary>
    public int[]? HieraStageDepths { get; set; }

    /// <summary>The first stage's attention heads (null: Tiny and Small 1, Base+ and Large 2); they double per stage.</summary>
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

    /// <summary>Throws if a setting is out of range. SAM 2.1 segments one object at a time, so <c>NumClasses</c> is 1; its
    /// regularizer is the stochastic depth <see cref="DropPath"/>, so <c>DropRate</c> stays 0.</summary>
    public void Validate()
    {
        if (NumClasses != 1) throw new ArgumentException("SAM 2.1 segments one prompted object at a time: NumClasses must be 1.", nameof(NumClasses));
        if (DropRate != 0) throw new ArgumentException("SAM 2.1 regularizes with stochastic depth: set DropPath, not DropRate.", nameof(DropRate));
        Sam2SettingsValidation.Validate(this);
    }
}
