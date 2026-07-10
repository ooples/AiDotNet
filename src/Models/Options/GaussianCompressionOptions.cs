using System;

namespace AiDotNet.Models.Options;

/// <summary>
/// Per-technique configuration for the GaussianSplatting post-training compression pass
/// (#1835 excellence goal #3).
/// </summary>
/// <remarks>
/// <para>
/// Every field is defaulted to a paper-friendly setting so the user can enable
/// <see cref="GaussianSplattingOptions.CompressOnBuildComplete"/> and get a meaningful
/// compressed cloud without touching this class. Advanced users tune per-technique on/off +
/// thresholds.
/// </para>
/// <para><b>For Beginners:</b> Compression takes the raw Gaussian cloud your training
/// produced and makes it smaller / faster to render without hurting quality much:
/// <list type="bullet">
///   <item><description>Prune — drop Gaussians that ended up nearly transparent.</description></item>
///   <item><description>Merge — combine Gaussians that overlap almost entirely into one.</description></item>
///   <item><description>Quantize — represent each color coefficient with 8 bits instead of 32.</description></item>
/// </list>
/// All three are on by default; toggle individual techniques off if you don't need them.
/// </para>
/// </remarks>
public sealed class GaussianCompressionOptions
{
    /// <summary>Prune Gaussians whose opacity has collapsed below the configured threshold.</summary>
    /// <value>true (paper default — collapsed Gaussians contribute nothing to renders).</value>
    /// <remarks>Threshold is <see cref="GaussianSplattingOptions.OpacityPruneThreshold"/>.</remarks>
    public bool PruneLowOpacity { get; set; } = true;

    /// <summary>Merge nearly-identical overlapping Gaussians into a single Gaussian.</summary>
    /// <value>true.</value>
    /// <remarks>Overlap fraction is measured via <see cref="MergeOverlapThreshold"/>.</remarks>
    public bool MergeOverlapping { get; set; } = true;

    /// <summary>Overlap fraction above which two Gaussians are candidates for merging.</summary>
    /// <value>0.9 (aggressive — only merges near-duplicates).</value>
    /// <remarks>Range [0, 1]. Values near 1 preserve individual Gaussians; values near 0 merge
    /// aggressively (may hurt quality). Validated at consumption time.</remarks>
    public double MergeOverlapThreshold { get; set; } = 0.9;

    /// <summary>Quantize per-Gaussian color / SH coefficients into a small integer range.</summary>
    /// <value>true (paper-standard for shipped cloud files).</value>
    public bool QuantizeSphericalHarmonics { get; set; } = true;

    /// <summary>Bits-per-coefficient after quantization.</summary>
    /// <value>8 (paper-standard). 4-bit is experimental (~2x further storage reduction with a small quality cost).</value>
    public int QuantizationBits { get; set; } = 8;

    /// <summary>Creates a new instance with paper-quality defaults.</summary>
    public GaussianCompressionOptions() { }

    /// <summary>Deep-copies every field from <paramref name="other"/>.</summary>
    /// <exception cref="ArgumentNullException"><paramref name="other"/> is null.</exception>
    public GaussianCompressionOptions(GaussianCompressionOptions other)
    {
        if (other is null) throw new ArgumentNullException(nameof(other));
        PruneLowOpacity           = other.PruneLowOpacity;
        MergeOverlapping          = other.MergeOverlapping;
        MergeOverlapThreshold     = other.MergeOverlapThreshold;
        QuantizeSphericalHarmonics = other.QuantizeSphericalHarmonics;
        QuantizationBits          = other.QuantizationBits;
    }
}

/// <summary>
/// Per-attribute learning-rate multipliers applied to a Gaussian's children immediately
/// after it splits (#1835 excellence goal #4).
/// </summary>
/// <remarks>
/// <para>
/// Paper (Kerbl et al. 2023 §5.3): position LR decreases (children are now local — small
/// movements matter more), scale LR increases (children are smaller and need faster updates
/// to converge), others unchanged. Values persist on the child Gaussian and compound across
/// second-generation splits.
/// </para>
/// <para><b>For Beginners:</b> When a Gaussian splits into two, the children need different
/// learning rates than their parent — smaller position steps (they're where they need to be)
/// and larger scale steps (they need to shrink to fit). Reference impls hard-code the paper
/// values; here you can override any of them.
/// </para>
/// </remarks>
public sealed class SplitChildLearningRateScales
{
    /// <summary>Position LR multiplier for split children.</summary>
    /// <value>0.7 (paper — decrease).</value>
    public double Position { get; set; } = 0.7;

    /// <summary>Scale LR multiplier for split children.</summary>
    /// <value>1.5 (paper — increase).</value>
    public double Scale { get; set; } = 1.5;

    /// <summary>Opacity LR multiplier for split children.</summary>
    /// <value>1.0 (paper — unchanged).</value>
    public double Opacity { get; set; } = 1.0;

    /// <summary>Rotation LR multiplier for split children.</summary>
    /// <value>1.0 (paper — unchanged).</value>
    public double Rotation { get; set; } = 1.0;

    /// <summary>Color LR multiplier for split children.</summary>
    /// <value>1.0 (paper — unchanged).</value>
    public double Color { get; set; } = 1.0;

    /// <summary>Creates a new instance with paper defaults.</summary>
    public SplitChildLearningRateScales() { }

    /// <summary>Deep-copies every field from <paramref name="other"/>.</summary>
    /// <exception cref="ArgumentNullException"><paramref name="other"/> is null.</exception>
    public SplitChildLearningRateScales(SplitChildLearningRateScales other)
    {
        if (other is null) throw new ArgumentNullException(nameof(other));
        Position = other.Position;
        Scale    = other.Scale;
        Opacity  = other.Opacity;
        Rotation = other.Rotation;
        Color    = other.Color;
    }
}
