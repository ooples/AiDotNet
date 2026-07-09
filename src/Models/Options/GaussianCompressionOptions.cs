namespace AiDotNet.Models.Options;

/// <summary>
/// Per-technique configuration for the GaussianSplatting post-training compression pass
/// (#1835 excellence goal #3). Every field is defaulted to a paper-friendly setting so the
/// user can enable <see cref="GaussianSplattingOptions.CompressOnBuildComplete"/> and get a
/// meaningful compressed cloud without touching this class. Advanced users tune per-technique
/// on/off + thresholds.
/// </summary>
public sealed class GaussianCompressionOptions
{
    /// <summary>
    /// Prune Gaussians whose opacity has collapsed below
    /// <see cref="GaussianSplattingOptions.OpacityPruneThreshold"/>. Default: on.
    /// </summary>
    public bool PruneLowOpacity { get; set; } = true;

    /// <summary>
    /// Merge Gaussians whose bounding ellipses overlap by more than
    /// <see cref="MergeOverlapThreshold"/>. Default: on.
    /// </summary>
    public bool MergeOverlapping { get; set; } = true;

    /// <summary>
    /// Overlap fraction above which two Gaussians are candidates for merging (0..1). Paper-
    /// standard default 0.9 keeps aggressive overlap.
    /// </summary>
    public double MergeOverlapThreshold { get; set; } = 0.9;

    /// <summary>
    /// Quantize per-Gaussian spherical harmonics coefficients from float32 to int8 with a
    /// per-Gaussian scale factor. Default: on (paper-standard for shipped cloud files).
    /// </summary>
    public bool QuantizeSphericalHarmonics { get; set; } = true;

    /// <summary>
    /// The bits-per-coefficient after quantization. 8-bit is paper-standard; 4-bit is
    /// experimental and unlocks ~2x further storage reduction with a small quality cost.
    /// </summary>
    public int QuantizationBits { get; set; } = 8;
}

/// <summary>
/// Per-attribute learning-rate multipliers applied to the two children immediately after a
/// Gaussian split (#1835 excellence goal #4). Paper: position decreases (children are now
/// local, small movements matter more), scale increases (they're smaller and need faster
/// updates to converge), others unchanged. Every field defaults to the paper value.
/// </summary>
public sealed class SplitChildLearningRateScales
{
    /// <summary>Position LR multiplier for split children. Paper: 0.7 (decrease).</summary>
    public double Position { get; set; } = 0.7;

    /// <summary>Scale LR multiplier for split children. Paper: 1.5 (increase).</summary>
    public double Scale { get; set; } = 1.5;

    /// <summary>Opacity LR multiplier. Paper: 1.0 (unchanged).</summary>
    public double Opacity { get; set; } = 1.0;

    /// <summary>Rotation LR multiplier. Paper: 1.0 (unchanged).</summary>
    public double Rotation { get; set; } = 1.0;

    /// <summary>Color LR multiplier. Paper: 1.0 (unchanged).</summary>
    public double Color { get; set; } = 1.0;
}
