using System;

namespace AiDotNet.Diffusion.StyleTransfer;

/// <summary>
/// Configuration for <see cref="UniVSTModel{T}"/>.
/// </summary>
/// <remarks>
/// <para>
/// Defaults are the published values from Song, Lin, Zhan, Yan, Cao and Ji, "UniVST: A Unified
/// Framework for Training-free Localized Video Style Transfer" (arXiv:2410.20084, TPAMI 2025):
/// Stable Diffusion v1.5, T = 50 DDIM steps, 512x512, 16 frames per batch.
/// </para>
/// <para>
/// Every timestep boundary is a FRACTION of T rather than an absolute step index, because that is
/// how the paper states them and because it keeps the schedule meaningful when
/// <see cref="DdimSteps"/> changes. A literal step number would silently mean something different
/// at T = 20 than at T = 50.
/// </para>
/// <para><b>For Beginners:</b> UniVST restyles only a chosen region of a video and leaves the rest
/// alone. It does not train anything — it steers a pretrained image diffusion model. These settings
/// control which region is tracked, how strongly the style is applied, and how much the result is
/// smoothed across frames so it does not flicker.</para>
/// </remarks>
public class UniVSTOptions
{
    /// <summary>Gets or sets the number of DDIM steps (T). Default 50, the paper's value.</summary>
    public int DdimSteps { get; set; } = 50;

    /// <summary>Gets or sets the number of frames processed together. Default 16, the paper's batch.</summary>
    public int NumFrames { get; set; } = 16;

    // ---------------------------------------------------------------- mask propagation

    /// <summary>
    /// Gets or sets the DDIM-inversion timestep, as a fraction of T, whose feature maps drive mask
    /// propagation. Default 0.4 (t0 = 0.4T), the paper's value.
    /// </summary>
    /// <remarks>
    /// The paper draws these features from the UNet's second upsampling block during INVERSION, not
    /// during denoising — see <see cref="MaskFeatureUpBlockIndex"/>.
    /// </remarks>
    public double MaskFeatureTimestepFraction { get; set; } = 0.4;

    /// <summary>
    /// Gets or sets which UNet upsampling block supplies the mask-propagation features.
    /// Default 2, the paper's "upsampling block-2".
    /// </summary>
    /// <remarks>
    /// Deeper blocks are too coarse to place a boundary and the earliest are too noisy to match
    /// reliably, so this is a real architectural choice rather than a tuning knob.
    /// </remarks>
    public int MaskFeatureUpBlockIndex { get; set; } = 2;

    /// <summary>
    /// Gets or sets k, the number of nearest neighbours matched per point. Default 10, the paper's value.
    /// </summary>
    public int MaskMatchNeighbors { get; set; } = 10;

    /// <summary>
    /// Gets or sets how many preceding frames act as anchors alongside the first frame.
    /// Default 9, giving the paper's "first frame + previous 9".
    /// </summary>
    /// <remarks>
    /// The first frame is always an anchor in addition to these. Keeping it pinned is what stops the
    /// mask drifting: a purely local chain would accumulate error frame over frame.
    /// </remarks>
    public int MaskAnchorHistory { get; set; } = 9;

    /// <summary>
    /// Gets or sets the random downsampling rate applied to candidate anchor points. Default 0.5,
    /// the paper's value.
    /// </summary>
    /// <remarks>
    /// The paper adjusts the sampling rates so the retained points keep the foreground/background
    /// proportions of the source mask. Uniform sampling would under-represent whichever region is
    /// smaller and bias the majority vote toward the larger one.
    /// </remarks>
    public double MaskDownsampleRate { get; set; } = 0.5;

    // ---------------------------------------------------------------- AdaIN stylization

    /// <summary>Gets or sets the start of the latent-level AdaIN window, as a fraction of T. Default 0.10.</summary>
    public double LatentAdaInStartFraction { get; set; } = 0.10;

    /// <summary>Gets or sets the end of the latent-level AdaIN window, as a fraction of T. Default 0.15.</summary>
    /// <remarks>
    /// A narrow late window on purpose: latent AdaIN applied over the whole trajectory overwhelms
    /// content structure, while applying it only near the end injects colour statistics after the
    /// layout has settled.
    /// </remarks>
    public double LatentAdaInEndFraction { get; set; } = 0.15;

    /// <summary>
    /// Gets or sets gamma, the query-blending weight between the stylized and content queries.
    /// Default 0.35, the paper's value. Applied at EVERY timestep.
    /// </summary>
    public double QueryBlendGamma { get; set; } = 0.35;

    /// <summary>Gets or sets the start of the key/value AdaIN ramp, as a fraction of T. Default 0.4.</summary>
    public double KeyValueAdaInStartFraction { get; set; } = 0.4;

    /// <summary>Gets or sets the end of the key/value AdaIN ramp, as a fraction of T. Default 1.0.</summary>
    public double KeyValueAdaInEndFraction { get; set; } = 1.0;

    /// <summary>Gets or sets beta at the start of the ramp. Default 0.1, the paper's value.</summary>
    public double BetaAtRampStart { get; set; } = 0.1;

    /// <summary>Gets or sets beta at the end of the ramp. Default 0.9, the paper's value.</summary>
    /// <remarks>
    /// beta interpolates LINEARLY between these across the ramp. A constant blend is the obvious
    /// simplification and it is wrong: early steps need mostly the raw style key/value to establish
    /// the appearance, later steps need mostly the AdaIN-aligned one to preserve content.
    /// </remarks>
    public double BetaAtRampEnd { get; set; } = 0.9;

    // ---------------------------------------------------------------- consistent smoothing

    /// <summary>Gets or sets the start of the smoothing window, as a fraction of T. Default 0.3.</summary>
    public double SmoothingStartFraction { get; set; } = 0.3;

    /// <summary>Gets or sets the end of the smoothing window, as a fraction of T. Default 0.4.</summary>
    public double SmoothingEndFraction { get; set; } = 0.4;

    /// <summary>
    /// Gets or sets m, the half-width of the smoothing window; the full window is 2m + 1 frames.
    /// Default 2, giving a 5-frame window.
    /// </summary>
    public int SmoothingHalfWindow { get; set; } = 2;

    /// <summary>
    /// Validates the configuration and throws when a value cannot describe a usable schedule.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">A value is out of range or an interval is inverted.</exception>
    public void Validate()
    {
        if (DdimSteps <= 0)
            throw new ArgumentOutOfRangeException(nameof(DdimSteps), DdimSteps, "DdimSteps must be positive.");
        if (NumFrames <= 0)
            throw new ArgumentOutOfRangeException(nameof(NumFrames), NumFrames, "NumFrames must be positive.");
        if (MaskFeatureUpBlockIndex < 0)
            throw new ArgumentOutOfRangeException(nameof(MaskFeatureUpBlockIndex), MaskFeatureUpBlockIndex,
                "MaskFeatureUpBlockIndex cannot be negative.");
        if (MaskMatchNeighbors <= 0)
            throw new ArgumentOutOfRangeException(nameof(MaskMatchNeighbors), MaskMatchNeighbors,
                "MaskMatchNeighbors must be positive.");
        if (MaskAnchorHistory < 0)
            throw new ArgumentOutOfRangeException(nameof(MaskAnchorHistory), MaskAnchorHistory,
                "MaskAnchorHistory cannot be negative.");
        if (MaskDownsampleRate is <= 0.0 or > 1.0)
            throw new ArgumentOutOfRangeException(nameof(MaskDownsampleRate), MaskDownsampleRate,
                "MaskDownsampleRate must be in (0, 1].");
        if (QueryBlendGamma is < 0.0 or > 1.0)
            throw new ArgumentOutOfRangeException(nameof(QueryBlendGamma), QueryBlendGamma,
                "QueryBlendGamma must be in [0, 1].");
        if (SmoothingHalfWindow < 0)
            throw new ArgumentOutOfRangeException(nameof(SmoothingHalfWindow), SmoothingHalfWindow,
                "SmoothingHalfWindow cannot be negative.");

        ValidateFraction(MaskFeatureTimestepFraction, nameof(MaskFeatureTimestepFraction));
        ValidateInterval(LatentAdaInStartFraction, LatentAdaInEndFraction,
            nameof(LatentAdaInStartFraction), nameof(LatentAdaInEndFraction));
        ValidateInterval(KeyValueAdaInStartFraction, KeyValueAdaInEndFraction,
            nameof(KeyValueAdaInStartFraction), nameof(KeyValueAdaInEndFraction));
        ValidateInterval(SmoothingStartFraction, SmoothingEndFraction,
            nameof(SmoothingStartFraction), nameof(SmoothingEndFraction));
    }

    private static void ValidateFraction(double value, string name)
    {
        if (value is < 0.0 or > 1.0 || double.IsNaN(value))
            throw new ArgumentOutOfRangeException(name, value, $"{name} must be a fraction of T in [0, 1].");
    }

    private static void ValidateInterval(double start, double end, string startName, string endName)
    {
        ValidateFraction(start, startName);
        ValidateFraction(end, endName);
        if (start > end)
            throw new ArgumentOutOfRangeException(startName, start,
                $"{startName} ({start}) must not exceed {endName} ({end}).");
    }
}
