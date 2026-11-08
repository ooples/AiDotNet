using AiDotNet.Attributes;

namespace AiDotNet.Safety.Watermarking;

/// <summary>
/// Configuration of a <see cref="SynthIdImageWatermarker{T}"/>. It adds SynthID-Image's detection objective and
/// calibrated decision to the StegaStamp networks and distortions configured by <see cref="StegaStampOptions"/>.
/// </summary>
/// <remarks>
/// The defaults follow what Gowal et al. (2025, Sec. 8) state for SynthID-O: a 136-bit payload in 512 x 512 images,
/// with thresholds set for a 0.1% false positive rate. The paper does not describe error correction, so all 136 bits are
/// payload here. It does not publish its networks, losses or training schedule. Those come from
/// <see cref="StegaStampOptions"/>.
/// </remarks>
[PaperDefault(nameof(ImageSize), 512, Source = "Sec. 8: SynthID-O can encode 136-bit payloads within 512 x 512 pixel images")]
[PaperDefault(nameof(SecretBits), 136, Source = "Sec. 8: SynthID-O can encode 136-bit payloads within 512 x 512 pixel images")]
public class SynthIdImageOptions : StegaStampOptions
{
    /// <summary>SynthID-O's stated sizes, 512 x 512 images and 136 payload bits, with no error correction.</summary>
    public SynthIdImageOptions()
    {
        ImageSize = 512;
        SecretBits = 136;
        UseErrorCorrection = false;
    }

    /// <summary>A copy of <paramref name="other"/>.</summary>
    public SynthIdImageOptions(SynthIdImageOptions other) : base(other)
    {
        DetectionLossScale = other.DetectionLossScale;
        SpecificityLossScale = other.SpecificityLossScale;
        FalsePositiveRate = other.FalsePositiveRate;
    }

    /// <summary>
    /// Weight of the detection loss: cross-entropy of the detection logit on distorted watermarked images (label 1)
    /// and on the same distortions of the clean images (label 0). Like the specificity loss, it ramps in with the image
    /// loss, after the secret-only phase.
    /// </summary>
    public double DetectionLossScale { get; set; } = 1.0;

    /// <summary>
    /// Weight of the specificity loss. One image's residual is added to another image in the batch, and that
    /// composite is trained as unwatermarked, so a residual lifted onto other content is not detected. The paper
    /// asks for a content-dependent mark but does not give its loss. It is a fraction of the detection loss
    /// and ramps in with it. Set this to 0 to turn it off.
    /// </summary>
    public double SpecificityLossScale { get; set; } = 0.5;

    /// <summary>
    /// The level alpha of the calibrated decision. <see cref="SynthIdImageWatermarker{T}.Decide"/> reports watermarked
    /// only when the clean-image p-value is at most alpha. This bounds the false positive rate on images exchangeable
    /// with the clean calibration set. The paper's experiments use 0.1%.
    /// </summary>
    [PaperDefault(0.001, Source = "Sec. 8: unless stated otherwise, the models are calibrated to a 0.1% false positive rate")]
    public double FalsePositiveRate { get; set; } = 0.001;

    /// <inheritdoc />
    internal override StegaStampOptions Copy() => new SynthIdImageOptions(this);

    internal void ValidateDetection()
    {
        Validate();
        if (double.IsNaN(DetectionLossScale) || DetectionLossScale <= 0)
            throw new ArgumentOutOfRangeException(nameof(DetectionLossScale), DetectionLossScale, "The detection loss needs a positive weight.");
        if (double.IsNaN(SpecificityLossScale) || SpecificityLossScale < 0)
            throw new ArgumentOutOfRangeException(nameof(SpecificityLossScale), SpecificityLossScale, "The specificity loss weight cannot be negative.");
        if (double.IsNaN(ImageLossScale) || ImageLossScale <= 0)
            throw new ArgumentOutOfRangeException(nameof(ImageLossScale), ImageLossScale, "The detection loss ramps with the image loss, which needs a positive weight.");
        if (!(FalsePositiveRate > 0 && FalsePositiveRate < 1))
            throw new ArgumentOutOfRangeException(nameof(FalsePositiveRate), FalsePositiveRate, "The false positive rate lies in (0, 1).");
    }
}
