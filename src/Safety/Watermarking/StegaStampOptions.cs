using AiDotNet.Models.Options;

namespace AiDotNet.Safety.Watermarking;

/// <summary>
/// Configuration of a <see cref="StegaStampWatermarker{T}"/>: the image and secret sizes and the training schedule of
/// Tancik et al. (2020). Defaults are the paper's; the ramps count training steps.
/// </summary>
public class StegaStampOptions : NeuralNetworkOptions
{
    /// <summary>The paper's settings.</summary>
    public StegaStampOptions() { }

    /// <summary>A copy of <paramref name="other"/>.</summary>
    public StegaStampOptions(StegaStampOptions other) : base(other)
    {
        if (other is null) throw new ArgumentNullException(nameof(other));
        ImageSize = other.ImageSize;
        SecretBits = other.SecretBits;
        UseErrorCorrection = other.UseErrorCorrection;
        LearningRate = other.LearningRate;
        CriticLearningRate = other.CriticLearningRate;
        CriticWeightClip = other.CriticWeightClip;
        ImageLossScale = other.ImageLossScale;
        SecretLossScale = other.SecretLossScale;
        CriticLossScale = other.CriticLossScale;
        ImageLossRampSteps = other.ImageLossRampSteps;
        CriticLossRampSteps = other.CriticLossRampSteps;
        NoImageLossSteps = other.NoImageLossSteps;
        EdgeLossGain = other.EdgeLossGain;
        YuvLossScales = (double[])other.YuvLossScales.Clone();
        PerspectiveJitter = other.PerspectiveJitter;
        PerspectiveRampSteps = other.PerspectiveRampSteps;
        BrightnessShift = other.BrightnessShift;
        HueShift = other.HueShift;
        SaturationMix = other.SaturationMix;
        NoiseStdDev = other.NoiseStdDev;
        ContrastLow = other.ContrastLow;
        ContrastHigh = other.ContrastHigh;
        MinimumJpegQuality = other.MinimumJpegQuality;
        ColorRampSteps = other.ColorRampSteps;
        BlurProbability = other.BlurProbability;
    }

    /// <summary>Side of the square image the networks work at (the paper uses 400). A multiple of 16.</summary>
    public int ImageSize { get; set; } = 400;

    /// <summary>Bits the encoder hides (the paper uses 100: 56 data bits, BCH parity, padding).</summary>
    public int SecretBits { get; set; } = 100;

    /// <summary>
    /// Protects the keyed payload with the paper's BCH code (56 data bits, 35 parity bits, corrects 5 errors). Needs
    /// <see cref="SecretBits"/> of at least 91; the remaining bits are zero padding, as in the paper.
    /// </summary>
    public bool UseErrorCorrection { get; set; } = true;

    /// <summary>Adam learning rate of the encoder and decoder (the paper: 1e-4).</summary>
    public double LearningRate { get; set; } = 1e-4;

    /// <summary>Learning rate of the WGAN critic.</summary>
    public double CriticLearningRate { get; set; } = 1e-4;

    /// <summary>The WGAN critic's weights are clipped to [-c, c] after each of its steps.</summary>
    public double CriticWeightClip { get; set; } = 0.01;

    /// <summary>Weight of the residual's L2 image loss once ramped (the paper: 2).</summary>
    public double ImageLossScale { get; set; } = 2.0;

    /// <summary>Weight of the secret's cross-entropy (the paper: 1.5).</summary>
    public double SecretLossScale { get; set; } = 1.5;

    /// <summary>Weight of the critic (adversarial) loss once ramped (the paper: 0.5).</summary>
    public double CriticLossScale { get; set; } = 0.5;

    /// <summary>Steps over which the image loss ramps to its full weight after <see cref="NoImageLossSteps"/>.</summary>
    public int ImageLossRampSteps { get; set; } = 15000;

    /// <summary>Steps over which the critic loss ramps to its full weight after <see cref="NoImageLossSteps"/>.</summary>
    public int CriticLossRampSteps { get; set; } = 15000;

    /// <summary>Initial steps that train the secret alone, so decoding is learned before imperceptibility.</summary>
    public int NoImageLossSteps { get; set; } = 1500;

    /// <summary>Extra L2 weight on the residual near the image border (the reference code's l2_edge_gain).</summary>
    public double EdgeLossGain { get; set; } = 10.0;

    /// <summary>Weights of the Y, U and V channels in the image loss.</summary>
    public double[] YuvLossScales { get; set; } = { 1.0, 1.0, 1.0 };

    /// <summary>Largest corner displacement of the random perspective warp, as a fraction of the side.</summary>
    public double PerspectiveJitter { get; set; } = 0.1;

    /// <summary>Steps over which the perspective warp ramps to full strength.</summary>
    public int PerspectiveRampSteps { get; set; } = 10000;

    /// <summary>Largest brightness offset.</summary>
    public double BrightnessShift { get; set; } = 0.3;

    /// <summary>Largest per-channel (hue) offset.</summary>
    public double HueShift { get; set; } = 0.1;

    /// <summary>Largest mix toward the luminance image (desaturation).</summary>
    public double SaturationMix { get; set; } = 1.0;

    /// <summary>Largest standard deviation of the additive Gaussian noise.</summary>
    public double NoiseStdDev { get; set; } = 0.02;

    /// <summary>Smallest contrast factor.</summary>
    public double ContrastLow { get; set; } = 0.5;

    /// <summary>Largest contrast factor.</summary>
    public double ContrastHigh { get; set; } = 1.5;

    /// <summary>Lowest quality of the differentiable JPEG approximation.</summary>
    public double MinimumJpegQuality { get; set; } = 50;

    /// <summary>Steps over which the colour, noise and JPEG perturbations ramp to full strength.</summary>
    public int ColorRampSteps { get; set; } = 1000;

    /// <summary>Chance of a Gaussian blur, and separately of a motion (line) blur, on a training step.</summary>
    public double BlurProbability { get; set; } = 0.25;

    /// <summary>A copy of these options with the runtime type preserved.</summary>
    internal virtual StegaStampOptions Copy() => new StegaStampOptions(this);

    internal void Validate()
    {
        if (ImageSize < 16 || ImageSize % 16 != 0)
            throw new ArgumentOutOfRangeException(nameof(ImageSize), ImageSize, "The image side must be a positive multiple of 16.");
        if (SecretBits < 8) throw new ArgumentOutOfRangeException(nameof(SecretBits), SecretBits, "At least 8 secret bits are needed.");
        if (UseErrorCorrection && SecretBits < 91)
            throw new ArgumentOutOfRangeException(nameof(SecretBits), SecretBits, "The BCH payload needs at least 91 secret bits.");
        if (YuvLossScales is null || YuvLossScales.Length != 3)
            throw new ArgumentException("Three YUV loss scales are needed.", nameof(YuvLossScales));
        if (!(ContrastLow > 0 && ContrastHigh >= ContrastLow))
            throw new ArgumentOutOfRangeException(nameof(ContrastLow), "Contrast factors must be positive and ordered.");
        if (!(MinimumJpegQuality >= 1 && MinimumJpegQuality <= 100))
            throw new ArgumentOutOfRangeException(nameof(MinimumJpegQuality), MinimumJpegQuality, "JPEG quality lies in [1, 100].");
        if (!(BlurProbability >= 0 && BlurProbability <= 0.5))
            throw new ArgumentOutOfRangeException(nameof(BlurProbability), BlurProbability, "Each blur's probability lies in [0, 0.5].");
    }
}
