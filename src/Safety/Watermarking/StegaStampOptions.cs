using AiDotNet.Attributes;
using AiDotNet.Models.Options;

namespace AiDotNet.Safety.Watermarking;

/// <summary>
/// Configuration of a <see cref="StegaStampWatermarker{T}"/>: the image and secret sizes and the training schedule of
/// Tancik et al. (2020).
/// </summary>
/// <remarks>
/// The image size, the message length and the BCH protection are the paper's (Sec. 4.1, 5.4). The paper gives no
/// learning rate, loss weights or ramps. Those defaults are the authors' reference training script
/// (scripts/base.sh in github.com/tancik/StegaStamp). The ramps count training steps.
/// </remarks>
public class StegaStampOptions : NeuralNetworkOptions
{
    /// <summary>The paper's sizes and the reference script's training schedule.</summary>
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
        LearningRateWarmupSteps = other.LearningRateWarmupSteps;
        CriticWeightClip = other.CriticWeightClip;
        CriticGradientClip = other.CriticGradientClip;
        ImageLossScale = other.ImageLossScale;
        SecretLossScale = other.SecretLossScale;
        CriticLossScale = other.CriticLossScale;
        ImageLossRampSteps = other.ImageLossRampSteps;
        CriticLossRampSteps = other.CriticLossRampSteps;
        NoImageLossSteps = other.NoImageLossSteps;
        EdgeLossGain = other.EdgeLossGain;
        EdgeLossDelaySteps = other.EdgeLossDelaySteps;
        EdgeLossRampSteps = other.EdgeLossRampSteps;
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
    [PaperDefault(400, Source = "Sec. 4.1: the encoder receives a 400 x 400 pixel input")]
    public virtual int ImageSize { get; set; } = 400;

    /// <summary>Bits the encoder hides (the paper uses 100: 56 data bits, BCH parity, padding).</summary>
    [PaperDefault(100, Source = "Sec. 4.1 and 5.4: the message is a 100 bit binary string")]
    public virtual int SecretBits { get; set; } = 100;

    /// <summary>
    /// Protects the keyed payload with a BCH code correcting 5 errors, as the paper does for its 56 data bits
    /// (Sec. 5.4). This code works over GF(2^7): 56 data bits and 35 parity bits, so it needs
    /// <see cref="SecretBits"/> of at least 91, and the remaining bits are zero padding. The paper's reference code
    /// packs its parity in bytes, 40 bits.
    /// </summary>
    public bool UseErrorCorrection { get; set; } = true;

    /// <summary>
    /// Adam learning rate of the encoder and decoder (the reference script: 1e-4). As in the reference, the
    /// secret-only phase and the full loss each train with their own Adam, so the second starts with fresh moments.
    /// </summary>
    public double LearningRate { get; set; } = 1e-4;

    /// <summary>
    /// Steps over which the learning rate of the encoder and decoder's first optimizer (the secret-only phase's, or
    /// the full loss's when <see cref="NoImageLossSteps"/> is 0) rises linearly from 0 to <see cref="LearningRate"/>;
    /// 0 (the reference script) starts at the full rate. The first Adam steps move every weight by about the learning
    /// rate whatever its gradient, so a short run at a learning rate above the reference's can saturate the residual
    /// before the decoder learns to read it; a warmup avoids that.
    /// </summary>
    public int LearningRateWarmupSteps { get; set; }

    /// <summary>
    /// RMSProp learning rate of the WGAN critic (the reference script: 1e-5, with TensorFlow 1's RMSProp: decay 0.9,
    /// epsilon 1e-10, mean square starting at 1). The critic trains once the secret-only phase ends, after each
    /// encoder and decoder step, as in the reference.
    /// </summary>
    public double CriticLearningRate { get; set; } = 1e-5;

    /// <summary>The WGAN critic's weights are clipped to [-c, c] after each of its steps.</summary>
    public double CriticWeightClip { get; set; } = 0.01;

    /// <summary>Each element of the critic's gradient is clipped to [-c, c] before its step (the reference script: 0.25).</summary>
    public double CriticGradientClip { get; set; } = 0.25;

    /// <summary>Weight of the residual's L2 image loss once ramped (the reference script: 2).</summary>
    public double ImageLossScale { get; set; } = 2.0;

    /// <summary>Weight of the secret's cross-entropy (the reference script: 1.5).</summary>
    public double SecretLossScale { get; set; } = 1.5;

    /// <summary>Weight of the critic (adversarial) loss once ramped (the reference script: 0.5).</summary>
    public double CriticLossScale { get; set; } = 0.5;

    /// <summary>
    /// Steps over which the image loss's weight ramps linearly to <see cref="ImageLossScale"/>, counted from the start
    /// of training as in the reference script; the loss is off until <see cref="NoImageLossSteps"/>, so it switches
    /// on at the ramp's value there.
    /// </summary>
    public int ImageLossRampSteps { get; set; } = 15000;

    /// <summary>
    /// Steps over which the critic loss's weight ramps linearly to <see cref="CriticLossScale"/>, counted from the start
    /// of training as in the reference script; off until <see cref="NoImageLossSteps"/>.
    /// </summary>
    public int CriticLossRampSteps { get; set; } = 15000;

    /// <summary>Initial steps that train the secret alone, so decoding is learned before imperceptibility.</summary>
    public int NoImageLossSteps { get; set; } = 1500;

    /// <summary>
    /// Extra weight on the residual near the image border once ramped (the reference code's l2_edge_gain): the
    /// residual is scaled by 1 + gain * (1 - falloff) before it is squared.
    /// </summary>
    public double EdgeLossGain { get; set; } = 10.0;

    /// <summary>Steps before the edge gain starts to ramp in (the reference script: 80000).</summary>
    public int EdgeLossDelaySteps { get; set; } = 80000;

    /// <summary>Steps over which the edge gain ramps to <see cref="EdgeLossGain"/> after the delay (the reference script: 10000).</summary>
    public int EdgeLossRampSteps { get; set; } = 10000;

    /// <summary>
    /// Weights of the Y, U and V channels in the image loss. The reference script's 1, 100, 100 keep the residual
    /// almost entirely in luminance.
    /// </summary>
    public double[] YuvLossScales { get; set; } = { 1.0, 100.0, 100.0 };

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
        if (LearningRateWarmupSteps < 0)
            throw new ArgumentOutOfRangeException(nameof(LearningRateWarmupSteps), LearningRateWarmupSteps, "The warmup cannot be negative.");
        if (!(CriticGradientClip > 0))
            throw new ArgumentOutOfRangeException(nameof(CriticGradientClip), CriticGradientClip, "The critic's gradient clip must be positive.");
        if (EdgeLossDelaySteps < 0 || EdgeLossRampSteps < 0)
            throw new ArgumentOutOfRangeException(nameof(EdgeLossDelaySteps), "The edge loss delay and ramp cannot be negative.");
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
