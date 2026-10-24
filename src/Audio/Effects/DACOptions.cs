using AiDotNet.Audio.Codecs;

namespace AiDotNet.Audio.Effects;

/// <summary>
/// Options for DAC, the Descript Audio Codec ("High-Fidelity Audio Compression with Improved RVQGAN", Kumar et al. 2023).
/// The defaults are the paper's 44.1 kHz model: strides (2, 4, 8, 8), decoder width 1536, nine 1024-entry codebooks with
/// 8-dimensional factorized codes (about 7.75 kbps), quantizer dropout, and the multi-period plus multi-band multi-scale STFT
/// discriminators.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> DAC turns 44.1 kHz audio into about 86 frames of codes per second and back, at high
/// fidelity for speech, music and general sounds. Fewer codebooks give a lower bitrate.</para>
/// </remarks>
public class DACOptions : NeuralAudioCodecOptions
{
    /// <summary>Creates the paper's 44.1 kHz configuration.</summary>
    public DACOptions()
    {
        SampleRate = 44100;
        Channels = 1;
        NumQuantizers = 9;
        CodebookSize = 1024;
        TargetBandwidthKbps = 8.0;
        SegmentSize = 16758;            // 0.38 s at 44.1 kHz (§4.3)
    }

    // ---------------------------------------------------------------- encoder / decoder (§4.3)

    /// <summary>Gets or sets the encoder's first width (64; reference).</summary>
    public int EncoderDim { get; set; } = 64;

    /// <summary>Gets or sets the encoder strides (2, 4, 8, 8).</summary>
    public int[] EncoderRates { get; set; } = [2, 4, 8, 8];

    /// <summary>Gets or sets the latent dimension (0: the encoder's final width, EncoderDim · 2^blocks = 1024; reference).</summary>
    public int LatentDim { get; set; }

    /// <summary>Gets or sets the decoder width (1536).</summary>
    public int DecoderDim { get; set; } = 1536;

    /// <summary>Gets or sets the decoder rates (8, 8, 4, 2).</summary>
    public int[] DecoderRates { get; set; } = [8, 8, 4, 2];

    // ---------------------------------------------------------------- quantizer (§3.2–3.3, App. A)

    /// <summary>Gets or sets the factorized code dimension M (8).</summary>
    public int CodebookDim { get; set; } = 8;

    /// <summary>Gets or sets the probability that a training example uses n_q ~ U[1, N_q] codebooks (1.0: the proposed
    /// model's quantizer dropout).</summary>
    public double QuantizerDropout { get; set; } = 1.0;

    /// <summary>Gets or sets whether the codebook and commitment losses compare raw vectors (the reference code) rather
    /// than L2-normalized ones (App. A; false).</summary>
    public bool CodebookLossesOnRawVectors { get; set; }

    // ---------------------------------------------------------------- discriminators (§3.4, §4.3)

    /// <summary>Gets or sets the multi-period discriminator's periods (2, 3, 5, 7, 11).</summary>
    public int[] DiscriminatorPeriods { get; set; } = [2, 3, 5, 7, 11];

    /// <summary>Gets or sets the multi-scale STFT discriminator's windows (2048, 1024, 512; hop a quarter).</summary>
    public int[] DiscriminatorWindows { get; set; } = [2048, 1024, 512];

    /// <summary>Gets or sets the STFT discriminator's band edges as fractions of the bins (0, .1, .25, .5, .75, 1).</summary>
    public double[] DiscriminatorBands { get; set; } = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0];

    /// <summary>Gets or sets the STFT discriminator's channels (32; reference).</summary>
    public int DiscriminatorChannels { get; set; } = 32;

    /// <summary>Gets or sets a divisor on the period discriminators' widths (1: the paper's).</summary>
    public int DiscriminatorWidthDivisor { get; set; } = 1;

    // ---------------------------------------------------------------- losses (§3.5)

    /// <summary>Gets or sets the mel losses' windows (32 … 2048; hop a quarter).</summary>
    public int[] MelWindows { get; set; } = [32, 64, 128, 256, 512, 1024, 2048];

    /// <summary>Gets or sets the mel bins per window (5, 10, 20, 40, 80, 160, 320).</summary>
    public int[] MelBins { get; set; } = [5, 10, 20, 40, 80, 160, 320];

    /// <summary>Gets or sets the multi-scale mel loss weight (15).</summary>
    public double MelLossWeight { get; set; } = 15.0;

    /// <summary>Gets or sets the feature-matching weight (2).</summary>
    public double FeatureLossWeight { get; set; } = 2.0;

    /// <summary>Gets or sets the adversarial weight (1).</summary>
    public double AdversarialLossWeight { get; set; } = 1.0;

    /// <summary>Gets or sets the codebook loss weight (1).</summary>
    public double CodebookLossWeight { get; set; } = 1.0;

    /// <summary>Gets or sets the commitment loss weight (0.25).</summary>
    public double CommitmentLossWeight { get; set; } = 0.25;

    // ---------------------------------------------------------------- optimizer (§4.3)

    /// <summary>Gets or sets AdamW's learning rate (1e-4).</summary>
    public double LearningRate { get; set; } = 1e-4;

    /// <summary>Gets or sets AdamW's β1 (0.8).</summary>
    public double Beta1 { get; set; } = 0.8;

    /// <summary>Gets or sets AdamW's β2 (0.9; the reference configuration uses 0.99).</summary>
    public double Beta2 { get; set; } = 0.9;

    /// <summary>Gets or sets AdamW's weight decay (0.01: PyTorch's default, which the reference keeps).</summary>
    public double WeightDecay { get; set; } = 0.01;

    /// <summary>Gets or sets the learning rate's decay per step (γ = 0.999996).</summary>
    public double LearningRateDecay { get; set; } = 0.999996;
}
