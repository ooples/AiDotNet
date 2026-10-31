namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>Options for MelGAN (Kumar et al. 2019): a fully convolutional GAN vocoder with multi-scale window-based
/// discriminators, trained on the hinge loss with feature matching.</summary>
/// <remarks>
/// <para>
/// Defaults are the paper's (§2, App. A–B) and, where it is silent, its released code (descriptinc/melgan-neurips):
/// ngf 32 (a first width of 512), upsampling ratios 8, 8, 2, 2, three residual blocks per stack; three discriminators
/// of base width 16; λ_FM = 10; Adam (β = 0.5, 0.9) at 1e-4 for both networks, batch 16; 8192-sample segments; the
/// input is the reference's <c>Audio2Mel</c> (22.05 kHz, 1024-point FFT and window, hop 256, 80 Slaney mel bands,
/// <c>log10(max(x, 1e-5))</c>).
/// </para>
/// <para><b>For Beginners:</b> These options configure the MelGAN model. Default values follow the original paper settings.</para>
/// </remarks>
public class MelGANOptions : VocoderOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public MelGANOptions(MelGANOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        Ngf = other.Ngf;
        ResidualLayers = other.ResidualLayers;
        NumDiscriminators = other.NumDiscriminators;
        DiscriminatorWidthDivisor = other.DiscriminatorWidthDivisor;
        FeatureMatchingWeight = other.FeatureMatchingWeight;
        Beta1 = other.Beta1;
        Beta2 = other.Beta2;
        SegmentSize = other.SegmentSize;
        WindowSize = other.WindowSize;
        SamplingSeed = other.SamplingSeed;
    }

    /// <summary>Creates the paper's configuration.</summary>
    public MelGANOptions()
    {
        SampleRate = 22050;
        MelChannels = 80;
        HopSize = 256;
        FftSize = 1024;
        UpsampleRates = [8, 8, 2, 2];
        LearningRate = 1e-4;
        WeightDecay = 0.0;
    }

    /// <summary>Gets or sets the generator's base width ngf (32); the first convolution has ngf · 2^ratios channels.</summary>
    public int Ngf { get; set; } = 32;

    /// <summary>Gets or sets the residual blocks per stack (3: dilations 1, 3, 9).</summary>
    public int ResidualLayers { get; set; } = 3;

    /// <summary>Gets or sets the number of discriminators (3: raw audio, ×2 and ×4 downsampled).</summary>
    public int NumDiscriminators { get; set; } = 3;

    /// <summary>Gets or sets a divisor on every discriminator width (1: the paper's widths).</summary>
    public int DiscriminatorWidthDivisor { get; set; } = 1;

    /// <summary>Gets or sets λ, the feature-matching weight (10).</summary>
    public double FeatureMatchingWeight { get; set; } = 10.0;

    /// <summary>Gets or sets Adam's β₁ (0.5).</summary>
    public double Beta1 { get; set; } = 0.5;

    /// <summary>Gets or sets Adam's β₂ (0.9).</summary>
    public double Beta2 { get; set; } = 0.9;

    /// <summary>Gets or sets the training segment in samples (8192).</summary>
    public int SegmentSize { get; set; } = 8192;

    /// <summary>Gets or sets the STFT window of the input features (1024).</summary>
    public int WindowSize { get; set; } = 1024;

    /// <summary>Gets or sets the seed of the training segment draws.</summary>
    public int SamplingSeed { get; set; }
}
