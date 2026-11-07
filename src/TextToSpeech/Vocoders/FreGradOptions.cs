namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>Options for FreGrad (Nguyen et al. 2024): a lightweight diffusion vocoder that denoises the two Haar
/// wavelet sub-bands of the waveform with frequency-aware dilated convolutions, a separate energy prior per band, a
/// zero-terminal-SNR schedule and a multi-resolution STFT magnitude loss.</summary>
/// <remarks>
/// <para>
/// Defaults are the paper's (§3–4.1): 30 frequency-aware residual blocks with a dilation cycle of 7 and 32 channels;
/// DiffWave's step embedding and mel upsampler with the upsampling halved (16 × 8 = 128 per frame on the half-length
/// sub-bands); the β schedule Linear(1e-4, 0.05, 50) shifted to zero terminal SNR with γ = 1e-4 (Eq. 8); a prior per
/// band from the lower and upper halves of the mel bands with PriorGrad's energy normalization; the loss
/// Σ_{l,h} L_diff + 0.1 · L_mag with L_mag the log-magnitude term of a 3-resolution STFT loss (FFT 512, 1024, 2048;
/// windows 240, 600, 1200); 22.05 kHz audio, 80 mel bands from a 1024-point FFT with hop 256 between 80 Hz and 8 kHz;
/// Adam (0.9, 0.999) at 2e-4, batch 16; 50 sampling steps.
/// </para>
/// <para>What the paper leaves open follows kaistmm/fregrad and the works it cites: the STFT loss hops 50, 120, 240
/// (Parallel WaveGAN's, whose resolutions the paper's FFT and window sizes are); the feature pipeline, energy statistics
/// and 62-frame crops of PriorGrad's reference; no gradient clipping; each sampling step clamped to [−1, 1]. The
/// reference's biquad filtering of the priors and of the sampled sub-bands is not in the paper and is not applied.</para>
/// <para><b>For Beginners:</b> These options configure the FreGrad model. Default values follow the original paper settings.</para>
/// </remarks>
public class FreGradOptions : PriorGradOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public FreGradOptions(FreGradOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        MagnitudeLossWeight = other.MagnitudeLossWeight;
        StftFftSizes = (int[])other.StftFftSizes.Clone();
        StftHopSizes = (int[])other.StftHopSizes.Clone();
        StftWindowSizes = (int[])other.StftWindowSizes.Clone();
    }

    /// <summary>Creates the paper's FreGrad configuration.</summary>
    public FreGradOptions()
    {
        NumResLayers = 30;
        ResChannels = 32;
        DilationCycle = 7;
        UpsampleStrides = [16, 8];
        MelMaxFrequency = 8000;
        NoiseSchedule = ZeroTerminalSnr(Linear(1e-4, 0.05, 50), 1e-4);
    }

    /// <summary>Gets or sets the weight λ of the STFT magnitude loss (0.1).</summary>
    public double MagnitudeLossWeight { get; set; } = 0.1;

    /// <summary>Gets or sets the STFT loss FFT sizes (512, 1024, 2048).</summary>
    public int[] StftFftSizes { get; set; } = [512, 1024, 2048];

    /// <summary>Gets or sets the STFT loss hops (50, 120, 240).</summary>
    public int[] StftHopSizes { get; set; } = [50, 120, 240];

    /// <summary>Gets or sets the STFT loss windows (240, 600, 1200).</summary>
    public int[] StftWindowSizes { get; set; } = [240, 600, 1200];

    /// <summary>The β schedule whose √ᾱ is shifted to reach (nearly) zero at the last step (Eq. 8, Lin et al. 2024):
    /// <c>√ᾱ' = √ᾱ_1 (√ᾱ − √ᾱ_T + γ) / (√ᾱ_1 − √ᾱ_T + γ)</c>, the first step unchanged.</summary>
    public static double[] ZeroTerminalSnr(double[] betas, double gamma)
    {
        int n = betas.Length;
        var root = new double[n];
        double product = 1;
        for (int i = 0; i < n; i++)
        {
            product *= 1 - betas[i];
            root[i] = Math.Sqrt(product);
        }
        double first = root[0], last = root[n - 1] - gamma;
        var shifted = new double[n];
        for (int i = 0; i < n; i++)
        {
            double r = (root[i] - last) * first / (first - last);
            shifted[i] = r * r;
        }
        var result = new double[n];
        result[0] = 1 - shifted[0];
        for (int i = 1; i < n; i++) result[i] = 1 - shifted[i] / shifted[i - 1];
        return result;
    }
}
