namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>What DiffWave's noise predictor is conditioned on (Kong et al. 2021, §3.2–3.3).</summary>
public enum DiffWaveConditioner
{
    /// <summary>The mel spectrogram: DiffWave as a neural vocoder (§3.2 local conditioner, §5.1).</summary>
    MelSpectrogram,

    /// <summary>Nothing: unconditional waveform generation (§3.3, §5.2).</summary>
    Unconditional,

    /// <summary>A global discrete label such as a word or speaker ID (§3.2 global conditioner, §5.3).</summary>
    ClassLabel,
}

/// <summary>Options for DiffWave (Kong et al. 2021): a diffusion vocoder whose noise predictor is a non-autoregressive,
/// bidirectional dilated-convolution network conditioned on the mel spectrogram.</summary>
/// <remarks>
/// <para>
/// Defaults are the paper's DiffWave BASE with T = 50 (§5.1, Table 1): 30 residual layers of C = 64 channels, kernel 3,
/// dilation cycle 1, 2, …, 512; a linear β schedule from 1e-4 to 0.05 over 50 steps; 22.05 kHz audio, 80 mel bands
/// from a 1024-point FFT with hop 256; the mel upsampled ×256 by two transposed 2-D convolutions of stride 16 and
/// (3, 32) kernels with leaky ReLU 0.4; Adam at 2e-4, batch 16, about 16,000-sample clips; the fast-sampling schedule
/// 1e-4, 1e-3, 0.01, 0.05, 0.2, 0.5 (App. B; set <see cref="UseFastSampling"/>).
/// </para>
/// <para>What the paper leaves open follows lmnt-com/diffwave (<c>params.py</c>, <c>preprocess.py</c>): 62-frame crops;
/// mel features from torchaudio (centred, window-normalized magnitude, HTK bands from 20 Hz to Nyquist, a 1024-sample
/// window) as <c>clamp((20 log10(max(x, 1e-5)) − 20 + 100) / 100, 0, 1)</c>; each sampling step clamped to [−1, 1]; no
/// gradient clipping. The paper's L2 noise loss is used (the reference's code uses L1).</para>
/// <para>The paper's other two tasks have their own configurations: <see cref="Unconditional"/> (§5.2) and
/// <see cref="ClassConditional"/> (§5.3). Both generate whole utterances (the paper trains on full one-second SC09
/// clips), so a model input is the class label (or, unconditionally, ignored) rather than a spectrogram.</para>
/// <para><b>For Beginners:</b> These options configure the DiffWave model. Default values follow the original paper settings.</para>
/// </remarks>
public class DiffWaveOptions : VocoderOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public DiffWaveOptions(DiffWaveOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        NumResLayers = other.NumResLayers;
        ResChannels = other.ResChannels;
        DilationCycle = other.DilationCycle;
        NoiseSchedule = (double[])other.NoiseSchedule.Clone();
        InferenceNoiseSchedule = (double[])other.InferenceNoiseSchedule.Clone();
        UseFastSampling = other.UseFastSampling;
        UpsampleStrides = (int[])other.UpsampleStrides.Clone();
        CropFrames = other.CropFrames;
        WindowSize = other.WindowSize;
        MelMinFrequency = other.MelMinFrequency;
        SamplingSeed = other.SamplingSeed;
        Conditioner = other.Conditioner;
        NumClasses = other.NumClasses;
        UtteranceSamples = other.UtteranceSamples;
    }

    /// <summary>The paper's unconditional generation configuration (§5.2): 16 kHz one-second utterances
    /// (L = 16,000), 36 residual layers of C = 256 channels, dilation cycle 1, 2, …, 2048, T = 200 steps with β linear
    /// from 1e-4 to 0.02, sampled over the full schedule; Adam at 2e-4 with batch 16.</summary>
    public static DiffWaveOptions Unconditional() => new()
    {
        Conditioner = DiffWaveConditioner.Unconditional,
        SampleRate = 16000,
        UtteranceSamples = 16000,
        NumResLayers = 36,
        ResChannels = 256,
        DilationCycle = 12,
        NoiseSchedule = Linear(1e-4, 0.02, 200),
        NumDiffusionSteps = 200,
        UseFastSampling = false,
    };

    /// <summary>The paper's class-conditional configuration (§5.3): the unconditional model's hyperparameters with a
    /// shared 128-wide embedding of one of <paramref name="numClasses"/> labels (10 digits for SC09) added in every
    /// residual layer.</summary>
    public static DiffWaveOptions ClassConditional(int numClasses)
    {
        if (numClasses <= 0) throw new ArgumentOutOfRangeException(nameof(numClasses), "A class-conditional DiffWave needs at least one class.");
        var options = Unconditional();
        options.Conditioner = DiffWaveConditioner.ClassLabel;
        options.NumClasses = numClasses;
        return options;
    }

    /// <summary>Gets or sets what the noise predictor is conditioned on (the mel spectrogram).</summary>
    public DiffWaveConditioner Conditioner { get; set; } = DiffWaveConditioner.MelSpectrogram;

    /// <summary>Gets or sets the number of labels of the class conditioner (no default: it is the dataset's; SC09 has
    /// 10 digits).</summary>
    public int NumClasses { get; set; }

    /// <summary>Gets or sets the length L of a generated utterance without a spectrogram (16,000 in §5.2).</summary>
    public int UtteranceSamples { get; set; } = 16000;

    /// <summary>Creates the paper's DiffWave BASE (T = 50) configuration.</summary>
    public DiffWaveOptions()
    {
        SampleRate = 22050;
        MelChannels = 80;
        HopSize = 256;
        FftSize = 1024;
        LearningRate = 2e-4;
        WeightDecay = 0.0;
        NumDiffusionSteps = 50;
    }

    /// <summary>Gets or sets the residual layers N (30).</summary>
    public int NumResLayers { get; set; } = 30;

    /// <summary>Gets or sets the residual channels C (64 for BASE; 128 for LARGE).</summary>
    public int ResChannels { get; set; } = 64;

    /// <summary>Gets or sets the dilation cycle length (10: dilations 1 … 512).</summary>
    public int DilationCycle { get; set; } = 10;

    /// <summary>Gets or sets the training β schedule (50 values linearly spaced from 1e-4 to 0.05).</summary>
    public double[] NoiseSchedule { get; set; } = Linear(1e-4, 0.05, 50);

    /// <summary>Gets or sets the fast-sampling β schedule (1e-4, 1e-3, 0.01, 0.05, 0.2, 0.5).</summary>
    public double[] InferenceNoiseSchedule { get; set; } = [1e-4, 1e-3, 0.01, 0.05, 0.2, 0.5];

    /// <summary>Gets or sets whether synthesis uses the fast-sampling schedule (false: the training schedule).</summary>
    public bool UseFastSampling { get; set; }

    /// <summary>Gets or sets the spectrogram upsampler's strides (16, 16: their product is the hop).</summary>
    public int[] UpsampleStrides { get; set; } = [16, 16];

    /// <summary>Gets or sets the mel frames of a training crop (62).</summary>
    public int CropFrames { get; set; } = 62;

    /// <summary>Gets or sets the analysis window of the input features (1024).</summary>
    public int WindowSize { get; set; } = 1024;

    /// <summary>Gets or sets the lowest mel frequency (20 Hz).</summary>
    public double MelMinFrequency { get; set; } = 20;

    /// <summary>Gets or sets the seed of the training draws and of the synthesis noise.</summary>
    public int SamplingSeed { get; set; }

    /// <summary>Values linearly spaced from <paramref name="from"/> to <paramref name="to"/> (numpy <c>linspace</c>).</summary>
    public static double[] Linear(double from, double to, int count)
    {
        var values = new double[count];
        for (int i = 0; i < count; i++) values[i] = count == 1 ? from : from + (to - from) * i / (count - 1);
        return values;
    }
}
