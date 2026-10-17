namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>Options for WaveGrad (Chen et al. 2021): a diffusion vocoder that conditions its noise predictor on the
/// continuous noise level √ᾱ, so one trained model samples with any inference schedule.</summary>
/// <remarks>
/// <para>
/// Defaults are the paper's WaveGrad Base conditioned on the continuous noise level (§2.2, §4, App. A–B): 24 kHz
/// audio, 128 mel bands from a 2048-point FFT with a 50 ms window and a 12.5 ms hop, 20 Hz to 12 kHz; UBlocks
/// upsampling 5, 5, 3, 2, 2 with 512, 512, 256, 128, 128 channels and dilations 1, 2, 4, 8 in the first three and
/// 1, 2, 1, 2 in the rest; mirrored DBlocks with dilations 1, 2, 4; a 5×1 convolution of 32 channels over the waveform
/// and a 3×1 convolution of 768 over the mel spectrogram; the noise level scaled by C = 5000 in the positional
/// encoding; S = 1000 training levels from the β schedule Linear(1e-6, 0.01, 1000), a level drawn uniformly between
/// two neighbouring ones; the L1 loss; 24-frame (7,200-sample) crops at batch 256. <see cref="Large"/> builds
/// WaveGrad Large.
/// </para>
/// <para>What the paper leaves open follows lmnt-com/wavegrad: Adam at 2e-4, the gradient norm clipped to 1; leaky
/// ReLU 0.2; features from torchaudio (centred, window-normalized magnitude, HTK bands) as
/// <c>clamp((20 log10(max(x, 1e-5)) − 20 + 100) / 100, 0, 1)</c>; each sampling step clamped to [−1, 1]; synthesis over
/// the training schedule unless <see cref="InferenceNoiseSchedule"/> is set. The paper's six-iteration schedule was
/// found by a grid search over {1, …, 9} × 10^{−6 … −1} per step (App. B) and its values are not published.</para>
/// <para><b>For Beginners:</b> These options configure the WaveGrad model. Default values follow the original paper settings.</para>
/// </remarks>
public class WaveGradOptions : VocoderOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public WaveGradOptions(WaveGradOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        UpsampleFactors = (int[])other.UpsampleFactors.Clone();
        UpsampleChannels = (int[])other.UpsampleChannels.Clone();
        UpsampleDilations = other.UpsampleDilations.Select(d => (int[])d.Clone()).ToArray();
        RepeatBlocks = other.RepeatBlocks;
        MelProjectionChannels = other.MelProjectionChannels;
        WaveformChannels = other.WaveformChannels;
        NoiseLevelScale = other.NoiseLevelScale;
        LeakySlope = other.LeakySlope;
        NoiseSchedule = (double[])other.NoiseSchedule.Clone();
        InferenceNoiseSchedule = other.InferenceNoiseSchedule is null ? null : (double[])other.InferenceNoiseSchedule.Clone();
        CropFrames = other.CropFrames;
        WindowSize = other.WindowSize;
        MelMinFrequency = other.MelMinFrequency;
        MelMaxFrequency = other.MelMaxFrequency;
        GradientClipNorm = other.GradientClipNorm;
        SamplingSeed = other.SamplingSeed;
    }

    /// <summary>Creates the paper's WaveGrad Base configuration.</summary>
    public WaveGradOptions()
    {
        SampleRate = 24000;
        MelChannels = 128;
        HopSize = 300;
        FftSize = 2048;
        LearningRate = 2e-4;
        WeightDecay = 0.0;
        NumDiffusionSteps = 1000;
    }

    /// <summary>WaveGrad Large (§4): every UBlock and DBlock followed by one that does not resample, dilations
    /// 1, 2, 4, 8 in every UBlock, 60-frame (18,000-sample) crops.</summary>
    public static WaveGradOptions Large() => new()
    {
        RepeatBlocks = true,
        UpsampleDilations = [[1, 2, 4, 8], [1, 2, 4, 8], [1, 2, 4, 8], [1, 2, 4, 8], [1, 2, 4, 8]],
        CropFrames = 60,
    };

    /// <summary>Gets or sets the UBlocks' upsampling factors (5, 5, 3, 2, 2; their product is the hop).</summary>
    public int[] UpsampleFactors { get; set; } = [5, 5, 3, 2, 2];

    /// <summary>Gets or sets the UBlocks' output channels (512, 512, 256, 128, 128).</summary>
    public int[] UpsampleChannels { get; set; } = [512, 512, 256, 128, 128];

    /// <summary>Gets or sets the four dilations of each UBlock (1, 2, 4, 8 for the first three; 1, 2, 1, 2 after).</summary>
    public int[][] UpsampleDilations { get; set; } = [[1, 2, 4, 8], [1, 2, 4, 8], [1, 2, 4, 8], [1, 2, 1, 2], [1, 2, 1, 2]];

    /// <summary>Gets or sets whether each UBlock and DBlock is followed by a non-resampling one (WaveGrad Large).</summary>
    public bool RepeatBlocks { get; set; }

    /// <summary>Gets or sets the channels of the 3×1 convolution over the mel spectrogram (768).</summary>
    public int MelProjectionChannels { get; set; } = 768;

    /// <summary>Gets or sets the channels of the 5×1 convolution over the noisy waveform (32).</summary>
    public int WaveformChannels { get; set; } = 32;

    /// <summary>Gets or sets the linear scale C of the noise level in the positional encoding (5000).</summary>
    public double NoiseLevelScale { get; set; } = 5000;

    /// <summary>Gets or sets the leaky-ReLU slope (0.2).</summary>
    public double LeakySlope { get; set; } = 0.2;

    /// <summary>Gets or sets the β schedule whose S noise levels the training draws between (Linear(1e-6, 0.01, 1000)).</summary>
    public double[] NoiseSchedule { get; set; } = Linear(1e-6, 0.01, 1000);

    /// <summary>Gets or sets the inference β schedule, or null to sample with <see cref="NoiseSchedule"/>.</summary>
    public double[]? InferenceNoiseSchedule { get; set; }

    /// <summary>Gets or sets the mel frames of a training crop (24).</summary>
    public int CropFrames { get; set; } = 24;

    /// <summary>Gets or sets the analysis window of the input features (1200: 50 ms).</summary>
    public int WindowSize { get; set; } = 1200;

    /// <summary>Gets or sets the lowest mel frequency (20 Hz).</summary>
    public double MelMinFrequency { get; set; } = 20;

    /// <summary>Gets or sets the highest mel frequency (12 kHz).</summary>
    public double MelMaxFrequency { get; set; } = 12000;

    /// <summary>Gets or sets the global gradient-norm clip of a training step (1).</summary>
    public double GradientClipNorm { get; set; } = 1.0;

    /// <summary>Gets or sets the seed of the training draws and of the synthesis noise.</summary>
    public int SamplingSeed { get; set; }

    /// <summary>Values linearly spaced from <paramref name="from"/> to <paramref name="to"/> (numpy <c>linspace</c>).</summary>
    public static double[] Linear(double from, double to, int count) => DiffWaveOptions.Linear(from, to, count);
}
