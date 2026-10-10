namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>Options for the WaveNet vocoder (van den Oord et al. 2016): an autoregressive stack of dilated causal
/// convolutions predicting a softmax over μ-law classes, locally conditioned on the mel spectrogram.</summary>
/// <remarks>
/// <para>
/// The paper fixes the model's form (§2): 256 μ-law classes and a softmax; dilated causal convolutions whose dilation
/// doubles to 512 and repeats, with filter width 2 (each 1, 2, …, 512 block has a receptive field of 1024); gated
/// tanh × sigmoid units; residual and parameterized skip connections; ReLU → 1×1 → ReLU → 1×1 → softmax at the output;
/// local conditioning upsampled by a transposed convolutional network and added in every gate through 1×1 convolutions.
/// </para>
/// <para>What it leaves open follows r9y9/wavenet_vocoder's μ-law preset (<c>mulaw256_wavenet.json</c>): 30 layers in
/// 3 dilation cycles, 128 residual, 256 gate (128 per half) and 128 skip channels; upsampling by 4 × 4 × 4 × 4 = 256;
/// 22.05 kHz audio with 80-band log10 mel spectrograms (librosa, 1024-point FFT, hop 256, 125 Hz–7.6 kHz); Adam at
/// 1e-3 halved every 200k steps, batch 8, 10,240-sample segments, no gradient clipping.</para>
/// <para><b>For Beginners:</b> These options configure the WaveNet model. Default values follow the original paper settings.</para>
/// </remarks>
public class WaveNetOptions : VocoderOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public WaveNetOptions(WaveNetOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        NumDilatedLayers = other.NumDilatedLayers;
        DilationCycle = other.DilationCycle;
        KernelSize = other.KernelSize;
        ResidualChannels = other.ResidualChannels;
        GateChannels = other.GateChannels;
        SkipChannels = other.SkipChannels;
        MuLawLevels = other.MuLawLevels;
        UpsampleScales = (int[])other.UpsampleScales.Clone();
        SegmentSamples = other.SegmentSamples;
        WindowSize = other.WindowSize;
        MelMinFrequency = other.MelMinFrequency;
        MelMaxFrequency = other.MelMaxFrequency;
        LearningRateHalvingSteps = other.LearningRateHalvingSteps;
        SamplingSeed = other.SamplingSeed;
    }

    /// <summary>Creates the configuration described in the remarks.</summary>
    public WaveNetOptions()
    {
        SampleRate = 22050;
        MelChannels = 80;
        HopSize = 256;
        FftSize = 1024;
        LearningRate = 1e-3;
        WeightDecay = 0.0;
    }

    /// <summary>Gets or sets the dilated causal convolution layers (30).</summary>
    public int NumDilatedLayers { get; set; } = 30;

    /// <summary>Gets or sets the layers per dilation cycle (10: dilations 1 … 512).</summary>
    public int DilationCycle { get; set; } = 10;

    /// <summary>Gets or sets the causal convolutions' filter width (2).</summary>
    public int KernelSize { get; set; } = 2;

    /// <summary>Gets or sets the residual channels (128).</summary>
    public int ResidualChannels { get; set; } = 128;

    /// <summary>Gets or sets the channels of each half of the gated unit (128).</summary>
    public int GateChannels { get; set; } = 128;

    /// <summary>Gets or sets the skip channels (128).</summary>
    public int SkipChannels { get; set; } = 128;

    /// <summary>Gets or sets the μ-law classes (256).</summary>
    public int MuLawLevels { get; set; } = 256;

    /// <summary>Gets or sets the transposed convolutions' upsampling factors (4, 4, 4, 4; their product is the hop).</summary>
    public int[] UpsampleScales { get; set; } = [4, 4, 4, 4];

    /// <summary>Gets or sets the samples of a training segment (10,240; rounded down to whole frames).</summary>
    public int SegmentSamples { get; set; } = 10240;

    /// <summary>Gets or sets the analysis window of the input features (1024).</summary>
    public int WindowSize { get; set; } = 1024;

    /// <summary>Gets or sets the lowest mel frequency (125 Hz).</summary>
    public double MelMinFrequency { get; set; } = 125;

    /// <summary>Gets or sets the highest mel frequency (7.6 kHz).</summary>
    public double MelMaxFrequency { get; set; } = 7600;

    /// <summary>Gets or sets the steps after which the learning rate halves (200,000).</summary>
    public int LearningRateHalvingSteps { get; set; } = 200000;

    /// <summary>Gets or sets the seed of the synthesis draws.</summary>
    public int SamplingSeed { get; set; }
}
