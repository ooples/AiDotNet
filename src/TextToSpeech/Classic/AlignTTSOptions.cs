namespace AiDotNet.TextToSpeech.Classic;

/// <summary>Options for AlignTTS (feed-forward TTS that learns its own alignment with a mix density network).</summary>
/// <remarks>
/// <para>Defaults are the paper's configuration (Zeng et al. 2020, §4.2): 6 FFT blocks on each side with every
/// dimension 768, 2 heads and kernel 3; a duration predictor of 2 FFT blocks with dimension 128; a mix density network
/// with hidden size 256 whose output is a mean and a variance per mel channel.</para>
/// <para><b>For Beginners:</b> These options configure the AlignTTS model. Default values follow the original paper settings.</para>
/// </remarks>
public class AlignTTSOptions : AcousticModelOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public AlignTTSOptions(AlignTTSOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        FftFilterSize = other.FftFilterSize;
        FftKernelSize = other.FftKernelSize;
        DurationPredictorDim = other.DurationPredictorDim;
        DurationPredictorLayers = other.DurationPredictorLayers;
        MixDensityHiddenSize = other.MixDensityHiddenSize;
        MixDensityHiddenLayers = other.MixDensityHiddenLayers;
    }

    public AlignTTSOptions()
    {
        EncoderDim = 768;
        DecoderDim = 80;
        HiddenDim = 768;
        NumEncoderLayers = 6;
        NumDecoderLayers = 6;
        NumHeads = 2;
        UsePostnet = false; // AlignTTS has no postnet.
    }

    /// <summary>Gets or sets the FFT blocks' convolution width ("the dimension of each network ... 768").</summary>
    public int FftFilterSize { get; set; } = 768;

    /// <summary>Gets or sets the kernel of both FFT convolutions (3).</summary>
    public int FftKernelSize { get; set; } = 3;

    /// <summary>Gets or sets the duration predictor's dimension (128).</summary>
    public int DurationPredictorDim { get; set; } = 128;

    /// <summary>Gets or sets the duration predictor's FFT blocks (2).</summary>
    public int DurationPredictorLayers { get; set; } = 2;

    /// <summary>Gets or sets the mix density network's hidden size (256).</summary>
    public int MixDensityHiddenSize { get; set; } = 256;

    /// <summary>
    /// Gets or sets the mix density network's hidden linear layers. The paper says "multiple stacked linear layers"
    /// without a count; the default is two.
    /// </summary>
    public int MixDensityHiddenLayers { get; set; } = 2;
}
