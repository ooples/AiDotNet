namespace AiDotNet.TextToSpeech.Classic;

/// <summary>Options for FastSpeech (non-autoregressive TTS with duration predictor).</summary>
/// <remarks>
/// <para>Defaults are the paper's LJSpeech configuration (Ren et al. 2019, §4.2 and Table 5): 6 FFT blocks on each
/// side, hidden 384, 2 heads, convolutional FFN 384→1536→384 with kernel 3 in both layers, duration predictor of
/// 256 filters and kernel 3, dropout 0.1.</para>
/// <para><b>For Beginners:</b> These options configure the FastSpeech model. Default values follow the original paper settings.</para>
/// </remarks>
public class FastSpeechOptions : AcousticModelOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public FastSpeechOptions(FastSpeechOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        DurationPredictorFilterSize = other.DurationPredictorFilterSize;
        DurationPredictorKernelSize = other.DurationPredictorKernelSize;
        DurationPredictorDropout = other.DurationPredictorDropout;
        DurationScale = other.DurationScale;
        FftFilterSize = other.FftFilterSize;
        FftKernelSizes = (int[])other.FftKernelSizes.Clone();
    }

    public FastSpeechOptions()
    {
        EncoderDim = 384;
        DecoderDim = 80;
        HiddenDim = 384;
        NumEncoderLayers = 6;
        NumDecoderLayers = 6;
        NumHeads = 2;
        DropoutRate = 0.1;
        UsePostnet = false; // FastSpeech has no postnet.
    }

    /// <summary>
    /// Gets or sets the duration predictor filter size. The paper gives 384 in §4.2 and 256 in Table 5; the default
    /// follows Table 5.
    /// </summary>
    public int DurationPredictorFilterSize { get; set; } = 256;

    /// <summary>Gets or sets the duration predictor kernel size (3).</summary>
    public int DurationPredictorKernelSize { get; set; } = 3;

    /// <summary>Gets or sets the duration predictor dropout (the paper's dropout, 0.1).</summary>
    public double DurationPredictorDropout { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the speed control α applied to predicted durations at inference (Ren et al. 2019, §3.2): 1 is
    /// normal speed, larger is slower, smaller is faster.
    /// </summary>
    public double DurationScale { get; set; } = 1.0;

    /// <summary>Gets or sets the filter size of the FFT blocks' first convolution (1536).</summary>
    public int FftFilterSize { get; set; } = 1536;

    /// <summary>Gets or sets the kernel sizes of the FFT blocks' two convolutions (3 and 3).</summary>
    public int[] FftKernelSizes { get; set; } = { 3, 3 };
}
