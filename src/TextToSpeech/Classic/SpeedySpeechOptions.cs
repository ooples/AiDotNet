namespace AiDotNet.TextToSpeech.Classic;

/// <summary>Options for SpeedySpeech's student network (fully convolutional non-autoregressive TTS).</summary>
/// <remarks>
/// <para>
/// Defaults are the paper's (Vainer &amp; Dušek 2020, §4.3) with what it leaves unstated taken from the authors'
/// implementation (github.com/janvainer/speedyspeech, <c>hparam.HPFertility</c>, <c>HPStft</c>): 128 channels; 13 encoder
/// residual blocks of two convolutions with dilations <c>4 × [1, 2, 4] + [1]</c> (the paper's "26 encoder blocks with
/// dilations 1, 1, 2, 2, 4, 4"), 17 decoder blocks with <c>4 × [1, 2, 4, 8] + [1]</c> (its "34 decoder blocks"), kernel 4,
/// a duration predictor of three one-convolution blocks with kernels 4, 3 and 1 (the paper's "dilations 4, 3, 1");
/// LJSpeech audio at 22.05 kHz, 1024-point STFT, hop 256, 80 mel bins; log-mel targets standardized with the LJSpeech
/// statistics (mean −5.522, standard deviation 2.063); Adam at 0.002 with gradients clipped to norm 1.
/// </para>
/// <para><b>For Beginners:</b> These options configure the SpeedySpeech model. Default values follow the original paper settings.</para>
/// </remarks>
public class SpeedySpeechOptions : AcousticModelOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public SpeedySpeechOptions(SpeedySpeechOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        EncoderDilations = (int[])other.EncoderDilations.Clone();
        DecoderDilations = (int[])other.DecoderDilations.Clone();
        EncoderKernelSize = other.EncoderKernelSize;
        DecoderKernelSize = other.DecoderKernelSize;
        DurationPredictorKernelSizes = (int[])other.DurationPredictorKernelSizes.Clone();
        MelMean = other.MelMean;
        MelStd = other.MelStd;
        GradientClipNorm = other.GradientClipNorm;
    }

    /// <summary>Creates the paper's LJSpeech configuration.</summary>
    public SpeedySpeechOptions()
    {
        EncoderDim = 128;
        DecoderDim = 80;
        HiddenDim = 128;
        SampleRate = 22050;
        MelChannels = 80;
        HopSize = 256;
        FftSize = 1024;
        UsePostnet = false;
        LearningRate = 0.002;
    }

    /// <summary>Gets or sets the dilation of each encoder residual block (two convolutions each).</summary>
    public int[] EncoderDilations { get; set; } = { 1, 2, 4, 1, 2, 4, 1, 2, 4, 1, 2, 4, 1 };

    /// <summary>Gets or sets the dilation of each decoder residual block (two convolutions each).</summary>
    public int[] DecoderDilations { get; set; } = { 1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8, 1 };

    /// <summary>Gets or sets the kernel of the encoder's convolutions (4).</summary>
    public int EncoderKernelSize { get; set; } = 4;

    /// <summary>Gets or sets the kernel of the decoder's convolutions (4).</summary>
    public int DecoderKernelSize { get; set; } = 4;

    /// <summary>Gets or sets the kernel of each duration predictor block (4, 3, 1).</summary>
    public int[] DurationPredictorKernelSizes { get; set; } = { 4, 3, 1 };

    /// <summary>Gets or sets the mean the log-mel targets are standardized with (LJSpeech: −5.522).</summary>
    public double MelMean { get; set; } = -5.522;

    /// <summary>Gets or sets the standard deviation the log-mel targets are standardized with (LJSpeech: 2.063).</summary>
    public double MelStd { get; set; } = 2.063;

    /// <summary>Gets or sets the gradient-norm clip (1).</summary>
    public double GradientClipNorm { get; set; } = 1.0;
}
