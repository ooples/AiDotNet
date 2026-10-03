namespace AiDotNet.TextToSpeech.Classic;

/// <summary>Options for Deep Voice 3 (fully convolutional attention-based TTS), single-speaker configuration.</summary>
/// <remarks>
/// <para>
/// Defaults are Table 4's single-speaker column (Ping et al. 2018, App. C): 48 kHz audio, 4096-point FFT, 2400-sample
/// window and 600-sample shift, r = 4, 80 mel bands; character embedding 256; encoder 7 layers of width 5 with 64
/// channels; decoder pre-net 128 and 256, 4 decoder layers of width 5, attention hidden size 128; position weight 1.0
/// and initial key rate 6.3 (query rate 1); converter 5 layers of width 5 with 256 channels; dropout keep probability
/// 0.95; Adam at 0.001 with maximum gradient norm 100 and gradient clipping value 5. The decoding limits follow the
/// reference implementation (r9y9/deepvoice3_pytorch: at most 200 steps, at least 10).
/// </para>
/// <para><b>For Beginners:</b> These options configure the DeepVoice3 model. Default values follow the original paper settings.</para>
/// </remarks>
public class DeepVoice3Options : AcousticModelOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public DeepVoice3Options(DeepVoice3Options other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        ConvKernelSize = other.ConvKernelSize;
        EmbeddingDim = other.EmbeddingDim;
        EncoderChannels = other.EncoderChannels;
        DecoderPrenetSizes = (int[])other.DecoderPrenetSizes.Clone();
        AttentionDim = other.AttentionDim;
        PositionWeight = other.PositionWeight;
        KeyPositionRate = other.KeyPositionRate;
        QueryPositionRate = other.QueryPositionRate;
        NumConverterLayers = other.NumConverterLayers;
        ConverterChannels = other.ConverterChannels;
        WindowSize = other.WindowSize;
        MaxGradientNorm = other.MaxGradientNorm;
        GradientClipValue = other.GradientClipValue;
        MaxDecoderSteps = other.MaxDecoderSteps;
        MinDecoderSteps = other.MinDecoderSteps;
    }

    /// <summary>Creates the paper's single-speaker configuration.</summary>
    public DeepVoice3Options()
    {
        SampleRate = 48000;
        FftSize = 4096;
        HopSize = 600;
        MelChannels = 80;
        OutputsPerStep = 4;
        NumEncoderLayers = 7;
        NumDecoderLayers = 4;
        DropoutRate = 0.05;
        LearningRate = 0.001;
        UsePostnet = false;
    }

    /// <summary>Gets or sets the width of every convolution (5).</summary>
    public int ConvKernelSize { get; set; } = 5;

    /// <summary>Gets or sets the character embedding width (256).</summary>
    public int EmbeddingDim { get; set; } = 256;

    /// <summary>Gets or sets the encoder convolution channels (64).</summary>
    public int EncoderChannels { get; set; } = 64;

    /// <summary>Gets or sets the decoder pre-net's fully connected widths (128, 256); the last is the decoder width.</summary>
    public int[] DecoderPrenetSizes { get; set; } = { 128, 256 };

    /// <summary>Gets or sets the attention hidden size (128).</summary>
    public int AttentionDim { get; set; } = 128;

    /// <summary>Gets or sets the positional encodings' weight (1.0).</summary>
    public double PositionWeight { get; set; } = 1.0;

    /// <summary>Gets or sets the keys' position rate ω_key (6.3, the dataset's output-to-input timestep ratio).</summary>
    public double KeyPositionRate { get; set; } = 6.3;

    /// <summary>Gets or sets the queries' position rate ω_query (1).</summary>
    public double QueryPositionRate { get; set; } = 1.0;

    /// <summary>Gets or sets the converter's convolution blocks (5).</summary>
    public int NumConverterLayers { get; set; } = 5;

    /// <summary>Gets or sets the converter's channels (256).</summary>
    public int ConverterChannels { get; set; } = 256;

    /// <summary>Gets or sets the analysis window in samples (2400).</summary>
    public int WindowSize { get; set; } = 2400;

    /// <summary>Gets or sets the maximum gradient norm (100).</summary>
    public double MaxGradientNorm { get; set; } = 100.0;

    /// <summary>Gets or sets the gradient clipping value applied to every component after the norm clip (5).</summary>
    public double GradientClipValue { get; set; } = 5.0;

    /// <summary>Gets or sets the most decoder steps at inference (200).</summary>
    public int MaxDecoderSteps { get; set; } = 200;

    /// <summary>Gets or sets the fewest decoder steps before the done prediction may stop decoding (10).</summary>
    public int MinDecoderSteps { get; set; } = 10;
}
