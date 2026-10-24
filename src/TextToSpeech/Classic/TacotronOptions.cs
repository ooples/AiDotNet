namespace AiDotNet.TextToSpeech.Classic;

/// <summary>Options for Tacotron (attention-based sequence-to-sequence TTS with CBHG modules).</summary>
/// <remarks>
/// <para>
/// Defaults are the paper's (Wang et al. 2017, §4, Table 1): 24 kHz audio, 50 ms Hann frames (1200 samples) with a
/// 12.5 ms shift (300), 2048-point FFT and 0.97 pre-emphasis; 256-wide character embedding; pre-nets FC-256 and FC-128
/// with ReLU and dropout 0.5; an encoder CBHG with K = 16 and 128-wide convolutions, highways and GRUs; a 256-cell
/// attention GRU, 2 residual 256-cell decoder GRUs and r = 2 frames per step; a post-processing CBHG with K = 8 and
/// projections of 256 and 80; Griffin–Lim with 50 iterations on magnitudes raised to 1.2; Adam from 0.001.
/// </para>
/// <para><b>For Beginners:</b> These options configure the Tacotron model. Default values follow the original paper settings.</para>
/// </remarks>
public class TacotronOptions : AcousticModelOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public TacotronOptions(TacotronOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        EmbeddingDim = other.EmbeddingDim;
        PrenetSizes = (int[])other.PrenetSizes.Clone();
        PrenetDropout = other.PrenetDropout;
        EncoderBankSize = other.EncoderBankSize;
        PostBankSize = other.PostBankSize;
        CbhgChannels = other.CbhgChannels;
        PostProjectionChannels = other.PostProjectionChannels;
        AttentionDim = other.AttentionDim;
        DecoderRnnDim = other.DecoderRnnDim;
        WindowSize = other.WindowSize;
        PreEmphasis = other.PreEmphasis;
        MaxDecoderSteps = other.MaxDecoderSteps;
        GriffinLimIterations = other.GriffinLimIterations;
        GriffinLimPower = other.GriffinLimPower;
    }

    /// <summary>Creates the paper's configuration.</summary>
    public TacotronOptions()
    {
        SampleRate = 24000;
        HopSize = 300;
        FftSize = 2048;
        MelChannels = 80;
        OutputsPerStep = 2;
        LearningRate = 0.001;
        UsePostnet = true;
    }

    /// <summary>Gets or sets the character embedding width (256).</summary>
    public int EmbeddingDim { get; set; } = 256;

    /// <summary>Gets or sets the widths of the encoder and decoder pre-nets' two layers (256, 128).</summary>
    public int[] PrenetSizes { get; set; } = { 256, 128 };

    /// <summary>Gets or sets the pre-nets' dropout (0.5).</summary>
    public double PrenetDropout { get; set; } = 0.5;

    /// <summary>Gets or sets the encoder CBHG's convolution bank size K (16).</summary>
    public int EncoderBankSize { get; set; } = 16;

    /// <summary>Gets or sets the post-processing CBHG's convolution bank size K (8).</summary>
    public int PostBankSize { get; set; } = 8;

    /// <summary>Gets or sets the width of the CBHG bank convolutions, highways and each GRU direction (128).</summary>
    public int CbhgChannels { get; set; } = 128;

    /// <summary>Gets or sets the post-processing CBHG's first projection width (256).</summary>
    public int PostProjectionChannels { get; set; } = 256;

    /// <summary>Gets or sets the attention GRU and attention width (256).</summary>
    public int AttentionDim { get; set; } = 256;

    /// <summary>Gets or sets the decoder GRUs' width (256).</summary>
    public int DecoderRnnDim { get; set; } = 256;

    /// <summary>Gets or sets the analysis window in samples (50 ms at 24 kHz: 1200).</summary>
    public int WindowSize { get; set; } = 1200;

    /// <summary>Gets or sets the pre-emphasis coefficient (0.97).</summary>
    public double PreEmphasis { get; set; } = 0.97;

    /// <summary>Gets or sets the decoder steps synthesis runs (the paper states no stopping rule; the reference
    /// implementation decodes a fixed 1000 frames, which is 500 steps at r = 2).</summary>
    public int MaxDecoderSteps { get; set; } = 500;

    /// <summary>Gets or sets the Griffin–Lim iterations (50).</summary>
    public int GriffinLimIterations { get; set; } = 50;

    /// <summary>Gets or sets the power the predicted magnitudes are raised to before Griffin–Lim (1.2).</summary>
    public double GriffinLimPower { get; set; } = 1.2;
}
