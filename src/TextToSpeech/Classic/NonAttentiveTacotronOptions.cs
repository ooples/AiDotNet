namespace AiDotNet.TextToSpeech.Classic;

/// <summary>Options for Non-Attentive Tacotron (supervised-duration variant).</summary>
/// <remarks>
/// <para>
/// Defaults are Table 6 of Shen et al. 2020 (Appendix A): 24 kHz audio, 50 ms Hann frames (1200 samples), 12.5 ms hop
/// (300), 2048-point FFT, 128 mel channels; token embedding 512; encoder convolutions [512, 512, 512] of kernel 5 with
/// batch normalization (decay 0.999) and no activation, a 512 × 2 bidirectional LSTM; duration and range predictors of
/// two 512 × 2 bidirectional LSTMs; positional embedding 32; decoder pre-net [256, 256] with ReLU and dropout 0.5, two
/// 1024-unit LSTMs with zoneout 0.1 and cell cap 10; post-net [512, 512, 512, 512, K] of kernel 5; λ_dur = 2.0;
/// reduction factor 2 (§4.1).
/// </para>
/// <para><b>For Beginners:</b> These options configure the NonAttentiveTacotron model. Default values follow the original paper settings.</para>
/// </remarks>
public class NonAttentiveTacotronOptions : AcousticModelOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public NonAttentiveTacotronOptions(NonAttentiveTacotronOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        EmbeddingDim = other.EmbeddingDim;
        EncoderConvChannels = (int[])other.EncoderConvChannels.Clone();
        EncoderKernelSize = other.EncoderKernelSize;
        BatchNormDecay = other.BatchNormDecay;
        EncoderLstmDim = other.EncoderLstmDim;
        DurationLstmDim = other.DurationLstmDim;
        RangeLstmDim = other.RangeLstmDim;
        PositionalEmbeddingDim = other.PositionalEmbeddingDim;
        PrenetSizes = (int[])other.PrenetSizes.Clone();
        PrenetDropout = other.PrenetDropout;
        DecoderLstmDim = other.DecoderLstmDim;
        ZoneoutProbability = other.ZoneoutProbability;
        LstmCellClip = other.LstmCellClip;
        PostnetKernelSize = other.PostnetKernelSize;
        PostnetDropout = other.PostnetDropout;
        DurationLossWeight = other.DurationLossWeight;
        WindowSize = other.WindowSize;
        DurationScale = other.DurationScale;
    }

    /// <summary>Creates the paper's configuration.</summary>
    public NonAttentiveTacotronOptions()
    {
        SampleRate = 24000;
        HopSize = 300;
        FftSize = 2048;
        MelChannels = 128;
        OutputsPerStep = 2;
        PostnetDim = 512;
        PostnetLayers = 5;
        DropoutRate = 0.5;
        LearningRate = 0.001;
        UsePostnet = true;
    }

    /// <summary>Gets or sets the token embedding width (512).</summary>
    public int EmbeddingDim { get; set; } = 512;

    /// <summary>Gets or sets the encoder convolutions' channels ([512, 512, 512]).</summary>
    public int[] EncoderConvChannels { get; set; } = { 512, 512, 512 };

    /// <summary>Gets or sets the encoder convolutions' kernel (5).</summary>
    public int EncoderKernelSize { get; set; } = 5;

    /// <summary>Gets or sets the encoder batch normalization decay (0.999).</summary>
    public double BatchNormDecay { get; set; } = 0.999;

    /// <summary>Gets or sets the encoder bidirectional LSTM width per direction (512).</summary>
    public int EncoderLstmDim { get; set; } = 512;

    /// <summary>Gets or sets the duration predictor's bidirectional LSTM width per direction (512).</summary>
    public int DurationLstmDim { get; set; } = 512;

    /// <summary>Gets or sets the range predictor's bidirectional LSTM width per direction (512).</summary>
    public int RangeLstmDim { get; set; } = 512;

    /// <summary>Gets or sets the positional embedding width (32).</summary>
    public int PositionalEmbeddingDim { get; set; } = 32;

    /// <summary>Gets or sets the decoder pre-net widths ([256, 256], supervised).</summary>
    public int[] PrenetSizes { get; set; } = { 256, 256 };

    /// <summary>Gets or sets the pre-net dropout (0.5).</summary>
    public double PrenetDropout { get; set; } = 0.5;

    /// <summary>Gets or sets the decoder LSTM width (1024).</summary>
    public int DecoderLstmDim { get; set; } = 1024;

    /// <summary>Gets or sets the LSTM zoneout probability (0.1).</summary>
    public double ZoneoutProbability { get; set; } = 0.1;

    /// <summary>Gets or sets the LSTM cell absolute value cap (10).</summary>
    public double LstmCellClip { get; set; } = 10.0;

    /// <summary>Gets or sets the post-net kernel (5).</summary>
    public int PostnetKernelSize { get; set; } = 5;

    /// <summary>Gets or sets the post-net dropout (not stated; 0).</summary>
    public double PostnetDropout { get; set; }

    /// <summary>Gets or sets λ_dur (2.0, supervised).</summary>
    public double DurationLossWeight { get; set; } = 2.0;

    /// <summary>Gets or sets the analysis window in samples (50 ms at 24 kHz: 1200).</summary>
    public int WindowSize { get; set; } = 1200;

    /// <summary>Gets or sets the pace control applied to predicted durations (1; §4.2 divides durations by a factor).</summary>
    public double DurationScale { get; set; } = 1.0;
}
