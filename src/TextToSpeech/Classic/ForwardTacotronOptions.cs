namespace AiDotNet.TextToSpeech.Classic;

/// <summary>Options for ForwardTacotron (non-autoregressive Tacotron with duration, pitch and energy predictors).</summary>
/// <remarks>
/// <para>
/// Defaults are the reference implementation's single-speaker configuration (as-ideas/ForwardTacotron,
/// <c>configs/singlespeaker.yaml</c>): 22.05 kHz audio, 1024-point STFT, hop 256, 80 mel bins; embedding 256; series
/// predictors with a 64-wide embedding, 256-channel convolutions, dropout 0.5 and bidirectional GRUs of 64 (duration),
/// 128 (pitch) and 64 (energy); a CBHG pre-net (K = 16, 256 channels, 4 highways, dropout 0.5); a 512-unit
/// bidirectional LSTM; a CBHG post-net (K = 8, 256 channels, 4 highways, no dropout); pitch and energy strength 1;
/// variance loss factor 0.1; Adam with the progressive schedule 5e-5 for 150k steps then 1e-5; gradient norm clip 1;
/// voiced pitch kept within 30–600 Hz.
/// </para>
/// <para><b>For Beginners:</b> These options configure the ForwardTacotron model. Default values follow the reference implementation.</para>
/// </remarks>
public class ForwardTacotronOptions : AcousticModelOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public ForwardTacotronOptions(ForwardTacotronOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        EmbeddingDim = other.EmbeddingDim;
        SeriesEmbeddingDim = other.SeriesEmbeddingDim;
        DurationConvDim = other.DurationConvDim;
        DurationRnnDim = other.DurationRnnDim;
        DurationDropout = other.DurationDropout;
        PitchConvDim = other.PitchConvDim;
        PitchRnnDim = other.PitchRnnDim;
        PitchDropout = other.PitchDropout;
        PitchStrength = other.PitchStrength;
        EnergyConvDim = other.EnergyConvDim;
        EnergyRnnDim = other.EnergyRnnDim;
        EnergyDropout = other.EnergyDropout;
        EnergyStrength = other.EnergyStrength;
        PrenetDim = other.PrenetDim;
        PrenetBankSize = other.PrenetBankSize;
        PrenetDropout = other.PrenetDropout;
        RnnDim = other.RnnDim;
        PostnetChannels = other.PostnetChannels;
        PostnetBankSize = other.PostnetBankSize;
        PostnetDropout = other.PostnetDropout;
        VarianceLossFactor = other.VarianceLossFactor;
        SecondStageLearningRate = other.SecondStageLearningRate;
        SecondStageStartStep = other.SecondStageStartStep;
        GradientClipNorm = other.GradientClipNorm;
        PitchMinHz = other.PitchMinHz;
        PitchMaxHz = other.PitchMaxHz;
        PitchMean = other.PitchMean;
        PitchStd = other.PitchStd;
        EnergyMean = other.EnergyMean;
        EnergyStd = other.EnergyStd;
        DurationScale = other.DurationScale;
    }

    /// <summary>Creates the reference single-speaker configuration.</summary>
    public ForwardTacotronOptions()
    {
        SampleRate = 22050;
        HopSize = 256;
        FftSize = 1024;
        MelChannels = 80;
        LearningRate = 5e-5;
        UsePostnet = true;
    }

    /// <summary>Gets or sets the phoneme embedding width (256).</summary>
    public int EmbeddingDim { get; set; } = 256;

    /// <summary>Gets or sets the series predictors' embedding width (64).</summary>
    public int SeriesEmbeddingDim { get; set; } = 64;

    /// <summary>Gets or sets the duration predictor's convolution channels (256).</summary>
    public int DurationConvDim { get; set; } = 256;

    /// <summary>Gets or sets the duration predictor's GRU width (64).</summary>
    public int DurationRnnDim { get; set; } = 64;

    /// <summary>Gets or sets the duration predictor's dropout (0.5).</summary>
    public double DurationDropout { get; set; } = 0.5;

    /// <summary>Gets or sets the pitch predictor's convolution channels (256).</summary>
    public int PitchConvDim { get; set; } = 256;

    /// <summary>Gets or sets the pitch predictor's GRU width (128).</summary>
    public int PitchRnnDim { get; set; } = 128;

    /// <summary>Gets or sets the pitch predictor's dropout (0.5).</summary>
    public double PitchDropout { get; set; } = 0.5;

    /// <summary>Gets or sets the pitch conditioning strength (1; 0 disables it).</summary>
    public double PitchStrength { get; set; } = 1.0;

    /// <summary>Gets or sets the energy predictor's convolution channels (256).</summary>
    public int EnergyConvDim { get; set; } = 256;

    /// <summary>Gets or sets the energy predictor's GRU width (64).</summary>
    public int EnergyRnnDim { get; set; } = 64;

    /// <summary>Gets or sets the energy predictor's dropout (0.5).</summary>
    public double EnergyDropout { get; set; } = 0.5;

    /// <summary>Gets or sets the energy conditioning strength (1; 0 disables it).</summary>
    public double EnergyStrength { get; set; } = 1.0;

    /// <summary>Gets or sets the CBHG pre-net's channels (256).</summary>
    public int PrenetDim { get; set; } = 256;

    /// <summary>Gets or sets the CBHG pre-net's bank size K (16).</summary>
    public int PrenetBankSize { get; set; } = 16;

    /// <summary>Gets or sets the CBHG pre-net's dropout (0.5).</summary>
    public double PrenetDropout { get; set; } = 0.5;

    /// <summary>Gets or sets the bidirectional LSTM's width per direction (512).</summary>
    public int RnnDim { get; set; } = 512;

    /// <summary>Gets or sets the CBHG post-net's channels (256).</summary>
    public int PostnetChannels { get; set; } = 256;

    /// <summary>Gets or sets the CBHG post-net's bank size K (8).</summary>
    public int PostnetBankSize { get; set; } = 8;

    /// <summary>Gets or sets the CBHG post-net's dropout (0).</summary>
    public double PostnetDropout { get; set; }

    /// <summary>Gets or sets the weight of the duration, pitch and energy losses (0.1 each).</summary>
    public double VarianceLossFactor { get; set; } = 0.1;

    /// <summary>Gets or sets the learning rate of the second schedule stage (1e-5).</summary>
    public double SecondStageLearningRate { get; set; } = 1e-5;

    /// <summary>Gets or sets the step at which the second schedule stage begins (150,000).</summary>
    public int SecondStageStartStep { get; set; } = 150_000;

    /// <summary>Gets or sets the gradient norm clip (1).</summary>
    public double GradientClipNorm { get; set; } = 1.0;

    /// <summary>Gets or sets the lowest voiced F0 counted in a phoneme's pitch (30 Hz).</summary>
    public double PitchMinHz { get; set; } = 30.0;

    /// <summary>Gets or sets the highest voiced F0 counted in a phoneme's pitch (600 Hz).</summary>
    public double PitchMaxHz { get; set; } = 600.0;

    /// <summary>Gets or sets the training set's mean nonzero phoneme pitch, used to normalize pitch (0 leaves it as is).</summary>
    public double PitchMean { get; set; }

    /// <summary>Gets or sets the training set's standard deviation of nonzero phoneme pitch (1 leaves it as is).</summary>
    public double PitchStd { get; set; } = 1.0;

    /// <summary>Gets or sets the training set's mean nonzero phoneme energy (0 leaves it as is).</summary>
    public double EnergyMean { get; set; }

    /// <summary>Gets or sets the training set's standard deviation of nonzero phoneme energy (1 leaves it as is).</summary>
    public double EnergyStd { get; set; } = 1.0;

    /// <summary>Gets or sets the speaking-rate factor applied to predicted durations (1; the reference's alpha divides
    /// the prediction, so 1/alpha here).</summary>
    public double DurationScale { get; set; } = 1.0;
}
