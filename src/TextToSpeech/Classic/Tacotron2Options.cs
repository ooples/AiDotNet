namespace AiDotNet.TextToSpeech.Classic;

/// <summary>Options for Tacotron 2 (location-sensitive attention with WaveNet vocoder).</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the Tacotron2 model. Default values follow the original paper settings.</para>
/// </remarks>
public class Tacotron2Options : AcousticModelOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public Tacotron2Options(Tacotron2Options other)
        : base(other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        PrenetDim = other.PrenetDim;
        AttentionRnnDim = other.AttentionRnnDim;
        DecoderRnnDim = other.DecoderRnnDim;
        AttentionDimension = other.AttentionDimension;
        AttentionLocationChannels = other.AttentionLocationChannels;
        StopThreshold = other.StopThreshold;
        EmbeddingDim = other.EmbeddingDim;
        PrenetDropout = other.PrenetDropout;
        ZoneoutProbability = other.ZoneoutProbability;
        ConvolutionDropout = other.ConvolutionDropout;
        AttentionKernelSize = other.AttentionKernelSize;
        SamplingSeed = other.SamplingSeed;
    }

    public Tacotron2Options()
    {
        // Paper training configuration (Shen et al., 2018, S3): Adam at 10^-3, with exponential
        // decay beginning after iteration 50,000 and ending at 10^-5. The paper does not publish the
        // decay rate, so this option pins the fully specified initial phase without inventing a curve.
        LearningRate = 1e-3;

        // Same section: "L2 regularization with weight 10^-6". The generic TTS default is 0.01,
        // four orders of magnitude heavier than the paper's, so pin Tacotron 2's published value.
        WeightDecay = 1e-6;

        EncoderDim = 512;
        DecoderDim = 80;
        HiddenDim = 512;
        NumEncoderLayers = 3;
        NumDecoderLayers = 2;
        NumHeads = 1;
        OutputsPerStep = 1;
        UsePostnet = true;
        PostnetDim = 512;
        PostnetLayers = 5;
    }

    /// <summary>Gets or sets the prenet dimension.</summary>
    public int PrenetDim { get; set; } = 256;

    /// <summary>Gets or sets the attention RNN dimension.</summary>
    public int AttentionRnnDim { get; set; } = 1024;

    /// <summary>Gets or sets the decoder RNN dimension.</summary>
    public int DecoderRnnDim { get; set; } = 1024;

    /// <summary>Gets or sets the attention projection dimension.</summary>
    public int AttentionDimension { get; set; } = 128;

    /// <summary>Gets or sets the attention location feature channels.</summary>
    public int AttentionLocationChannels { get; set; } = 32;

    /// <summary>Gets or sets the inference stop-token probability threshold.</summary>
    public double StopThreshold { get; set; } = 0.5;

    /// <summary>Gets or sets the character embedding width (512, §2.2).</summary>
    public int EmbeddingDim { get; set; } = 512;

    /// <summary>Gets or sets the decoder pre-net's dropout, applied in training and at inference (0.5, §2.2).</summary>
    public double PrenetDropout { get; set; } = 0.5;

    /// <summary>Gets or sets the decoder LSTMs' zoneout probability (0.1, §2.2).</summary>
    public double ZoneoutProbability { get; set; } = 0.1;

    /// <summary>Gets or sets the dropout after every encoder and post-net convolution (0.5, §2.2).</summary>
    public double ConvolutionDropout { get; set; } = 0.5;

    /// <summary>Gets or sets the location convolution's odd filter length (31, §2.2).</summary>
    public int AttentionKernelSize { get; set; } = 31;

    /// <summary>Gets or sets the seed of the inference pre-net dropout, so the same text synthesizes the same output.</summary>
    public int SamplingSeed { get; set; }
}
