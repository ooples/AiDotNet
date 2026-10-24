namespace AiDotNet.TextToSpeech.Classic;

/// <summary>Options for Transformer TTS (Li et al. 2019).</summary>
/// <remarks>
/// <para>
/// Defaults are the paper's where it states them (Li et al. 2019, §3–4): 512-wide phoneme embeddings and encoder pre-net
/// (Tacotron 2's three 5-wide convolutions with batch normalization, ReLU and dropout, then a linear projection); a
/// decoder pre-net of two 256-unit fully connected layers with ReLU, then a linear projection; 6 encoder and 6 decoder
/// layers with 8 heads; mel and stop linear projections and Tacotron 2's 5-layer post-net; a positive stop-token weight
/// in 5.0–8.0; 16 kHz audio at 80 frames per second (hop 200). Where the paper defers to the works it builds on, their
/// values are used: the Transformer base model's d_model 512, d_ff 2048 and dropout 0.1 (Vaswani et al. 2017), and
/// Tacotron 2's dropout 0.5 in the pre-nets and post-net, its 5-layer 512-channel post-net (PostnetLayers, PostnetDim) and 1024-point FFT with a 50 ms window
/// (Shen et al. 2018).
/// </para>
/// <para><b>For Beginners:</b> These options configure the TransformerTTS model. Default values follow the original paper settings.</para>
/// </remarks>
public class TransformerTTSOptions : AcousticModelOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public TransformerTTSOptions(TransformerTTSOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        FeedForwardDim = other.FeedForwardDim;
        EncoderPrenetLayers = other.EncoderPrenetLayers;
        EncoderPrenetChannels = other.EncoderPrenetChannels;
        PrenetKernelSize = other.PrenetKernelSize;
        DecoderPrenetSizes = (int[])other.DecoderPrenetSizes.Clone();
        PrenetDropout = other.PrenetDropout;
        PostnetKernelSize = other.PostnetKernelSize;
        PostnetDropout = other.PostnetDropout;
        StopTokenPositiveWeight = other.StopTokenPositiveWeight;
        MaxDecoderSteps = other.MaxDecoderSteps;
    }

    /// <summary>Creates the paper's configuration.</summary>
    public TransformerTTSOptions()
    {
        EncoderDim = 512;
        HiddenDim = 512;
        NumEncoderLayers = 6;
        NumDecoderLayers = 6;
        NumHeads = 8;
        DropoutRate = 0.1;
        SampleRate = 16000;
        HopSize = 200;
        FftSize = 1024;
        MelChannels = 80;
        UsePostnet = true;
    }

    /// <summary>Gets or sets the inner width d_ff of the feed-forward networks (2048, the Transformer base model).</summary>
    public int FeedForwardDim { get; set; } = 2048;

    /// <summary>Gets or sets the encoder pre-net's convolutions (3).</summary>
    public int EncoderPrenetLayers { get; set; } = 3;

    /// <summary>Gets or sets the encoder pre-net's channels (512).</summary>
    public int EncoderPrenetChannels { get; set; } = 512;

    /// <summary>Gets or sets the encoder pre-net's kernel (5).</summary>
    public int PrenetKernelSize { get; set; } = 5;

    /// <summary>Gets or sets the decoder pre-net's fully connected widths (256, 256).</summary>
    public int[] DecoderPrenetSizes { get; set; } = { 256, 256 };

    /// <summary>Gets or sets the pre-nets' dropout (0.5).</summary>
    public double PrenetDropout { get; set; } = 0.5;

    /// <summary>Gets or sets the post-net's kernel (5).</summary>
    public int PostnetKernelSize { get; set; } = 5;

    /// <summary>Gets or sets the post-net's dropout (0.5).</summary>
    public double PostnetDropout { get; set; } = 0.5;

    /// <summary>Gets or sets the positive weight of the final "stop" frame in the stop-token loss (the paper uses 5.0–8.0).</summary>
    public double StopTokenPositiveWeight { get; set; } = 5.0;

    /// <summary>Gets or sets the most decoder steps synthesis runs before the stop token fires.</summary>
    public int MaxDecoderSteps { get; set; } = 1000;
}
