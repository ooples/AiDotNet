namespace AiDotNet.TextToSpeech.EndToEnd;

/// <summary>Options for VITS2 (Kong et al. 2023): VITS with an adversarially trained stochastic duration predictor,
/// noise-scaled monotonic alignment search, a Transformer block in the normalizing flows and a speaker-conditioned text
/// encoder.</summary>
/// <remarks>
/// <para>
/// The shared settings are in <see cref="VitsModelOptions"/> with VITS2's changes (§3): no blank token, the 80-band mel
/// spectrogram as the posterior encoder's input, 256 training instances per step (49 updates per LJSpeech epoch of 12,500
/// clips), the waveform networks trained for 800k steps and the duration predictor for 30k steps after them ("separately
/// trained as the last training step", §2.1).
/// </para>
/// <para>
/// What the paper leaves open follows its community reference implementation (p0p4k/vits2_pytorch,
/// <c>configs/vits2_ljs_nosdp.json</c>): the duration predictor's layers (<c>DurationPredictor(hidden, 256, 3, 0.5)</c>),
/// the time-step-wise discriminator (<c>DurationDiscriminatorV2(hidden, hidden, 3)</c>), the alignment noise schedule
/// (the paper's 0.01 falling by 2e-6 per step), the flow's Transformer (<c>pre_conv</c>: 2 layers, 2 heads, kernel 3,
/// dropout 0.1 over the coupling's half channels) and the speaker conditioning at the third encoder block.
/// </para>
/// <para><b>For Beginners:</b> These options configure the VITS2 model. Default values follow the original paper settings.</para>
/// </remarks>
public class VITS2Options : VitsModelOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public VITS2Options(VITS2Options other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        DurationPredictorFilterChannels = other.DurationPredictorFilterChannels;
        DurationDiscriminatorKernelSize = other.DurationDiscriminatorKernelSize;
        DurationNoiseScale = other.DurationNoiseScale;
        AlignmentNoiseScale = other.AlignmentNoiseScale;
        AlignmentNoiseDecay = other.AlignmentNoiseDecay;
        FlowTransformerLayers = other.FlowTransformerLayers;
        FlowTransformerHeads = other.FlowTransformerHeads;
        FlowTransformerKernelSize = other.FlowTransformerKernelSize;
        FlowTransformerDropout = other.FlowTransformerDropout;
        SpeakerConditionedEncoderBlock = other.SpeakerConditionedEncoderBlock;
        AcousticTrainingSteps = other.AcousticTrainingSteps;
    }

    /// <summary>Creates the paper's configuration.</summary>
    public VITS2Options()
    {
        AddBlank = false;
        UpdatesPerEpoch = 49;
    }

    /// <summary>Gets or sets the duration predictor's convolution width (256).</summary>
    public int DurationPredictorFilterChannels { get; set; } = 256;

    /// <summary>Gets or sets the duration discriminator's kernel (3; its width is the text encoder's).</summary>
    public int DurationDiscriminatorKernelSize { get; set; } = 3;

    /// <summary>Gets or sets the scale of the duration predictor's noise z_d at synthesis (1: the training distribution).</summary>
    public double DurationNoiseScale { get; set; } = 1.0;

    /// <summary>Gets or sets the alignment noise scale at the first step (0.01, §2.2).</summary>
    public double AlignmentNoiseScale { get; set; } = 0.01;

    /// <summary>Gets or sets how much the alignment noise scale falls per step (2e-6, §2.2); it stops at zero.</summary>
    public double AlignmentNoiseDecay { get; set; } = 2e-6;

    /// <summary>Gets or sets the Transformer layers in each flow coupling (2; 0 removes the block).</summary>
    public int FlowTransformerLayers { get; set; } = 2;

    /// <summary>Gets or sets the flow Transformer's heads (2).</summary>
    public int FlowTransformerHeads { get; set; } = 2;

    /// <summary>Gets or sets the flow Transformer's feed-forward kernel (3).</summary>
    public int FlowTransformerKernelSize { get; set; } = 3;

    /// <summary>Gets or sets the flow Transformer's dropout (0.1).</summary>
    public double FlowTransformerDropout { get; set; } = 0.1;

    /// <summary>Gets or sets the text encoder block before which a multi-speaker model adds the speaker embedding (2: the
    /// third block, §2.4).</summary>
    public int SpeakerConditionedEncoderBlock { get; set; } = 2;

    /// <summary>Gets or sets the steps that train the waveform networks before the duration predictor trains on its own
    /// (800,000; the duration predictor then trains for the paper's 30,000 steps).</summary>
    public long AcousticTrainingSteps { get; set; } = 800_000;
}
