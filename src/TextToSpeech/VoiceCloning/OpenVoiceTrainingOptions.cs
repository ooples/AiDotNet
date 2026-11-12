namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>Paper section 2.3 training graph. Loss/optimizer settings not published by
/// OpenVoice use configurable VITS/HiFi-GAN baseline choices, not claimed recovered training settings.</summary>
public sealed class OpenVoiceTrainingOptions
{
    /// <summary>Creates an instance with the default settings.</summary>
    public OpenVoiceTrainingOptions() { }
    /// <summary>Copies the settings from another options instance.</summary>
    public OpenVoiceTrainingOptions(OpenVoiceTrainingOptions other)
    {
        if (other is null) throw new ArgumentNullException(nameof(other));
        IncludeTrainingComponents = other.IncludeTrainingComponents; IpaVocabularySize = other.IpaVocabularySize;
        IpaFilterChannels = other.IpaFilterChannels; IpaHeads = other.IpaHeads; IpaLayers = other.IpaLayers;
        IpaKernel = other.IpaKernel; IpaDropout = other.IpaDropout; SegmentFrames = other.SegmentFrames;
        MelChannels = other.MelChannels; MelMaxFrequency = other.MelMaxFrequency;
        MelLossWeight = other.MelLossWeight; FeatureMatchingWeight = other.FeatureMatchingWeight; KlWeight = other.KlWeight;
        DiscriminatorWidthDivisor = other.DiscriminatorWidthDivisor; LearningRate = other.LearningRate;
        Beta1 = other.Beta1; Beta2 = other.Beta2; Epsilon = other.Epsilon; WeightDecay = other.WeightDecay;
        LearningRateDecay = other.LearningRateDecay; UpdatesPerEpoch = other.UpdatesPerEpoch;
        SamplingSeed = other.SamplingSeed; TrainingSteps = other.TrainingSteps;
    }

    /// <summary>Builds the auxiliary prior/discriminators before first training; training enables and retains this flag automatically.</summary>
    public bool IncludeTrainingComponents { get; set; }
    /// <summary>Size of the caller's universal IPA phoneme-token inventory. Training tokens must use this inventory, not an ARPAbet replacement.</summary>
    public int IpaVocabularySize { get; set; } = 256;
    /// <summary>Gets or sets the IPA encoder filter channels. Defaults to <c>768</c>.</summary>
    public int IpaFilterChannels { get; set; } = 768;
    /// <summary>Gets or sets the number of attention heads in the IPA encoder. Defaults to <c>2</c>.</summary>
    public int IpaHeads { get; set; } = 2;
    /// <summary>Gets or sets the number of layers in the IPA encoder. Defaults to <c>6</c>.</summary>
    public int IpaLayers { get; set; } = 6;
    /// <summary>Gets or sets the IPA encoder kernel. Defaults to <c>3</c>.</summary>
    public int IpaKernel { get; set; } = 3;
    /// <summary>Gets or sets the dropout probability for the IPA encoder. Defaults to <c>.1</c>.</summary>
    public double IpaDropout { get; set; } = .1;
    /// <summary>Gets or sets the segment frames. Defaults to <c>32</c>.</summary>
    public int SegmentFrames { get; set; } = 32;
    /// <summary>Gets or sets the mel channels. Defaults to <c>80</c>.</summary>
    public int MelChannels { get; set; } = 80;
    /// <summary>Gets or sets the mel max frequency. Defaults to <c>8000</c>.</summary>
    public double MelMaxFrequency { get; set; } = 8000;
    /// <summary>Gets or sets the weight of the mel loss. Defaults to <c>45</c>.</summary>
    public double MelLossWeight { get; set; } = 45;
    /// <summary>Gets or sets the weight of the feature matching objective. Defaults to <c>2</c>.</summary>
    public double FeatureMatchingWeight { get; set; } = 2;
    /// <summary>Gets or sets the weight of the kl objective. Defaults to <c>1</c>.</summary>
    public double KlWeight { get; set; } = 1;
    /// <summary>One retains all discriminator channels. Larger values support explicit reduced-graph conformance fixtures.</summary>
    public int DiscriminatorWidthDivisor { get; set; } = 1;
    /// <summary>Gets or sets the optimizer learning rate. Defaults to <c>2e-4</c>.</summary>
    public double LearningRate { get; set; } = 2e-4;
    /// <summary>Gets or sets the first-moment decay for the training optimizer. Defaults to <c>.8</c>.</summary>
    public double Beta1 { get; set; } = .8;
    /// <summary>Gets or sets the second-moment decay for the training optimizer. Defaults to <c>.99</c>.</summary>
    public double Beta2 { get; set; } = .99;
    /// <summary>Gets or sets the numerical stability constant of the optimizer. Defaults to <c>1e-9</c>.</summary>
    public double Epsilon { get; set; } = 1e-9;
    /// <summary>Gets or sets the optimizer weight decay. Defaults to <c>.01</c>.</summary>
    public double WeightDecay { get; set; } = .01;
    /// <summary>Gets or sets the learning rate decay. Defaults to <c>.999875</c>.</summary>
    public double LearningRateDecay { get; set; } = .999875;
    /// <summary>Gets or sets the updates per epoch. Defaults to <c>1000</c>.</summary>
    public int UpdatesPerEpoch { get; set; } = 1000;
    /// <summary>Gets or sets the random seed used during sampling. Defaults to <c>2294</c>.</summary>
    public int SamplingSeed { get; set; } = 2294;
    /// <summary>Completed utterance steps, retained with the options for draw/schedule continuation.</summary>
    public long TrainingSteps { get; set; }
}
