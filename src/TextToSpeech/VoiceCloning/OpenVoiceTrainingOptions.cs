namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>Paper section 2.3 training graph. Loss/optimizer settings not published by
/// OpenVoice use configurable VITS/HiFi-GAN baseline choices, not claimed recovered training settings.</summary>
public sealed class OpenVoiceTrainingOptions
{
    public OpenVoiceTrainingOptions() { }
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
    public int IpaFilterChannels { get; set; } = 768;
    public int IpaHeads { get; set; } = 2;
    public int IpaLayers { get; set; } = 6;
    public int IpaKernel { get; set; } = 3;
    public double IpaDropout { get; set; } = .1;
    public int SegmentFrames { get; set; } = 32;
    public int MelChannels { get; set; } = 80;
    public double MelMaxFrequency { get; set; } = 8000;
    public double MelLossWeight { get; set; } = 45;
    public double FeatureMatchingWeight { get; set; } = 2;
    public double KlWeight { get; set; } = 1;
    /// <summary>One retains all discriminator channels. Larger values support explicit reduced-graph conformance fixtures.</summary>
    public int DiscriminatorWidthDivisor { get; set; } = 1;
    public double LearningRate { get; set; } = 2e-4;
    public double Beta1 { get; set; } = .8;
    public double Beta2 { get; set; } = .99;
    public double Epsilon { get; set; } = 1e-9;
    public double WeightDecay { get; set; } = .01;
    public double LearningRateDecay { get; set; } = .999875;
    public int UpdatesPerEpoch { get; set; } = 1000;
    public int SamplingSeed { get; set; } = 2294;
    /// <summary>Completed utterance steps, retained with the options for draw/schedule continuation.</summary>
    public long TrainingSteps { get; set; }
}
