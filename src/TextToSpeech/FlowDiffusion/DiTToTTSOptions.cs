using AiDotNet.TextToSpeech.EndToEnd;

namespace AiDotNet.TextToSpeech.FlowDiffusion;

/// <summary>The four diffusion-transformer sizes evaluated by DiTTo-TTS.</summary>
public enum DiTToModelSize { Small, Base, Large, ExtraLarge }

/// <summary>Diffusion and length prediction, plus the separately pretrained and optionally semantically finetuned codec.</summary>
public enum DiTToTrainingStage { Diffusion, LengthPredictor, MelVaePretraining, MelVaeSemanticFinetuning }

/// <summary>
/// Options for DiTTo-TTS. Defaults are the English Base system from Lee et al. (2024): frozen
/// <c>microsoft/speecht5_tts</c> text features, CLaM-TTS's 512-wide 10.76 Hz Mel-VAE latent,
/// the 12-layer/768-wide DiT, the separate length predictor, and 25-step classifier-free DDIM.
/// </summary>
public class DiTToTTSOptions : EndToEndTtsOptions
{
    public DiTToTTSOptions()
    {
        SampleRate = 22050; MelChannels = 80; HopSize = 256; FftSize = 1024;
        HiddenDim = DiTHiddenDim = EncoderDim = DecoderDim = 768;
        NumEncoderLayers = TextEncoderLayers = 12; NumDecoderLayers = DiTLayers = 12;
        NumHeads = DiTHeads = 12; VocabSize = TextVocabulary = 81; MaxTextLength = 600;
        MaxMelLength = MaxLatentLength = 2048; DropoutRate = 0.1;
        NumDiffusionSteps = TrainingDiffusionSteps; NumFlowSteps = InferenceSteps; WeightDecay = 0;
    }

    public DiTToTTSOptions(DiTToTTSOptions other) : base(other)
    {
        Validation.Guard.NotNull(other);
        TextHiddenDim = other.TextHiddenDim; TextFeedForwardDim = other.TextFeedForwardDim;
        TextHeads = other.TextHeads; TextEncoderLayers = other.TextEncoderLayers; TextVocabulary = other.TextVocabulary;
        TextPadId = other.TextPadId; TextEosId = other.TextEosId; SentencePieceModelPath = other.SentencePieceModelPath;
        LatentDim = other.LatentDim; MelVaeChannels = other.MelVaeChannels;
        MelVaeMultipliers = (int[])other.MelVaeMultipliers.Clone(); MelVaeDownsampleFactor = other.MelVaeDownsampleFactor;
        RvqDepth = other.RvqDepth; RvqVocabulary = other.RvqVocabulary;
        DiTHiddenDim = other.DiTHiddenDim; DiTLayers = other.DiTLayers; DiTHeads = other.DiTHeads;
        DiTFeedForwardMultiplier = other.DiTFeedForwardMultiplier;
        LengthHiddenDim = other.LengthHiddenDim; LengthLayers = other.LengthLayers; LengthHeads = other.LengthHeads;
        LengthFeedForwardDim = other.LengthFeedForwardDim; MaxLatentLength = other.MaxLatentLength; LengthTopK = other.LengthTopK;
        TrainingDiffusionSteps = other.TrainingDiffusionSteps; InferenceSteps = other.InferenceSteps;
        NoiseScheduleScale = other.NoiseScheduleScale; GuidanceScale = other.GuidanceScale;
        ClassifierFreeDropProbability = other.ClassifierFreeDropProbability;
        FullMaskProbability = other.FullMaskProbability; MinimumMaskRatio = other.MinimumMaskRatio;
        MaximumMaskRatio = other.MaximumMaskRatio; SamplingSeed = other.SamplingSeed;
        BigVganInitialChannels = other.BigVganInitialChannels;
        BigVganUpsampleRates = (int[])other.BigVganUpsampleRates.Clone();
        BigVganUpsampleKernels = (int[])other.BigVganUpsampleKernels.Clone();
        BigVganResidualKernels = (int[])other.BigVganResidualKernels.Clone();
        BigVganResidualDilations = (int[])other.BigVganResidualDilations.Clone();
        TrainingStage = other.TrainingStage;
        CodecTraining = new DiTToCodecTrainingOptions(other.CodecTraining);
    }

    public int TextHiddenDim { get; set; } = 768;
    public int TextFeedForwardDim { get; set; } = 3072;
    public int TextHeads { get; set; } = 12;
    public int TextEncoderLayers { get; set; } = 12;
    public int TextVocabulary { get; set; } = 81;
    public int TextPadId { get; set; } = 1;
    public int TextEosId { get; set; } = 2;
    public string? SentencePieceModelPath { get; set; }

    public int LatentDim { get; set; } = 512;
    public int MelVaeChannels { get; set; } = 256;
    public int[] MelVaeMultipliers { get; set; } = { 1, 1, 2, 2 };
    public int MelVaeDownsampleFactor { get; set; } = 8;
    public int RvqDepth { get; set; } = 32;
    public int RvqVocabulary { get; set; } = 1024;

    public int DiTHiddenDim { get; set; } = 768;
    public int DiTLayers { get; set; } = 12;
    public int DiTHeads { get; set; } = 12;
    public int DiTFeedForwardMultiplier { get; set; } = 4;

    public int LengthHiddenDim { get; set; } = 768;
    public int LengthLayers { get; set; } = 4;
    public int LengthHeads { get; set; } = 12;
    public int LengthFeedForwardDim { get; set; } = 2048;
    /// <summary>The maximum remaining-length class N. Prompt frames are added after length prediction.</summary>
    public int MaxLatentLength { get; set; } = 2048;
    public int LengthTopK { get; set; } = 20;

    public int TrainingDiffusionSteps { get; set; } = 1000;
    public int InferenceSteps { get; set; } = 25;
    public double NoiseScheduleScale { get; set; } = 0.3;
    public double GuidanceScale { get; set; } = 5.0;
    public double ClassifierFreeDropProbability { get; set; } = 0.1;
    public double FullMaskProbability { get; set; } = 0.1;
    public double MinimumMaskRatio { get; set; } = 0.7;
    public double MaximumMaskRatio { get; set; } = 1.0;
    public int SamplingSeed { get; set; } = 2406;
    public DiTToTrainingStage TrainingStage { get; set; } = DiTToTrainingStage.Diffusion;
    public DiTToCodecTrainingOptions CodecTraining { get; set; } = new();

    public int BigVganInitialChannels { get; set; } = 1536;
    public int[] BigVganUpsampleRates { get; set; } = { 4, 4, 2, 2, 2, 2 };
    public int[] BigVganUpsampleKernels { get; set; } = { 8, 8, 4, 4, 4, 4 };
    public int[] BigVganResidualKernels { get; set; } = { 3, 7, 11 };
    public int[] BigVganResidualDilations { get; set; } = { 1, 3, 5 };

    /// <summary>Compatibility name used by the old placeholder.</summary>
    public int FlowDim { get => DiTHiddenDim; set { DiTHiddenDim = value; HiddenDim = value; } }
    /// <summary>Compatibility name used by the old placeholder.</summary>
    public int NumFlowLayers { get => DiTLayers; set { DiTLayers = value; NumDecoderLayers = value; } }

    /// <summary>Applies one of the exact DiT size rows in Table 1.</summary>
    public DiTToTTSOptions UseSize(DiTToModelSize size)
    {
        (DiTLayers, DiTHiddenDim, DiTHeads) = size switch
        {
            DiTToModelSize.Small => (12, 384, 6), DiTToModelSize.Base => (12, 768, 12),
            DiTToModelSize.Large => (24, 1024, 16), DiTToModelSize.ExtraLarge => (28, 1152, 16),
            _ => throw new ArgumentOutOfRangeException(nameof(size))
        };
        HiddenDim = DiTHiddenDim; NumDecoderLayers = DiTLayers; NumHeads = DiTHeads;
        return this;
    }
}
