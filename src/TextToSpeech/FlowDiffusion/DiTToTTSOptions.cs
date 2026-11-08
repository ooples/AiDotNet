using AiDotNet.TextToSpeech.EndToEnd;

namespace AiDotNet.TextToSpeech.FlowDiffusion;

/// <summary>The four diffusion-transformer sizes evaluated by DiTTo-TTS.</summary>
public enum DiTToModelSize
{
    /// <summary>The Small diffusion-transformer configuration.</summary>
    Small,
    /// <summary>The Base diffusion-transformer configuration.</summary>
    Base,
    /// <summary>The Large diffusion-transformer configuration.</summary>
    Large,
    /// <summary>The Extra Large diffusion-transformer configuration.</summary>
    ExtraLarge,
}

/// <summary>Diffusion and length prediction, plus the separately pretrained and optionally semantically finetuned codec.</summary>
public enum DiTToTrainingStage
{
    /// <summary>Train the diffusion transformer.</summary>
    Diffusion,
    /// <summary>Train the latent-length predictor.</summary>
    LengthPredictor,
    /// <summary>Pretrain the mel VAE codec.</summary>
    MelVaePretraining,
    /// <summary>Finetune the mel VAE with semantic supervision.</summary>
    MelVaeSemanticFinetuning,
}

/// <summary>
/// Options for DiTTo-TTS. Defaults are the English Base system from Lee et al. (2024): frozen
/// <c>microsoft/speecht5_tts</c> text features, CLaM-TTS's 512-wide 10.76 Hz Mel-VAE latent,
/// the 12-layer/768-wide DiT, the separate length predictor, and 25-step classifier-free DDIM.
/// </summary>
public class DiTToTTSOptions : EndToEndTtsOptions
{
    /// <summary>Creates an instance with the default settings.</summary>
    public DiTToTTSOptions()
    {
        SampleRate = 22050; MelChannels = 80; HopSize = 256; FftSize = 1024;
        HiddenDim = DiTHiddenDim = EncoderDim = DecoderDim = 768;
        NumEncoderLayers = TextEncoderLayers = 12; NumDecoderLayers = DiTLayers = 12;
        NumHeads = DiTHeads = 12; VocabSize = TextVocabulary = 81; MaxTextLength = 600;
        MaxMelLength = MaxLatentLength = 2048; DropoutRate = 0.1;
        NumDiffusionSteps = TrainingDiffusionSteps; NumFlowSteps = InferenceSteps; WeightDecay = 0;
    }

    /// <summary>Copies the settings from another options instance.</summary>
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

    /// <summary>Gets or sets the feature width of the text hidden. Defaults to <c>768</c>.</summary>
    public int TextHiddenDim { get; set; } = 768;
    /// <summary>Gets or sets the feature width of the text feed forward. Defaults to <c>3072</c>.</summary>
    public int TextFeedForwardDim { get; set; } = 3072;
    /// <summary>Gets or sets the number of attention heads in the text. Defaults to <c>12</c>.</summary>
    public int TextHeads { get; set; } = 12;
    /// <summary>Gets or sets the number of layers in the text encoder. Defaults to <c>12</c>.</summary>
    public int TextEncoderLayers { get => NumEncoderLayers; set => NumEncoderLayers = value; }
    /// <summary>Gets or sets the text vocabulary size. Defaults to <c>81</c>.</summary>
    public int TextVocabulary { get; set; } = 81;
    /// <summary>Gets or sets the token ID for text pad. Defaults to <c>1</c>.</summary>
    public int TextPadId { get; set; } = 1;
    /// <summary>Gets or sets the token ID for text eos. Defaults to <c>2</c>.</summary>
    public int TextEosId { get; set; } = 2;
    /// <summary>Gets or sets the path to the SentencePiece tokenizer model.</summary>
    public string? SentencePieceModelPath { get; set; }

    /// <summary>Gets or sets the feature width of the latent. Defaults to <c>512</c>.</summary>
    public int LatentDim { get; set; } = 512;
    /// <summary>Gets or sets the mel VAE channels. Defaults to <c>256</c>.</summary>
    public int MelVaeChannels { get; set; } = 256;
    /// <summary>Gets or sets the mel VAE multipliers.</summary>
    public int[] MelVaeMultipliers { get; set; } = { 1, 1, 2, 2 };
    /// <summary>Gets or sets the mel VAE downsample factor. Defaults to <c>8</c>.</summary>
    public int MelVaeDownsampleFactor { get; set; } = 8;
    /// <summary>Gets or sets the residual vector quantizer depth. Defaults to <c>32</c>.</summary>
    public int RvqDepth { get; set; } = 32;
    /// <summary>Gets or sets the residual vector quantizer vocabulary. Defaults to <c>1024</c>.</summary>
    public int RvqVocabulary { get; set; } = 1024;

    /// <summary>Gets or sets the feature width of the diffusion transformer hidden. Defaults to <c>768</c>.</summary>
    public int DiTHiddenDim { get => HiddenDim; set => HiddenDim = value; }
    /// <summary>Gets or sets the number of layers in the diffusion transformer. Defaults to <c>12</c>.</summary>
    public int DiTLayers { get => NumDecoderLayers; set => NumDecoderLayers = value; }
    /// <summary>Gets or sets the number of attention heads in the diffusion transformer. Defaults to <c>12</c>.</summary>
    public int DiTHeads { get => NumHeads; set => NumHeads = value; }
    /// <summary>Gets or sets the diffusion transformer feed forward multiplier. Defaults to <c>4</c>.</summary>
    public int DiTFeedForwardMultiplier { get; set; } = 4;

    /// <summary>Gets or sets the feature width of the length hidden. Defaults to <c>768</c>.</summary>
    public int LengthHiddenDim { get; set; } = 768;
    /// <summary>Gets or sets the number of layers in the length. Defaults to <c>4</c>.</summary>
    public int LengthLayers { get; set; } = 4;
    /// <summary>Gets or sets the number of attention heads in the length. Defaults to <c>12</c>.</summary>
    public int LengthHeads { get; set; } = 12;
    /// <summary>Gets or sets the feature width of the length feed forward. Defaults to <c>2048</c>.</summary>
    public int LengthFeedForwardDim { get; set; } = 2048;
    /// <summary>The maximum remaining-length class N. Prompt frames are added after length prediction.</summary>
    public int MaxLatentLength { get; set; } = 2048;
    /// <summary>Gets or sets the length top k. Defaults to <c>20</c>.</summary>
    public int LengthTopK { get; set; } = 20;

    /// <summary>Gets or sets the training diffusion steps. Defaults to <c>1000</c>.</summary>
    public int TrainingDiffusionSteps { get; set; } = 1000;
    /// <summary>Gets or sets the inference steps. Defaults to <c>25</c>.</summary>
    public int InferenceSteps { get; set; } = 25;
    /// <summary>Gets or sets the noise schedule scale. Defaults to <c>0.3</c>.</summary>
    public double NoiseScheduleScale { get; set; } = 0.3;
    /// <summary>Gets or sets the guidance scale. Defaults to <c>5.0</c>.</summary>
    public double GuidanceScale { get; set; } = 5.0;
    /// <summary>Gets or sets the classifier free drop probability. Defaults to <c>0.1</c>.</summary>
    public double ClassifierFreeDropProbability { get; set; } = 0.1;
    /// <summary>Gets or sets the full mask probability. Defaults to <c>0.1</c>.</summary>
    public double FullMaskProbability { get; set; } = 0.1;
    /// <summary>Gets or sets the minimum mask ratio. Defaults to <c>0.7</c>.</summary>
    public double MinimumMaskRatio { get; set; } = 0.7;
    /// <summary>Gets or sets the maximum mask ratio. Defaults to <c>1.0</c>.</summary>
    public double MaximumMaskRatio { get; set; } = 1.0;
    /// <summary>Gets or sets the random seed used during sampling. Defaults to <c>2406</c>.</summary>
    public int SamplingSeed { get; set; } = 2406;
    /// <summary>Gets or sets the stage to optimize during training. Defaults to <c>DiTToTrainingStage.Diffusion</c>.</summary>
    public DiTToTrainingStage TrainingStage { get; set; } = DiTToTrainingStage.Diffusion;
    /// <summary>Gets or sets the codec training settings.</summary>
    public DiTToCodecTrainingOptions CodecTraining { get; set; } = new();

    /// <summary>Gets or sets the BigVGAN initial channels. Defaults to <c>1536</c>.</summary>
    public int BigVganInitialChannels { get; set; } = 1536;
    /// <summary>Gets or sets the BigVGAN upsample rates.</summary>
    public int[] BigVganUpsampleRates { get; set; } = { 4, 4, 2, 2, 2, 2 };
    /// <summary>Gets or sets the convolution kernel sizes for the BigVGAN upsample.</summary>
    public int[] BigVganUpsampleKernels { get; set; } = { 8, 8, 4, 4, 4, 4 };
    /// <summary>Gets or sets the convolution kernel sizes for the BigVGAN residual.</summary>
    public int[] BigVganResidualKernels { get; set; } = { 3, 7, 11 };
    /// <summary>Gets or sets the convolution dilations for the BigVGAN residual.</summary>
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
