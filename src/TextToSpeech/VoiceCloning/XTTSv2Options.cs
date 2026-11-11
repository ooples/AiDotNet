using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>Configuration of Casanova et al.'s XTTS-v2 and Coqui's released checkpoint.</summary>
/// <remarks>
/// Defaults describe <c>coqui/XTTS-v2</c>: a 30-layer, 1024-wide GPT-2, a 1024-entry single-codebook
/// dVAE, the six-block conditioning encoder and 32-latent Perceiver, H/ASP speaker encoder, and the
/// speaker-conditioned 24 kHz HiFi-GAN decoder. The paper reports 6,681 text symbols; the released
/// <c>vocab.json</c> and checkpoint both have 6,681 embedding rows.
/// </remarks>
public class XTTSv2Options : CodecTtsOptions
{
    /// <summary>The GPT is decoder-only; use ConditioningBlocks for the reference encoder.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("XTTS GPT is decoder-only and requires NumEncoderLayers=0; native contracts reject other values. ConditioningBlocks controls the separate reference encoder.")]
    public new int NumEncoderLayers { get => base.NumEncoderLayers; set => base.NumEncoderLayers = value; }

    /// <summary>The text embedding rows, paired with the loaded tokenizer vocabulary.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("VocabSize must match the loaded tokenizer's rows; native contracts reject unpaired counts and load paired tokenizer/embedding vocabularies.")]
    public new int VocabSize { get => base.VocabSize; set => base.VocabSize = value; }

    /// <summary>The released dVAE's single codebook.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("The native XTTS dVAE supports one codebook; native contracts reject other counts and encode/decode its single-stream tokens.")]
    public new int NumCodebooks { get => base.NumCodebooks; set => base.NumCodebooks = value; }

    /// <summary>The dVAE vocabulary, paired with the audio vocabulary and special IDs.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("CodebookSize must pair with AudioVocabularySize=CodebookSize+2 and the following start/stop IDs; native tests reject unpaired values and encode/decode paired vocabularies.")]
    public new int CodebookSize { get => base.CodebookSize; set => base.CodebookSize = value; }

    /// <summary>The language ids in the released checkpoint, in model-card order.</summary>
    public static readonly string[] ReleasedLanguages =
        ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "hu", "ko", "ja", "hi"];

    /// <summary>Creates the released XTTS-v2 configuration and the paper's full-training optimizer recipe.</summary>
    public XTTSv2Options()
    {
        SampleRate = 24000;
        InputSampleRate = 22050;
        MelChannels = 80;
        HopSize = 256;
        NumCodebooks = 1;
        CodebookSize = 1024;
        CodecFrameRate = 22;
        NumEncoderLayers = 0;
        LLMDim = 1024;
        NumLLMLayers = 30;
        NumHeads = 16;
        SpeakerEmbeddingDim = 512;
        VocabSize = 6681;
        MaxTextLength = 402;
        MaxCodecFrames = 605;
        LearningRate = 5e-5;
        WeightDecay = 1e-2;
        DropoutRate = 0.1;
        LanguageModelName = "GPT-2";
    }

    /// <summary>Creates a deep copy.</summary>
    public XTTSv2Options(XTTSv2Options other) : base(other)
    {
        Validation.Guard.NotNull(other);
        InputSampleRate = other.InputSampleRate;
        PaperTextVocabularySize = other.PaperTextVocabularySize;
        _audioVocabularySize = other._audioVocabularySize;
        StartTextToken = other.StartTextToken;
        StopTextToken = other.StopTextToken;
        _startAudioToken = other._startAudioToken;
        _stopAudioToken = other._stopAudioToken;
        MaxPromptTokens = other.MaxPromptTokens;
        ConditioningBlocks = other.ConditioningBlocks;
        ConditioningHeads = other.ConditioningHeads;
        PerceiverDepth = other.PerceiverDepth;
        PerceiverLatents = other.PerceiverLatents;
        PerceiverHeads = other.PerceiverHeads;
        PerceiverHeadDim = other.PerceiverHeadDim;
        PerceiverFeedForwardMultiple = other.PerceiverFeedForwardMultiple;
        DvaeHiddenDim = other.DvaeHiddenDim;
        DvaeCodebookDim = other.DvaeCodebookDim;
        DvaeLayers = other.DvaeLayers;
        DvaeResidualBlocks = other.DvaeResidualBlocks;
        GptCodeStride = other.GptCodeStride;
        ConditioningFftSize = other.ConditioningFftSize;
        ConditioningWindowSize = other.ConditioningWindowSize;
        ConditioningHopSize = other.ConditioningHopSize;
        DvaeFftSize = other.DvaeFftSize;
        DvaeWindowSize = other.DvaeWindowSize;
        DvaeHopSize = other.DvaeHopSize;
        MelMaxFrequency = other.MelMaxFrequency;
        MaximumReferenceSeconds = other.MaximumReferenceSeconds;
        ConditioningSeconds = other.ConditioningSeconds;
        ConditioningChunkSeconds = other.ConditioningChunkSeconds;
        MinimumConditioningChunkSeconds = other.MinimumConditioningChunkSeconds;
        SpeakerEncoderFilters = (int[])other.SpeakerEncoderFilters.Clone();
        DecoderInitialChannels = other.DecoderInitialChannels;
        DecoderUpsampleRates = (int[])other.DecoderUpsampleRates.Clone();
        DecoderUpsampleKernels = (int[])other.DecoderUpsampleKernels.Clone();
        DecoderResblockKernels = (int[])other.DecoderResblockKernels.Clone();
        DecoderResblockDilations = other.DecoderResblockDilations.Select(x => (int[])x.Clone()).ToArray();
        Temperature = other.Temperature;
        RepetitionPenalty = other.RepetitionPenalty;
        TopK = other.TopK;
        TopP = other.TopP;
        LengthPenalty = other.LengthPenalty;
        Speed = other.Speed;
        Language = other.Language;
        SamplingSeed = other.SamplingSeed;
        AdamBeta1 = other.AdamBeta1;
        AdamBeta2 = other.AdamBeta2;
        AdamEpsilon = other.AdamEpsilon;
        TextLossWeight = other.TextLossWeight;
        AudioLossWeight = other.AudioLossWeight;
        GradientAccumulationSteps = other.GradientAccumulationSteps;
        LearningRateMilestones = (int[])other.LearningRateMilestones.Clone();
        LearningRateGamma = other.LearningRateGamma;
        TokenizerPath = other.TokenizerPath;
        MelStatsPath = other.MelStatsPath;
    }

    /// <summary>The public GPT fine-tuning recipe (the frozen dVAE and decoder are not updated).</summary>
    public static XTTSv2Options OfficialGptFineTuning() => new() { LearningRate = 5e-6 };

    /// <summary>Input/reference and dVAE sample rate (22.05 kHz); generated audio is 24 kHz.</summary>
    public int InputSampleRate { get; set; }

    /// <summary>Embedding rows in the released checkpoint.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("TextVocabularySize must match the loaded tokenizer; native contracts reject unpaired counts and exercise a paired tokenizer/embedding vocabulary.")]
    public int TextVocabularySize { get => VocabSize; set => VocabSize = value; }
    /// <summary>Vocabulary size stated by the paper and present in the pinned official checkpoint.</summary>
    public int PaperTextVocabularySize { get; set; } = 6681;
    /// <summary>The dVAE codes plus start and stop: <see cref="CodebookSize"/> + 2 (1026) unless set.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("AudioVocabularySize must equal CodebookSize+2 with the following special IDs; native tests reject unpaired values and round-trip paired vocabularies.")]
    public int AudioVocabularySize { get => _audioVocabularySize ?? CodebookSize + 2; set => _audioVocabularySize = value; }
    private int? _audioVocabularySize;
    /// <summary>Resolved from <c>[START]</c> in the released tokenizer when it is loaded.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("The loaded tokenizer owns [START]'s ID; native tests load tokenizer assets with exchanged start/stop IDs and verify the resolved IDs.")]
    public int StartTextToken { get; set; } = -1;
    /// <summary>Resolved from <c>[STOP]</c> in the released tokenizer when it is loaded.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("The loaded tokenizer owns [STOP]'s ID; native tests load tokenizer assets with exchanged start/stop IDs and verify the resolved IDs.")]
    public int StopTextToken { get; set; } = -1;
    /// <summary>Gets or sets the token ID for start audio: <see cref="CodebookSize"/> (1024) unless set.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("StartAudioToken must follow the dVAE codes, before StopAudioToken; native tests reject unpaired IDs and encode/decode paired audio vocabularies.")]
    public int StartAudioToken { get => _startAudioToken ?? CodebookSize; set => _startAudioToken = value; }
    private int? _startAudioToken;
    /// <summary>Gets or sets the token ID for stop audio: <see cref="CodebookSize"/> + 1 (1025) unless set.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("StopAudioToken must follow StartAudioToken at CodebookSize+1; native tests reject unpaired IDs and encode/decode paired audio vocabularies.")]
    public int StopAudioToken { get => _stopAudioToken ?? CodebookSize + 1; set => _stopAudioToken = value; }
    private int? _stopAudioToken;
    /// <summary>Gets or sets the max prompt tokens. Defaults to <c>70</c>.</summary>
    public int MaxPromptTokens { get; set; } = 70;

    /// <summary>Gets or sets the conditioning blocks. Defaults to <c>6</c>.</summary>
    public int ConditioningBlocks { get; set; } = 6;
    /// <summary>Gets or sets the number of attention heads in the conditioning. Defaults to <c>16</c>.</summary>
    public int ConditioningHeads { get; set; } = 16;
    /// <summary>Gets or sets the perceiver depth. Defaults to <c>2</c>.</summary>
    public int PerceiverDepth { get; set; } = 2;
    /// <summary>Gets or sets the perceiver latents. Defaults to <c>32</c>.</summary>
    public int PerceiverLatents { get; set; } = 32;
    /// <summary>Gets or sets the number of attention heads in the perceiver. Defaults to <c>8</c>.</summary>
    public int PerceiverHeads { get; set; } = 8;
    /// <summary>Gets or sets the feature width of the perceiver head. Defaults to <c>64</c>.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("The released XTTS Perceiver uses 64-wide heads; native rejection contracts cover other widths and the native sampling contracts exercise the supported graph.")]
    public int PerceiverHeadDim { get; set; } = 64;
    /// <summary>Gets or sets the perceiver feed forward multiple. Defaults to <c>4</c>.</summary>
    public int PerceiverFeedForwardMultiple { get; set; } = 4;

    /// <summary>Gets or sets the feature width of the discrete VAE hidden. Defaults to <c>512</c>.</summary>
    public int DvaeHiddenDim { get; set; } = 512;
    /// <summary>Gets or sets the feature width of the discrete VAE codebook. Defaults to <c>512</c>.</summary>
    public int DvaeCodebookDim { get; set; } = 512;
    /// <summary>Gets or sets the number of layers in the discrete VAE. Defaults to <c>2</c>.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("The released dVAE has two stride-2 layers; native contracts reject other depths and verify the supported fourfold downsampling.")]
    public int DvaeLayers { get; set; } = 2;
    /// <summary>Gets or sets the discrete VAE residual blocks. Defaults to <c>3</c>.</summary>
    public int DvaeResidualBlocks { get; set; } = 3;
    /// <summary>Samples represented by one GPT/dVAE code at 22.05 kHz (21.5332 codes/s).</summary>
    public int GptCodeStride { get; set; } = 1024;
    /// <summary>Gets or sets the acoustic token rate.</summary>
    public double AcousticTokenRate => (double)InputSampleRate / GptCodeStride;

    /// <summary>Perceiver conditioning mel parameters from the released inference path.</summary>
    public int ConditioningFftSize { get; set; } = 2048;
    /// <summary>Gets or sets the conditioning window size. Defaults to <c>1024</c>.</summary>
    public int ConditioningWindowSize { get; set; } = 1024;
    /// <summary>Gets or sets the conditioning hop size. Defaults to <c>256</c>.</summary>
    public int ConditioningHopSize { get; set; } = 256;
    /// <summary>Gets or sets the discrete VAE FFT size. Defaults to <c>1024</c>.</summary>
    public int DvaeFftSize { get; set; } = 1024;
    /// <summary>Gets or sets the discrete VAE window size. Defaults to <c>1024</c>.</summary>
    public int DvaeWindowSize { get; set; } = 1024;
    /// <summary>Gets or sets the discrete VAE hop size. Defaults to <c>256</c>.</summary>
    public int DvaeHopSize { get; set; } = 256;
    /// <summary>Gets or sets the mel max frequency. Defaults to <c>8000</c>.</summary>
    public double MelMaxFrequency { get; set; } = 8000;
    /// <summary>Gets or sets the maximum reference duration, in seconds. Defaults to <c>30</c>.</summary>
    public double MaximumReferenceSeconds { get; set; } = 30;
    /// <summary>Gets or sets the conditioning duration, in seconds. Defaults to <c>6</c>.</summary>
    public double ConditioningSeconds { get; set; } = 6;
    /// <summary>Gets or sets the conditioning chunk duration, in seconds. Defaults to <c>6</c>.</summary>
    public double ConditioningChunkSeconds { get; set; } = 6;
    /// <summary>Gets or sets the minimum conditioning chunk duration, in seconds. Defaults to <c>0.33</c>.</summary>
    public double MinimumConditioningChunkSeconds { get; set; } = 0.33;
    /// <summary>H/ASP SE-ResNet stage widths from the released <c>config_se.json</c>.</summary>
    public int[] SpeakerEncoderFilters { get; set; } = [32, 64, 128, 256];

    /// <summary>Gets or sets the decoder initial channels. Defaults to <c>512</c>.</summary>
    public int DecoderInitialChannels { get; set; } = 512;
    /// <summary>Gets or sets the decoder upsample rates.</summary>
    public int[] DecoderUpsampleRates { get; set; } = [8, 8, 2, 2];
    /// <summary>Gets or sets the convolution kernel sizes for the decoder upsample.</summary>
    public int[] DecoderUpsampleKernels { get; set; } = [16, 16, 4, 4];
    /// <summary>Gets or sets the convolution kernel sizes for the decoder resblock.</summary>
    public int[] DecoderResblockKernels { get; set; } = [3, 7, 11];
    /// <summary>Gets or sets the convolution dilations for the decoder resblock.</summary>
    public int[][] DecoderResblockDilations { get; set; } = [[1, 3, 5], [1, 3, 5], [1, 3, 5]];

    /// <summary>Gets or sets the logit temperature used during sampling. Defaults to <c>0.75</c>.</summary>
    public double Temperature { get; set; } = 0.75;
    /// <summary>Gets or sets the penalty applied to repeated tokens during generation. Defaults to <c>10</c>.</summary>
    public double RepetitionPenalty { get; set; } = 10;
    /// <summary>Gets or sets the maximum number of highest-probability tokens retained during sampling. Defaults to <c>50</c>.</summary>
    public int TopK { get; set; } = 50;
    /// <summary>Gets or sets the cumulative probability threshold for nucleus sampling. Defaults to <c>0.85</c>.</summary>
    public double TopP { get; set; } = 0.85;
    /// <summary>Gets or sets the length penalty. Defaults to <c>1</c>.</summary>
    public double LengthPenalty { get; set; } = 1;
    /// <summary>Gets or sets the speed. Defaults to <c>1</c>.</summary>
    public double Speed { get; set; } = 1;
    /// <summary>Gets or sets the language code used for speech processing. Defaults to <c>"en"</c>.</summary>
    public string Language { get; set; } = "en";
    /// <summary>Gets or sets the random seed used during sampling. Defaults to <c>1</c>.</summary>
    public int SamplingSeed { get; set; } = 1;

    /// <summary>Gets or sets the first-moment decay for the adam optimizer. Defaults to <c>0.9</c>.</summary>
    public double AdamBeta1 { get; set; } = 0.9;
    /// <summary>Gets or sets the second-moment decay for the adam optimizer. Defaults to <c>0.96</c>.</summary>
    public double AdamBeta2 { get; set; } = 0.96;
    /// <summary>Gets or sets the numerical stability constant for adam. Defaults to <c>1e-8</c>.</summary>
    public double AdamEpsilon { get; set; } = 1e-8;
    /// <summary>Gets or sets the weight of the text loss. Defaults to <c>0.01</c>.</summary>
    public double TextLossWeight { get; set; } = 0.01;
    /// <summary>Gets or sets the weight of the audio loss. Defaults to <c>1</c>.</summary>
    public double AudioLossWeight { get; set; } = 1;
    /// <summary>Gets or sets the gradient accumulation steps. Defaults to <c>16</c>.</summary>
    public int GradientAccumulationSteps { get; set; } = 16;
    /// <summary>Gets or sets the learning rate milestones.</summary>
    public int[] LearningRateMilestones { get; set; } = [5000, 150000, 300000];
    /// <summary>Gets or sets the learning rate gamma. Defaults to <c>0.5</c>.</summary>
    public double LearningRateGamma { get; set; } = 0.5;

    /// <summary>Optional Hugging Face/tokenizers tokenizer JSON shipped as <c>vocab.json</c>.</summary>
    public string? TokenizerPath { get; set; }
    /// <summary>Optional released <c>mel_stats.pth</c>.</summary>
    public string? MelStatsPath { get; set; }
}
