namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>The independently trainable stages of Spark-TTS.</summary>
public enum SparkTTSTrainingComponent
{
    /// <summary>Qwen2.5 causal next-token prediction over text, global speaker tokens and semantic tokens.</summary>
    LanguageModel,

    /// <summary>BiCodec reconstruction, semantic VQ and speaker-token objectives.</summary>
    BiCodec,
}

/// <summary>The five ordinal pitch and speed labels in the released Spark tokenizer.</summary>
public enum SparkTTSControlLevel
{
    VeryLow,
    Low,
    Moderate,
    High,
    VeryHigh,
}

/// <summary>Options for the released Spark-TTS 0.5B graph.</summary>
/// <remarks>
/// Defaults are taken from <c>SparkAudio/Spark-TTS-0.5B</c>: Qwen2 has width 896, 24 layers,
/// 14 query heads and 2 key/value heads; BiCodec emits one 8,192-way semantic stream at 50 Hz and
/// represents a speaker with 32 finite-scalar-quantized 4^6-way global tokens.
/// </remarks>
public class SparkTTSOptions : CodecTtsOptions
{
    public SparkTTSOptions()
    {
        SampleRate = 16000;
        MelChannels = 128;
        HopSize = 320;
        NumCodebooks = 1;
        CodebookSize = 8192;
        CodecFrameRate = 50;
        LLMDim = 896;
        NumLLMLayers = 24;
        NumHeads = 14;
        VocabSize = 166000;
        SpeakerEmbeddingDim = 1024;
        MaxCodecFrames = 3000;
        LanguageModelName = "Qwen2.5-0.5B";
        LearningRate = 2e-4;
        WeightDecay = 0.01;
    }

    // Qwen2.5-0.5B (LLM/config.json).
    public int LmIntermediateDim { get; set; } = 4864;
    public int KeyValueHeads { get; set; } = 2;
    public double RmsNormEpsilon { get; set; } = 1e-6;
    public double RopeTheta { get; set; } = 1_000_000;
    public int MaximumPositionEmbeddings { get; set; } = 32768;
    public bool TieWordEmbeddings { get; set; } = true;
    /// <summary>Original Qwen end-of-message token used by the released generator.</summary>
    public int EosToken { get; set; } = 151645;
    /// <summary>Optional speech-range-constrained decoding; false reproduces the released full-vocabulary sampling.</summary>
    public bool ConstrainedSpeechDecoding { get; set; }
    /// <summary>Optional literal character vocabulary for compact custom models; null uses the released Qwen BPE.</summary>
    public string[]? TextSymbols { get; set; }

    // The released tokenizer's exact added-token ranges.
    public int GlobalTokenOffset { get; set; } = 151665;
    public int SemanticTokenOffset { get; set; } = 155761;
    public int TaskTtsToken { get; set; } = 165137;
    public int TaskControllableTtsToken { get; set; } = 165143;
    public int StartContentToken { get; set; } = 165146;
    public int StartStyleLabelToken { get; set; } = 165147;
    public int StartStylePromptToken { get; set; } = 165148;
    public int StartGlobalToken { get; set; } = 165150;
    public int StartSemanticToken { get; set; } = 165151;
    public int EndContentToken { get; set; } = 165152;
    public int EndStyleLabelToken { get; set; } = 165153;
    public int EndStylePromptToken { get; set; } = 165154;
    public int EndGlobalToken { get; set; } = 165156;
    public int EndSemanticToken { get; set; } = 165157;
    public int PitchLabelOffset { get; set; } = 164965;
    public int SpeedLabelOffset { get; set; } = 164986;
    public int GenderTokenOffset { get; set; } = 165032;
    public int PitchValueOffset { get; set; } = 163953;
    public int SpeedValueOffset { get; set; } = 164975;

    // Wav2Vec2-large-xlsr-53. Spark averages transformer hidden states 11, 14 and 16.
    public int[] Wav2VecConvDims { get; set; } = { 512, 512, 512, 512, 512, 512, 512 };
    public int[] Wav2VecConvKernels { get; set; } = { 10, 3, 3, 3, 3, 2, 2 };
    public int[] Wav2VecConvStrides { get; set; } = { 5, 2, 2, 2, 2, 2, 2 };
    public int Wav2VecHiddenDim { get; set; } = 1024;
    public int Wav2VecLayers { get; set; } = 24;
    public int Wav2VecHeads { get; set; } = 16;
    public int Wav2VecIntermediateDim { get; set; } = 4096;
    public int[] Wav2VecMixedLayers { get; set; } = { 11, 14, 16 };

    // BiCodec semantic encoder / decoder (BiCodec/config.yaml).
    public int BiCodecDim { get; set; } = 1024;
    public int VocosDim { get; set; } = 384;
    public int VocosIntermediateDim { get; set; } = 2048;
    public int SemanticEncoderLayers { get; set; } = 12;
    public int PrenetLayers { get; set; } = 12;
    public int PostnetLayers { get; set; } = 6;
    public int SemanticCodeDimension { get; set; } = 8;
    public double SemanticCommitmentWeight { get; set; } = 0.25;
    public double SemanticCodebookWeight { get; set; } = 2.0;

    // Global speaker tokenizer.
    public int GlobalCodebookSize { get; set; } = 4096;
    public int GlobalTokenCount { get; set; } = 32;
    public int SpeakerLatentDim { get; set; } = 128;
    public int SpeakerEcapaChannels { get; set; } = 512;
    public int SpeakerContextDim => SpeakerEcapaChannels * 3;
    public int PerceiverHeads { get; set; } = 8;
    public int PerceiverDepth { get; set; } = 2;
    public int[] FsqLevels { get; set; } = { 4, 4, 4, 4, 4, 4 };

    // DAC-style neural waveform generator. Product of the rates is 320 samples = 50 Hz at 16 kHz.
    public int WaveDecoderChannels { get; set; } = 1536;
    public int[] WaveDecoderRates { get; set; } = { 8, 5, 4, 2 };
    public int[] WaveDecoderKernels { get; set; } = { 16, 11, 8, 4 };

    // Mel front end used only for the global speaker tokenizer.
    public int MelFftSize { get; set; } = 1024;
    public int MelWindowSize { get; set; } = 640;
    public double MelMinimumFrequency { get; set; } = 10;
    public double ReferenceSegmentSeconds { get; set; } = 6;

    // Released generation settings.
    public double Temperature { get; set; } = 0.8;
    public int TopK { get; set; } = 50;
    public double TopP { get; set; } = 0.95;
    public int GenerationSeed { get; set; } = 2025;
    public int MinimumGeneratedCodecFrames { get; set; } = 1;

    public SparkTTSTrainingComponent TrainingComponent { get; set; } = SparkTTSTrainingComponent.LanguageModel;

    /// <summary>DAC-style multi-scale mel analysis windows for codec training.</summary>
    public int[] TrainingMelWindows { get; set; } = [32, 64, 128, 256, 512, 1024, 2048];
    public int[] TrainingMelBins { get; set; } = [5, 10, 20, 40, 80, 160, 320];
    public int[] DiscriminatorWindows { get; set; } = [2048, 1024, 512];
    public double[] DiscriminatorBands { get; set; } = [0, 0.1, 0.25, 0.5, 0.75, 1];
    public int DiscriminatorChannels { get; set; } = 32;
    public int DiscriminatorWidthDivisor { get; set; } = 1;
    public double MelLossWeight { get; set; } = 15;
    public double FeatureMatchingWeight { get; set; } = 2;

    /// <summary>Teacher/global-FSQ warmup duration. The paper specifies a warmup but not its step count.</summary>
    public int GlobalTokenizerWarmupSteps { get; set; } = 10000;
}
