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
    /// <summary>The lowest pitch or speed label.</summary>
    VeryLow,
    /// <summary>A low pitch or speed label.</summary>
    Low,
    /// <summary>The middle pitch or speed label.</summary>
    Moderate,
    /// <summary>A high pitch or speed label.</summary>
    High,
    /// <summary>The highest pitch or speed label.</summary>
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
    /// <summary>Initializes a new instance with default values.</summary>
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
    /// <summary>Gets or sets the feature width of the language-model intermediate. Defaults to <c>4864</c>.</summary>
    public int LmIntermediateDim { get; set; } = 4864;
    /// <summary>Gets or sets the number of key/value attention heads. Defaults to <c>2</c>.</summary>
    public int KeyValueHeads { get; set; } = 2;
    /// <summary>Gets or sets the numerical stability constant for RMS norm. Defaults to <c>1e-6</c>.</summary>
    public double RmsNormEpsilon { get; set; } = 1e-6;
    /// <summary>Gets or sets the rotary positional-embedding frequency base for the model. Defaults to <c>1_000_000</c>.</summary>
    public double RopeTheta { get; set; } = 1_000_000;
    /// <summary>Gets or sets the maximum number of token positions. Defaults to <c>32768</c>.</summary>
    public int MaximumPositionEmbeddings { get; set; } = 32768;
    /// <summary>Gets or sets whether input and output token embeddings share their weights. Defaults to <c>true</c>.</summary>
    public bool TieWordEmbeddings { get; set; } = true;
    /// <summary>Original Qwen end-of-message token used by the released generator.</summary>
    public int EosToken { get; set; } = 151645;
    /// <summary>Optional speech-range-constrained decoding; false reproduces the released full-vocabulary sampling.</summary>
    public bool ConstrainedSpeechDecoding { get; set; }
    /// <summary>Optional literal character vocabulary for compact custom models; null uses the released Qwen BPE.</summary>
    public string[]? TextSymbols { get; set; }

    // The released tokenizer's exact added-token ranges.
    /// <summary>Gets or sets the global token offset. Defaults to <c>151665</c>.</summary>
    public int GlobalTokenOffset { get; set; } = 151665;
    /// <summary>Gets or sets the semantic token offset. Defaults to <c>155761</c>.</summary>
    public int SemanticTokenOffset { get; set; } = 155761;
    /// <summary>Gets or sets the token ID for task tts. Defaults to <c>165137</c>.</summary>
    public int TaskTtsToken { get; set; } = 165137;
    /// <summary>Gets or sets the token ID for task controllable tts. Defaults to <c>165143</c>.</summary>
    public int TaskControllableTtsToken { get; set; } = 165143;
    /// <summary>Gets or sets the token ID for start content. Defaults to <c>165146</c>.</summary>
    public int StartContentToken { get; set; } = 165146;
    /// <summary>Gets or sets the token ID for start style label. Defaults to <c>165147</c>.</summary>
    public int StartStyleLabelToken { get; set; } = 165147;
    /// <summary>Gets or sets the token ID for start style prompt. Defaults to <c>165148</c>.</summary>
    public int StartStylePromptToken { get; set; } = 165148;
    /// <summary>Gets or sets the token ID for start global. Defaults to <c>165150</c>.</summary>
    public int StartGlobalToken { get; set; } = 165150;
    /// <summary>Gets or sets the token ID for start semantic. Defaults to <c>165151</c>.</summary>
    public int StartSemanticToken { get; set; } = 165151;
    /// <summary>Gets or sets the token ID for end content. Defaults to <c>165152</c>.</summary>
    public int EndContentToken { get; set; } = 165152;
    /// <summary>Gets or sets the token ID for end style label. Defaults to <c>165153</c>.</summary>
    public int EndStyleLabelToken { get; set; } = 165153;
    /// <summary>Gets or sets the token ID for end style prompt. Defaults to <c>165154</c>.</summary>
    public int EndStylePromptToken { get; set; } = 165154;
    /// <summary>Gets or sets the token ID for end global. Defaults to <c>165156</c>.</summary>
    public int EndGlobalToken { get; set; } = 165156;
    /// <summary>Gets or sets the token ID for end semantic. Defaults to <c>165157</c>.</summary>
    public int EndSemanticToken { get; set; } = 165157;
    /// <summary>Gets or sets the pitch label offset. Defaults to <c>164965</c>.</summary>
    public int PitchLabelOffset { get; set; } = 164965;
    /// <summary>Gets or sets the speed label offset. Defaults to <c>164986</c>.</summary>
    public int SpeedLabelOffset { get; set; } = 164986;
    /// <summary>Gets or sets the gender token offset. Defaults to <c>165032</c>.</summary>
    public int GenderTokenOffset { get; set; } = 165032;
    /// <summary>Gets or sets the pitch value offset. Defaults to <c>163953</c>.</summary>
    public int PitchValueOffset { get; set; } = 163953;
    /// <summary>Gets or sets the speed value offset. Defaults to <c>164975</c>.</summary>
    public int SpeedValueOffset { get; set; } = 164975;

    // Wav2Vec2-large-xlsr-53. Spark averages transformer hidden states 11, 14 and 16.
    /// <summary>Gets or sets the Wav2Vec convolution channel widths.</summary>
    public int[] Wav2VecConvDims { get; set; } = { 512, 512, 512, 512, 512, 512, 512 };
    /// <summary>Gets or sets the convolution kernel sizes for the Wav2Vec convolution.</summary>
    public int[] Wav2VecConvKernels { get; set; } = { 10, 3, 3, 3, 3, 2, 2 };
    /// <summary>Gets or sets the convolution strides for the Wav2Vec convolution.</summary>
    public int[] Wav2VecConvStrides { get; set; } = { 5, 2, 2, 2, 2, 2, 2 };
    /// <summary>Gets or sets the feature width of the Wav2Vec hidden. Defaults to <c>1024</c>.</summary>
    public int Wav2VecHiddenDim { get; set; } = 1024;
    /// <summary>Gets or sets the number of layers in the Wav2Vec. Defaults to <c>24</c>.</summary>
    public int Wav2VecLayers { get; set; } = 24;
    /// <summary>Gets or sets the number of attention heads in the Wav2Vec. Defaults to <c>16</c>.</summary>
    public int Wav2VecHeads { get; set; } = 16;
    /// <summary>Gets or sets the feature width of the Wav2Vec intermediate. Defaults to <c>4096</c>.</summary>
    public int Wav2VecIntermediateDim { get; set; } = 4096;
    /// <summary>Gets or sets the number of layers in the Wav2Vec mixed.</summary>
    public int[] Wav2VecMixedLayers { get; set; } = { 11, 14, 16 };

    // BiCodec semantic encoder / decoder (BiCodec/config.yaml).
    /// <summary>Gets or sets the feature width of the BiCodec. Defaults to <c>1024</c>.</summary>
    public int BiCodecDim { get; set; } = 1024;
    /// <summary>Gets or sets the feature width of the vocos. Defaults to <c>384</c>.</summary>
    public int VocosDim { get; set; } = 384;
    /// <summary>Gets or sets the feature width of the vocos intermediate. Defaults to <c>2048</c>.</summary>
    public int VocosIntermediateDim { get; set; } = 2048;
    /// <summary>Gets or sets the number of layers in the semantic encoder. Defaults to <c>12</c>.</summary>
    public int SemanticEncoderLayers { get; set; } = 12;
    /// <summary>Gets or sets the number of layers in the prenet. Defaults to <c>12</c>.</summary>
    public int PrenetLayers { get; set; } = 12;
    /// <summary>Gets or sets the number of layers in the postnet. Defaults to <c>6</c>.</summary>
    public int PostnetLayers { get; set; } = 6;
    /// <summary>Gets or sets the feature width of the semantic code. Defaults to <c>8</c>.</summary>
    public int SemanticCodeDimension { get; set; } = 8;
    /// <summary>Gets or sets the weight of the semantic commitment objective. Defaults to <c>0.25</c>.</summary>
    public double SemanticCommitmentWeight { get; set; } = 0.25;
    /// <summary>Gets or sets the weight of the semantic codebook objective. Defaults to <c>2.0</c>.</summary>
    public double SemanticCodebookWeight { get; set; } = 2.0;

    // Global speaker tokenizer.
    /// <summary>Gets or sets the global codebook size. Defaults to <c>4096</c>.</summary>
    public int GlobalCodebookSize { get; set; } = 4096;
    /// <summary>Gets or sets the global token count. Defaults to <c>32</c>.</summary>
    public int GlobalTokenCount { get; set; } = 32;
    /// <summary>Gets or sets the feature width of the speaker latent. Defaults to <c>128</c>.</summary>
    public int SpeakerLatentDim { get; set; } = 128;
    /// <summary>Gets or sets the speaker ECAPA channels. Defaults to <c>512</c>.</summary>
    public int SpeakerEcapaChannels { get; set; } = 512;
    /// <summary>Gets or sets the feature width of the speaker context.</summary>
    public int SpeakerContextDim => SpeakerEcapaChannels * 3;
    /// <summary>Gets or sets the number of attention heads in the perceiver. Defaults to <c>8</c>.</summary>
    public int PerceiverHeads { get; set; } = 8;
    /// <summary>Gets or sets the perceiver depth. Defaults to <c>2</c>.</summary>
    public int PerceiverDepth { get; set; } = 2;
    /// <summary>Gets or sets the finite scalar quantizer levels.</summary>
    public int[] FsqLevels { get; set; } = { 4, 4, 4, 4, 4, 4 };

    // DAC-style neural waveform generator. Product of the rates is 320 samples = 50 Hz at 16 kHz.
    /// <summary>Gets or sets the wave decoder channels. Defaults to <c>1536</c>.</summary>
    public int WaveDecoderChannels { get; set; } = 1536;
    /// <summary>Gets or sets the wave decoder rates.</summary>
    public int[] WaveDecoderRates { get; set; } = { 8, 5, 4, 2 };
    /// <summary>Gets or sets the convolution kernel sizes for the wave decoder.</summary>
    public int[] WaveDecoderKernels { get; set; } = { 16, 11, 8, 4 };

    // Mel front end used only for the global speaker tokenizer.
    /// <summary>Gets or sets the mel FFT size. Defaults to <c>1024</c>.</summary>
    public int MelFftSize { get; set; } = 1024;
    /// <summary>Gets or sets the mel window size. Defaults to <c>640</c>.</summary>
    public int MelWindowSize { get; set; } = 640;
    /// <summary>Gets or sets the mel minimum frequency. Defaults to <c>10</c>.</summary>
    public double MelMinimumFrequency { get; set; } = 10;
    /// <summary>Gets or sets the reference segment duration, in seconds. Defaults to <c>6</c>.</summary>
    public double ReferenceSegmentSeconds { get; set; } = 6;

    // Released generation settings.
    /// <summary>Gets or sets the logit temperature used during sampling. Defaults to <c>0.8</c>.</summary>
    public double Temperature { get; set; } = 0.8;
    /// <summary>Gets or sets the maximum number of highest-probability tokens retained during sampling. Defaults to <c>50</c>.</summary>
    public int TopK { get; set; } = 50;
    /// <summary>Gets or sets the cumulative probability threshold for nucleus sampling. Defaults to <c>0.95</c>.</summary>
    public double TopP { get; set; } = 0.95;
    /// <summary>Gets or sets the random seed used during generation. Defaults to <c>2025</c>.</summary>
    public int GenerationSeed { get; set; } = 2025;
    /// <summary>Gets or sets the minimum generated codec frames. Defaults to <c>1</c>.</summary>
    public int MinimumGeneratedCodecFrames { get; set; } = 1;

    /// <summary>Gets or sets the component to optimize during training. Defaults to <c>SparkTTSTrainingComponent.LanguageModel</c>.</summary>
    public SparkTTSTrainingComponent TrainingComponent { get; set; } = SparkTTSTrainingComponent.LanguageModel;

    /// <summary>DAC-style multi-scale mel analysis windows for codec training.</summary>
    public int[] TrainingMelWindows { get; set; } = [32, 64, 128, 256, 512, 1024, 2048];
    /// <summary>Gets or sets the training mel bins.</summary>
    public int[] TrainingMelBins { get; set; } = [5, 10, 20, 40, 80, 160, 320];
    /// <summary>Gets or sets the discriminator windows.</summary>
    public int[] DiscriminatorWindows { get; set; } = [2048, 1024, 512];
    /// <summary>Gets or sets the discriminator bands.</summary>
    public double[] DiscriminatorBands { get; set; } = [0, 0.1, 0.25, 0.5, 0.75, 1];
    /// <summary>Gets or sets the discriminator channels. Defaults to <c>32</c>.</summary>
    public int DiscriminatorChannels { get; set; } = 32;
    /// <summary>Gets or sets the discriminator width divisor. Defaults to <c>1</c>.</summary>
    public int DiscriminatorWidthDivisor { get; set; } = 1;
    /// <summary>Gets or sets the weight of the mel loss. Defaults to <c>15</c>.</summary>
    public double MelLossWeight { get; set; } = 15;
    /// <summary>Gets or sets the weight of the feature matching objective. Defaults to <c>2</c>.</summary>
    public double FeatureMatchingWeight { get; set; } = 2;

    /// <summary>Teacher/global-FSQ warmup duration. The paper specifies a warmup but not its step count.</summary>
    public int GlobalTokenizerWarmupSteps { get; set; } = 10000;
}
