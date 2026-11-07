namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Options for UniAudio's universal codec and multi-scale Transformer.</summary>
/// <remarks>
/// Defaults follow Yang et al. 2024, Tables 8 and 18: three 1024-way RVQ codes at 50 Hz, a 24-layer global
/// and 8-layer local causal Transformer with 12 heads, top-30 sampling at temperature 0.8, and a three-second
/// speaker prompt. <see cref="OfficialV1"/> selects the different axes in the released checkpoint's config.yaml.
/// </remarks>
public class UniAudioOptions : CodecTtsOptions
{
    /// <summary>Creates the paper configuration. Use <see cref="OfficialV1"/> for the released checkpoint.</summary>
    public UniAudioOptions()
    {
        SampleRate = 16000;
        HopSize = 320;
        MelChannels = 80;
        NumCodebooks = 3;
        CodebookSize = 1024;
        CodecFrameRate = 50;
        LLMDim = 1536;
        NumEncoderLayers = 24;
        NumLLMLayers = 8;
        NumHeads = 12;
        GlobalFeedForwardDim = 6144;
        LocalFeedForwardDim = 6144;
        VocabSize = 4212;
        MaxTextLength = 500;
        MaxCodecFrames = 3000;
        DropoutRate = 0.0;
        LearningRate = 1e-4;
        WeightDecay = 1e-3;
    }

    /// <summary>Initializes an independent copy of another UniAudio configuration.</summary>
    public UniAudioOptions(UniAudioOptions other)
        : base(other)
    {
        Validation.Guard.NotNull(other);
        GlobalFeedForwardDim = other.GlobalFeedForwardDim;
        PrefixLanguageModel = other.PrefixLanguageModel;
        LocalFeedForwardDim = other.LocalFeedForwardDim;
        MaxContextPatches = other.MaxContextPatches;
        TopK = other.TopK;
        Temperature = other.Temperature;
        MaximumLengthRatio = other.MaximumLengthRatio;
        MinimumLengthRatio = other.MinimumLengthRatio;
        MaximumGeneratedFrames = other.MaximumGeneratedFrames;
        MinimumGeneratedFrames = other.MinimumGeneratedFrames;
        PromptSeconds = other.PromptSeconds;
        JointVocabularySize = other.JointVocabularySize;
        PhoneVocabularySize = other.PhoneVocabularySize;
        AudioTokenOffset = other.AudioTokenOffset;
        PhoneTokenOffset = other.PhoneTokenOffset;
        PadToken = other.PadToken;
        ContinuousToken = other.ContinuousToken;
        StartToken = other.StartToken;
        EndToken = other.EndToken;
        PhoneStartToken = other.PhoneStartToken;
        PhoneEndToken = other.PhoneEndToken;
        AudioStartToken = other.AudioStartToken;
        AudioEndToken = other.AudioEndToken;
        AudioPromptStartToken = other.AudioPromptStartToken;
        AudioPromptEndToken = other.AudioPromptEndToken;
        TtsTaskToken = other.TtsTaskToken;
        GlobalStartToken = other.GlobalStartToken;
        LocalStartToken = other.LocalStartToken;
        CodecFilters = other.CodecFilters;
        CodecDimension = other.CodecDimension;
        CodecTotalQuantizers = other.CodecTotalQuantizers;
        CodecRatios = (int[])other.CodecRatios.Clone();
        CodecKMeansIterations = other.CodecKMeansIterations;
        CodecDeadCodeThreshold = other.CodecDeadCodeThreshold;
        CodecCodebookDecay = other.CodecCodebookDecay;
        SamplingSeed = other.SamplingSeed;
        UseReleasedTextFrontend = other.UseReleasedTextFrontend;
        TrainingComponent = other.TrainingComponent;
        IncludeCodecDiscriminators = other.IncludeCodecDiscriminators;
        CodecTraining = new UniAudioCodecTrainingOptions(other.CodecTraining);
    }

    /// <summary>Gets or sets the global Transformer's feed-forward width (6,144).</summary>
    public int GlobalFeedForwardDim { get; set; } = 6144;

    /// <summary>Allows bidirectional condition attention when enabled. The paper and released v1 checkpoint are causal.</summary>
    public bool PrefixLanguageModel { get; set; }

    /// <summary>Uses the released Kaldi lexicon and alignment-phone inventory for raw text.</summary>
    public bool UseReleasedTextFrontend { get; set; } = true;

    /// <summary>Trains the language model or independently pretrains the universal codec.</summary>
    public UniAudioTrainingComponent TrainingComponent { get; set; }

    /// <summary>Retains the codec critics when reconstructing a model after switching back to LM training.</summary>
    public bool IncludeCodecDiscriminators { get; set; }

    /// <summary>The universal codec's reconstruction, discriminator and optimizer recipe.</summary>
    public UniAudioCodecTrainingOptions CodecTraining { get; set; } = new();

    /// <summary>The original ep5-iter340000 checkpoint and universal codec configuration.</summary>
    /// <remarks>The archive's config differs from Table 8 and the README announcement. Its trained embedding
    /// axes require 4,215 tokens and 8,000 positions, with 16 heads; the codec has eight 256-dimensional codebooks.</remarks>
    public static UniAudioOptions OfficialV1() => new()
    {
        NumHeads = 16,
        JointVocabularySize = 4215,
        VocabSize = 4215,
        MaxContextPatches = 8000,
        PhoneVocabularySize = 501,
        CodecDimension = 256,
        CodecTotalQuantizers = 8,
        PrefixLanguageModel = false,
        UseReleasedTextFrontend = true,
        LearningRate = 0.0005,
        MinimumGeneratedFrames = 0,
        CodecTraining = new UniAudioCodecTrainingOptions { Bandwidths = [0.5, 1, 1.5, 2, 4] },
    };

    /// <summary>Gets or sets the local Transformer's feed-forward width (6,144).</summary>
    public int LocalFeedForwardDim { get; set; } = 6144;

    /// <summary>Gets or sets the global context limit in three-token patches (3,000).</summary>
    public int MaxContextPatches { get; set; } = 3000;

    /// <summary>Gets or sets top-k sampling's candidate count (30).</summary>
    public int TopK { get; set; } = 30;

    /// <summary>Gets or sets sampling temperature (0.8).</summary>
    public double Temperature { get; set; } = 0.8;

    /// <summary>Gets or sets the release inference maximum-length ratio (7).</summary>
    public double MaximumLengthRatio { get; set; } = 7.0;

    /// <summary>Gets or sets the release inference minimum-length ratio (0.5).</summary>
    public double MinimumLengthRatio { get; set; } = 0.5;

    /// <summary>Gets or sets a hard cap on newly generated audio frames.</summary>
    public int MaximumGeneratedFrames { get; set; } = 3000;

    /// <summary>Gets or sets the minimum number of newly generated audio frames.</summary>
    public int MinimumGeneratedFrames { get; set; } = 1;

    /// <summary>Gets or sets the speaker-prompt duration (three seconds).</summary>
    public double PromptSeconds { get; set; } = 3.0;

    /// <summary>Gets or sets the paper's joint vocabulary size (4,212, including special tokens).</summary>
    public int JointVocabularySize { get; set; } = 4212;

    /// <summary>Gets or sets the reserved phoneme vocabulary (the release reserves 500 entries).</summary>
    public int PhoneVocabularySize { get; set; } = 500;

    /// <summary>Gets or sets the first joint-vocabulary id assigned to codec tokens.</summary>
    public int AudioTokenOffset { get; set; } = 128;

    /// <summary>Gets or sets the first joint-vocabulary id assigned to phonemes.</summary>
    public int PhoneTokenOffset { get; set; } = 3201;

    /// <summary>Gets or sets the released padding token id.</summary>
    public int PadToken { get; set; } = 0;
    /// <summary>Gets or sets the released continuous-placeholder token id.</summary>
    public int ContinuousToken { get; set; } = 1;
    /// <summary>Gets or sets the released sequence-start token id.</summary>
    public int StartToken { get; set; } = 2;
    /// <summary>Gets or sets the released sequence-end token id.</summary>
    public int EndToken { get; set; } = 3;
    /// <summary>Gets or sets the released phoneme-start token id.</summary>
    public int PhoneStartToken { get; set; } = 4;
    /// <summary>Gets or sets the released phoneme-end token id.</summary>
    public int PhoneEndToken { get; set; } = 5;
    /// <summary>Gets or sets the released audio-start token id.</summary>
    public int AudioStartToken { get; set; } = 8;
    /// <summary>Gets or sets the released audio-end token id.</summary>
    public int AudioEndToken { get; set; } = 9;
    /// <summary>Gets or sets the released audio-prompt-start token id.</summary>
    public int AudioPromptStartToken { get; set; } = 12;
    /// <summary>Gets or sets the released audio-prompt-end token id.</summary>
    public int AudioPromptEndToken { get; set; } = 13;
    /// <summary>Gets or sets the released TTS-task token id.</summary>
    public int TtsTaskToken { get; set; } = 18;
    /// <summary>Gets or sets the released global-transformer start token id.</summary>
    public int GlobalStartToken { get; set; } = 23;
    /// <summary>Gets or sets the released local-transformer start token id.</summary>
    public int LocalStartToken { get; set; } = 24;

    /// <summary>Gets or sets the universal codec's base SEANet width (32).</summary>
    public int CodecFilters { get; set; } = 32;

    /// <summary>Gets or sets the universal codec latent width (128).</summary>
    public int CodecDimension { get; set; } = 128;

    /// <summary>Gets or sets the codec's trained RVQ stages (12 at 6 kbps; inference selects the first three).</summary>
    public int CodecTotalQuantizers { get; set; } = 12;

    /// <summary>Gets or sets the decoder-order codec strides (8, 5, 4, 2).</summary>
    public int[] CodecRatios { get; set; } = [8, 5, 4, 2];

    /// <summary>Gets or sets first-batch k-means iterations for a newly trained codec (50).</summary>
    public int CodecKMeansIterations { get; set; } = 50;

    /// <summary>Gets or sets the EMA dead-code threshold (2).</summary>
    public int CodecDeadCodeThreshold { get; set; } = 2;

    /// <summary>Gets or sets the codec codebook EMA decay (0.99).</summary>
    public double CodecCodebookDecay { get; set; } = 0.99;

    /// <summary>Gets or sets deterministic sampling's seed.</summary>
    public int SamplingSeed { get; set; } = 888;
}
