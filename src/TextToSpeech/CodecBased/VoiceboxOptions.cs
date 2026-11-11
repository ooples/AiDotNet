namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Configuration of the English Voicebox audio and duration models.</summary>
/// <remarks>
/// Defaults reproduce Le et al. (2023), Appendix D, Tables D13-D14 and Appendix A: a 24-layer,
/// 1024-wide audio Transformer; an 8-layer, 512-wide regression duration Transformer; normalized
/// 80-bin log-mel features at 100 Hz; midpoint integration with 32 vector-field evaluations; and the
/// 16 kHz adaptation of HiFi-GAN V1. The paper did not publish weights, source code, the phone table,
/// or several ordinary Transformer implementation details; the corresponding choices are explicit
/// options so converted, independently trained checkpoints cannot be mistaken for Meta checkpoints.
/// </remarks>
public class VoiceboxOptions : TtsModelOptions
{
    /// <summary>Creates the paper's English Voicebox configuration.</summary>
    public VoiceboxOptions()
    {
        SampleRate = 16000;
        HopSize = 160;
        FftSize = 1024;
        MelChannels = 80;
        HiddenDim = 1024;
        VocabSize = 512;
        DropoutRate = 0.1;
        LearningRate = 1e-4;
        WeightDecay = 0.0;
    }

    /// <summary>Copies a Voicebox configuration.</summary>
    public VoiceboxOptions(VoiceboxOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        WindowLength = other.WindowLength;
        MelMinFrequency = other.MelMinFrequency;
        MelMaxFrequency = other.MelMaxFrequency;
        MelMean = other.MelMean;
        MelStandardDeviation = other.MelStandardDeviation;
        PhoneVocabularySize = other.PhoneVocabularySize;
        PhoneVocabulary = other.PhoneVocabulary is null ? null : new Dictionary<string, int>(other.PhoneVocabulary, StringComparer.Ordinal);
        UseNativeEnglishFallback = other.UseNativeEnglishFallback;
        PhoneEmbeddingDim = other.PhoneEmbeddingDim;
        AudioModelDim = other.AudioModelDim;
        AudioFeedForwardDim = other.AudioFeedForwardDim;
        AudioLayers = other.AudioLayers;
        AudioHeads = other.AudioHeads;
        AudioAttentionDropout = other.AudioAttentionDropout;
        AudioActivationDropout = other.AudioActivationDropout;
        AudioConvPositionWidth = other.AudioConvPositionWidth;
        AudioConvPositionGroups = other.AudioConvPositionGroups;
        AudioConvPositionDepth = other.AudioConvPositionDepth;
        DurationModelDim = other.DurationModelDim;
        DurationFeedForwardDim = other.DurationFeedForwardDim;
        DurationLayers = other.DurationLayers;
        DurationHeads = other.DurationHeads;
        DurationAttentionDropout = other.DurationAttentionDropout;
        DurationActivationDropout = other.DurationActivationDropout;
        DurationConvPositionWidth = other.DurationConvPositionWidth;
        DurationConvPositionGroups = other.DurationConvPositionGroups;
        DurationConvPositionDepth = other.DurationConvPositionDepth;
        SigmaMinimum = other.SigmaMinimum;
        OdeSteps = other.OdeSteps;
        GuidanceStrength = other.GuidanceStrength;
        FlowTimeScale = other.FlowTimeScale;
        ConditionalDropout = other.ConditionalDropout;
        SpectrogramDropProbability = other.SpectrogramDropProbability;
        SpectrogramMaskMinimum = other.SpectrogramMaskMinimum;
        SpectrogramMaskMaximum = other.SpectrogramMaskMaximum;
        DurationDropProbability = other.DurationDropProbability;
        DurationMaskMinimum = other.DurationMaskMinimum;
        DurationMaskMaximum = other.DurationMaskMaximum;
        MaximumAudioFrames = other.MaximumAudioFrames;
        MaximumPhoneDuration = other.MaximumPhoneDuration;
        WarmupSteps = other.WarmupSteps;
        TrainingUpdates = other.TrainingUpdates;
        GradientClipNorm = other.GradientClipNorm;
        VocoderInitialChannels = other.VocoderInitialChannels;
        VocoderUpsampleRates = (int[])other.VocoderUpsampleRates.Clone();
        VocoderUpsampleKernels = (int[])other.VocoderUpsampleKernels.Clone();
        VocoderResBlockKernels = (int[])other.VocoderResBlockKernels.Clone();
        VocoderResBlockDilations = other.VocoderResBlockDilations.Select(x => (int[])x.Clone()).ToArray();
        SamplingSeed = other.SamplingSeed;
        NumCodebooks = other.NumCodebooks;
        CodebookSize = other.CodebookSize;
    }

    /// <summary>Analysis window in samples (640, or 40 ms).</summary>
    public int WindowLength { get; set; } = 640;
    /// <summary>The dataset/checkpoint's position-dependent MFA phone IDs, including SIL. The paper does not publish its exact inventory.</summary>
    public Dictionary<string, int>? PhoneVocabulary { get; set; }
    /// <summary>Explicitly enables the repository's English ARPAbet frontend for independently trained compatible models; it is not the paper's MFA frontend.</summary>
    public bool UseNativeEnglishFallback { get; set; }
    /// <summary>Lowest mel frequency (0 Hz).</summary>
    public double MelMinFrequency { get; set; }
    /// <summary>Highest mel frequency (8 kHz).</summary>
    public double MelMaxFrequency { get; set; } = 8000;
    /// <summary>Global log-mel mean from Appendix A.3.</summary>
    public double MelMean { get; set; } = -5.8843;
    /// <summary>Global log-mel standard deviation from Appendix A.3.</summary>
    public double MelStandardDeviation { get; set; } = 2.2615;
    /// <summary>Capacity of the converted position-dependent phone table.</summary>
    public int PhoneVocabularySize { get; set; } = 512;
    /// <summary>Phone embedding width H; the paper does not publish H.</summary>
    public int PhoneEmbeddingDim { get; set; } = 256;
    /// <summary>Audio vector-field Transformer width (1024).</summary>
    public int AudioModelDim { get; set; } = 1024;
    /// <summary>Audio Transformer feed-forward width (4096).</summary>
    public int AudioFeedForwardDim { get; set; } = 4096;
    /// <summary>Audio Transformer layers (24).</summary>
    public int AudioLayers { get; set; } = 24;
    /// <summary>Audio Transformer attention heads (16).</summary>
    public int AudioHeads { get; set; } = 16;
    /// <summary>Audio attention dropout (0).</summary>
    public double AudioAttentionDropout { get; set; }
    /// <summary>Audio feed-forward activation dropout (0.1).</summary>
    public double AudioActivationDropout { get; set; } = 0.1;
    /// <summary>Audio convolutional-position kernel width (31).</summary>
    public int AudioConvPositionWidth { get; set; } = 31;
    /// <summary>Audio convolutional-position groups (16).</summary>
    public int AudioConvPositionGroups { get; set; } = 16;
    /// <summary>Audio convolutional-position depth (2).</summary>
    public int AudioConvPositionDepth { get; set; } = 2;
    /// <summary>English regression-duration Transformer width (512).</summary>
    public int DurationModelDim { get; set; } = 512;
    /// <summary>Duration Transformer feed-forward width (2048).</summary>
    public int DurationFeedForwardDim { get; set; } = 2048;
    /// <summary>English duration Transformer layers (8).</summary>
    public int DurationLayers { get; set; } = 8;
    /// <summary>Duration Transformer heads (8).</summary>
    public int DurationHeads { get; set; } = 8;
    /// <summary>Duration attention dropout (0.1).</summary>
    public double DurationAttentionDropout { get; set; } = 0.1;
    /// <summary>Duration feed-forward activation dropout (0.1).</summary>
    public double DurationActivationDropout { get; set; } = 0.1;
    /// <summary>Duration convolutional-position kernel width (15).</summary>
    public int DurationConvPositionWidth { get; set; } = 15;
    /// <summary>Duration convolutional-position groups (16).</summary>
    public int DurationConvPositionGroups { get; set; } = 16;
    /// <summary>Duration convolutional-position depth (2).</summary>
    public int DurationConvPositionDepth { get; set; } = 2;
    /// <summary>Terminal standard deviation of the optimal-transport path (1e-5).</summary>
    public double SigmaMinimum { get; set; } = 1e-5;
    /// <summary>Midpoint integration steps (16, therefore 32 vector-field evaluations).</summary>
    public int OdeSteps { get; set; } = 16;
    /// <summary>English zero-shot classifier-free guidance strength (0.7).</summary>
    public double GuidanceStrength { get; set; } = 0.7;
    /// <summary>Scale applied before sinusoidal encoding of the unit-interval flow time.</summary>
    public double FlowTimeScale { get; set; } = 1000.0;
    /// <summary>Probability of dropping all conditions during audio or duration training (0.2).</summary>
    public double ConditionalDropout { get; set; } = 0.2;
    /// <summary>Probability of masking the complete spectrogram (0.3).</summary>
    public double SpectrogramDropProbability { get; set; } = 0.3;
    /// <summary>Smallest otherwise sampled spectrogram mask fraction (0.7).</summary>
    public double SpectrogramMaskMinimum { get; set; } = 0.7;
    /// <summary>Largest otherwise sampled spectrogram mask fraction (1.0).</summary>
    public double SpectrogramMaskMaximum { get; set; } = 1.0;
    /// <summary>Probability of masking the complete duration sequence (0.2).</summary>
    public double DurationDropProbability { get; set; } = 0.2;
    /// <summary>Smallest otherwise sampled duration mask fraction (0.1).</summary>
    public double DurationMaskMinimum { get; set; } = 0.1;
    /// <summary>Largest otherwise sampled duration mask fraction (1.0).</summary>
    public double DurationMaskMaximum { get; set; } = 1.0;
    /// <summary>Training chunk limit in mel frames (1600).</summary>
    public int MaximumAudioFrames { get; set; } = 1600;
    /// <summary>Safety bound on an inferred phone duration; not specified by the paper.</summary>
    public int MaximumPhoneDuration { get; set; } = 1000;
    /// <summary>Linear learning-rate warmup updates (5000).</summary>
    public int WarmupSteps { get; set; } = 5000;
    /// <summary>English audio-model updates (500,000).</summary>
    public int TrainingUpdates { get; set; } = 500000;
    /// <summary>Audio-model gradient norm clip (0.2).</summary>
    public double GradientClipNorm { get; set; } = 0.2;
    /// <summary>HiFi-GAN V1 initial generator width (512).</summary>
    public int VocoderInitialChannels { get; set; } = 512;
    /// <summary>Voicebox's 160x HiFi-GAN upsampling factors.</summary>
    public int[] VocoderUpsampleRates { get; set; } = [5, 4, 4, 2];
    /// <summary>Voicebox's adjusted HiFi-GAN transposed-convolution kernels.</summary>
    public int[] VocoderUpsampleKernels { get; set; } = [11, 8, 8, 4];
    /// <summary>HiFi-GAN V1 multi-receptive-field kernels.</summary>
    public int[] VocoderResBlockKernels { get; set; } = [3, 7, 11];
    /// <summary>HiFi-GAN V1 type-1 residual dilations.</summary>
    public int[][] VocoderResBlockDilations { get; set; } = [[1, 3, 5], [1, 3, 5], [1, 3, 5]];
    /// <summary>Seed for native sampling and compact reproducibility tests.</summary>
    public int SamplingSeed { get; set; } = 1234;

    // Compatibility aliases retained for callers of the original placeholder options.
    /// <summary>Alias of <see cref="AudioModelDim"/>.</summary>
    public int LLMDim { get => AudioModelDim; set => AudioModelDim = value; }
    /// <summary>Alias of <see cref="AudioLayers"/>.</summary>
    public int NumLLMLayers { get => AudioLayers; set => AudioLayers = value; }
    /// <summary>Alias of <see cref="AudioHeads"/>.</summary>
    public override int NumHeads { get => AudioHeads; set => AudioHeads = value; }
    /// <summary>Alias of <see cref="PhoneEmbeddingDim"/>.</summary>
    public int TextEncoderDim { get => PhoneEmbeddingDim; set => PhoneEmbeddingDim = value; }

    /// <summary>Compatibility-only remnant of the former placeholder; Voicebox has no codec codebooks.</summary>
    public int NumCodebooks { get; set; }
    /// <summary>Compatibility-only remnant of the former placeholder; Voicebox has no codec vocabulary.</summary>
    public int CodebookSize { get; set; }
    /// <summary>Compatibility alias for the continuous mel-frame limit.</summary>
    public int MaxCodecFrames { get => MaximumAudioFrames; set => MaximumAudioFrames = value; }
}
