using AiDotNet.Models.Options;

namespace AiDotNet.Audio.Generation;

/// <summary>Options for AudioLM (Borsos et al. 2022, "AudioLM: a Language Modeling Approach to Audio Generation").</summary>
/// <remarks>
/// <para>
/// The defaults are the paper's speech model (§IV-B): semantic tokens are w2v-BERT XL's 7th masked-prediction layer
/// clustered into K = 1024 (25 a second); acoustic tokens are a 16 kHz SoundStream with strides (2, 4, 5, 8) and 12 RVQ
/// levels of 1024 codes (50 frames a second), split into Q′ = 4 coarse and 8 fine levels. Each stage is a decoder-only
/// Transformer with 12 layers, 16 heads, width 1024, feed-forward 4096, dropout 0.1 and T5-style relative positions,
/// trained on random crops of 30, 10 and 3 seconds; sampling uses temperatures 0.6, 0.8 and 0.6.
/// </para>
/// <para>
/// The paper states no optimizer or learning rate (16 TPUv4s, batch 256, 1M steps per stage) and released no code.
/// Those gaps follow lucidrains/audiolm-pytorch: Adam (β = 0.9 / 0.99, ε = 1e-8, no weight decay) at a constant 3e-4
/// with gradients clipped at norm 0.5, the gated-GELU (GEGLU) feed-forward and untied output heads.
/// </para>
/// <para><b>For Beginners:</b> AudioLM continues a recording: given a few seconds of speech or piano, it writes what
/// comes next, keeping the voice and the recording conditions. It has no text input; SPEAR-TTS adds one.</para>
/// </remarks>
public class AudioLMOptions : ModelOptions
{
    /// <summary>Creates the paper's options.</summary>
    public AudioLMOptions()
    {
    }

    /// <summary>Creates a copy of <paramref name="other"/>.</summary>
    public AudioLMOptions(AudioLMOptions other)
    {
        if (other is null) throw new ArgumentNullException(nameof(other));
        SampleRate = other.SampleRate;
        SemanticCodes = other.SemanticCodes;
        SemanticFrameRate = other.SemanticFrameRate;
        NumCodebooks = other.NumCodebooks;
        CoarseCodebooks = other.CoarseCodebooks;
        CodebookSize = other.CodebookSize;
        ModelDim = other.ModelDim;
        NumLayers = other.NumLayers;
        NumHeads = other.NumHeads;
        KeyValueDim = other.KeyValueDim;
        FeedForwardDim = other.FeedForwardDim;
        DropoutRate = other.DropoutRate;
        SemanticCropSeconds = other.SemanticCropSeconds;
        CoarseCropSeconds = other.CoarseCropSeconds;
        FineCropSeconds = other.FineCropSeconds;
        SemanticTemperature = other.SemanticTemperature;
        CoarseTemperature = other.CoarseTemperature;
        FineTemperature = other.FineTemperature;
        LearningRate = other.LearningRate;
        MaxGradientNorm = other.MaxGradientNorm;
        SoundStream = (SoundStreamOptions)AiDotNet.Models.CloneEngine.CopyConfiguration(other.SoundStream);
        Seed = other.Seed;
    }

    /// <summary>Audio sample rate (16 kHz).</summary>
    public int SampleRate { get; set; } = 16000;

    /// <summary>Semantic token vocabulary (K = 1024).</summary>
    public int SemanticCodes { get; set; } = 1024;

    /// <summary>Semantic tokens per second (25).</summary>
    public int SemanticFrameRate { get; set; } = 25;

    /// <summary>Q, SoundStream's RVQ levels (12).</summary>
    public int NumCodebooks { get; set; } = 12;

    /// <summary>Q′, the levels the coarse stage models (4).</summary>
    public int CoarseCodebooks { get; set; } = 4;

    /// <summary>N, codes per level (1024).</summary>
    public int CodebookSize { get; set; } = 1024;

    /// <summary>Transformer width (1024).</summary>
    public int ModelDim { get; set; } = 1024;

    /// <summary>Layers per stage (12).</summary>
    public int NumLayers { get; set; } = 12;

    /// <summary>Attention heads (16).</summary>
    public int NumHeads { get; set; } = 16;

    /// <summary>Width of each head (64: 16 heads over 1024).</summary>
    public int KeyValueDim { get; set; } = 64;

    /// <summary>Feed-forward width (4096).</summary>
    public int FeedForwardDim { get; set; } = 4096;

    /// <summary>Dropout (0.1).</summary>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>Training crop of the semantic stage, in seconds (30).</summary>
    public double SemanticCropSeconds { get; set; } = 30;

    /// <summary>Training crop of the coarse stage, in seconds (10).</summary>
    public double CoarseCropSeconds { get; set; } = 10;

    /// <summary>Training crop and generation chunk of the fine stage, in seconds (3).</summary>
    public double FineCropSeconds { get; set; } = 3;

    /// <summary>Sampling temperature of the semantic stage (0.6).</summary>
    public double SemanticTemperature { get; set; } = 0.6;

    /// <summary>Sampling temperature of the coarse stage (0.8).</summary>
    public double CoarseTemperature { get; set; } = 0.8;

    /// <summary>Sampling temperature of the fine stage (0.6).</summary>
    public double FineTemperature { get; set; } = 0.6;

    /// <summary>Learning rate (3e-4, lucidrains/audiolm-pytorch; constant).</summary>
    public double LearningRate { get; set; } = 3e-4;

    /// <summary>Gradient norm clip (0.5, lucidrains/audiolm-pytorch).</summary>
    public double MaxGradientNorm { get; set; } = 0.5;

    /// <summary>The SoundStream codec: 16 kHz, strides (2, 4, 5, 8), 12 levels of 1024 codes.</summary>
    public SoundStreamOptions SoundStream { get; set; } = new()
    {
        SampleRate = 16000, NumQuantizers = 12, CodebookSize = 1024, TargetBandwidthKbps = 6.0,
    };
}
