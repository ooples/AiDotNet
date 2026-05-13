namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for AST (Audio Spectrogram Transformer) models
/// (Gong et al. 2021).
/// </summary>
/// <remarks>
/// <para>
/// Defaults follow the published AST-Base recipe (Gong et al. 2021 §2):
/// 128-mel input × ~1024 frames, 12 transformer layers × 12 heads × 768
/// embedding dim, 16×16 patches (paper §2.2), trained on AudioSet
/// (527 classes).
/// </para>
/// </remarks>
public class ASTModelOptions : AudioNeuralNetworkOptions
{
    /// <summary>Audio sample rate (Hz). AST-Base trained at 16 kHz.</summary>
    public int SampleRate { get; init; } = 16_000;

    /// <summary>STFT window size in samples. AST default: 25 ms ≈ 400 samples at 16 kHz.</summary>
    public int StftWindowSize { get; init; } = 400;

    /// <summary>STFT hop length in samples. AST default: 10 ms = 160 samples at 16 kHz.</summary>
    public int HopLength { get; init; } = 160;

    /// <summary>Number of mel filterbank bands. Paper §2.1: 128.</summary>
    public int NumMelBands { get; init; } = 128;

    /// <summary>Target spectrogram length in frames (AudioSet clips are ~10 s).</summary>
    public int TargetLength { get; init; } = 1024;

    /// <summary>Patch size H × W for the ViT patch embedding (paper §2.2: 16).</summary>
    public int PatchSize { get; init; } = 16;

    /// <summary>Number of output classes. 527 for AudioSet (AST-Base default).</summary>
    public int NumClasses { get; init; } = 527;

    /// <summary>Embedding / hidden dimension. AST-Base: 768.</summary>
    public int EmbeddingDim { get; init; } = 768;

    /// <summary>Number of transformer encoder layers. AST-Base: 12.</summary>
    public int NumLayers { get; init; } = 12;

    /// <summary>Attention heads per transformer block. AST-Base: 12.</summary>
    public int NumHeads { get; init; } = 12;

    /// <summary>FFN hidden dimension (4× embedding per Vaswani 2017).</summary>
    public int FeedForwardDim { get; init; } = 3072;

    /// <summary>Dropout rate inside the transformer blocks.</summary>
    public double DropoutRate { get; init; } = 0.0;
}
