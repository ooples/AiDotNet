namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for PANNs (Pretrained Audio Neural Networks) models
/// (Kong et al. 2020).
/// </summary>
/// <remarks>
/// <para>
/// Defaults follow the published CNN14 recipe (Kong et al. 2020 §3): 64 mel
/// bands, 32 kHz sample rate, 1024-sample STFT window, 320-sample hop, four
/// CNN stages (64→128→256→512 channels) + global pool + embedding head +
/// 527-class AudioSet linear classifier.
/// </para>
/// </remarks>
public class PANNsModelOptions : AudioNeuralNetworkOptions
{
    /// <summary>Audio sample rate (Hz). PANNs CNN14 trained at 32 kHz.</summary>
    public int SampleRate { get; init; } = 32_000;

    /// <summary>STFT window size in samples. Paper §3: 1024.</summary>
    public int StftWindowSize { get; init; } = 1024;

    /// <summary>STFT hop length in samples. Paper §3: 320.</summary>
    public int HopLength { get; init; } = 320;

    /// <summary>Number of mel filterbank bands. Paper §3: 64.</summary>
    public int NumMelBands { get; init; } = 64;

    /// <summary>Number of output classes. AudioSet: 527.</summary>
    public int NumClasses { get; init; } = 527;

    /// <summary>Embedding dimension produced before the classification head. CNN14: 2048.</summary>
    public int EmbeddingDim { get; init; } = 2048;

    /// <summary>Dropout rate inside the CNN blocks. Paper §3.</summary>
    public double DropoutRate { get; init; } = 0.2;
}
