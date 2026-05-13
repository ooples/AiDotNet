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
/// <para><b>For Beginners:</b> PANNs is a family of pretrained convolutional
/// networks for audio tagging. CNN14 is the canonical balanced-accuracy
/// variant. The defaults here reproduce the AudioSet-pretrained checkpoint;
/// you usually only override <c>NumClasses</c> when fine-tuning on a
/// different label set.</para>
/// </remarks>
public class PANNsModelOptions : AudioNeuralNetworkOptions
{
    /// <summary>Initializes a new instance with PANNs CNN14 defaults.</summary>
    public PANNsModelOptions() { }

    /// <summary>
    /// Initializes a new instance by copying every property from
    /// <paramref name="other"/>. Throws when <paramref name="other"/> is null.
    /// </summary>
    /// <param name="other">Source options to copy. Must not be null.</param>
    public PANNsModelOptions(PANNsModelOptions other)
    {
        if (other is null) throw new ArgumentNullException(nameof(other));
        SampleRate = other.SampleRate;
        StftWindowSize = other.StftWindowSize;
        HopLength = other.HopLength;
        NumMelBands = other.NumMelBands;
        NumClasses = other.NumClasses;
        EmbeddingDim = other.EmbeddingDim;
        DropoutRate = other.DropoutRate;
    }

    /// <summary>Audio sample rate in Hz used by the STFT frontend.</summary>
    /// <value>Default 32000 — the PANNs CNN14 training rate (Kong et al. 2020).</value>
    /// <remarks><para><b>For Beginners:</b> How many audio samples per
    /// second. PANNs was trained at 32 kHz; downsampling to 16 kHz will lose
    /// the high-frequency band the CNN14 filters expect.</para></remarks>
    public int SampleRate { get; init; } = 32_000;

    /// <summary>STFT window size in samples (analysis frame length).</summary>
    /// <value>Default 1024 (paper §3, ≈ 32 ms at 32 kHz).</value>
    /// <remarks><para><b>For Beginners:</b> The STFT slides this many
    /// samples and runs an FFT. Larger windows give sharper frequency
    /// resolution but smear out time. 1024 is the CNN14 default.</para></remarks>
    public int StftWindowSize { get; init; } = 1024;

    /// <summary>STFT hop length in samples between successive frames.</summary>
    /// <value>Default 320 (paper §3, ≈ 10 ms at 32 kHz).</value>
    /// <remarks><para><b>For Beginners:</b> How far the analysis window
    /// shifts between frames. 10 ms is the canonical default.</para></remarks>
    public int HopLength { get; init; } = 320;

    /// <summary>Number of mel filterbank bands per spectrogram frame.</summary>
    /// <value>Default 64 (paper §3).</value>
    /// <remarks><para><b>For Beginners:</b> Mel-spaced frequency bins
    /// (perceptually-uniform). PANNs CNN14 was trained with 64 bands; the
    /// pretrained weights expect this exact value.</para></remarks>
    public int NumMelBands { get; init; } = 64;

    /// <summary>Number of output classes for the classification head.</summary>
    /// <value>Default 527 (the AudioSet ontology PANNs trained on).</value>
    /// <remarks><para><b>For Beginners:</b> Set this to your label count
    /// when fine-tuning on a non-AudioSet dataset.</para></remarks>
    public int NumClasses { get; init; } = 527;

    /// <summary>Embedding dimension produced before the classification head.</summary>
    /// <value>Default 2048 (CNN14, after the global pooling step).</value>
    /// <remarks><para><b>For Beginners:</b> Size of the pooled audio
    /// embedding emitted by the CNN trunk. CNN14 produces 2048; other
    /// PANNs variants emit 512 or 1024.</para></remarks>
    public int EmbeddingDim { get; init; } = 2048;

    /// <summary>Dropout rate inside the CNN blocks.</summary>
    /// <value>Default 0.2 (paper §3).</value>
    /// <remarks><para><b>For Beginners:</b> Dropout randomly zeroes a
    /// fraction of activations during training to discourage overfitting.
    /// 0.2 is the PANNs CNN14 default; tune up to 0.5 if you see severe
    /// overfitting on a small dataset.</para></remarks>
    public double DropoutRate { get; init; } = 0.2;
}
