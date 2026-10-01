using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.TextToSpeech;

/// <summary>A batch of token sequences paired with mel spectrograms and their true lengths.</summary>
/// <remarks>
/// Tokens have shape [batch, padded tokens], and mels [batch, padded frames, mel channels].
/// Lengths and optional duration metadata are copied; the caller retains ownership of tensors.
/// Padding is not part of the objective. Explicit durations must assign at least one frame
/// to each active token and sum to that example's mel length.
/// </remarks>
public sealed class AlignedMelBatch<T>
{
    /// <summary>Integer token identifiers. Only positions below each token length are consumed.</summary>
    public Tensor<int> Tokens { get; }
    /// <summary>Target mel spectrograms, with the mel channel last.</summary>
    public Tensor<T> TargetMels { get; }
    /// <summary>True token lengths, excluding padding.</summary>
    public IReadOnlyList<int> TokenLengths { get; }
    /// <summary>True mel lengths, excluding padding.</summary>
    public IReadOnlyList<int> MelLengths { get; }
    internal int[][]? Durations { get; }

    /// <summary>Creates a paired batch, optionally supplying one duration per active token.</summary>
    public AlignedMelBatch(Tensor<int> tokens, IReadOnlyList<int> tokenLengths,
        Tensor<T> targetMels, IReadOnlyList<int> melLengths,
        IReadOnlyList<IReadOnlyList<int>>? tokenDurations = null)
    {
        Tokens = tokens ?? throw new ArgumentNullException(nameof(tokens));
        TargetMels = targetMels ?? throw new ArgumentNullException(nameof(targetMels));
        if (tokenLengths is null) throw new ArgumentNullException(nameof(tokenLengths));
        if (melLengths is null) throw new ArgumentNullException(nameof(melLengths));
        TokenLengths = Array.AsReadOnly(tokenLengths.ToArray());
        MelLengths = Array.AsReadOnly(melLengths.ToArray());
        if (tokenDurations is not null)
        {
            Durations = new int[tokenDurations.Count][];
            for (int i = 0; i < Durations.Length; i++)
                Durations[i] = tokenDurations[i]?.ToArray()
                    ?? throw new ArgumentException("Duration rows must not be null.", nameof(tokenDurations));
        }
    }
}

/// <summary>Duration-expanded mel output; padding is zero and true lengths are explicit.</summary>
public sealed class AlignedMelOutput<T>
{
    /// <summary>Mel spectrogram [batch, maximum mel length, mel channels], not a waveform.</summary>
    public Tensor<T> MelSpectrogram { get; }
    /// <summary>Predicted log durations [batch, padded tokens], zero outside true token lengths.</summary>
    public Tensor<T> LogDurations { get; }
    /// <summary>Integer durations [batch, padded tokens], zero outside true token lengths.</summary>
    public Tensor<int> TokenDurations { get; }
    /// <summary>True output mel lengths.</summary>
    public IReadOnlyList<int> MelLengths { get; }

    internal AlignedMelOutput(Tensor<T> melSpectrogram, Tensor<T> logDurations,
        Tensor<int> tokenDurations, int[] melLengths)
    {
        MelSpectrogram = melSpectrogram;
        LogDurations = logDurations;
        TokenDurations = tokenDurations;
        MelLengths = Array.AsReadOnly((int[])melLengths.Clone());
    }
}

/// <summary>Actual tape-connected losses for paired token-to-mel training.</summary>
public sealed class AlignedMelObjective<T>
{
    /// <summary>Decoded mel squared error, averaged over active frames and mel channels.</summary>
    public Tensor<T> MelLoss { get; }
    /// <summary>Aligned Gaussian-prior squared error, averaged over active frames and channels.</summary>
    public Tensor<T> PriorLoss { get; }
    /// <summary>Log-duration squared error, averaged over active tokens.</summary>
    public Tensor<T> DurationLoss { get; }
    /// <summary>Sum of mel, prior, and duration losses.</summary>
    public Tensor<T> TotalLoss { get; }
    /// <summary>Predictions and the actual alignment used for this objective.</summary>
    public AlignedMelOutput<T> Output { get; }

    internal AlignedMelObjective(Tensor<T> melLoss, Tensor<T> priorLoss, Tensor<T> durationLoss,
        Tensor<T> totalLoss, AlignedMelOutput<T> output)
    {
        MelLoss = melLoss;
        PriorLoss = priorLoss;
        DurationLoss = durationLoss;
        TotalLoss = totalLoss;
        Output = output;
    }
}
