namespace AiDotNet.TextToSpeech;

/// <summary>
/// An utterance's acoustic training targets, aligned to its mel frames.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public sealed class AcousticTargets<T>
{
    internal AcousticTargets(Tensor<T> mel, int melFrames, Tensor<T>? pitch, Tensor<T>? energy,
        Tensor<T>? linearSpectrogram = null)
    {
        LinearSpectrogram = linearSpectrogram;
        Mel = mel;
        MelFrames = melFrames;
        Pitch = pitch;
        Energy = energy;
    }

    /// <summary>Target mel spectrogram, <c>[frames, melChannels]</c>.</summary>
    /// <summary>The log-magnitude linear spectrogram, <c>[frames, fftSize / 2 + 1]</c>, when the sample carries it or
    /// its recording.</summary>
    public Tensor<T>? LinearSpectrogram { get; }

    public Tensor<T> Mel { get; }

    /// <summary>Number of mel frames.</summary>
    public int MelFrames { get; }

    /// <summary>F0 per frame in Hz, <c>[frames]</c> (0 where unvoiced), or null when neither supplied nor derivable.</summary>
    public Tensor<T>? Pitch { get; }

    /// <summary>Energy per frame, <c>[frames]</c>, or null when neither supplied nor derivable.</summary>
    public Tensor<T>? Energy { get; }
}
