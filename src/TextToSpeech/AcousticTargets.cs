namespace AiDotNet.TextToSpeech;

/// <summary>
/// An utterance's acoustic training targets, aligned to its mel frames.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public sealed class AcousticTargets<T>
{
    internal AcousticTargets(Tensor<T> mel, int melFrames, double[]? pitch, double[]? energy)
    {
        Mel = mel;
        MelFrames = melFrames;
        Pitch = pitch;
        Energy = energy;
    }

    /// <summary>Target mel spectrogram, <c>[frames, melChannels]</c>.</summary>
    public Tensor<T> Mel { get; }

    /// <summary>Number of mel frames.</summary>
    public int MelFrames { get; }

    /// <summary>F0 per frame in Hz (0 where unvoiced), or null when neither supplied nor derivable.</summary>
    public double[]? Pitch { get; }

    /// <summary>Energy per frame, or null when neither supplied nor derivable.</summary>
    public double[]? Energy { get; }
}
