namespace AiDotNet.Audio.MusicGen;

/// <summary>One MusicGen training example: a music clip, its text description and optional metadata.</summary>
/// <typeparam name="T">The numeric type.</typeparam>
public sealed class MusicGenTrainingSample<T>
{
    /// <summary>The clip as a waveform at the model's sample rate (or give <see cref="Codes"/>).</summary>
    public Tensor<T>? Audio { get; init; }

    /// <summary>The clip's EnCodec codes <c>[frames, codebooks]</c>, used instead of encoding <see cref="Audio"/>.</summary>
    public Tensor<T>? Codes { get; init; }

    /// <summary>The text description; null trains the step unconditionally.</summary>
    public string? Description { get; init; }

    /// <summary>Metadata merged into the description by condition merging (§3.1): any of <c>key</c>, <c>bpm</c>,
    /// <c>genre</c>, <c>moods</c>, <c>instrument</c> and <c>keywords</c> (lists comma-separated).</summary>
    public IReadOnlyDictionary<string, string>? Metadata { get; init; }
}
