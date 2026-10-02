namespace AiDotNet.TextToSpeech;

/// <summary>
/// One utterance of text-to-speech training data, with whatever supervision the model's paper trains on.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// A plain <c>Train(tokens, mel)</c> call carries the text and the target spectrogram and nothing else, but most
/// acoustic models are trained on more: FastSpeech 2 on forced-alignment durations and frame-level pitch and
/// energy (Ren et al. 2021, §2.3), codec language models on codec tokens and a speaker prompt, zero-shot models on
/// reference audio. This type carries all of it. Anything a paper derives from the recording itself — pitch with
/// WORLD, energy from the STFT, the mel spectrogram — the model derives from <see cref="Audio"/> when it is not
/// supplied; anything a paper takes from an external tool (FastSpeech 2's Montreal Forced Aligner durations)
/// must be supplied, and the model says so by name when it is missing.
/// </para>
/// <para><b>For Beginners:</b> Fill in the text tokens and either the recording or its mel spectrogram. Add
/// durations when your model needs an alignment; the error message tells you if it does.</para>
/// </remarks>
public sealed class TtsTrainingSample<T>
{
    /// <summary>Text (phoneme or character) token ids, <c>[tokens]</c>.</summary>
    public required Tensor<T> Tokens { get; init; }

    /// <summary>The recording, <c>[samples]</c>, at the model's sample rate.</summary>
    public Tensor<T>? Audio { get; init; }

    /// <summary>Target mel spectrogram, <c>[frames, melChannels]</c>. Derived from <see cref="Audio"/> when absent.</summary>
    public Tensor<T>? Mel { get; init; }

    /// <summary>Frames per token from a forced alignment; the values sum to the number of mel frames.</summary>
    public int[]? Durations { get; init; }

    /// <summary>F0 per mel frame in Hz, 0 where unvoiced. Derived from <see cref="Audio"/> when absent.</summary>
    public double[]? Pitch { get; init; }

    /// <summary>Energy per mel frame. Derived from <see cref="Audio"/> when absent.</summary>
    public double[]? Energy { get; init; }

    /// <summary>Reference audio or embedding for the target speaker or style.</summary>
    public Tensor<T>? SpeakerReference { get; init; }

    /// <summary>Discrete codec tokens of the recording, <c>[frames, codebooks]</c>.</summary>
    public Tensor<T>? CodecTokens { get; init; }
}
