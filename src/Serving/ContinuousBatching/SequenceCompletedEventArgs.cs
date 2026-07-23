namespace AiDotNet.Serving.ContinuousBatching;

/// <summary>
/// Event args for sequence completion.
/// </summary>
public class SequenceCompletedEventArgs<T> : EventArgs
{
    /// <summary>The completed sequence.</summary>
    public required SequenceState<T> Sequence { get; set; }

    /// <summary>The generation result.</summary>
    public required GenerationResult<T> Result { get; set; }
}
