namespace AiDotNet.Serving.ContinuousBatching;

/// <summary>
/// Event args for token generation.
/// </summary>
public class TokenGeneratedEventArgs<T> : EventArgs
{
    /// <summary>The sequence that generated the token.</summary>
    public required SequenceState<T> Sequence { get; set; }

    /// <summary>The generated token ID.</summary>
    public required int TokenId { get; set; }
}
