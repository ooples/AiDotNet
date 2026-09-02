using AiDotNet.Validation;

namespace AiDotNet.Agentic.Embeddings;

/// <summary>The outcome of one embedding request: either every vector, or an explicit bounded failure.</summary>
/// <remarks>
/// <para>
/// A batch is a discriminated outcome rather than a possibly-empty list. The reference implementation returns an
/// empty vector (in fact a <c>(list, float)</c> tuple, which does not match its own declared return type) when a
/// request fails, stores that value on the program record, and lets every later cosine comparison against it return
/// zero — so one transient network error both fails open for the current decision and permanently poisons that
/// record's similarity for the rest of the run. Here a failure is a value the caller must look at: nothing is
/// cached, nothing is stored, and the caller decides whether to fail open or closed.
/// </para>
/// <para>
/// <see cref="FailureReason"/> is bounded and carries only a status code or an exception type name. A provider's
/// response body can echo request content or an authorization header, so it never reaches this string, a log, or a
/// prompt.
/// </para>
/// <para><b>For Beginners:</b> When you ask a provider to turn text into vectors, the call can fail. This type says
/// clearly which happened: <see cref="Succeeded"/> true with one vector per input, or false with a short reason.
/// Because the failure is explicit you can decide what a failure means for your run instead of silently getting
/// back something that looks like a valid but meaningless answer.</para>
/// </remarks>
public sealed class EmbeddingBatch
{
    private static readonly EmbeddingVector[] NoVectors = Array.Empty<EmbeddingVector>();

    private readonly EmbeddingVector[] _vectors;

    private EmbeddingBatch(EmbeddingVector[] vectors, bool succeeded, string failureReason)
    {
        _vectors = vectors;
        Succeeded = succeeded;
        FailureReason = failureReason;
    }

    /// <summary>The longest failure reason retained, in characters.</summary>
    public const int MaxFailureReasonLength = 200;

    /// <summary>Gets whether the request produced a vector for every input.</summary>
    public bool Succeeded { get; }

    /// <summary>Gets the vectors, in the order the inputs were supplied; empty when the request failed.</summary>
    public IReadOnlyList<EmbeddingVector> Vectors => _vectors;

    /// <summary>Gets the bounded failure reason, or an empty string when the request succeeded.</summary>
    public string FailureReason { get; }

    /// <summary>Creates a successful batch.</summary>
    /// <param name="vectors">One vector per input, in input order.</param>
    /// <returns>A batch reporting success.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="vectors"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException"><paramref name="vectors"/> contains a <c>null</c> entry.</exception>
    public static EmbeddingBatch Success(IEnumerable<EmbeddingVector> vectors)
    {
        Guard.NotNull(vectors);
        EmbeddingVector[] copy = vectors.ToArray();
        foreach (EmbeddingVector vector in copy)
        {
            if (vector is null)
            {
                throw new ArgumentException("An embedding batch cannot contain a null vector.", nameof(vectors));
            }
        }

        return new EmbeddingBatch(copy, succeeded: true, failureReason: string.Empty);
    }

    /// <summary>Creates a failed batch carrying a bounded reason.</summary>
    /// <param name="reason">Why the request failed; truncated to <see cref="MaxFailureReasonLength"/> characters.</param>
    /// <returns>A batch reporting failure and holding no vectors.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="reason"/> is <c>null</c>.</exception>
    public static EmbeddingBatch Failure(string reason)
    {
        Guard.NotNull(reason);
        string bounded = reason.Length <= MaxFailureReasonLength
            ? reason
            : reason.Substring(0, MaxFailureReasonLength);
        return new EmbeddingBatch(NoVectors, succeeded: false, failureReason: bounded);
    }

    /// <summary>Returns the outcome and vector count, never the components.</summary>
    /// <returns>A short diagnostic label for this batch.</returns>
    public override string ToString() => Succeeded
        ? "embeddings(" + _vectors.Length.ToString(System.Globalization.CultureInfo.InvariantCulture) + ")"
        : "embeddings(failed: " + FailureReason + ")";
}
