namespace AiDotNet.Agentic.Models.Local;

/// <summary>
/// A constraint that always restricts generation to a fixed set of token ids, regardless of context — for
/// example, "only emit digit tokens" or "only emit tokens from this closed label vocabulary".
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> The simplest gate: a permanent allow-list. Whatever has been generated, only
/// these tokens are ever permitted next. Handy when the whole answer must come from a small known set.
/// </para>
/// </remarks>
public sealed class AllowedTokenSetConstraint : ITokenConstraint
{
    private readonly IReadOnlyCollection<int> _allowed;

    /// <summary>
    /// Initializes a new constraint permitting only the given token ids.
    /// </summary>
    /// <param name="allowedTokenIds">The permitted token ids. Must be non-null and non-empty.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="allowedTokenIds"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">Thrown when <paramref name="allowedTokenIds"/> is empty.</exception>
    public AllowedTokenSetConstraint(IEnumerable<int> allowedTokenIds)
    {
        Guard.NotNull(allowedTokenIds);
        // Dedup via HashSet, then freeze: returning the mutable backing set
        // would let a caller cast it back and silently change decoding
        // behavior mid-generation.
        var deduplicated = new HashSet<int>(allowedTokenIds);
        if (deduplicated.Count == 0)
        {
            throw new ArgumentException("At least one allowed token id is required.", nameof(allowedTokenIds));
        }

        var frozen = new int[deduplicated.Count];
        deduplicated.CopyTo(frozen);
        _allowed = Array.AsReadOnly(frozen);
    }

    /// <inheritdoc/>
    public IReadOnlyCollection<int>? AllowedNextTokens(IReadOnlyList<int> generatedTokenIds) => _allowed;
}
