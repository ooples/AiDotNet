namespace AiDotNet.Agentic.Pipeline;

/// <summary>
/// A simple <see cref="IContentModerator"/> that blocks content containing any of a configured set of banned
/// terms. Zero-config and deterministic — useful as a baseline guardrail or for tests; swap in a classifier
/// (e.g., an <c>src/Safety</c>-backed moderator) for nuanced moderation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> A blocklist checker: if the text contains a forbidden word, it's blocked.
/// Fast and predictable, but it only catches exact terms — not paraphrases or intent.
/// </para>
/// </remarks>
public sealed class DenyListContentModerator : IContentModerator
{
    private readonly IReadOnlyList<string> _bannedTerms;
    private readonly StringComparison _comparison;

    /// <summary>
    /// Initializes a new deny-list moderator.
    /// </summary>
    /// <param name="bannedTerms">The terms that, if present, block the content. Must be non-null.</param>
    /// <param name="caseInsensitive">Whether matching ignores case. Default <c>true</c>.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="bannedTerms"/> is <c>null</c>.</exception>
    public DenyListContentModerator(IEnumerable<string> bannedTerms, bool caseInsensitive = true)
    {
        Guard.NotNull(bannedTerms);
        // Trim and drop whitespace-only entries: a stray " " would otherwise
        // become a near-global block. Dedup so the same term isn't scanned twice.
        _bannedTerms = bannedTerms
            .Where(t => !string.IsNullOrWhiteSpace(t))
            .Select(t => t.Trim())
            .Distinct(caseInsensitive ? StringComparer.OrdinalIgnoreCase : StringComparer.Ordinal)
            .ToList();
        _comparison = caseInsensitive ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal;
    }

    /// <inheritdoc/>
    public Task<ModerationVerdict> CheckAsync(string content, CancellationToken cancellationToken = default)
    {
        Guard.NotNull(content);
        foreach (var term in _bannedTerms)
        {
            if (ContainsTerm(content, term, _comparison))
            {
                return Task.FromResult(ModerationVerdict.Block($"Content contains a disallowed term: '{term}'."));
            }
        }

        return Task.FromResult(ModerationVerdict.Allow());
    }

    // Boundary-aware matching: the term must not be embedded inside a larger
    // word, so a banned "ass" does not block "class". Boundaries are
    // non-alphanumeric characters or the ends of the content, which also works
    // for multi-word phrases (boundaries apply at the phrase's edges).
    private static bool ContainsTerm(string content, string term, StringComparison comparison)
    {
        var searchFrom = 0;
        while (searchFrom <= content.Length - term.Length)
        {
            var found = content.IndexOf(term, searchFrom, comparison);
            if (found < 0)
            {
                return false;
            }

            var startsAtBoundary = found == 0 || !char.IsLetterOrDigit(content[found - 1]);
            var endIndex = found + term.Length;
            var endsAtBoundary = endIndex >= content.Length || !char.IsLetterOrDigit(content[endIndex]);
            if (startsAtBoundary && endsAtBoundary)
            {
                return true;
            }

            searchFrom = found + 1;
        }

        return false;
    }
}
