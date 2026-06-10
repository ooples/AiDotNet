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
        _bannedTerms = bannedTerms.Where(t => t is not null && t.Length > 0).ToList();
        _comparison = caseInsensitive ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal;
    }

    /// <inheritdoc/>
    public Task<ModerationVerdict> CheckAsync(string content, CancellationToken cancellationToken = default)
    {
        Guard.NotNull(content);
        foreach (var term in _bannedTerms)
        {
            if (content.IndexOf(term, _comparison) >= 0)
            {
                return Task.FromResult(ModerationVerdict.Block($"Content contains a disallowed term: '{term}'."));
            }
        }

        return Task.FromResult(ModerationVerdict.Allow());
    }
}
