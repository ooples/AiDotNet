using AiDotNet.Enums;
using AiDotNet.Validation;

namespace AiDotNet.Evolution.Programs.Novelty;

/// <summary>One novelty decision, with the rung that produced it and what that rung cost.</summary>
/// <remarks>
/// <para>
/// The verdict alone is not enough to run a search responsibly. A gate that admits everything because its provider
/// is unreachable looks identical, from the outside, to a gate that admits everything because the candidates really
/// are new — unless the decision says which rung answered and how many paid comparisons it made. Those counts are
/// what a run reports, what a cost claim is proved with, and what makes a misconfigured threshold visible.
/// </para>
/// <para>
/// <see cref="Reason"/> is bounded and describes the mechanism, never the program: it is safe to log. Program text
/// never appears here at all, only <see cref="NearestGenomeId"/>, which is a hash.
/// </para>
/// <para><b>For Beginners:</b> This is the answer to "should we bother evaluating this candidate?", plus the
/// receipt. <see cref="IsNovel"/> is the answer, <see cref="DecidedBy"/> says which check settled it, and the three
/// count properties say how much that cost — the cheap check is free, the other two are not.</para>
/// </remarks>
public sealed class ProgramNoveltyDecision
{
    /// <summary>The longest reason retained, in characters.</summary>
    public const int MaxReasonLength = 240;

    /// <summary>Initializes a novelty decision.</summary>
    /// <param name="isNovel">Whether the candidate should proceed.</param>
    /// <param name="decidedBy">The rung that produced the verdict.</param>
    /// <param name="reason">A bounded, program-free explanation; truncated to <see cref="MaxReasonLength"/>.</param>
    /// <param name="nearestGenomeId">The identity of the closest known genome, or <c>null</c> when none was compared.</param>
    /// <param name="nearestStructuralDistance">The smallest structural distance found, or <c>null</c> when none was computed.</param>
    /// <param name="embeddingSimilarity">The highest cosine similarity found, or <c>null</c> when none was computed.</param>
    /// <param name="structuralComparisons">How many structural distances were computed.</param>
    /// <param name="embeddingRequests">How many embedding requests were issued.</param>
    /// <param name="judgeRequests">How many judging requests were issued.</param>
    /// <exception cref="ArgumentNullException"><paramref name="reason"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentOutOfRangeException">
    /// <paramref name="decidedBy"/> is not a defined value, a count is negative, or a supplied metric is not finite.
    /// </exception>
    public ProgramNoveltyDecision(
        bool isNovel,
        ProgramNoveltyStage decidedBy,
        string reason,
        string? nearestGenomeId = null,
        double? nearestStructuralDistance = null,
        double? embeddingSimilarity = null,
        int structuralComparisons = 0,
        int embeddingRequests = 0,
        int judgeRequests = 0)
    {
        Guard.NotNull(reason);
        if (!Enum.IsDefined(typeof(ProgramNoveltyStage), decidedBy))
        {
            throw new ArgumentOutOfRangeException(nameof(decidedBy), decidedBy, "Value must be a defined stage.");
        }

        RequireNonNegative(structuralComparisons, nameof(structuralComparisons));
        RequireNonNegative(embeddingRequests, nameof(embeddingRequests));
        RequireNonNegative(judgeRequests, nameof(judgeRequests));
        RequireFinite(nearestStructuralDistance, nameof(nearestStructuralDistance));
        RequireFinite(embeddingSimilarity, nameof(embeddingSimilarity));

        IsNovel = isNovel;
        DecidedBy = decidedBy;
        Reason = reason.Length <= MaxReasonLength ? reason : reason.Substring(0, MaxReasonLength);
        NearestGenomeId = nearestGenomeId;
        NearestStructuralDistance = nearestStructuralDistance;
        EmbeddingSimilarity = embeddingSimilarity;
        StructuralComparisons = structuralComparisons;
        EmbeddingRequests = embeddingRequests;
        JudgeRequests = judgeRequests;
    }

    /// <summary>Gets whether the candidate is novel enough to proceed.</summary>
    public bool IsNovel { get; }

    /// <summary>Gets the rung of the novelty ladder that produced the verdict.</summary>
    public ProgramNoveltyStage DecidedBy { get; }

    /// <summary>Gets the bounded, program-free explanation of the verdict.</summary>
    public string Reason { get; }

    /// <summary>Gets the identity of the closest known genome, or <c>null</c> when nothing was compared.</summary>
    public string? NearestGenomeId { get; }

    /// <summary>Gets the smallest structural distance found, or <c>null</c> when none was computed.</summary>
    public double? NearestStructuralDistance { get; }

    /// <summary>Gets the highest embedding cosine similarity found, or <c>null</c> when none was computed.</summary>
    public double? EmbeddingSimilarity { get; }

    /// <summary>Gets how many structural distances this decision computed; each one is local arithmetic.</summary>
    public int StructuralComparisons { get; }

    /// <summary>Gets how many embedding requests this decision issued; zero unless the cheap rung was inconclusive.</summary>
    public int EmbeddingRequests { get; }

    /// <summary>Gets how many judging requests this decision issued; zero unless both cheaper rungs were inconclusive.</summary>
    public int JudgeRequests { get; }

    /// <summary>Gets whether this decision reached neither a provider nor a model.</summary>
    public bool WasFree => EmbeddingRequests == 0 && JudgeRequests == 0;

    /// <summary>Returns the verdict, the deciding rung, and the paid-call counts.</summary>
    /// <returns>A short diagnostic label for this decision.</returns>
    public override string ToString() =>
        (IsNovel ? "novel" : "not-novel") + " by " + DecidedBy +
        " (embeddings=" + EmbeddingRequests.ToString(System.Globalization.CultureInfo.InvariantCulture) +
        ", judgements=" + JudgeRequests.ToString(System.Globalization.CultureInfo.InvariantCulture) + ")";

    private static void RequireNonNegative(int value, string parameterName)
    {
        if (value < 0) throw new ArgumentOutOfRangeException(parameterName, value, "Value cannot be negative.");
    }

    private static void RequireFinite(double? value, string parameterName)
    {
        if (value is { } number && (double.IsNaN(number) || double.IsInfinity(number)))
        {
            throw new ArgumentOutOfRangeException(parameterName, number, "Value must be finite.");
        }
    }
}
