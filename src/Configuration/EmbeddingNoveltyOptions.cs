using System.Globalization;

namespace AiDotNet.Configuration;

/// <summary>Immutable, constructor-validated thresholds for the structural, embedding, and judge novelty stages.</summary>
/// <remarks>
/// <para>
/// Every value is checked in the constructor and every property is read-only, so an instance that exists is an
/// instance that is valid. That matters here more than it does for a mutable options bag: these thresholds gate
/// spending, and a value validated only when a run starts would let a mistyped threshold sit dormant until the
/// first candidate arrives — by which time the cheap gate might be admitting everything and paying for an
/// embedding request per proposal.
/// </para>
/// <para>
/// The stages form a cost ladder. A candidate whose nearest structural neighbour is at least
/// <see cref="StructuralNoveltyThreshold"/> away is novel immediately, with no embedding request and no model call;
/// this rung has no equivalent upstream, where the very first thing the insertion path does is embed the candidate.
/// Only a structural near-duplicate reaches the embedding rung, which compares it against at most
/// <see cref="MaxEmbeddingComparisons"/> of its nearest neighbours rather than every member of its island, and only
/// a candidate whose cosine similarity reaches <see cref="EmbeddingSimilarityThreshold"/> reaches the judge.
/// </para>
/// <para>
/// <see cref="FailOpenOnEmbeddingFailure"/> and <see cref="FailOpenOnJudgeFailure"/> default to <c>true</c>, which
/// reproduces the reference implementation's effective behaviour — there, an embedding error yields an empty vector
/// whose similarity to everything is zero, and a judge error returns "novel" — but here it is a decision the run
/// makes explicitly and can reverse. Set either to <c>false</c> when spending an evaluation on a duplicate costs
/// more than discarding a possibly-new candidate.
/// </para>
/// <para><b>For Beginners:</b> These numbers decide when a new candidate counts as "different enough to be worth
/// trying". The first one is a cheap text comparison and does almost all the work. The second only matters if you
/// supplied an embedding model, and the last two say what should happen when that model or the judging model is
/// unreachable — accept the candidate and risk a duplicate, or reject it and risk losing a good idea. The defaults
/// accept.</para>
/// </remarks>
public sealed class EmbeddingNoveltyOptions
{
    /// <summary>The default structural distance at or above which a candidate is novel without further checks.</summary>
    public const double DefaultStructuralNoveltyThreshold = 0.15;

    /// <summary>The default cosine similarity at or above which a candidate is treated as a possible duplicate.</summary>
    /// <remarks>Matches the reference implementation's <c>similarity_threshold</c> default of 0.99.</remarks>
    public const double DefaultEmbeddingSimilarityThreshold = 0.99;

    /// <summary>The default number of nearest neighbours compared by embedding.</summary>
    public const int DefaultMaxEmbeddingComparisons = 8;

    /// <summary>The default number of accepted genomes a gate remembers.</summary>
    public const int DefaultMaxTrackedGenomes = 512;

    /// <summary>Initializes a validated set of novelty thresholds.</summary>
    /// <param name="structuralNoveltyThreshold">
    /// The structural distance at or above which a candidate is novel outright; 0 to 1. Zero disables the cheap
    /// rung, sending every candidate to the embedding or judge stage.
    /// </param>
    /// <param name="embeddingSimilarityThreshold">
    /// The cosine similarity at or above which a structural near-duplicate is treated as a possible duplicate; 0 to 1.
    /// </param>
    /// <param name="maxEmbeddingComparisons">How many nearest neighbours are compared by embedding; 1 to 256.</param>
    /// <param name="failOpenOnEmbeddingFailure">Whether an unreachable embedding provider admits the candidate.</param>
    /// <param name="failOpenOnJudgeFailure">Whether an unusable judge answer admits the candidate.</param>
    /// <param name="maxTrackedGenomes">How many accepted genomes a gate remembers; 1 to 1,000,000.</param>
    /// <exception cref="ArgumentOutOfRangeException">A threshold is not finite or falls outside its permitted range.</exception>
    public EmbeddingNoveltyOptions(
        double structuralNoveltyThreshold = DefaultStructuralNoveltyThreshold,
        double embeddingSimilarityThreshold = DefaultEmbeddingSimilarityThreshold,
        int maxEmbeddingComparisons = DefaultMaxEmbeddingComparisons,
        bool failOpenOnEmbeddingFailure = true,
        bool failOpenOnJudgeFailure = true,
        int maxTrackedGenomes = DefaultMaxTrackedGenomes)
    {
        ValidateUnitInterval(structuralNoveltyThreshold, nameof(structuralNoveltyThreshold));
        ValidateUnitInterval(embeddingSimilarityThreshold, nameof(embeddingSimilarityThreshold));

        if (maxEmbeddingComparisons < 1 || maxEmbeddingComparisons > 256)
        {
            throw new ArgumentOutOfRangeException(nameof(maxEmbeddingComparisons), maxEmbeddingComparisons,
                "Value must be between 1 and 256.");
        }

        if (maxTrackedGenomes < 1 || maxTrackedGenomes > 1_000_000)
        {
            throw new ArgumentOutOfRangeException(nameof(maxTrackedGenomes), maxTrackedGenomes,
                "Value must be between 1 and 1000000.");
        }

        StructuralNoveltyThreshold = structuralNoveltyThreshold;
        EmbeddingSimilarityThreshold = embeddingSimilarityThreshold;
        MaxEmbeddingComparisons = maxEmbeddingComparisons;
        FailOpenOnEmbeddingFailure = failOpenOnEmbeddingFailure;
        FailOpenOnJudgeFailure = failOpenOnJudgeFailure;
        MaxTrackedGenomes = maxTrackedGenomes;
    }

    /// <summary>Gets the structural distance at or above which a candidate is novel without any paid comparison.</summary>
    public double StructuralNoveltyThreshold { get; }

    /// <summary>Gets the cosine similarity at or above which a structural near-duplicate is a possible duplicate.</summary>
    public double EmbeddingSimilarityThreshold { get; }

    /// <summary>Gets how many nearest neighbours are compared by embedding.</summary>
    public int MaxEmbeddingComparisons { get; }

    /// <summary>Gets whether an unreachable embedding provider admits the candidate.</summary>
    public bool FailOpenOnEmbeddingFailure { get; }

    /// <summary>Gets whether an unusable judge answer admits the candidate.</summary>
    public bool FailOpenOnJudgeFailure { get; }

    /// <summary>Gets how many accepted genomes a novelty gate remembers before discarding the oldest.</summary>
    public int MaxTrackedGenomes { get; }

    /// <summary>Gets a stable hash of these thresholds for checkpoint and version comparison.</summary>
    /// <returns>A short invariant string that changes whenever any threshold changes.</returns>
    public string ToVersionString() => string.Join("|", new[]
    {
        StructuralNoveltyThreshold.ToString("R", CultureInfo.InvariantCulture),
        EmbeddingSimilarityThreshold.ToString("R", CultureInfo.InvariantCulture),
        MaxEmbeddingComparisons.ToString(CultureInfo.InvariantCulture),
        FailOpenOnEmbeddingFailure ? "embed-open" : "embed-closed",
        FailOpenOnJudgeFailure ? "judge-open" : "judge-closed",
        MaxTrackedGenomes.ToString(CultureInfo.InvariantCulture)
    });

    private static void ValidateUnitInterval(double value, string parameterName)
    {
        if (double.IsNaN(value) || double.IsInfinity(value) || value < 0.0 || value > 1.0)
        {
            throw new ArgumentOutOfRangeException(parameterName, value,
                "Value must be a finite number between 0 and 1.");
        }
    }
}
