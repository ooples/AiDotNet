using AiDotNet.Models.Options;

namespace AiDotNet.FederatedLearning.PSI;

/// <summary>
/// Defines the interface for approximate entity matching in PSI.
/// </summary>
/// <remarks>
/// <para>Fuzzy matchers handle the common real-world scenario where entity identifiers
/// aren't perfectly identical across parties due to typos, formatting differences,
/// transliteration, or data entry errors.</para>
///
/// <para><b>For Beginners:</b> In an ideal world, "Patient #12345" at Hospital A is exactly
/// the same string at Hospital B. In practice, one might store "John Smith" and the other
/// "Jon Smith" or "SMITH, JOHN". Fuzzy matching bridges these gaps by finding IDs that
/// are similar enough to be the same entity.</para>
/// </remarks>
public interface IFuzzyMatcher
{
    /// <summary>
    /// Gets the name of this fuzzy matching strategy.
    /// </summary>
    string StrategyName { get; }

    /// <summary>
    /// Computes the similarity score between two identifiers.
    /// </summary>
    /// <param name="id1">The first identifier.</param>
    /// <param name="id2">The second identifier.</param>
    /// <param name="options">Fuzzy matching configuration options.</param>
    /// <returns>A similarity score. The interpretation depends on the strategy:
    /// for distance-based metrics, lower is more similar; for similarity-based metrics,
    /// higher (closer to 1.0) is more similar.</returns>
    double ComputeSimilarity(string id1, string id2, FuzzyMatchOptions options);

    /// <summary>
    /// Determines whether two identifiers are a match according to the configured threshold.
    /// </summary>
    /// <param name="id1">The first identifier.</param>
    /// <param name="id2">The second identifier.</param>
    /// <param name="options">Fuzzy matching configuration options.</param>
    /// <returns>True if the identifiers are similar enough to be considered the same entity.</returns>
    bool IsMatch(string id1, string id2, FuzzyMatchOptions options);

    /// <summary>
    /// Normalizes an identifier before comparison by applying case folding,
    /// whitespace normalization, and other transformations.
    /// </summary>
    /// <param name="id">The identifier to normalize.</param>
    /// <param name="options">Fuzzy matching configuration options.</param>
    /// <returns>The normalized identifier.</returns>
    string Normalize(string id, FuzzyMatchOptions options);

    /// <summary>
    /// Finds all matches for a given identifier from a set of candidates.
    /// </summary>
    /// <param name="id">The identifier to match against.</param>
    /// <param name="candidates">The candidate identifiers to search.</param>
    /// <param name="options">Fuzzy matching configuration options.</param>
    /// <returns>A list of (candidateIndex, similarityScore) pairs for all matches,
    /// sorted by best match first.</returns>
    IReadOnlyList<(int CandidateIndex, double Similarity)> FindMatches(
        string id, IReadOnlyList<string> candidates, FuzzyMatchOptions options);
}
