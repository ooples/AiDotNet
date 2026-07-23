namespace AiDotNet.Models;

using AiDotNet.Tensors.Helpers;

/// <summary>
/// Represents a request to select participating clients for a federated learning round.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> This object packages up the information needed to decide which clients should
/// join the next training round (who is available, how many to pick, and any optional hints like groups).
/// </remarks>
public class ClientSelectionRequest
{
    /// <summary>
    /// Gets or sets the round number (0-based) for which selection is being performed.
    /// </summary>
    public int RoundNumber { get; set; }

    /// <summary>
    /// Gets or sets the fraction of clients to select (0.0 to 1.0).
    /// </summary>
    public double FractionToSelect { get; set; }

    /// <summary>
    /// Gets or sets the full set of candidate client IDs.
    /// </summary>
    public IReadOnlyList<int> CandidateClientIds { get; set; } = Array.Empty<int>();

    /// <summary>
    /// Gets or sets client weights (typically proportional to sample count).
    /// </summary>
    public IReadOnlyDictionary<int, double> ClientWeights { get; set; } = new Dictionary<int, double>();

    /// <summary>
    /// Gets or sets optional group keys for stratified selection.
    /// </summary>
    public IReadOnlyDictionary<int, string>? ClientGroupKeys { get; set; }

    /// <summary>
    /// Gets or sets optional client availability probabilities (0.0 to 1.0).
    /// </summary>
    public IReadOnlyDictionary<int, double>? ClientAvailabilityProbabilities { get; set; }

    /// <summary>
    /// Gets or sets optional per-client performance scores (higher is better).
    /// </summary>
    public IReadOnlyDictionary<int, double>? ClientPerformanceScores { get; set; }

    /// <summary>
    /// Gets or sets optional per-client embeddings for cluster-based selection.
    /// </summary>
    public IReadOnlyDictionary<int, double[]>? ClientEmbeddings { get; set; }

    /// <summary>
    /// Gets or sets the random number generator to use for selection.
    /// </summary>
    public Random Random { get; set; } = RandomHelper.CreateSecureRandom();
}

