namespace AiDotNet.Serving.Models.Federated;

/// <summary>
/// Response after aggregating a federated round.
/// </summary>
public class AggregateFederatedRoundResponse
{
    /// <summary>
    /// Gets or sets the new round number after aggregation.
    /// </summary>
    public int NewCurrentRound { get; set; }

    /// <summary>
    /// Gets or sets the number of client updates that were aggregated.
    /// </summary>
    public int AggregatedClientCount { get; set; }
}

