using AiDotNet.Models;

namespace AiDotNet.FederatedLearning.Selection;

/// <summary>
/// Performance-aware client selection using an explore/exploit policy over historical scores.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> PerformanceAwareClientSelectionStrategy provides AI safety functionality. Default values follow the original paper settings.</para>
/// </remarks>
public sealed class PerformanceAwareClientSelectionStrategy : ClientSelectionStrategyBase
{
    private readonly double _explorationRate;

    public PerformanceAwareClientSelectionStrategy(double explorationRate = 0.1)
    {
        if (explorationRate < 0.0 || explorationRate > 1.0)
        {
            throw new ArgumentOutOfRangeException(nameof(explorationRate), "Exploration rate must be in [0, 1].");
        }

        _explorationRate = explorationRate;
    }

    public override List<int> SelectClients(ClientSelectionRequest request)
    {
        if (request == null)
        {
            throw new ArgumentNullException(nameof(request));
        }

        var candidates = request.CandidateClientIds ?? Array.Empty<int>();
        int desired = GetDesiredClientCount(candidates, request.FractionToSelect);

        var scores = request.ClientPerformanceScores;
        if (scores == null || scores.Count == 0)
        {
            return ShuffleAndTake(candidates, desired, request.Random);
        }

        // Explore: random selection.
        if (request.Random.NextDouble() < _explorationRate)
        {
            return ShuffleAndTake(candidates, desired, request.Random);
        }

        // Exploit: take highest-score clients.
        var ordered = candidates
            .Distinct()
            .Select(id => (ClientId: id, Score: scores.TryGetValue(id, out var s) ? s : double.NegativeInfinity))
            .OrderByDescending(x => x.Score)
            .ThenBy(x => x.ClientId)
            .Take(desired)
            .Select(x => x.ClientId)
            .ToList();

        ordered.Sort();
        return ordered;
    }

    public override string GetStrategyName() => "PerformanceAware";
}

