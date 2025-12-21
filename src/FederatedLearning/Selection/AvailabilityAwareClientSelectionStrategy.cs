using AiDotNet.Models;

namespace AiDotNet.FederatedLearning.Selection;

/// <summary>
/// Availability-aware client selection using per-client online probabilities.
/// </summary>
public sealed class AvailabilityAwareClientSelectionStrategy : ClientSelectionStrategyBase
{
    private readonly double _availabilityThreshold;

    public AvailabilityAwareClientSelectionStrategy(double availabilityThreshold = 0.0)
    {
        if (availabilityThreshold < 0.0 || availabilityThreshold > 1.0)
        {
            throw new ArgumentOutOfRangeException(nameof(availabilityThreshold), "Availability threshold must be in [0, 1].");
        }

        _availabilityThreshold = availabilityThreshold;
    }

    public override List<int> SelectClients(ClientSelectionRequest request)
    {
        if (request == null)
        {
            throw new ArgumentNullException(nameof(request));
        }

        var candidates = request.CandidateClientIds ?? Array.Empty<int>();
        int desired = GetDesiredClientCount(candidates, request.FractionToSelect);

        var availability = request.ClientAvailabilityProbabilities;
        if (availability == null || availability.Count == 0)
        {
            return ShuffleAndTake(candidates, desired, request.Random);
        }

        var eligible = new List<int>();
        var fallback = new List<(int ClientId, double P)>();

        foreach (var id in candidates.Distinct())
        {
            double p = availability.TryGetValue(id, out var pp) ? pp : 0.0;
            if (p < 0.0)
            {
                p = 0.0;
            }
            if (p > 1.0)
            {
                p = 1.0;
            }

            if (p >= _availabilityThreshold && request.Random.NextDouble() <= p)
            {
                eligible.Add(id);
            }

            fallback.Add((id, p));
        }

        if (eligible.Count >= desired)
        {
            return ShuffleAndTake(eligible, desired, request.Random);
        }

        // Fill remaining slots by highest availability probability.
        var remaining = fallback
            .OrderByDescending(x => x.P)
            .ThenBy(x => x.ClientId)
            .Select(x => x.ClientId)
            .Where(id => !eligible.Contains(id))
            .ToList();

        int need = desired - eligible.Count;
        if (need > 0 && remaining.Count > 0)
        {
            eligible.AddRange(remaining.Take(need));
        }

        eligible = eligible.Distinct().ToList();
        eligible.Sort();
        if (eligible.Count > desired)
        {
            eligible = eligible.Take(desired).ToList();
        }

        return eligible;
    }

    public override string GetStrategyName() => "AvailabilityAware";
}

