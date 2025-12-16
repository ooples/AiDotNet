using AiDotNet.Models;

namespace AiDotNet.FederatedLearning.Selection;

/// <summary>
/// Weighted random selection without replacement (typically weighted by sample count).
/// </summary>
public sealed class WeightedRandomClientSelectionStrategy : ClientSelectionStrategyBase
{
    public override List<int> SelectClients(ClientSelectionRequest request)
    {
        if (request == null)
        {
            throw new ArgumentNullException(nameof(request));
        }

        var candidates = request.CandidateClientIds ?? Array.Empty<int>();
        int desired = GetDesiredClientCount(candidates, request.FractionToSelect);
        if (desired >= candidates.Count)
        {
            return candidates.OrderBy(i => i).ToList();
        }

        var weights = request.ClientWeights ?? new Dictionary<int, double>();
        var remaining = candidates.Distinct().ToList();
        var selected = new List<int>(desired);

        for (int k = 0; k < desired; k++)
        {
            double total = remaining
                .Select(id => weights.TryGetValue(id, out var w) ? w : 0.0)
                .Where(w => w > 0.0)
                .Sum();

            if (total <= 0.0)
            {
                // Fallback to uniform if no usable weights.
                selected.AddRange(ShuffleAndTake(remaining, desired - selected.Count, request.Random));
                break;
            }

            double r = request.Random.NextDouble() * total;
            double cumulative = 0.0;
            int chosen = remaining[0];
            foreach (var id in remaining)
            {
                double w = (weights.TryGetValue(id, out var ww) && ww > 0.0) ? ww : 0.0;
                cumulative += w;
                if (cumulative >= r)
                {
                    chosen = id;
                    break;
                }
            }

            selected.Add(chosen);
            remaining.Remove(chosen);
            if (remaining.Count == 0)
            {
                break;
            }
        }

        selected.Sort();
        return selected;
    }

    public override string GetStrategyName() => "WeightedRandom";
}
