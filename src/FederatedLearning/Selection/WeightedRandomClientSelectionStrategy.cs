using AiDotNet.Models;

namespace AiDotNet.FederatedLearning.Selection;

/// <summary>
/// Weighted random selection without replacement (typically weighted by sample count).
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> WeightedRandomClientSelectionStrategy provides AI safety functionality. Default values follow the original paper settings.</para>
/// </remarks>
public sealed class WeightedRandomClientSelectionStrategy : ClientSelectionStrategyBase
{
    public override List<int> SelectClients(ClientSelectionRequest request)
    {
        if (request == null)
        {
            throw new ArgumentNullException(nameof(request));
        }

        var candidates = (request.CandidateClientIds ?? Array.Empty<int>()).Distinct().ToList();
        int desired = GetDesiredClientCount(candidates, request.FractionToSelect);
        if (desired >= candidates.Count)
        {
            candidates.Sort();
            return candidates;
        }

        var weights = request.ClientWeights ?? new Dictionary<int, double>();
        var remaining = candidates;
        var selected = new List<int>(desired);
        double total = remaining
            .Select(id => (weights.TryGetValue(id, out var w) && w > 0.0) ? w : 0.0)
            .Sum();

        for (int k = 0; k < desired; k++)
        {
            if (total <= 0.0)
            {
                // Fallback to uniform if no usable weights.
                selected.AddRange(ShuffleAndTake(remaining, desired - selected.Count, request.Random));
                break;
            }

            double r = request.Random.NextDouble() * total;
            double cumulative = 0.0;
            int chosen = remaining[0];
            int lastPositive = chosen;
            foreach (var id in remaining)
            {
                double w = (weights.TryGetValue(id, out var ww) && ww > 0.0) ? ww : 0.0;
                if (w > 0.0)
                {
                    lastPositive = id;
                }

                cumulative += w;
                if (cumulative >= r)
                {
                    chosen = id;
                    break;
                }
            }

            if (cumulative < r)
            {
                chosen = lastPositive;
            }

            selected.Add(chosen);
            remaining.Remove(chosen);

            if (weights.TryGetValue(chosen, out var chosenWeight) && chosenWeight > 0.0)
            {
                total = Math.Max(0.0, total - chosenWeight);
            }
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
