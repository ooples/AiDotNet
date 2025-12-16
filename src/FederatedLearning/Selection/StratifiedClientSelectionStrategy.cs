using AiDotNet.Models;

namespace AiDotNet.FederatedLearning.Selection;

/// <summary>
/// Stratified client selection using a client-to-group mapping.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> If clients are split into groups (for example by region or device type),
/// stratified sampling tries to pick clients from each group instead of accidentally picking only one group.
/// </remarks>
public sealed class StratifiedClientSelectionStrategy : ClientSelectionStrategyBase
{
    public override List<int> SelectClients(ClientSelectionRequest request)
    {
        if (request == null)
        {
            throw new ArgumentNullException(nameof(request));
        }

        var candidates = (request.CandidateClientIds ?? Array.Empty<int>()).Distinct().ToArray();
        int desired = GetDesiredClientCount(candidates, request.FractionToSelect);

        var groupKeys = request.ClientGroupKeys;
        if (groupKeys == null || groupKeys.Count == 0)
        {
            return ShuffleAndTake(candidates, desired, request.Random);
        }

        var groups = new Dictionary<string, List<int>>(StringComparer.OrdinalIgnoreCase);
        foreach (var id in candidates)
        {
            if (!groupKeys.TryGetValue(id, out var group) || string.IsNullOrWhiteSpace(group))
            {
                group = "default";
            }

            if (!groups.TryGetValue(group, out var list))
            {
                list = new List<int>();
                groups[group] = list;
            }

            list.Add(id);
        }

        if (groups.Count == 1)
        {
            return ShuffleAndTake(candidates, desired, request.Random);
        }

        int total = candidates.Length;
        var groupAllocations = new List<(string Group, int Count)>(groups.Count);
        int allocated = 0;

        foreach (var kvp in groups.OrderBy(k => k.Key, StringComparer.OrdinalIgnoreCase))
        {
            int groupSize = kvp.Value.Count;
            int count = Math.Max(1, (int)Math.Round((double)groupSize / total * desired));
            count = Math.Min(count, groupSize);
            groupAllocations.Add((kvp.Key, count));
            allocated += count;
        }

        // Adjust allocations to match desired count.
        while (allocated > desired)
        {
            for (int i = 0; i < groupAllocations.Count && allocated > desired; i++)
            {
                if (groupAllocations[i].Count > 1)
                {
                    groupAllocations[i] = (groupAllocations[i].Group, groupAllocations[i].Count - 1);
                    allocated--;
                }
            }

            // If all are at 1 and still too many, reduce any groups that can reduce.
            if (allocated > desired)
            {
                for (int i = 0; i < groupAllocations.Count && allocated > desired; i++)
                {
                    if (groupAllocations[i].Count > 0)
                    {
                        groupAllocations[i] = (groupAllocations[i].Group, groupAllocations[i].Count - 1);
                        allocated--;
                    }
                }
            }
        }

        while (allocated < desired)
        {
            foreach (var (group, _) in groupAllocations.ToArray())
            {
                if (allocated >= desired)
                {
                    break;
                }

                int idx = groupAllocations.FindIndex(g => string.Equals(g.Group, group, StringComparison.OrdinalIgnoreCase));
                int current = groupAllocations[idx].Count;
                int max = groups[group].Count;
                if (current < max)
                {
                    groupAllocations[idx] = (group, current + 1);
                    allocated++;
                }
            }

            if (allocated < desired && groupAllocations.Sum(g => g.Count) == allocated)
            {
                // No capacity left.
                break;
            }
        }

        var selected = new List<int>(desired);
        foreach (var (group, count) in groupAllocations)
        {
            if (count <= 0)
            {
                continue;
            }

            selected.AddRange(ShuffleAndTake(groups[group], Math.Min(count, groups[group].Count), request.Random));
        }

        if (selected.Count < desired)
        {
            var remaining = candidates.Except(selected).ToList();
            if (remaining.Count > 0)
            {
                selected.AddRange(ShuffleAndTake(remaining, Math.Min(desired - selected.Count, remaining.Count), request.Random));
            }
        }

        selected = selected.Distinct().ToList();
        if (selected.Count > desired)
        {
            selected = selected.Take(desired).ToList();
        }

        selected.Sort();
        return selected;
    }

    public override string GetStrategyName() => "Stratified";
}
