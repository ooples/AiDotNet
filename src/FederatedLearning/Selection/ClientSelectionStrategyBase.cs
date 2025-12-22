using AiDotNet.Interfaces;
using AiDotNet.Models;

namespace AiDotNet.FederatedLearning.Selection;

/// <summary>
/// Base class for client selection strategies.
/// </summary>
public abstract class ClientSelectionStrategyBase : IClientSelectionStrategy
{
    public abstract List<int> SelectClients(ClientSelectionRequest request);

    public abstract string GetStrategyName();

    protected static int GetDesiredClientCount(IReadOnlyList<int> candidates, double fraction)
    {
        if (candidates == null)
        {
            throw new ArgumentNullException(nameof(candidates));
        }

        if (fraction <= 0.0 || fraction > 1.0)
        {
            throw new ArgumentOutOfRangeException(nameof(fraction), "Fraction must be in (0, 1].");
        }

        return Math.Max(1, (int)Math.Ceiling(candidates.Count * fraction));
    }

    protected static List<int> ShuffleAndTake(IReadOnlyList<int> items, int count, Random random)
    {
        if (items == null)
        {
            throw new ArgumentNullException(nameof(items));
        }

        if (random == null)
        {
            throw new ArgumentNullException(nameof(random));
        }

        if (count <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(count), "Count must be positive.");
        }

        if (count >= items.Count)
        {
            return items.OrderBy(i => i).ToList();
        }

        var shuffled = items.OrderBy(_ => random.Next()).ToList();
        var selected = shuffled.Take(count).ToList();
        selected.Sort();
        return selected;
    }
}

