namespace AiDotNet.Tests.FederatedLearning;

using AiDotNet.FederatedLearning.Selection;
using AiDotNet.Models;

internal sealed class ClientSelectionStrategyBaseAccessor : ClientSelectionStrategyBase
{
    public override List<int> SelectClients(ClientSelectionRequest request)
    {
        throw new NotSupportedException("Test accessor only.");
    }

    public override string GetStrategyName() => "Accessor";

    public static int CallGetDesiredClientCount(IReadOnlyList<int> candidates, double fraction)
    {
        return GetDesiredClientCount(candidates, fraction);
    }

    public static List<int> CallShuffleAndTake(IReadOnlyList<int> items, int count, Random random)
    {
        return ShuffleAndTake(items, count, random);
    }
}

