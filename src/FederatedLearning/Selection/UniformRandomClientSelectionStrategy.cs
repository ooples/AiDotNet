using AiDotNet.Models;

namespace AiDotNet.FederatedLearning.Selection;

/// <summary>
/// Uniform random client selection (fractional participation).
/// </summary>
public sealed class UniformRandomClientSelectionStrategy : ClientSelectionStrategyBase
{
    public override List<int> SelectClients(ClientSelectionRequest request)
    {
        if (request == null)
        {
            throw new ArgumentNullException(nameof(request));
        }

        var candidates = request.CandidateClientIds ?? Array.Empty<int>();
        int desired = GetDesiredClientCount(candidates, request.FractionToSelect);
        return ShuffleAndTake(candidates, desired, request.Random);
    }

    public override string GetStrategyName() => "UniformRandom";
}

