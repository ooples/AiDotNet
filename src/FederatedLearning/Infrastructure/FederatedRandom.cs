using AiDotNet.Tensors.Helpers;

namespace AiDotNet.FederatedLearning.Infrastructure;

/// <summary>
/// Provides deterministic random sources for federated learning simulation.
/// </summary>
internal static class FederatedRandom
{
    public static Random CreateRoundRandom(int? baseSeed, int roundNumber, int salt = 0)
    {
        if (!baseSeed.HasValue)
        {
            return RandomHelper.ThreadSafeRandom;
        }

        unchecked
        {
            // Stable mixing without relying on HashCode.Combine (which is randomized per process).
            const int goldenRatioPrime = unchecked((int)0x9E3779B9);
            const int avalanchePrime = unchecked((int)0x85EBCA6B);

            int seed = baseSeed.Value;
            seed ^= (roundNumber + 1) * goldenRatioPrime;
            seed ^= salt * avalanchePrime;
            seed ^= (seed << 13);
            seed ^= (seed >> 17);
            seed ^= (seed << 5);
            return RandomHelper.CreateSeededRandom(seed);
        }
    }

    public static Random CreateClientRandom(int? baseSeed, int roundNumber, int clientId, int salt = 0)
    {
        if (!baseSeed.HasValue)
        {
            return RandomHelper.ThreadSafeRandom;
        }

        unchecked
        {
            const int goldenRatioPrime = unchecked((int)0x9E3779B9);
            const int avalanchePrime = unchecked((int)0x85EBCA6B);
            const int clientPrime = unchecked((int)0xC2B2AE35);

            int seed = baseSeed.Value;
            seed ^= (roundNumber + 1) * goldenRatioPrime;
            seed ^= clientId * clientPrime;
            seed ^= salt * avalanchePrime;
            seed ^= (seed << 13);
            seed ^= (seed >> 17);
            seed ^= (seed << 5);
            return RandomHelper.CreateSeededRandom(seed);
        }
    }
}
