using System.Security.Cryptography;
using System.Text;
using AiDotNet.FederatedLearning.Cryptography;
using AiDotNet.FederatedLearning.Infrastructure;

namespace AiDotNet.FederatedLearning.Privacy;

/// <summary>
/// Implements secure aggregation for vector-based model updates.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Secure aggregation lets the server compute the sum/average of client updates
/// without learning any single client's update. Each client adds pairwise masks derived from shared secrets.
/// The masks are constructed so they cancel out in the aggregate.
///
/// This implementation provides a synchronous, full-participation secure aggregation mode. If a client
/// drops out after masks are created, the round must be restarted (dropout-resilient unmasking is a separate
/// protocol step and is intentionally not part of this in-memory component).
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters (e.g., double, float).</typeparam>
public sealed class SecureAggregationVector<T> : FederatedLearningComponentBase<T>, IDisposable
{
    private readonly Dictionary<long, byte[]> _pairwiseMaskSeeds = new();
    private readonly int? _deterministicSeed;
    private int[] _clientIds = Array.Empty<int>();
    private readonly int _parameterCount;
    private bool _disposed;

    public SecureAggregationVector(int parameterCount, int? randomSeed = null)
    {
        if (parameterCount <= 0)
        {
            throw new ArgumentException("Parameter count must be positive.", nameof(parameterCount));
        }

        _parameterCount = parameterCount;
        _deterministicSeed = randomSeed;
    }

    ~SecureAggregationVector()
    {
        Dispose(disposing: false);
    }

    public void GeneratePairwiseSecrets(List<int> clientIds)
    {
        ThrowIfDisposed();

        if (clientIds == null || clientIds.Count < 2)
        {
            throw new ArgumentException("At least two clients are required for secure aggregation.", nameof(clientIds));
        }

        var distinct = clientIds
            .Where(id => id >= 0)
            .Distinct()
            .OrderBy(id => id)
            .ToArray();

        if (distinct.Length < 2)
        {
            throw new ArgumentException("At least two non-negative clients are required for secure aggregation.", nameof(clientIds));
        }

        ClearSecrets();
        _clientIds = distinct;

        if (_deterministicSeed.HasValue)
        {
            GenerateDeterministicPairwiseSeeds(distinct, _deterministicSeed.Value);
            return;
        }

        GenerateEphemeralPairwiseSeeds(distinct);
    }

    public Vector<T> MaskUpdate(int clientId, Vector<T> clientUpdate)
    {
        ThrowIfDisposed();
        return MaskUpdateInternal(clientId, clientUpdate, clientWeight: null);
    }

    public Vector<T> MaskUpdate(int clientId, Vector<T> clientUpdate, double clientWeight)
    {
        ThrowIfDisposed();
        if (clientWeight <= 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(clientWeight), "Client weight must be positive.");
        }

        return MaskUpdateInternal(clientId, clientUpdate, clientWeight);
    }

    public Vector<T> AggregateSecurely(Dictionary<int, Vector<T>> maskedUpdates, Dictionary<int, double> clientWeights)
    {
        ThrowIfDisposed();
        if (clientWeights == null || clientWeights.Count == 0)
        {
            throw new ArgumentException("Client weights cannot be null or empty.", nameof(clientWeights));
        }

        EnsureFullParticipation(maskedUpdates);

        var sum = AggregateSumSecurely(maskedUpdates);

        double totalWeight = 0.0;
        foreach (var clientId in maskedUpdates.Keys)
        {
            if (!clientWeights.TryGetValue(clientId, out var weight))
            {
                throw new ArgumentException($"Missing weight for client {clientId}.", nameof(clientWeights));
            }

            totalWeight += weight;
        }

        if (totalWeight <= 0.0)
        {
            throw new ArgumentException("Total weight must be positive.", nameof(clientWeights));
        }

        var invTotal = NumOps.FromDouble(1.0 / totalWeight);
        for (int i = 0; i < sum.Length; i++)
        {
            sum[i] = NumOps.Multiply(sum[i], invTotal);
        }

        return sum;
    }

    public Vector<T> AggregateSumSecurely(Dictionary<int, Vector<T>> maskedUpdates)
    {
        ThrowIfDisposed();
        if (maskedUpdates == null || maskedUpdates.Count == 0)
        {
            throw new ArgumentException("Masked updates cannot be null or empty.", nameof(maskedUpdates));
        }

        EnsureFullParticipation(maskedUpdates);

        var first = maskedUpdates.First().Value;
        EnsureParameterCountMatches(first.Length, nameof(maskedUpdates));

        var sum = new Vector<T>(first.Length);
        for (int i = 0; i < sum.Length; i++)
        {
            sum[i] = NumOps.Zero;
        }

        foreach (var update in maskedUpdates.Values)
        {
            EnsureParameterCountMatches(update.Length, nameof(maskedUpdates));

            for (int i = 0; i < sum.Length; i++)
            {
                sum[i] = NumOps.Add(sum[i], update[i]);
            }
        }

        return sum;
    }

    public void ClearSecrets()
    {
        ThrowIfDisposed();
        _pairwiseMaskSeeds.Clear();
        _clientIds = Array.Empty<int>();
    }

    public void Dispose()
    {
        Dispose(disposing: true);
        GC.SuppressFinalize(this);
    }

    private void Dispose(bool disposing)
    {
        if (_disposed)
        {
            return;
        }

        if (disposing)
        {
            _pairwiseMaskSeeds.Clear();
            _clientIds = Array.Empty<int>();
        }

        _disposed = true;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(SecureAggregationVector<T>));
        }
    }

    private Vector<T> MaskUpdateInternal(int clientId, Vector<T> clientUpdate, double? clientWeight)
    {
        if (clientUpdate == null)
        {
            throw new ArgumentNullException(nameof(clientUpdate));
        }

        EnsureParameterCountMatches(clientUpdate.Length, nameof(clientUpdate));
        EnsureClientIsKnown(clientId);

        var masked = new T[_parameterCount];
        for (int i = 0; i < _parameterCount; i++)
        {
            masked[i] = clientUpdate[i];
        }

        if (clientWeight.HasValue)
        {
            var weightT = NumOps.FromDouble(clientWeight.Value);
            for (int i = 0; i < _parameterCount; i++)
            {
                masked[i] = NumOps.Multiply(masked[i], weightT);
            }
        }

        foreach (var otherClientId in _clientIds)
        {
            if (otherClientId == clientId)
            {
                continue;
            }

            int min = Math.Min(clientId, otherClientId);
            int max = Math.Max(clientId, otherClientId);
            bool addMask = clientId == min;

            var seed = GetPairwiseSeed(min, max);
            ApplyPairwiseMask(seed, addMask, masked);
        }

        return new Vector<T>(masked);
    }

    private void EnsureParameterCountMatches(int actualCount, string paramName)
    {
        if (actualCount != _parameterCount)
        {
            throw new ArgumentException($"Parameter count mismatch. Expected {_parameterCount}, got {actualCount}.", paramName);
        }
    }

    private void EnsureClientIsKnown(int clientId)
    {
        if (_clientIds.Length == 0)
        {
            throw new ArgumentException($"No secrets found for client {clientId}. Call GeneratePairwiseSecrets first.", nameof(clientId));
        }

        if (Array.IndexOf(_clientIds, clientId) < 0)
        {
            throw new ArgumentException($"No secrets found for client {clientId}. Call GeneratePairwiseSecrets first.", nameof(clientId));
        }
    }

    private void EnsureFullParticipation(Dictionary<int, Vector<T>> maskedUpdates)
    {
        if (_clientIds.Length == 0)
        {
            throw new InvalidOperationException("Secure aggregation secrets are not initialized. Call GeneratePairwiseSecrets first.");
        }

        if (maskedUpdates.Count != _clientIds.Length)
        {
            throw new InvalidOperationException(
                $"Secure aggregation requires full participation for this mode. Expected {_clientIds.Length} masked updates, got {maskedUpdates.Count}.");
        }

        foreach (var clientId in _clientIds)
        {
            if (!maskedUpdates.ContainsKey(clientId))
            {
                throw new InvalidOperationException(
                    $"Secure aggregation requires full participation for this mode. Missing masked update for client {clientId}.");
            }
        }
    }

    private byte[] GetPairwiseSeed(int minClientId, int maxClientId)
    {
        var key = PairKey(minClientId, maxClientId);
        if (!_pairwiseMaskSeeds.TryGetValue(key, out var seed))
        {
            throw new InvalidOperationException($"Missing pairwise mask seed for pair ({minClientId},{maxClientId}).");
        }

        return seed;
    }

    private void ApplyPairwiseMask(byte[] seed, bool addMask, T[] masked)
    {
        using var prg = new HmacSha256Prg(seed);
        for (int i = 0; i < _parameterCount; i++)
        {
            var mask = NumOps.FromDouble((prg.NextUnitIntervalDouble() - 0.5) * 2.0);
            masked[i] = addMask ? NumOps.Add(masked[i], mask) : NumOps.Subtract(masked[i], mask);
        }
    }

    private void GenerateDeterministicPairwiseSeeds(int[] sortedClientIds, int baseSeed)
    {
        foreach (var (min, max) in EnumeratePairs(sortedClientIds))
        {
            _pairwiseMaskSeeds[PairKey(min, max)] = DeriveDeterministicSeed(baseSeed, min, max);
        }
    }

    private void GenerateEphemeralPairwiseSeeds(int[] sortedClientIds)
    {
        var keyPairs = new Dictionary<int, ECDiffieHellman>();
        try
        {
            foreach (var clientId in sortedClientIds)
            {
                keyPairs[clientId] = ECDiffieHellman.Create();
            }

            var salt = Encoding.UTF8.GetBytes("AiDotNet.SecAgg.v1");
            foreach (var (min, max) in EnumeratePairs(sortedClientIds))
            {
                var shared = keyPairs[min].DeriveKeyMaterial(keyPairs[max].PublicKey);
                try
                {
                    var info = Encoding.UTF8.GetBytes($"pair:{min}:{max}");
                    _pairwiseMaskSeeds[PairKey(min, max)] = HkdfSha256.DeriveKey(shared, salt, info, length: 32);
                }
                finally
                {
                    Array.Clear(shared, 0, shared.Length);
                }
            }
        }
        finally
        {
            foreach (var kp in keyPairs.Values)
            {
                kp.Dispose();
            }
        }
    }

    private static IEnumerable<(int min, int max)> EnumeratePairs(int[] sortedClientIds)
    {
        for (int i = 0; i < sortedClientIds.Length; i++)
        {
            for (int j = i + 1; j < sortedClientIds.Length; j++)
            {
                yield return (sortedClientIds[i], sortedClientIds[j]);
            }
        }
    }

    private static long PairKey(int minClientId, int maxClientId)
    {
        unchecked
        {
            return ((long)minClientId << 32) | (uint)maxClientId;
        }
    }

    private static byte[] DeriveDeterministicSeed(int baseSeed, int minClientId, int maxClientId)
    {
        var keyBytes = BitConverter.GetBytes(baseSeed);
        var data = Encoding.UTF8.GetBytes($"AiDotNet.SecAgg.v1:pair:{minClientId}:{maxClientId}");
        using var hmac = new HMACSHA256(keyBytes);
        return hmac.ComputeHash(data);
    }
}
