using System.Security.Cryptography;
using AiDotNet.FederatedLearning.Infrastructure;

namespace AiDotNet.FederatedLearning.Privacy;

/// <summary>
/// Implements secure aggregation for vector-based model updates.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Secure aggregation lets the server compute the sum/average of client updates
/// without learning any single client's update. Clients add pairwise "masks" that cancel out in the sum.
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters (e.g., double, float).</typeparam>
public sealed class SecureAggregationVector<T> : FederatedLearningComponentBase<T>
{
    private readonly Dictionary<int, Dictionary<int, T[]>> _pairwiseSecrets;
    private readonly Random? _testRandom;
    private readonly RandomNumberGenerator? _secureRandom;
    private readonly int _parameterCount;

    public SecureAggregationVector(int parameterCount, int? randomSeed = null)
    {
        if (parameterCount <= 0)
        {
            throw new ArgumentException("Parameter count must be positive.", nameof(parameterCount));
        }

        _parameterCount = parameterCount;
        _pairwiseSecrets = new Dictionary<int, Dictionary<int, T[]>>();

        if (randomSeed.HasValue)
        {
            _testRandom = new Random(randomSeed.Value);
            _secureRandom = null;
        }
        else
        {
            _testRandom = null;
            _secureRandom = RandomNumberGenerator.Create();
        }
    }

    public void GeneratePairwiseSecrets(List<int> clientIds)
    {
        if (clientIds == null || clientIds.Count < 2)
        {
            throw new ArgumentException("At least two clients are required for secure aggregation.", nameof(clientIds));
        }

        clientIds = clientIds.Distinct().OrderBy(id => id).ToList();
        ClearSecrets();

        foreach (var clientI in clientIds)
        {
            _pairwiseSecrets[clientI] = new Dictionary<int, T[]>();
        }

        for (int i = 0; i < clientIds.Count; i++)
        {
            for (int j = i + 1; j < clientIds.Count; j++)
            {
                int clientI = clientIds[i];
                int clientJ = clientIds[j];

                var secret = new T[_parameterCount];
                for (int k = 0; k < _parameterCount; k++)
                {
                    secret[k] = NumOps.FromDouble((NextUnitIntervalDouble() - 0.5) * 2.0);
                }

                _pairwiseSecrets[clientI][clientJ] = secret;

                var neg = new T[_parameterCount];
                for (int k = 0; k < _parameterCount; k++)
                {
                    neg[k] = NumOps.Negate(secret[k]);
                }
                _pairwiseSecrets[clientJ][clientI] = neg;
            }
        }
    }

    public Vector<T> MaskUpdate(int clientId, Vector<T> clientUpdate)
    {
        return MaskUpdateInternal(clientId, clientUpdate, clientWeight: null);
    }

    public Vector<T> MaskUpdate(int clientId, Vector<T> clientUpdate, double clientWeight)
    {
        if (clientWeight <= 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(clientWeight), "Client weight must be positive.");
        }

        return MaskUpdateInternal(clientId, clientUpdate, clientWeight);
    }

    public Vector<T> AggregateSecurely(Dictionary<int, Vector<T>> maskedUpdates, Dictionary<int, double> clientWeights)
    {
        if (clientWeights == null || clientWeights.Count == 0)
        {
            throw new ArgumentException("Client weights cannot be null or empty.", nameof(clientWeights));
        }

        var sum = AggregateSumSecurely(maskedUpdates);

        double totalWeight = 0.0;
        foreach (var clientId in maskedUpdates.Keys)
        {
            if (!clientWeights.TryGetValue(clientId, out var w))
            {
                throw new ArgumentException($"Missing weight for client {clientId}.", nameof(clientWeights));
            }

            totalWeight += w;
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
        if (maskedUpdates == null || maskedUpdates.Count == 0)
        {
            throw new ArgumentException("Masked updates cannot be null or empty.", nameof(maskedUpdates));
        }

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
        _pairwiseSecrets.Clear();
    }

    private Vector<T> MaskUpdateInternal(int clientId, Vector<T> clientUpdate, double? clientWeight)
    {
        if (clientUpdate == null)
        {
            throw new ArgumentNullException(nameof(clientUpdate));
        }

        EnsureParameterCountMatches(clientUpdate.Length, nameof(clientUpdate));

        if (!_pairwiseSecrets.TryGetValue(clientId, out var secretsForClient))
        {
            throw new ArgumentException($"No secrets found for client {clientId}. Call GeneratePairwiseSecrets first.", nameof(clientId));
        }

        var masked = new T[_parameterCount];
        for (int i = 0; i < _parameterCount; i++)
        {
            masked[i] = clientUpdate[i];
        }

        if (clientWeight.HasValue)
        {
            var w = NumOps.FromDouble(clientWeight.Value);
            for (int i = 0; i < _parameterCount; i++)
            {
                masked[i] = NumOps.Multiply(masked[i], w);
            }
        }

        foreach (var secret in secretsForClient.Values)
        {
            if (secret.Length != _parameterCount)
            {
                throw new InvalidOperationException($"Pairwise secret length mismatch. Expected {_parameterCount}, got {secret.Length}.");
            }

            for (int i = 0; i < _parameterCount; i++)
            {
                masked[i] = NumOps.Add(masked[i], secret[i]);
            }
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

    private double NextUnitIntervalDouble()
    {
        if (_testRandom != null)
        {
            return _testRandom.NextDouble();
        }

        var bytes = new byte[8];
        _secureRandom!.GetBytes(bytes);
        ulong value = BitConverter.ToUInt64(bytes, 0) >> 11;
        return value / (double)(1UL << 53);
    }
}
