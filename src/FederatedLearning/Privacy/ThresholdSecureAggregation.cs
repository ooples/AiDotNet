using AiDotNet.FederatedLearning.Infrastructure;

namespace AiDotNet.FederatedLearning.Privacy;

/// <summary>
/// Implements dropout-resilient secure aggregation for structured (layered) model updates.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Some models expose parameters as a dictionary of named arrays (for example, one array per layer).
/// This wrapper adapts that representation to the vector-based secure aggregation core.
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters (e.g., double, float).</typeparam>
public sealed class ThresholdSecureAggregation<T> : FederatedLearningComponentBase<T>, IDisposable
{
    private readonly ThresholdSecureAggregationVector<T> _inner;
    private readonly int _parameterCount;
    private bool _disposed;

    public ThresholdSecureAggregation(int parameterCount, int? randomSeed = null)
    {
        if (parameterCount <= 0)
        {
            throw new ArgumentException("Parameter count must be positive.", nameof(parameterCount));
        }

        _parameterCount = parameterCount;
        _inner = new ThresholdSecureAggregationVector<T>(parameterCount, randomSeed);
    }

    public int MinimumUploaderCount => _inner.MinimumUploaderCount;

    public int ReconstructionThreshold => _inner.ReconstructionThreshold;

    public void InitializeRound(
        List<int> clientIds,
        int minimumUploaderCount = 0,
        int reconstructionThreshold = 0,
        double maxDropoutFraction = 0.2)
    {
        ThrowIfDisposed();
        _inner.InitializeRound(clientIds, minimumUploaderCount, reconstructionThreshold, maxDropoutFraction);
    }

    public Dictionary<string, T[]> MaskUpdate(int clientId, Dictionary<string, T[]> clientUpdate)
    {
        ThrowIfDisposed();
        return MaskUpdateInternal(clientId, clientUpdate, clientWeight: null);
    }

    public Dictionary<string, T[]> MaskUpdate(int clientId, Dictionary<string, T[]> clientUpdate, double clientWeight)
    {
        ThrowIfDisposed();
        if (clientWeight <= 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(clientWeight), "Client weight must be positive.");
        }

        return MaskUpdateInternal(clientId, clientUpdate, clientWeight);
    }

    public Dictionary<string, T[]> AggregateSecurely(
        Dictionary<int, Dictionary<string, T[]>> maskedUpdates,
        Dictionary<int, double> clientWeights)
    {
        ThrowIfDisposed();
        return AggregateSecurely(maskedUpdates, clientWeights, unmaskingClientIds: null);
    }

    public Dictionary<string, T[]> AggregateSecurely(
        Dictionary<int, Dictionary<string, T[]>> maskedUpdates,
        Dictionary<int, double> clientWeights,
        IReadOnlyCollection<int>? unmaskingClientIds)
    {
        ThrowIfDisposed();

        if (clientWeights == null || clientWeights.Count == 0)
        {
            throw new ArgumentException("Client weights cannot be null or empty.", nameof(clientWeights));
        }

        var sum = AggregateSumSecurely(maskedUpdates, unmaskingClientIds);

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

        var totalWeightT = NumOps.FromDouble(totalWeight);
        foreach (var aggregatedParams in sum.Values)
        {
            for (int i = 0; i < aggregatedParams.Length; i++)
            {
                aggregatedParams[i] = NumOps.Divide(aggregatedParams[i], totalWeightT);
            }
        }

        return sum;
    }

    public Dictionary<string, T[]> AggregateSumSecurely(Dictionary<int, Dictionary<string, T[]>> maskedUpdates)
    {
        ThrowIfDisposed();
        return AggregateSumSecurely(maskedUpdates, unmaskingClientIds: null);
    }

    public Dictionary<string, T[]> AggregateSumSecurely(
        Dictionary<int, Dictionary<string, T[]>> maskedUpdates,
        IReadOnlyCollection<int>? unmaskingClientIds)
    {
        ThrowIfDisposed();
        if (maskedUpdates == null || maskedUpdates.Count == 0)
        {
            throw new ArgumentException("Masked updates cannot be null or empty.", nameof(maskedUpdates));
        }

        var firstUpdate = maskedUpdates.First().Value;
        if (firstUpdate == null || firstUpdate.Count == 0)
        {
            throw new ArgumentException("Masked updates cannot contain empty client updates.", nameof(maskedUpdates));
        }

        ValidateStructure(maskedUpdates, firstUpdate);

        var maskedVectors = new Dictionary<int, Vector<T>>(maskedUpdates.Count);
        foreach (var (clientId, update) in maskedUpdates)
        {
            maskedVectors[clientId] = new Vector<T>(FlattenParameters(update));
        }

        var summed = _inner.AggregateSumSecurely(maskedVectors, unmaskingClientIds);
        return UnflattenParameters(firstUpdate, summed);
    }

    public void ClearSecrets()
    {
        ThrowIfDisposed();
        _inner.ClearSecrets();
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _inner.Dispose();
        _disposed = true;
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(ThresholdSecureAggregation<T>));
        }
    }

    private Dictionary<string, T[]> MaskUpdateInternal(int clientId, Dictionary<string, T[]> clientUpdate, double? clientWeight)
    {
        if (clientUpdate == null || clientUpdate.Count == 0)
        {
            throw new ArgumentException("Client update cannot be null or empty.", nameof(clientUpdate));
        }

        var flatParams = FlattenParameters(clientUpdate);
        EnsureParameterCountMatches(flatParams.Length, nameof(clientUpdate));

        var maskedVector = clientWeight.HasValue
            ? _inner.MaskUpdate(clientId, new Vector<T>(flatParams), clientWeight.Value)
            : _inner.MaskUpdate(clientId, new Vector<T>(flatParams));

        return UnflattenParameters(clientUpdate, maskedVector);
    }

    private void EnsureParameterCountMatches(int actualCount, string paramName)
    {
        if (actualCount != _parameterCount)
        {
            throw new ArgumentException(
                $"Model parameter count mismatch. Expected {_parameterCount}, got {actualCount}. " +
                "Ensure ThresholdSecureAggregation is constructed with the correct parameterCount.",
                paramName);
        }
    }

    private static void ValidateStructure(
        Dictionary<int, Dictionary<string, T[]>> maskedUpdates,
        Dictionary<string, T[]> reference)
    {
        foreach (var kvp in maskedUpdates)
        {
            int clientId = kvp.Key;
            var update = kvp.Value;

            if (update == null || update.Count == 0)
            {
                throw new ArgumentException($"Masked update for client {clientId} cannot be null or empty.", nameof(maskedUpdates));
            }

            if (update.Count != reference.Count)
            {
                throw new InvalidOperationException(
                    $"Client {clientId} has inconsistent model structure. Expected {reference.Count} layers, got {update.Count}.");
            }

            foreach (var layerName in reference.Keys)
            {
                if (!update.TryGetValue(layerName, out var clientLayer) || clientLayer == null)
                {
                    throw new InvalidOperationException(
                        $"Client {clientId} is missing layer '{layerName}' or it is null. All clients must send the same layer names.");
                }

                int expectedLength = reference[layerName].Length;
                if (clientLayer.Length != expectedLength)
                {
                    throw new InvalidOperationException(
                        $"Client {clientId} layer '{layerName}' has {clientLayer.Length} parameters but expected {expectedLength}.");
                }
            }
        }
    }

    private static T[] FlattenParameters(Dictionary<string, T[]> parameters)
    {
        var keys = parameters.Keys.OrderBy(name => name, StringComparer.Ordinal).ToArray();
        int total = 0;
        foreach (var k in keys)
        {
            total += parameters[k].Length;
        }

        var flat = new T[total];
        int idx = 0;
        foreach (var k in keys)
        {
            var layer = parameters[k];
            Array.Copy(layer, 0, flat, idx, layer.Length);
            idx += layer.Length;
        }

        return flat;
    }

    private static Dictionary<string, T[]> UnflattenParameters(Dictionary<string, T[]> template, Vector<T> flat)
    {
        var output = new Dictionary<string, T[]>();
        int idx = 0;
        foreach (var layerName in template.Keys.OrderBy(name => name, StringComparer.Ordinal))
        {
            var layerTemplate = template[layerName];
            var layer = new T[layerTemplate.Length];
            for (int i = 0; i < layer.Length; i++, idx++)
            {
                layer[i] = flat[idx];
            }

            output[layerName] = layer;
        }

        return output;
    }
}

