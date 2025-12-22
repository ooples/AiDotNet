namespace AiDotNet.FederatedLearning.Privacy;

using System;
using System.Security.Cryptography;
using System.Text;
using AiDotNet.FederatedLearning.Cryptography;
using AiDotNet.FederatedLearning.Infrastructure;

/// <summary>
/// Implements secure aggregation for federated learning using cryptographic techniques.
/// </summary>
/// <remarks>
/// Secure aggregation is a cryptographic protocol that allows a server to compute the sum
/// of client updates without seeing individual contributions. Only the final aggregate is
/// visible to the server.
///
/// <b>For Beginners:</b> Secure aggregation is like a secret ballot election where votes
/// are counted but individual votes remain private.
///
/// How it works (simplified):
/// 1. Each client generates pairwise secret keys with other clients
/// 2. Clients mask their model updates with these secret keys
/// 3. Server receives masked updates: masked_update_i = update_i + Σ(secrets_ij)
/// 4. Secret masks cancel out when summing: Σ(masked_update_i) = Σ(update_i)
/// 5. Server gets the sum without seeing individual updates
///
/// Example with 3 clients:
/// - Client 1 shares secrets: s₁₂ with Client 2, s₁₃ with Client 3
/// - Client 2 shares secrets: s₂₁ with Client 1, s₂₃ with Client 3
/// - Client 3 shares secrets: s₃₁ with Client 1, s₃₂ with Client 2
///
/// Note: s₁₂ = -s₂₁ (secrets cancel in pairs)
///
/// Client 1 sends: update₁ + s₁₂ + s₁₃
/// Client 2 sends: update₂ + s₂₁ + s₂₃
/// Client 3 sends: update₃ + s₃₁ + s₃₂
///
/// Server computes sum:
/// (update₁ + s₁₂ + s₁₃) + (update₂ + s₂₁ + s₂₃) + (update₃ + s₃₁ + s₃₂)
/// = update₁ + update₂ + update₃ + (s₁₂ + s₂₁) + (s₁₃ + s₃₁) + (s₂₃ + s₃₂)
/// = update₁ + update₂ + update₃ + 0 + 0 + 0
/// = Σ(updates) ← Only this is visible to server!
///
/// This implementation derives pairwise mask seeds from per-round ephemeral ECDH shared secrets and expands them
/// via HKDF + a deterministic PRG. Pairwise masks cancel in the aggregate as long as all selected clients participate
/// in the round (synchronous, full-participation mode).
///
/// Benefits:
/// - Server cannot see individual client updates
/// - Protects against honest-but-curious server
/// - No trusted third party needed
/// - Computation overhead is reasonable
///
/// Limitations:
/// - Requires coordination between clients
/// - All (or threshold) clients must participate for masks to cancel
/// - Dropout handling requires additional mechanisms
/// - Communication overhead for key exchange
///
/// When to use Secure Aggregation:
/// - Don't fully trust the central server
/// - Regulatory requirements for data protection
/// - Want cryptographic privacy guarantees
/// - Willing to handle additional complexity
///
/// Can be combined with differential privacy for stronger protection:
/// - Secure aggregation: Protects individual updates from server
/// - Differential privacy: Protects individual data points from anyone
///
/// Reference: Bonawitz, K., et al. (2017). "Practical Secure Aggregation for Privacy-Preserving
/// Machine Learning." CCS 2017.
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters (e.g., double, float).</typeparam>
public class SecureAggregation<T> : FederatedLearningComponentBase<T>, IDisposable
{
    private readonly Dictionary<long, byte[]> _pairwiseMaskSeeds;
    private readonly int? _deterministicSeed;
    private int[] _clientIds = Array.Empty<int>();
    private readonly int _parameterCount;
    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of the <see cref="SecureAggregation{T}"/> class.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Sets up the secure aggregation protocol for a specific
    /// number of model parameters.
    ///
    /// In practice, this would involve:
    /// - Secure key exchange between clients
    /// - Authenticated channels
    /// - Agreement on random seed for deterministic mask generation
    ///
    /// In this in-memory implementation, pairwise masks are generated per-round and must be re-generated for each round.
    /// </remarks>
    /// <param name="parameterCount">The total number of model parameters to protect.</param>
    /// <param name="randomSeed">Optional random seed for reproducibility.</param>
    public SecureAggregation(int parameterCount, int? randomSeed = null)
    {
        if (parameterCount <= 0)
        {
            throw new ArgumentException("Parameter count must be positive.", nameof(parameterCount));
        }

        _parameterCount = parameterCount;
        _deterministicSeed = randomSeed;
        _pairwiseMaskSeeds = new Dictionary<long, byte[]>();
    }

    ~SecureAggregation()
    {
        Dispose(disposing: false);
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
            throw new ObjectDisposedException(nameof(SecureAggregation<T>));
        }
    }

    /// <summary>
    /// Generates pairwise secrets between all clients.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This creates secret keys that clients will use to mask
    /// their updates. The secrets are designed so they cancel out when aggregated.
    ///
    /// For each pair of clients (i, j):
    /// - Generate random secret s_ij
    /// - Set s_ji = -s_ij (so they cancel: s_ij + s_ji = 0)
    ///
    /// In production, this would use:
    /// - Diffie-Hellman key exchange
    /// - Public key infrastructure
    /// - Secure random number generation
    /// </remarks>
    /// <param name="clientIds">List of all participating client IDs.</param>
    public void GeneratePairwiseSecrets(List<int> clientIds)
    {
        ThrowIfDisposed();
        if (clientIds == null || clientIds.Count < 2)
        {
            throw new ArgumentException("Need at least 2 clients for secure aggregation.", nameof(clientIds));
        }

        var distinct = clientIds
            .Where(id => id >= 0)
            .Distinct()
            .OrderBy(id => id)
            .ToArray();

        if (distinct.Length < 2)
        {
            throw new ArgumentException("Need at least 2 distinct non-negative clients for secure aggregation.", nameof(clientIds));
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

    /// <summary>
    /// Masks a client's model update with pairwise secrets.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This adds secret masks to the client's update so the
    /// server can't see the original values. Only after aggregating all clients
    /// do the masks cancel out.
    ///
    /// Mathematical operation:
    /// masked_update = original_update + Σ(secrets_with_other_clients)
    ///
    /// For example, Client 1 with 3 clients total:
    /// - Original update: [0.5, -0.3, 0.8]
    /// - Secret with Client 2: [0.1, 0.2, -0.1]
    /// - Secret with Client 3: [-0.2, 0.1, 0.15]
    /// - Masked update: [0.4, 0.0, 0.85]
    ///
    /// Server sees: [0.4, 0.0, 0.85] ← Cannot recover original [0.5, -0.3, 0.8]
    /// But after aggregating all clients, secrets cancel and server gets correct sum.
    /// </remarks>
    /// <param name="clientId">The ID of the client whose update to mask.</param>
    /// <param name="clientUpdate">The client's model update.</param>
    /// <returns>The masked model update.</returns>
    public Dictionary<string, T[]> MaskUpdate(int clientId, Dictionary<string, T[]> clientUpdate)
    {
        ThrowIfDisposed();
        return MaskUpdateInternal(clientId, clientUpdate, clientWeight: null);
    }

    /// <summary>
    /// Masks a client's model update with pairwise secrets, applying the client's aggregation weight before masking.
    /// </summary>
    /// <remarks>
    /// For secure weighted averaging, clients must apply weights to their updates <i>before</i> masking so secrets still cancel.
    /// This overload multiplies the update by <paramref name="clientWeight"/> and then adds the pairwise masks.
    /// </remarks>
    /// <param name="clientId">The ID of the client whose update to mask.</param>
    /// <param name="clientUpdate">The client's (unweighted) model update.</param>
    /// <param name="clientWeight">The aggregation weight to apply to this client's update (e.g., sample count).</param>
    /// <returns>The masked (and weighted) model update.</returns>
    public Dictionary<string, T[]> MaskUpdate(int clientId, Dictionary<string, T[]> clientUpdate, double clientWeight)
    {
        ThrowIfDisposed();
        if (clientWeight <= 0.0)
        {
            throw new ArgumentException("Client weight must be positive.", nameof(clientWeight));
        }

        return MaskUpdateInternal(clientId, clientUpdate, clientWeight);
    }

    private Dictionary<string, T[]> MaskUpdateInternal(int clientId, Dictionary<string, T[]> clientUpdate, double? clientWeight)
    {
        if (clientUpdate == null || clientUpdate.Count == 0)
        {
            throw new ArgumentException("Client update cannot be null or empty.", nameof(clientUpdate));
        }

        EnsureClientIsKnown(clientId);

        // Create masked update
        var maskedUpdate = new Dictionary<string, T[]>();

        // Flatten all parameters to apply masks
        var flatParams = FlattenParameters(clientUpdate);
        EnsureParameterCountMatches(flatParams.Length, nameof(clientUpdate));

        var maskedFlatParams = new T[flatParams.Length];
        Array.Copy(flatParams, maskedFlatParams, flatParams.Length);

        // Apply weight to the update (not to the masks) so masks still cancel in the sum.
        if (clientWeight.HasValue)
        {
            var weightT = NumOps.FromDouble(clientWeight.Value);
            for (int i = 0; i < maskedFlatParams.Length; i++)
            {
                maskedFlatParams[i] = NumOps.Multiply(maskedFlatParams[i], weightT);
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
            ApplyPairwiseMask(seed, addMask, maskedFlatParams);
        }

        // Unflatten back to original structure
        int paramIndex = 0;
        foreach (var layerName in clientUpdate.Keys.OrderBy(name => name, StringComparer.Ordinal))
        {
            var originalLayer = clientUpdate[layerName];
            var maskedLayer = new T[originalLayer.Length];

            for (int i = 0; i < originalLayer.Length; i++, paramIndex++)
            {
                maskedLayer[i] = maskedFlatParams[paramIndex];
            }

            maskedUpdate[layerName] = maskedLayer;
        }

        return maskedUpdate;
    }

    /// <summary>
    /// Aggregates masked updates from all clients, returning a weighted average.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This sums up all the masked updates. Because the secret
    /// masks cancel out, the server recovers the sum of the underlying (possibly weighted)
    /// client updates without ever seeing any individual update.
    ///
    /// To compute a <i>weighted average</i> securely:
    /// - Each client must apply its weight to its update <i>before</i> masking (use <see cref="MaskUpdate(int, Dictionary{string, T[]}, double)"/>).
    /// - The server then divides the summed masked updates by the total weight.
    ///
    /// If you need the raw (un-normalized) sum of updates, use <see cref="AggregateSumSecurely"/>.
    ///
    /// Mathematical property:
    /// Σ(masked_update_i) = Σ(update_i + secrets_i)
    ///                     = Σ(update_i) + Σ(secrets_i)
    ///                     = Σ(update_i) + 0  ← secrets cancel
    ///                     = True sum of updates
    ///
    /// The server performs this aggregation without ever seeing individual updates!
    ///
    /// For example with 2 clients:
    /// Client 1 masked: [0.4, 0.0, 0.85] = [0.5, -0.3, 0.8] + [-0.1, 0.3, 0.05]
    /// Client 2 masked: [0.7, 0.1, 1.05] = [0.6, 0.4, 1.1] + [0.1, -0.3, -0.05]
    ///
    /// Sum of masked: [1.1, 0.1, 1.9]
    /// True sum: [0.5, -0.3, 0.8] + [0.6, 0.4, 1.1] = [1.1, 0.1, 1.9] ← Matches!
    /// (Note: Secrets [-0.1, 0.3, 0.05] + [0.1, -0.3, -0.05] = [0, 0, 0] ← Cancelled)
    /// </remarks>
    /// <param name="maskedUpdates">Dictionary of client IDs to their masked updates.</param>
    /// <param name="clientWeights">Dictionary of client IDs to their aggregation weights.</param>
    /// <returns>The securely aggregated model (weighted average if clients pre-weighted their updates before masking).</returns>
    public Dictionary<string, T[]> AggregateSecurely(
        Dictionary<int, Dictionary<string, T[]>> maskedUpdates,
        Dictionary<int, double> clientWeights)
    {
        ThrowIfDisposed();
        if (clientWeights == null || clientWeights.Count == 0)
        {
            throw new ArgumentException("Client weights cannot be null or empty.", nameof(clientWeights));
        }

        EnsureFullParticipation(maskedUpdates);

        var aggregatedUpdate = AggregateSumSecurely(maskedUpdates);

        // If the client updates were weighted before masking (recommended for secure weighted averaging),
        // divide by total weight to return a weighted average.
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
        foreach (var aggregatedParams in aggregatedUpdate.Values)
        {
            for (int i = 0; i < aggregatedParams.Length; i++)
            {
                aggregatedParams[i] = NumOps.Divide(aggregatedParams[i], totalWeightT);
            }
        }

        return aggregatedUpdate;
    }

    /// <summary>
    /// Aggregates masked updates from all clients, returning the raw sum with masks cancelled.
    /// </summary>
    /// <remarks>
    /// This method does not divide by any weight. It returns the sum of the underlying updates
    /// after the pairwise masks cancel out.
    /// </remarks>
    /// <param name="maskedUpdates">Dictionary of client IDs to their masked updates.</param>
    /// <returns>The securely aggregated model (sum of underlying updates with masks cancelled).</returns>
    public Dictionary<string, T[]> AggregateSumSecurely(Dictionary<int, Dictionary<string, T[]>> maskedUpdates)
    {
        ThrowIfDisposed();
        if (maskedUpdates == null || maskedUpdates.Count == 0)
        {
            throw new ArgumentException("Masked updates cannot be null or empty.", nameof(maskedUpdates));
        }

        EnsureFullParticipation(maskedUpdates);

        // Get model structure from first client
        var firstUpdate = maskedUpdates.First().Value;
        var aggregatedUpdate = new Dictionary<string, T[]>();

        // Initialize aggregated update with zeros
        foreach (var layerName in firstUpdate.Keys)
        {
            var layer = new T[firstUpdate[layerName].Length];
            for (int i = 0; i < layer.Length; i++)
            {
                layer[i] = NumOps.Zero;
            }

            aggregatedUpdate[layerName] = layer;
        }

        // Validate all clients have the same structure and parameter counts.
        foreach (var kvp in maskedUpdates)
        {
            int clientId = kvp.Key;
            var maskedUpdate = kvp.Value;

            if (maskedUpdate == null || maskedUpdate.Count == 0)
            {
                throw new ArgumentException($"Masked update for client {clientId} cannot be null or empty.", nameof(maskedUpdates));
            }

            if (maskedUpdate.Count != firstUpdate.Count)
            {
                throw new InvalidOperationException(
                    $"Client {clientId} has inconsistent model structure. Expected {firstUpdate.Count} layers, got {maskedUpdate.Count}.");
            }

            foreach (var layerName in firstUpdate.Keys)
            {
                if (!maskedUpdate.TryGetValue(layerName, out var clientLayer) || clientLayer == null)
                {
                    throw new InvalidOperationException(
                        $"Client {clientId} is missing layer '{layerName}' or it is null. All clients must send the same layer names.");
                }

                int expectedLength = firstUpdate[layerName].Length;
                if (clientLayer.Length != expectedLength)
                {
                    throw new InvalidOperationException(
                        $"Client {clientId} layer '{layerName}' has {clientLayer.Length} parameters but expected {expectedLength}.");
                }
            }
        }

        // Sum all masked updates
        // The pairwise secrets will cancel out, leaving only the true sum
        foreach (var maskedUpdate in maskedUpdates.Values)
        {
            foreach (var layerName in maskedUpdate.Keys)
            {
                var maskedParams = maskedUpdate[layerName];
                var aggregatedParams = aggregatedUpdate[layerName];

                for (int i = 0; i < maskedParams.Length; i++)
                {
                    aggregatedParams[i] = NumOps.Add(aggregatedParams[i], maskedParams[i]);
                }
            }
        }

        return aggregatedUpdate;
    }

    private void EnsureParameterCountMatches(int actualCount, string paramName)
    {
        if (actualCount != _parameterCount)
        {
            throw new ArgumentException(
                $"Model parameter count mismatch. Expected {_parameterCount}, got {actualCount}. " +
                "Ensure SecureAggregation is constructed with the correct parameterCount.",
                paramName);
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

    private void EnsureFullParticipation(Dictionary<int, Dictionary<string, T[]>> maskedUpdates)
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

    private void ApplyPairwiseMask(byte[] seed, bool addMask, T[] maskedFlatParams)
    {
        using var prg = new HmacSha256Prg(seed);
        for (int i = 0; i < maskedFlatParams.Length; i++)
        {
            var mask = NumOps.FromDouble((prg.NextUnitIntervalDouble() - 0.5) * 2.0);
            maskedFlatParams[i] = addMask ? NumOps.Add(maskedFlatParams[i], mask) : NumOps.Subtract(maskedFlatParams[i], mask);
        }
    }

    /// <summary>
    /// Flattens a hierarchical model structure into a single parameter array.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Converts the model from a dictionary of layers to a
    /// single flat array of all parameters. Makes it easier to apply masks uniformly.
    /// </remarks>
    /// <param name="model">The model to flatten.</param>
    /// <returns>A flat array of all parameters.</returns>
    private T[] FlattenParameters(Dictionary<string, T[]> model)
    {
        var orderedLayerNames = model.Keys.OrderBy(name => name, StringComparer.Ordinal).ToArray();
        return orderedLayerNames.SelectMany(layerName => model[layerName]).ToArray();
    }

    /// <summary>
    /// Clears all stored pairwise secrets.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Removes all secret keys from memory. Should be called
    /// after aggregation is complete for security.
    ///
    /// Security best practice:
    /// - Generate fresh secrets for each round
    /// - Clear old secrets to prevent reuse
    /// - Minimize time secrets are stored in memory
    /// </remarks>
    public void ClearSecrets()
    {
        ThrowIfDisposed();
        _pairwiseMaskSeeds.Clear();
        _clientIds = Array.Empty<int>();
    }

    /// <summary>
    /// Gets the number of clients with stored secrets.
    /// </summary>
    /// <returns>The count of clients.</returns>
    public int GetClientCount()
    {
        ThrowIfDisposed();
        return _clientIds.Length;
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
