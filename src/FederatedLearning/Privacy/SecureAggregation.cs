namespace AiDotNet.FederatedLearning.Privacy;

using System;
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
/// This implementation uses a simplified version with random masking for demonstration.
/// Production systems should use proper cryptographic protocols like:
/// - Bonawitz et al.'s Secure Aggregation protocol
/// - Threshold homomorphic encryption
/// - Secret sharing schemes
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
public class SecureAggregation<T> : FederatedLearningComponentBase<T>
{
    private readonly Dictionary<int, Dictionary<int, T[]>> _pairwiseSecrets;
    private readonly Random _random;
    private readonly int _parameterCount;

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
    /// This simplified implementation uses pseudorandom masks that cancel out.
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
        _pairwiseSecrets = new Dictionary<int, Dictionary<int, T[]>>();
        _random = randomSeed.HasValue ? new Random(randomSeed.Value) : new Random();
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
        if (clientIds == null || clientIds.Count < 2)
        {
            throw new ArgumentException("Need at least 2 clients for secure aggregation.", nameof(clientIds));
        }

        _pairwiseSecrets.Clear();

        // Generate pairwise secrets for each pair of clients
        for (int i = 0; i < clientIds.Count; i++)
        {
            int clientI = clientIds[i];
            _pairwiseSecrets[clientI] = new Dictionary<int, T[]>();

            for (int j = i + 1; j < clientIds.Count; j++)
            {
                int clientJ = clientIds[j];

                // Generate random secret for this pair
                var secret = new T[_parameterCount];
                for (int k = 0; k < _parameterCount; k++)
                {
                    // Use cryptographically secure random in production
                    secret[k] = NumOps.FromDouble((_random.NextDouble() - 0.5) * 2.0); // Range: [-1, 1]
                }

                // Store secret for client i with respect to client j
                _pairwiseSecrets[clientI][clientJ] = secret;

                // Store negated secret for client j with respect to client i
                // This ensures secrets cancel: secret_ij + secret_ji = 0
                if (!_pairwiseSecrets.ContainsKey(clientJ))
                {
                    _pairwiseSecrets[clientJ] = new Dictionary<int, T[]>();
                }

                var negatedSecret = new T[_parameterCount];
                for (int k = 0; k < _parameterCount; k++)
                {
                    negatedSecret[k] = NumOps.Negate(secret[k]);
                }
                _pairwiseSecrets[clientJ][clientI] = negatedSecret;
            }
        }
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
        if (clientUpdate == null || clientUpdate.Count == 0)
        {
            throw new ArgumentException("Client update cannot be null or empty.", nameof(clientUpdate));
        }

        if (!_pairwiseSecrets.ContainsKey(clientId))
        {
            throw new ArgumentException($"No secrets found for client {clientId}. Call GeneratePairwiseSecrets first.", nameof(clientId));
        }

        // Create masked update
        var maskedUpdate = new Dictionary<string, T[]>();

        // Flatten all parameters to apply masks
        var flatParams = FlattenParameters(clientUpdate);
        var maskedFlatParams = new T[flatParams.Length];
        Array.Copy(flatParams, maskedFlatParams, flatParams.Length);

        // Add all pairwise secrets for this client
        foreach (var otherClientSecrets in _pairwiseSecrets[clientId].Values)
        {
            for (int i = 0; i < Math.Min(maskedFlatParams.Length, otherClientSecrets.Length); i++)
            {
                maskedFlatParams[i] = NumOps.Add(maskedFlatParams[i], otherClientSecrets[i]);
            }
        }

        // Unflatten back to original structure
        int paramIndex = 0;
        foreach (var layerName in clientUpdate.Keys)
        {
            var originalLayer = clientUpdate[layerName];
            var maskedLayer = new T[originalLayer.Length];

            for (int i = 0; i < originalLayer.Length && paramIndex < maskedFlatParams.Length; i++, paramIndex++)
            {
                maskedLayer[i] = maskedFlatParams[paramIndex];
            }

            maskedUpdate[layerName] = maskedLayer;
        }

        return maskedUpdate;
    }

    /// <summary>
    /// Aggregates masked updates from all clients, recovering the true sum.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This sums up all the masked updates. Because the secret
    /// masks cancel out, the result is the true sum of client updates.
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
    /// Client 2 masked: [0.7, 0.7, 1.05] = [0.6, 0.4, 1.1] + [0.1, -0.3, -0.05]
    ///
    /// Sum of masked: [1.1, 0.7, 1.9]
    /// True sum: [0.5, -0.3, 0.8] + [0.6, 0.4, 1.1] = [1.1, 0.1, 1.9] ← Matches!
    /// (Note: Secrets [-0.1, 0.3, 0.05] + [0.1, -0.3, -0.05] = [0, 0, 0] ← Cancelled)
    /// </remarks>
    /// <param name="maskedUpdates">Dictionary of client IDs to their masked updates.</param>
    /// <param name="clientWeights">Dictionary of client IDs to their aggregation weights.</param>
    /// <returns>The securely aggregated model (sum of original updates with masks cancelled).</returns>
    public Dictionary<string, T[]> AggregateSecurely(
        Dictionary<int, Dictionary<string, T[]>> maskedUpdates,
        Dictionary<int, double> clientWeights)
    {
        if (maskedUpdates == null || maskedUpdates.Count == 0)
        {
            throw new ArgumentException("Masked updates cannot be null or empty.", nameof(maskedUpdates));
        }

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

        // Sum all masked updates
        // The pairwise secrets will cancel out, leaving only the true sum
        foreach (var clientId in maskedUpdates.Keys)
        {
            var maskedUpdate = maskedUpdates[clientId];

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

        // If using weighted aggregation, divide by total weight
        if (clientWeights != null && clientWeights.Count > 0)
        {
            double totalWeight = clientWeights.Values.Sum();

            if (totalWeight > 0)
            {
                var totalWeightT = NumOps.FromDouble(totalWeight);
                foreach (var layerName in aggregatedUpdate.Keys)
                {
                    var aggregatedParams = aggregatedUpdate[layerName];

                    for (int i = 0; i < aggregatedParams.Length; i++)
                    {
                        aggregatedParams[i] = NumOps.Divide(aggregatedParams[i], totalWeightT);
                    }
                }
            }
        }

        return aggregatedUpdate;
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
        int totalParams = model.Values.Sum(layer => layer.Length);
        var flatParams = new T[totalParams];

        int index = 0;
        foreach (var layer in model.Values)
        {
            foreach (var param in layer)
            {
                flatParams[index++] = param;
            }
        }

        return flatParams;
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
        _pairwiseSecrets.Clear();
    }

    /// <summary>
    /// Gets the number of clients with stored secrets.
    /// </summary>
    /// <returns>The count of clients.</returns>
    public int GetClientCount()
    {
        return _pairwiseSecrets.Count;
    }
}
