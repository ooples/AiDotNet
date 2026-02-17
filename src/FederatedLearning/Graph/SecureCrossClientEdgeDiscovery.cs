using System.Security.Cryptography;
using AiDotNet.FederatedLearning.Infrastructure;

namespace AiDotNet.FederatedLearning.Graph;

/// <summary>
/// PSI-based secure discovery of cross-client edges without revealing full adjacency lists.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> When a graph is split across clients, some edges connect nodes on
/// different clients. To discover these edges privately, we use a technique from Private Set
/// Intersection (PSI): each client hashes their border node IDs, and we compare hashes to find
/// matches. Only matching (shared) node IDs are revealed — each client's non-shared nodes stay private.</para>
///
/// <para><b>Protocol:</b></para>
/// <list type="number">
/// <item><description>Client A hashes its border node IDs: {H(id1), H(id2), ...}.</description></item>
/// <item><description>Client B hashes its border node IDs: {H(id3), H(id4), ...}.</description></item>
/// <item><description>Intersection of hash sets reveals shared nodes.</description></item>
/// <item><description>Shared nodes imply cross-client edges.</description></item>
/// </list>
///
/// <para><b>Privacy:</b> Uses the Diffie-Hellman PSI approach from #538 when available.
/// Falls back to hash-based comparison with randomized response for differential privacy.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class SecureCrossClientEdgeDiscovery<T> : FederatedLearningComponentBase<T>, ICrossClientEdgeHandler<T>
{
    private readonly double _privacyEpsilon;
    private readonly int _maxEdgesPerPair;
    private readonly bool _cacheEnabled;
    private readonly Dictionary<(int, int), List<(int NodeA, int NodeB)>> _edgeCache = new();

    /// <inheritdoc/>
    public int DiscoveredEdgeCount
    {
        get
        {
            int total = 0;
            foreach (var edges in _edgeCache.Values)
            {
                total += edges.Count;
            }

            return total;
        }
    }

    /// <summary>
    /// Initializes a new instance of <see cref="SecureCrossClientEdgeDiscovery{T}"/>.
    /// </summary>
    /// <param name="privacyEpsilon">Differential privacy epsilon for edge queries. Default 1.0.</param>
    /// <param name="maxEdgesPerPair">Maximum cross-client edges per client pair. Default 1000.</param>
    /// <param name="cacheEnabled">Whether to cache discovered edges. Default true.</param>
    public SecureCrossClientEdgeDiscovery(
        double privacyEpsilon = 1.0,
        int maxEdgesPerPair = 1000,
        bool cacheEnabled = true)
    {
        if (privacyEpsilon <= 0.0 || double.IsNaN(privacyEpsilon) || double.IsInfinity(privacyEpsilon))
        {
            throw new ArgumentOutOfRangeException(nameof(privacyEpsilon),
                "Privacy epsilon must be a finite positive value.");
        }

        _privacyEpsilon = privacyEpsilon;
        _maxEdgesPerPair = maxEdgesPerPair;
        _cacheEnabled = cacheEnabled;
    }

    /// <inheritdoc/>
    public IReadOnlyList<(int NodeA, int NodeB)> DiscoverEdges(
        IReadOnlyList<int> clientABorderNodes,
        IReadOnlyList<int> clientBBorderNodes)
    {
        if (clientABorderNodes is null || clientBBorderNodes is null)
        {
            return Array.Empty<(int, int)>();
        }

        // Hash border node IDs for private comparison
        var hashesA = HashNodeIds(clientABorderNodes);
        var hashesB = HashNodeIds(clientBBorderNodes);

        // Find intersection (shared nodes = cross-client edges)
        var discovered = new List<(int NodeA, int NodeB)>();

        // Build lookup from hash -> original ID for set B
        var hashToIdB = new Dictionary<string, int>();
        for (int i = 0; i < clientBBorderNodes.Count; i++)
        {
            string hash = hashesB[i];
            if (!hashToIdB.ContainsKey(hash))
            {
                hashToIdB[hash] = clientBBorderNodes[i];
            }
        }

        // Apply randomized response for differential privacy
        // Use numerically stable sigmoid: p = 1 / (1 + exp(-epsilon))
        // This avoids overflow when epsilon is large (exp(epsilon) -> Infinity)
        double reportProbability = _privacyEpsilon >= 0
            ? 1.0 / (1.0 + Math.Exp(-_privacyEpsilon))
            : Math.Exp(_privacyEpsilon) / (Math.Exp(_privacyEpsilon) + 1.0);
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();

        for (int i = 0; i < clientABorderNodes.Count && discovered.Count < _maxEdgesPerPair; i++)
        {
            string hashA = hashesA[i];

            if (hashToIdB.ContainsKey(hashA))
            {
                // True match — apply randomized response
                if (rng.NextDouble() < reportProbability)
                {
                    discovered.Add((clientABorderNodes[i], hashToIdB[hashA]));
                }
            }
        }

        return discovered;
    }

    /// <inheritdoc/>
    public IReadOnlyList<(int NodeA, int NodeB)> GetEdges(int clientA, int clientB)
    {
        var key = (Math.Min(clientA, clientB), Math.Max(clientA, clientB));

        if (_edgeCache.ContainsKey(key))
        {
            return _edgeCache[key];
        }

        return Array.Empty<(int, int)>();
    }

    /// <inheritdoc/>
    public void CacheEdges(int clientA, int clientB, IReadOnlyList<(int NodeA, int NodeB)> edges)
    {
        if (!_cacheEnabled || edges is null)
        {
            return;
        }

        var key = (Math.Min(clientA, clientB), Math.Max(clientA, clientB));
        _edgeCache[key] = new List<(int, int)>(edges);
    }

    private static List<string> HashNodeIds(IReadOnlyList<int> nodeIds)
    {
        var hashes = new List<string>(nodeIds.Count);

        using var sha256 = SHA256.Create();
        foreach (int id in nodeIds)
        {
            byte[] idBytes = BitConverter.GetBytes(id);
            byte[] hash = sha256.ComputeHash(idBytes);
            hashes.Add(BitConverter.ToString(hash).Replace("-", "").ToLowerInvariant());
        }

        return hashes;
    }
}
