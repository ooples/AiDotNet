using System.Security.Cryptography;
using System.Text;
using AiDotNet.FederatedLearning.Cryptography;
using AiDotNet.FederatedLearning.Infrastructure;

namespace AiDotNet.FederatedLearning.Privacy;

/// <summary>
/// Implements dropout-resilient secure aggregation for vector-based model updates.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Secure aggregation lets the server compute the sum/average of client updates
/// without learning any single client's update. Each client adds random "masks" to its update. When the
/// server combines all masked updates, the masks cancel out and the server recovers only the aggregate.
///
/// This variant is <i>dropout-resilient</i>:
/// - Some clients may fail to upload masked updates (upload dropout).
/// - Some clients may upload but fail to complete the unmasking step (unmasking dropout).
///
/// As long as enough clients complete the unmasking step (the reconstruction threshold), the server can
/// still recover the aggregate by reconstructing missing self-masks using Shamir secret sharing and by
/// removing leftover pairwise masks for clients that did not upload.
///
/// Reference: Bonawitz, K., et al. (2017). "Practical Secure Aggregation for Privacy-Preserving Machine Learning."
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters (e.g., double, float).</typeparam>
public sealed class ThresholdSecureAggregationVector<T> : FederatedLearningComponentBase<T>, IDisposable
{
    private const int SelfMaskSeedLength = 32;

    private readonly Dictionary<long, byte[]> _pairwiseMaskSeeds = new();
    private readonly Dictionary<int, byte[]> _selfMaskSeeds = new();
    private readonly Dictionary<int, Dictionary<int, byte[]>> _selfMaskSharesByOwner = new();
    private readonly Dictionary<int, int> _xByClientId = new();

    private readonly int? _deterministicSeed;
    private int[] _clientIds = Array.Empty<int>();
    private readonly int _parameterCount;
    private int _minimumUploaderCount;
    private int _reconstructionThreshold;
    private bool _disposed;

    public ThresholdSecureAggregationVector(int parameterCount, int? randomSeed = null)
    {
        if (parameterCount <= 0)
        {
            throw new ArgumentException("Parameter count must be positive.", nameof(parameterCount));
        }

        _parameterCount = parameterCount;
        _deterministicSeed = randomSeed;
        _minimumUploaderCount = 0;
        _reconstructionThreshold = 0;
    }

    ~ThresholdSecureAggregationVector()
    {
        Dispose(disposing: false);
    }

    /// <summary>
    /// Gets the minimum number of clients that must upload masked updates for the round to succeed.
    /// </summary>
    public int MinimumUploaderCount => _minimumUploaderCount;

    /// <summary>
    /// Gets the reconstruction threshold required to complete unmasking.
    /// </summary>
    public int ReconstructionThreshold => _reconstructionThreshold;

    /// <summary>
    /// Initializes a new secure aggregation round by generating the required cryptographic material.
    /// </summary>
    /// <remarks>
    /// If <paramref name="minimumUploaderCount"/> or <paramref name="reconstructionThreshold"/> are not set (0),
    /// industry-standard defaults are computed based on the selected clients.
    /// </remarks>
    /// <param name="clientIds">Selected clients for the round.</param>
    /// <param name="minimumUploaderCount">Minimum number of clients that must upload masked updates (0 = auto).</param>
    /// <param name="reconstructionThreshold">Minimum number of clients that must complete unmasking (0 = auto).</param>
    /// <param name="maxDropoutFraction">Used only when <paramref name="minimumUploaderCount"/> is 0 (auto).</param>
    public void InitializeRound(
        List<int> clientIds,
        int minimumUploaderCount = 0,
        int reconstructionThreshold = 0,
        double maxDropoutFraction = 0.2)
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

        _minimumUploaderCount = ResolveMinimumUploaderCount(distinct.Length, minimumUploaderCount, maxDropoutFraction);
        _reconstructionThreshold = ResolveReconstructionThreshold(distinct.Length, reconstructionThreshold, _minimumUploaderCount);

        _xByClientId.Clear();
        for (int i = 0; i < distinct.Length; i++)
        {
            _xByClientId[distinct[i]] = i + 1;
        }

        if (_deterministicSeed.HasValue)
        {
            GenerateDeterministicRoundSecrets(distinct, _deterministicSeed.Value);
            return;
        }

        GenerateEphemeralRoundSecrets(distinct);
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
        return AggregateSecurely(maskedUpdates, clientWeights, unmaskingClientIds: null);
    }

    public Vector<T> AggregateSecurely(
        Dictionary<int, Vector<T>> maskedUpdates,
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
        return AggregateSumSecurely(maskedUpdates, unmaskingClientIds: null);
    }

    public Vector<T> AggregateSumSecurely(Dictionary<int, Vector<T>> maskedUpdates, IReadOnlyCollection<int>? unmaskingClientIds)
    {
        ThrowIfDisposed();
        if (maskedUpdates == null || maskedUpdates.Count == 0)
        {
            throw new ArgumentException("Masked updates cannot be null or empty.", nameof(maskedUpdates));
        }

        EnsureInitialized();
        EnsureUploadersAreValid(maskedUpdates);

        var unmaskers = ResolveUnmaskers(maskedUpdates, unmaskingClientIds);

        var first = maskedUpdates.First().Value;
        EnsureParameterCountMatches(first.Length, nameof(maskedUpdates));

        var sum = new T[_parameterCount];
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

        UnmaskInPlace(sum, maskedUpdates.Keys, unmaskers);

        return new Vector<T>(sum);
    }

    public void ClearSecrets()
    {
        ThrowIfDisposed();

        _pairwiseMaskSeeds.Clear();
        _selfMaskSeeds.Clear();
        _selfMaskSharesByOwner.Clear();
        _xByClientId.Clear();
        _clientIds = Array.Empty<int>();
        _minimumUploaderCount = 0;
        _reconstructionThreshold = 0;
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
            _selfMaskSeeds.Clear();
            _selfMaskSharesByOwner.Clear();
            _xByClientId.Clear();
            _clientIds = Array.Empty<int>();
        }

        _disposed = true;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(ThresholdSecureAggregationVector<T>));
        }
    }

    private Vector<T> MaskUpdateInternal(int clientId, Vector<T> clientUpdate, double? clientWeight)
    {
        if (clientUpdate == null)
        {
            throw new ArgumentNullException(nameof(clientUpdate));
        }

        EnsureInitialized();
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

        ApplyMaskSeed(GetSelfMaskSeed(clientId), addMask: true, masked);

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
            ApplyMaskSeed(seed, addMask, masked);
        }

        return new Vector<T>(masked);
    }

    private void UnmaskInPlace(T[] sum, IEnumerable<int> uploaders, IReadOnlyCollection<int> unmaskers)
    {
        var uploaderSet = uploaders as ICollection<int> ?? uploaders.ToArray();
        var uploadersSorted = uploaderSet.OrderBy(id => id).ToArray();
        var unmaskerSet = unmaskers is HashSet<int> hs ? hs : new HashSet<int>(unmaskers);

        if (uploadersSorted.Length < _minimumUploaderCount)
        {
            throw new InvalidOperationException(
                $"Secure aggregation requires at least {_minimumUploaderCount} client uploads for this mode. Got {uploadersSorted.Length}.");
        }

        if (unmaskerSet.Count < _reconstructionThreshold)
        {
            throw new InvalidOperationException(
                $"Secure aggregation requires at least {_reconstructionThreshold} clients to complete unmasking for this mode. Got {unmaskerSet.Count}.");
        }

        // 1) Remove self-masks for uploaders (direct for unmaskers, reconstruct for uploaders who dropped after upload).
        foreach (var clientId in uploadersSorted)
        {
            if (unmaskerSet.Contains(clientId))
            {
                ApplyMaskSeed(GetSelfMaskSeed(clientId), addMask: false, sum);
                continue;
            }

            var reconstructed = ReconstructSelfMaskSeed(clientId, unmaskerSet);
            try
            {
                ApplyMaskSeed(reconstructed, addMask: false, sum);
            }
            finally
            {
                Array.Clear(reconstructed, 0, reconstructed.Length);
            }
        }

        // 2) Remove leftover pairwise masks for clients that did not upload.
        var uploaderLookup = new HashSet<int>(uploadersSorted);
        foreach (var expectedClientId in _clientIds)
        {
            if (uploaderLookup.Contains(expectedClientId))
            {
                continue;
            }

            foreach (var uploaderId in uploadersSorted)
            {
                int min = Math.Min(expectedClientId, uploaderId);
                int max = Math.Max(expectedClientId, uploaderId);
                bool uploaderWasMin = uploaderId == min;

                var seed = GetPairwiseSeed(min, max);

                // In MaskUpdate, the min client adds the mask and the max client subtracts it.
                // Since the missing client did not upload, the uploader-side contribution remains in the sum.
                // To remove it, we apply the inverse operation.
                bool addToSum = !uploaderWasMin;
                ApplyMaskSeed(seed, addMask: addToSum, sum);
            }
        }
    }

    private byte[] ReconstructSelfMaskSeed(int ownerClientId, HashSet<int> unmaskerSet)
    {
        if (!_selfMaskSharesByOwner.TryGetValue(ownerClientId, out var sharesByRecipient))
        {
            throw new InvalidOperationException($"Missing self-mask shares for client {ownerClientId}.");
        }

        var provided = new Dictionary<int, byte[]>();
        foreach (var recipientId in unmaskerSet)
        {
            if (sharesByRecipient.TryGetValue(recipientId, out var share))
            {
                provided[recipientId] = share;
            }
        }

        return ShamirSecretSharing.CombineShares(
            sharesByRecipient: provided,
            xByRecipient: _xByClientId,
            threshold: _reconstructionThreshold,
            secretLength: SelfMaskSeedLength);
    }

    private IReadOnlyCollection<int> ResolveUnmaskers(
        Dictionary<int, Vector<T>> maskedUpdates,
        IReadOnlyCollection<int>? unmaskingClientIds)
    {
        if (unmaskingClientIds == null)
        {
            return maskedUpdates.Keys.ToArray();
        }

        if (unmaskingClientIds.Count == 0)
        {
            throw new ArgumentException("Unmasking client list cannot be empty when provided.", nameof(unmaskingClientIds));
        }

        var uploaders = new HashSet<int>(maskedUpdates.Keys);
        foreach (var clientId in unmaskingClientIds)
        {
            if (!uploaders.Contains(clientId))
            {
                throw new ArgumentException($"Client {clientId} is not an uploader and cannot be an unmasker.", nameof(unmaskingClientIds));
            }
        }

        return unmaskingClientIds;
    }

    private void EnsureInitialized()
    {
        if (_clientIds.Length == 0)
        {
            throw new InvalidOperationException("Secure aggregation secrets are not initialized. Call InitializeRound first.");
        }
    }

    private void EnsureUploadersAreValid(Dictionary<int, Vector<T>> maskedUpdates)
    {
        if (maskedUpdates.Count < _minimumUploaderCount && _minimumUploaderCount > 0)
        {
            throw new InvalidOperationException(
                $"Secure aggregation requires at least {_minimumUploaderCount} client uploads for this mode. Got {maskedUpdates.Count}.");
        }

        foreach (var clientId in maskedUpdates.Keys)
        {
            EnsureClientIsKnown(clientId);
        }
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
        if (Array.IndexOf(_clientIds, clientId) < 0)
        {
            throw new ArgumentException($"No secrets found for client {clientId}. Call InitializeRound first.", nameof(clientId));
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

    private byte[] GetSelfMaskSeed(int clientId)
    {
        if (!_selfMaskSeeds.TryGetValue(clientId, out var seed))
        {
            throw new InvalidOperationException($"Missing self mask seed for client {clientId}.");
        }

        return seed;
    }

    private void ApplyMaskSeed(byte[] seed, bool addMask, T[] target)
    {
        using var prg = new HmacSha256Prg(seed);
        for (int i = 0; i < _parameterCount; i++)
        {
            var mask = NumOps.FromDouble((prg.NextUnitIntervalDouble() - 0.5) * 2.0);
            target[i] = addMask ? NumOps.Add(target[i], mask) : NumOps.Subtract(target[i], mask);
        }
    }

    private void GenerateDeterministicRoundSecrets(int[] sortedClientIds, int baseSeed)
    {
        foreach (var clientId in sortedClientIds)
        {
            _selfMaskSeeds[clientId] = DeriveDeterministicSeed(baseSeed, $"self:{clientId}");
        }

        foreach (var (min, max) in EnumeratePairs(sortedClientIds))
        {
            _pairwiseMaskSeeds[PairKey(min, max)] = DeriveDeterministicSeed(baseSeed, $"pair:{min}:{max}");
        }

        foreach (var clientId in sortedClientIds)
        {
            var shares = ShamirSecretSharing.SplitSecret(
                secret: _selfMaskSeeds[clientId],
                xByRecipient: _xByClientId,
                threshold: _reconstructionThreshold,
                deterministicSeed: baseSeed,
                info: $"AiDotNet.SecAgg.v2:selfmask:{clientId}");

            _selfMaskSharesByOwner[clientId] = shares;
        }
    }

    private void GenerateEphemeralRoundSecrets(int[] sortedClientIds)
    {
        using (var rng = RandomNumberGenerator.Create())
        {
            foreach (var clientId in sortedClientIds)
            {
                var seed = new byte[SelfMaskSeedLength];
                rng.GetBytes(seed);
                _selfMaskSeeds[clientId] = seed;
            }
        }

        var keyPairs = new Dictionary<int, ECDiffieHellman>();
        try
        {
            foreach (var clientId in sortedClientIds)
            {
                keyPairs[clientId] = ECDiffieHellman.Create();
            }

            var salt = Encoding.UTF8.GetBytes("AiDotNet.SecAgg.v2");
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

        foreach (var clientId in sortedClientIds)
        {
            var shares = ShamirSecretSharing.SplitSecret(
                secret: _selfMaskSeeds[clientId],
                xByRecipient: _xByClientId,
                threshold: _reconstructionThreshold,
                deterministicSeed: null,
                info: $"AiDotNet.SecAgg.v2:selfmask:{clientId}");

            _selfMaskSharesByOwner[clientId] = shares;
        }
    }

    private static int ResolveMinimumUploaderCount(int totalClients, int configuredMinimum, double maxDropoutFraction)
    {
        if (totalClients < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(totalClients), "Total clients must be at least 2.");
        }

        if (configuredMinimum > 0)
        {
            if (configuredMinimum < 2 || configuredMinimum > totalClients)
            {
                throw new ArgumentOutOfRangeException(nameof(configuredMinimum), "Minimum uploader count must be between 2 and total clients.");
            }

            return configuredMinimum;
        }

        double f = maxDropoutFraction;
        if (double.IsNaN(f) || double.IsInfinity(f))
        {
            f = 0.2;
        }

        f = Math.Max(0.0, Math.Min(0.9, f));
        int allowedDropouts = (int)Math.Floor(totalClients * f);

        if (allowedDropouts <= 0 && totalClients >= 3 && f > 0.0)
        {
            allowedDropouts = 1;
        }

        allowedDropouts = Math.Max(0, Math.Min(totalClients - 2, allowedDropouts));

        int min = totalClients - allowedDropouts;
        min = Math.Max(2, Math.Min(totalClients, min));
        return min;
    }

    private static int ResolveReconstructionThreshold(int totalClients, int configuredThreshold, int minimumUploaderCount)
    {
        int threshold = configuredThreshold > 0 ? configuredThreshold : minimumUploaderCount;

        if (threshold < 2 || threshold > totalClients)
        {
            throw new ArgumentOutOfRangeException(nameof(configuredThreshold), "Reconstruction threshold must be between 2 and total clients.");
        }

        if (threshold > minimumUploaderCount)
        {
            throw new ArgumentOutOfRangeException(
                nameof(configuredThreshold),
                "Reconstruction threshold cannot exceed the minimum uploader count for this mode.");
        }

        return threshold;
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

    private static byte[] DeriveDeterministicSeed(int baseSeed, string label)
    {
        var keyBytes = BitConverter.GetBytes(baseSeed);
        var data = Encoding.UTF8.GetBytes($"AiDotNet.SecAgg.v2:{label}");
        using var hmac = new HMACSHA256(keyBytes);
        return hmac.ComputeHash(data);
    }
}
