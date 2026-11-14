using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.FederatedLearning.Cryptography;
using AiDotNet.FederatedLearning.Infrastructure;
using AiDotNet.Models.Options;

namespace AiDotNet.FederatedLearning.Privacy;

/// <summary>
/// In-process secure aggregation following Bonawitz et al. (2017): one <see cref="SecureAggregationServer{T}"/> and one
/// <see cref="SecureAggregationClient{T}"/> per participant, with every message passed through the server.
/// </summary>
/// <remarks>
/// <para>
/// Bonawitz, K., Ivanov, V., Kreuter, B., Marcedone, A., McMahan, H. B., Patel, S., Ramage, D., Segal, A. and Seth, K.
/// (2017). "Practical Secure Aggregation for Privacy-Preserving Machine Learning." CCS 2017. Honest-but-curious
/// protocol, figure 4:
/// </para>
/// <list type="bullet">
/// <item><description><see cref="InitializeRound"/> runs round 0 (AdvertiseKeys) and round 1 (ShareKeys) for every
/// client: P-256 Diffie-Hellman keys, Shamir t-of-n shares of <c>s_u^SK</c> and <c>b_u</c>, encrypted and routed by
/// the server.</description></item>
/// <item><description><see cref="MaskUpdate"/> runs round 2 (MaskedInputCollection) for one client: fixed-point
/// encoding into <c>Z_{2^b}</c>, plus <c>PRG(b_u)</c> and the signed pairwise masks <c>PRG(s_{u,v})</c>, all modulo
/// <c>2^b</c>. A client that never calls it has dropped out after sharing keys.</description></item>
/// <item><description><see cref="AggregateRingSum"/>, <see cref="AggregateSum"/> and
/// <see cref="AggregateWeightedAverage"/> run round 4 (Unmasking): the server learns <c>U_3</c>, the unmasking clients
/// release shares, and the server reconstructs the dropped clients' pairwise masks and the survivors' self masks.</description></item>
/// </list>
/// <para>
/// The modular sum is exact, so the decoded aggregate equals the sum of the survivors' fixed-point encodings bit for
/// bit, whatever the masks were. It is decoded in <see cref="double"/> and rounded once to <typeparamref name="T"/>.
/// </para>
/// <para><b>For Beginners:</b> Secure aggregation is like a secret ballot: the server can count the votes but never
/// read a single ballot. Each client adds large random numbers to its update; the numbers are arranged so they cancel
/// exactly when every update is added together, and secret sharing lets the server finish the count even when a few
/// clients disconnect.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters (e.g., double, float, Half).</typeparam>
[ComponentType(ComponentType.FederatedAggregator)]
[PipelineStage(PipelineStage.Training)]
public sealed class SecureAggregation<T> : FederatedLearningComponentBase<T>, IDisposable
{
    private readonly int _parameterCount;
    private readonly SecureAggregationMode _mode;
    private readonly int _configuredThreshold;
    private readonly ISecureAggregationPrg _prg;
    private readonly SortedDictionary<int, SecureAggregationClient<T>> _clients = new();
    private IReadOnlyDictionary<int, IReadOnlyList<SecureAggregationEncryptedShare>> _inboxes =
        new Dictionary<int, IReadOnlyList<SecureAggregationEncryptedShare>>();
    private SecureAggregationServer<T>? _server;
    private bool _disposed;

    /// <summary>
    /// Creates secure aggregation for updates of <paramref name="parameterCount"/> values.
    /// </summary>
    /// <param name="parameterCount">The update length <c>m</c>.</param>
    /// <param name="options">Protocol options; <c>null</c> uses the defaults of <see cref="SecureAggregationOptions"/>.</param>
    public SecureAggregation(int parameterCount, SecureAggregationOptions? options = null)
    {
        if (parameterCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(parameterCount), "Parameter count must be positive.");
        }

        var resolved = options ?? new SecureAggregationOptions();
        _parameterCount = parameterCount;
        _mode = resolved.Mode;
        _configuredThreshold = resolved.ReconstructionThreshold;
        _prg = resolved.Prg ?? new AesCtrSecureAggregationPrg();
        Encoding = new FixedPointRingEncoding(resolved.BitWidth, resolved.FractionalBits, resolved.ClipBound);
    }

    /// <summary>Gets the fixed-point ring encoding.</summary>
    public FixedPointRingEncoding Encoding { get; }

    /// <summary>Gets the threshold <c>t</c> of the current round (0 before <see cref="InitializeRound"/>).</summary>
    public int Threshold { get; private set; }

    /// <summary>Gets the clients of the current round, in ascending order.</summary>
    public IReadOnlyList<int> ClientIds => _clients.Keys.ToArray();

    /// <summary>
    /// Starts a round: runs AdvertiseKeys and ShareKeys for every client in <paramref name="clientIds"/>.
    /// </summary>
    /// <exception cref="ArgumentException">Fewer than two distinct client ids, or the ring is too small for this many
    /// clients.</exception>
    public void InitializeRound(IEnumerable<int> clientIds)
    {
        ThrowIfDisposed();
        if (clientIds is null)
        {
            throw new ArgumentNullException(nameof(clientIds));
        }

        var distinct = clientIds.Distinct().OrderBy(id => id).ToArray();
        if (distinct.Length < 2)
        {
            throw new ArgumentException("Secure aggregation needs at least two distinct clients.", nameof(clientIds));
        }

        ClearSecrets();
        Threshold = SecureAggregationProtocol.ResolveThreshold(distinct.Length, _mode, _configuredThreshold);
        Encoding.EnsureSumFits(distinct.Length);

        var server = new SecureAggregationServer<T>(_parameterCount, Threshold, Encoding, _prg);
        foreach (var id in distinct)
        {
            _clients[id] = new SecureAggregationClient<T>(id, _parameterCount, Threshold, Encoding, _prg);
        }

        var broadcast = server.CollectAdvertisedKeys(_clients.Values.Select(client => client.AdvertiseKeys()).ToArray());
        var shares = _clients.Values.SelectMany(client => client.ShareKeys(broadcast)).ToArray();
        _inboxes = server.CollectShares(shares);
        _server = server;
    }

    /// <summary>
    /// Round 2 for client <paramref name="clientId"/>: returns its masked update.
    /// </summary>
    /// <param name="clientId">A client of the current round.</param>
    /// <param name="update">The client's update.</param>
    /// <param name="weight">A positive factor applied before encoding; for weighted averaging pass the client's
    /// share of the total weight so the encoded values stay within <see cref="FixedPointRingEncoding.ClipBound"/>.</param>
    public SecureAggregationMaskedInput MaskUpdate(int clientId, Vector<T> update, double weight = 1.0)
    {
        ThrowIfDisposed();
        if (_server is null)
        {
            throw new InvalidOperationException("Call InitializeRound before masking updates.");
        }

        if (!_clients.TryGetValue(clientId, out var client) || !_inboxes.TryGetValue(clientId, out var inbox))
        {
            throw new ArgumentException($"Client {clientId} is not part of the current round.", nameof(clientId));
        }

        return client.MaskInput(update, inbox, weight);
    }

    /// <summary>
    /// Round 4: unmasks the sum and returns it in the ring, <c>sum_{u in U_3} Encode(w_u * x_u) mod 2^b</c>.
    /// </summary>
    /// <param name="maskedInputs">The masked inputs the server received; their senders form <c>U_3</c>.</param>
    /// <param name="unmaskingClientIds">The clients of <c>U_3</c> that answer the unmasking round; <c>null</c> means
    /// all of <c>U_3</c>.</param>
    /// <exception cref="SecureAggregationAbortedException">Fewer than <c>t</c> clients remain at some round.</exception>
    public ulong[] AggregateRingSum(
        IEnumerable<SecureAggregationMaskedInput> maskedInputs,
        IEnumerable<int>? unmaskingClientIds = null)
    {
        ThrowIfDisposed();
        var server = _server ?? throw new InvalidOperationException("Call InitializeRound before aggregating.");
        try
        {
            var survivors = server.CollectMaskedInputs(maskedInputs);
            var survivorSet = new HashSet<int>(survivors);
            var unmaskers = unmaskingClientIds is null
                ? survivors.ToArray()
                : unmaskingClientIds.Distinct().ToArray();
            if (unmaskers.Any(id => !survivorSet.Contains(id)))
            {
                throw new ArgumentException("Every unmasking client must have sent a masked input.", nameof(unmaskingClientIds));
            }

            var responses = unmaskers.Select(id => _clients[id].Unmask(survivors)).ToArray();
            return server.AggregateRingSum(responses);
        }
        finally
        {
            ClearSecrets();
        }
    }

    /// <summary>
    /// Round 4: unmasks the sum, decodes it in <see cref="double"/> and rounds once to <typeparamref name="T"/>.
    /// </summary>
    public Vector<T> AggregateSum(
        IEnumerable<SecureAggregationMaskedInput> maskedInputs,
        IEnumerable<int>? unmaskingClientIds = null)
    {
        var ring = AggregateRingSum(maskedInputs, unmaskingClientIds);
        return Decode(ring, divisor: 1.0);
    }

    /// <summary>
    /// Round 4 for weighted averaging: the clients masked <c>w_u * x_u</c> (see <see cref="MaskUpdate"/>); this unmasks
    /// the sum and divides it, in <see cref="double"/>, by the total weight of the clients in <c>U_3</c>, then rounds
    /// once to <typeparamref name="T"/>.
    /// </summary>
    /// <param name="maskedInputs">The masked inputs the server received.</param>
    /// <param name="clientWeights">The weight each client applied in <see cref="MaskUpdate"/>.</param>
    /// <param name="unmaskingClientIds">The unmasking clients; <c>null</c> means all senders.</param>
    public Vector<T> AggregateWeightedAverage(
        IEnumerable<SecureAggregationMaskedInput> maskedInputs,
        IReadOnlyDictionary<int, double> clientWeights,
        IEnumerable<int>? unmaskingClientIds = null)
    {
        if (maskedInputs is null)
        {
            throw new ArgumentNullException(nameof(maskedInputs));
        }

        if (clientWeights is null)
        {
            throw new ArgumentNullException(nameof(clientWeights));
        }

        var inputs = maskedInputs.ToArray();
        double totalWeight = 0.0;
        foreach (var input in inputs)
        {
            if (input is null || !clientWeights.TryGetValue(input.ClientId, out var weight))
            {
                throw new ArgumentException("Every masked input needs a client weight.", nameof(clientWeights));
            }

            if (double.IsNaN(weight) || double.IsInfinity(weight) || weight <= 0.0)
            {
                throw new ArgumentOutOfRangeException(nameof(clientWeights), "Client weights must be positive finite numbers.");
            }

            totalWeight += weight;
        }

        var ring = AggregateRingSum(inputs, unmaskingClientIds);
        return Decode(ring, totalWeight);
    }

    /// <summary>Drops all clients and server state of the current round.</summary>
    public void ClearSecrets()
    {
        foreach (var client in _clients.Values)
        {
            client.Dispose();
        }

        _clients.Clear();
        _inboxes = new Dictionary<int, IReadOnlyList<SecureAggregationEncryptedShare>>();
        _server = null;
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        ClearSecrets();
        _disposed = true;
    }

    private Vector<T> Decode(ulong[] ringSum, double divisor)
    {
        var values = new T[ringSum.Length];
        for (int i = 0; i < ringSum.Length; i++)
        {
            values[i] = NumOps.FromDouble(Encoding.Decode(ringSum[i]) / divisor);
        }

        return new Vector<T>(values);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(SecureAggregation<T>));
        }
    }
}
