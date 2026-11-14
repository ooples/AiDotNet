using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.FederatedLearning.Cryptography;
using AiDotNet.FederatedLearning.Infrastructure;

namespace AiDotNet.FederatedLearning.Privacy;

/// <summary>
/// The server side of one round of Bonawitz et al. (2017) secure aggregation (honest-but-curious protocol).
/// </summary>
/// <remarks>
/// <para>
/// "Practical Secure Aggregation for Privacy-Preserving Machine Learning" (CCS 2017), figure 4. Call in order:
/// </para>
/// <list type="number">
/// <item><description><see cref="CollectAdvertisedKeys"/> (round 0): the clients that advertised keys form
/// <c>U_1</c>; the returned list is broadcast to them.</description></item>
/// <item><description><see cref="CollectShares"/> (round 1): the clients that sent a complete set of encrypted shares
/// form <c>U_2</c>; the result routes each ciphertext to its recipient.</description></item>
/// <item><description><see cref="CollectMaskedInputs"/> (round 2): the clients that sent a masked input form
/// <c>U_3</c>; the returned list is sent to them.</description></item>
/// <item><description><see cref="AggregateRingSum"/> or <see cref="AggregateSum"/> (round 4): with responses from at
/// least <c>t</c> clients of <c>U_3</c>, reconstruct <c>s_v^SK</c> for every dropped <c>v</c> in <c>U_2 \ U_3</c> and
/// <c>b_u</c> for every <c>u</c> in <c>U_3</c>, and output
/// <c>z = sum_{u in U_3} y_u - sum_{u in U_3} PRG(b_u) + sum_{u in U_3, v in U_2 \ U_3} p_{v,u} mod R</c>.</description></item>
/// </list>
/// <para>
/// Every round aborts with <see cref="SecureAggregationAbortedException"/> when fewer than <c>t</c> clients remain.
/// The server sees only public keys, ciphertexts, masked inputs and the shares the honest clients release, which by
/// the paper's security analysis (section 6) reveal nothing beyond the sum over <c>U_3</c>.
/// </para>
/// <para><b>For Beginners:</b> The server coordinates the rounds and adds up the masked vectors. Only once enough
/// clients have answered the final round can it strip the masks, and what it gets is the total, never one client's
/// update.</para>
/// </remarks>
/// <typeparam name="T">The numeric type of the model parameters.</typeparam>
[ComponentType(ComponentType.FederatedAggregator)]
[PipelineStage(PipelineStage.Training)]
public sealed class SecureAggregationServer<T> : FederatedLearningComponentBase<T>
{
    private readonly int _parameterCount;
    private readonly int _threshold;
    private readonly FixedPointRingEncoding _encoding;
    private readonly ISecureAggregationPrg _prg;

    private readonly SortedDictionary<int, SecureAggregationAdvertisedKeys> _u1 = new();
    private int[] _u2 = Array.Empty<int>();
    private readonly SortedDictionary<int, ulong[]> _maskedInputs = new();

    /// <summary>
    /// Creates the server for one aggregation over <paramref name="parameterCount"/> values.
    /// </summary>
    /// <param name="parameterCount">The input length <c>m</c>.</param>
    /// <param name="threshold">The Shamir threshold <c>t</c> (at least 2).</param>
    /// <param name="encoding">The fixed-point ring encoding shared by all participants.</param>
    /// <param name="prg">The PRG shared by all participants; <c>null</c> selects <see cref="AesCtrSecureAggregationPrg"/>.</param>
    public SecureAggregationServer(
        int parameterCount,
        int threshold,
        FixedPointRingEncoding encoding,
        ISecureAggregationPrg? prg = null)
    {
        if (parameterCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(parameterCount), "Parameter count must be positive.");
        }

        if (threshold < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(threshold), "The threshold must be at least 2.");
        }

        _parameterCount = parameterCount;
        _threshold = threshold;
        _encoding = encoding ?? throw new ArgumentNullException(nameof(encoding));
        _prg = prg ?? new AesCtrSecureAggregationPrg();
        Round = SecureAggregationRound.AdvertiseKeys;
    }

    /// <summary>Gets the next round the server expects to run.</summary>
    public SecureAggregationRound Round { get; private set; }

    /// <summary>Gets the threshold <c>t</c>.</summary>
    public int Threshold => _threshold;

    /// <summary>Gets the encoding used to decode the aggregate.</summary>
    public FixedPointRingEncoding Encoding => _encoding;

    /// <summary>Gets <c>U_3</c>, the clients whose masked inputs are in the aggregate (empty before round 2).</summary>
    public IReadOnlyList<int> SurvivingClientIds => _maskedInputs.Keys.ToArray();

    /// <summary>
    /// Round 0: records the advertised keys of <c>U_1</c> and returns the list to broadcast, ordered by client id.
    /// </summary>
    /// <exception cref="SecureAggregationAbortedException"><c>|U_1| &lt; t</c>.</exception>
    public IReadOnlyList<SecureAggregationAdvertisedKeys> CollectAdvertisedKeys(IEnumerable<SecureAggregationAdvertisedKeys> advertisedKeys)
    {
        ExpectRound(SecureAggregationRound.AdvertiseKeys);
        if (advertisedKeys is null)
        {
            throw new ArgumentNullException(nameof(advertisedKeys));
        }

        var seenKeys = new HashSet<string>(StringComparer.Ordinal);
        foreach (var keys in advertisedKeys)
        {
            if (keys is null)
            {
                throw new ArgumentException("The advertised key list contains a null entry.", nameof(advertisedKeys));
            }

            if (_u1.ContainsKey(keys.ClientId))
            {
                throw new ArgumentException($"Client {keys.ClientId} advertised keys twice.", nameof(advertisedKeys));
            }

            P256KeyAgreement.ValidatePublicKey(keys.CipherPublicKey, nameof(advertisedKeys));
            P256KeyAgreement.ValidatePublicKey(keys.MaskingPublicKey, nameof(advertisedKeys));
            if (!seenKeys.Add(Convert.ToBase64String(keys.CipherPublicKey)) || !seenKeys.Add(Convert.ToBase64String(keys.MaskingPublicKey)))
            {
                throw new ArgumentException("Two advertised public keys are identical.", nameof(advertisedKeys));
            }

            _u1[keys.ClientId] = keys;
        }

        RequireThreshold(_u1.Count, SecureAggregationRound.AdvertiseKeys, "advertised keys");
        _encoding.EnsureSumFits(_u1.Count);
        Round = SecureAggregationRound.ShareKeys;
        return _u1.Values.ToArray();
    }

    /// <summary>
    /// Round 1: collects every client's encrypted shares. A client joins <c>U_2</c> only if it sent exactly one
    /// ciphertext to each other client of <c>U_1</c>. Returns, for every client of <c>U_2</c>, the ciphertexts from
    /// the other members of <c>U_2</c> addressed to it.
    /// </summary>
    /// <exception cref="SecureAggregationAbortedException"><c>|U_2| &lt; t</c>.</exception>
    public IReadOnlyDictionary<int, IReadOnlyList<SecureAggregationEncryptedShare>> CollectShares(
        IEnumerable<SecureAggregationEncryptedShare> shares)
    {
        ExpectRound(SecureAggregationRound.ShareKeys);
        if (shares is null)
        {
            throw new ArgumentNullException(nameof(shares));
        }

        var bySender = new Dictionary<int, Dictionary<int, SecureAggregationEncryptedShare>>();
        foreach (var share in shares)
        {
            if (share is null)
            {
                throw new ArgumentException("The share list contains a null entry.", nameof(shares));
            }

            if (!_u1.ContainsKey(share.SenderId) || !_u1.ContainsKey(share.RecipientId) || share.SenderId == share.RecipientId)
            {
                throw new ArgumentException($"A share from {share.SenderId} to {share.RecipientId} is not between two distinct clients of U_1.", nameof(shares));
            }

            if (!bySender.TryGetValue(share.SenderId, out var sent))
            {
                sent = new Dictionary<int, SecureAggregationEncryptedShare>();
                bySender[share.SenderId] = sent;
            }

            if (sent.ContainsKey(share.RecipientId))
            {
                throw new ArgumentException($"Client {share.SenderId} sent two shares to client {share.RecipientId}.", nameof(shares));
            }

            sent[share.RecipientId] = share;
        }

        _u2 = bySender
            .Where(pair => pair.Value.Count == _u1.Count - 1)
            .Select(pair => pair.Key)
            .OrderBy(id => id)
            .ToArray();
        RequireThreshold(_u2.Length, SecureAggregationRound.ShareKeys, "complete encrypted shares");

        var routed = new Dictionary<int, IReadOnlyList<SecureAggregationEncryptedShare>>(_u2.Length);
        foreach (var recipientId in _u2)
        {
            var inbox = new List<SecureAggregationEncryptedShare>(_u2.Length - 1);
            foreach (var senderId in _u2)
            {
                if (senderId != recipientId)
                {
                    inbox.Add(bySender[senderId][recipientId]);
                }
            }

            routed[recipientId] = inbox;
        }

        Round = SecureAggregationRound.MaskedInputCollection;
        return routed;
    }

    /// <summary>
    /// Round 2: collects the masked inputs. Their senders form <c>U_3</c>, which is returned and must be sent to them.
    /// </summary>
    /// <exception cref="SecureAggregationAbortedException"><c>|U_3| &lt; t</c>.</exception>
    public IReadOnlyList<int> CollectMaskedInputs(IEnumerable<SecureAggregationMaskedInput> maskedInputs)
    {
        ExpectRound(SecureAggregationRound.MaskedInputCollection);
        if (maskedInputs is null)
        {
            throw new ArgumentNullException(nameof(maskedInputs));
        }

        var u2 = new HashSet<int>(_u2);
        foreach (var input in maskedInputs)
        {
            if (input is null)
            {
                throw new ArgumentException("The masked input list contains a null entry.", nameof(maskedInputs));
            }

            if (!u2.Contains(input.ClientId))
            {
                throw new ArgumentException($"Client {input.ClientId} is not in U_2 and cannot contribute a masked input.", nameof(maskedInputs));
            }

            if (_maskedInputs.ContainsKey(input.ClientId))
            {
                throw new ArgumentException($"Client {input.ClientId} sent two masked inputs.", nameof(maskedInputs));
            }

            if (input.Values.Length != _parameterCount)
            {
                throw new ArgumentException($"Client {input.ClientId} sent {input.Values.Length} values; expected {_parameterCount}.", nameof(maskedInputs));
            }

            _maskedInputs[input.ClientId] = (ulong[])input.Values.Clone();
        }

        RequireThreshold(_maskedInputs.Count, SecureAggregationRound.MaskedInputCollection, "masked inputs");
        Round = SecureAggregationRound.Unmasking;
        return _maskedInputs.Keys.ToArray();
    }

    /// <summary>
    /// Round 4: removes all masks and returns <c>sum_{u in U_3} x_u mod R</c>, the exact modular sum of the survivors'
    /// fixed-point encodings.
    /// </summary>
    /// <exception cref="SecureAggregationAbortedException">Fewer than <c>t</c> clients of <c>U_3</c> responded, or
    /// fewer than <c>t</c> shares exist for a secret that must be reconstructed.</exception>
    public ulong[] AggregateRingSum(IEnumerable<SecureAggregationUnmaskingResponse> responses)
    {
        ExpectRound(SecureAggregationRound.Unmasking);
        if (responses is null)
        {
            throw new ArgumentNullException(nameof(responses));
        }

        var u3 = new HashSet<int>(_maskedInputs.Keys);
        var dropped = _u2.Where(id => !u3.Contains(id)).ToArray();
        var droppedSet = new HashSet<int>(dropped);
        var byResponder = new SortedDictionary<int, SecureAggregationUnmaskingResponse>();
        foreach (var response in responses)
        {
            if (response is null)
            {
                throw new ArgumentException("The response list contains a null entry.", nameof(responses));
            }

            if (!u3.Contains(response.ClientId) || byResponder.ContainsKey(response.ClientId))
            {
                throw new ArgumentException($"Client {response.ClientId} is not in U_3 or responded twice.", nameof(responses));
            }

            if (response.MaskingKeyShares.Keys.Any(id => !droppedSet.Contains(id)) ||
                response.SelfMaskShares.Keys.Any(id => !u3.Contains(id)))
            {
                throw new ArgumentException(
                    $"Client {response.ClientId} released a masking-key share for a survivor or a self-mask share for a dropped client.",
                    nameof(responses));
            }

            byResponder[response.ClientId] = response;
        }

        RequireThreshold(byResponder.Count, SecureAggregationRound.Unmasking, "unmasking responses");

        var x = SecureAggregationProtocol.XCoordinates(_u1.Keys.ToArray());
        var sum = new ulong[_parameterCount];
        foreach (var maskedInput in _maskedInputs.Values)
        {
            _encoding.AddInPlace(sum, maskedInput);
        }

        var scratch = new ulong[_parameterCount];

        // Dropped clients v in U_2 \ U_3: every survivor u added p_{u,v} = +PRG(s_{u,v}) if u < v, else -PRG(s_{u,v}).
        // Recompute s_{u,v} from the reconstructed s_v^SK and u's public key, and take p_{u,v} back out.
        foreach (var droppedId in dropped)
        {
            var maskingKey = Reconstruct(byResponder.Values, r => r.MaskingKeyShares, droppedId, x, "masking key");
            try
            {
                var droppedPublicKey = _u1[droppedId].MaskingPublicKey;
                if (!P256Sha256Agreement.IsKeyPair(maskingKey, droppedPublicKey))
                {
                    throw Abort(SecureAggregationRound.Unmasking, $"The masking key reconstructed for client {droppedId} does not match its public key.");
                }

                foreach (var survivorId in _maskedInputs.Keys)
                {
                    var seed = SecureAggregationProtocol.DerivePairwiseSeed(
                        maskingKey, droppedPublicKey, _u1[survivorId].MaskingPublicKey, droppedId, survivorId);
                    try
                    {
                        SecureAggregationProtocol.ApplyMask(_prg, _encoding, seed, sum, scratch, subtract: survivorId < droppedId);
                    }
                    finally
                    {
                        Array.Clear(seed, 0, seed.Length);
                    }
                }
            }
            finally
            {
                Array.Clear(maskingKey, 0, maskingKey.Length);
            }
        }

        // Survivors u in U_3: remove PRG(b_u).
        foreach (var survivorId in _maskedInputs.Keys)
        {
            var selfMaskSeed = Reconstruct(byResponder.Values, r => r.SelfMaskShares, survivorId, x, "self-mask seed");
            try
            {
                SecureAggregationProtocol.ApplyMask(_prg, _encoding, selfMaskSeed, sum, scratch, subtract: true);
            }
            finally
            {
                Array.Clear(selfMaskSeed, 0, selfMaskSeed.Length);
            }
        }

        _maskedInputs.Clear();
        Round = SecureAggregationRound.Completed;
        return sum;
    }

    /// <summary>
    /// Round 4, decoded: <see cref="AggregateRingSum"/> decoded to <see cref="double"/> and rounded once to
    /// <typeparamref name="T"/>.
    /// </summary>
    public Vector<T> AggregateSum(IEnumerable<SecureAggregationUnmaskingResponse> responses)
        => Decode(AggregateRingSum(responses), divisor: 1.0);

    /// <summary>
    /// Decodes a ring vector: each element is decoded to <see cref="double"/>, divided by <paramref name="divisor"/>
    /// in <see cref="double"/>, and rounded once to <typeparamref name="T"/>.
    /// </summary>
    public Vector<T> Decode(ulong[] ringSum, double divisor)
    {
        if (ringSum is null)
        {
            throw new ArgumentNullException(nameof(ringSum));
        }

        if (double.IsNaN(divisor) || double.IsInfinity(divisor) || divisor <= 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(divisor), "The divisor must be a positive finite number.");
        }

        var values = new T[ringSum.Length];
        for (int i = 0; i < ringSum.Length; i++)
        {
            values[i] = NumOps.FromDouble(_encoding.Decode(ringSum[i]) / divisor);
        }

        return new Vector<T>(values);
    }

    private byte[] Reconstruct(
        IEnumerable<SecureAggregationUnmaskingResponse> responses,
        Func<SecureAggregationUnmaskingResponse, IReadOnlyDictionary<int, byte[]>> select,
        int ownerId,
        IReadOnlyDictionary<int, int> x,
        string what)
    {
        var shares = new Dictionary<int, byte[]>();
        foreach (var response in responses)
        {
            if (select(response).TryGetValue(ownerId, out var share) && share is not null)
            {
                shares[response.ClientId] = share;
            }
        }

        if (shares.Count < _threshold)
        {
            throw Abort(
                SecureAggregationRound.Unmasking,
                $"Only {shares.Count} shares of client {ownerId}'s {what} were released; the threshold is {_threshold}.");
        }

        return ShamirSecretSharing.CombineShares(shares, x, _threshold, SecureAggregationProtocol.SeedLength);
    }

    private void RequireThreshold(int count, SecureAggregationRound round, string what)
    {
        if (count < _threshold)
        {
            throw Abort(round, $"Secure aggregation aborted in round {round}: {count} clients sent {what}; the threshold is {_threshold}.");
        }
    }

    private SecureAggregationAbortedException Abort(SecureAggregationRound round, string message)
    {
        _maskedInputs.Clear();
        Round = SecureAggregationRound.Completed;
        return new SecureAggregationAbortedException(round, message);
    }

    private void ExpectRound(SecureAggregationRound expected)
    {
        if (Round != expected)
        {
            throw new InvalidOperationException($"The server expected round {Round}, not {expected}.");
        }
    }
}
