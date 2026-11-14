using System.Security.Cryptography;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.FederatedLearning.Cryptography;
using AiDotNet.FederatedLearning.Infrastructure;

namespace AiDotNet.FederatedLearning.Privacy;

/// <summary>
/// The client side of one round of Bonawitz et al. (2017) secure aggregation (honest-but-curious protocol).
/// </summary>
/// <remarks>
/// <para>
/// "Practical Secure Aggregation for Privacy-Preserving Machine Learning" (CCS 2017), figure 4. One instance is one
/// client <c>u</c> in one aggregation; call the rounds in order:
/// </para>
/// <list type="number">
/// <item><description><see cref="AdvertiseKeys"/> (round 0): generate the P-256 key pairs <c>(c_u^PK, c_u^SK)</c> and
/// <c>(s_u^PK, s_u^SK)</c> and publish the public halves.</description></item>
/// <item><description><see cref="ShareKeys"/> (round 1): given the broadcast <c>U_1</c>, draw the self-mask seed
/// <c>b_u</c>, split <c>s_u^SK</c> and <c>b_u</c> into t-of-|U_1| Shamir shares, and encrypt each recipient's pair of
/// shares under <c>KA.agree(c_u^SK, c_v^PK)</c>.</description></item>
/// <item><description><see cref="MaskInput"/> (round 2): given the ciphertexts addressed to <c>u</c> (whose senders
/// form <c>U_2</c>), send <c>y_u = x_u + PRG(b_u) + sum_v p_{u,v} mod R</c> with
/// <c>p_{u,v} = +PRG(s_{u,v})</c> if <c>u &lt; v</c>, else <c>-PRG(s_{u,v})</c>.</description></item>
/// <item><description><see cref="Unmask"/> (round 4): given <c>U_3</c>, release <c>s_{v,u}^SK</c> for
/// <c>v</c> in <c>U_2 \ U_3</c> and <c>b_{v,u}</c> for <c>v</c> in <c>U_3</c>.</description></item>
/// </list>
/// <para>
/// All mask arithmetic happens in <c>Z_R</c> on <see cref="ulong"/>, never in <typeparamref name="T"/>: the input is
/// converted once to <see cref="double"/> and fixed-point encoded by <see cref="FixedPointRingEncoding"/>. Every
/// secret comes from the operating system's cryptographic random number generator.
/// </para>
/// <para><b>For Beginners:</b> Each client hides its model update behind random masks that only cancel when the
/// server adds everyone's contributions together. Secret-sharing the mask secrets lets the server finish even when
/// some clients disconnect, without ever being able to remove the masks of a single client.</para>
/// </remarks>
/// <typeparam name="T">The numeric type of the model parameters.</typeparam>
[ComponentType(ComponentType.FederatedAggregator)]
[PipelineStage(PipelineStage.Training)]
public sealed class SecureAggregationClient<T> : FederatedLearningComponentBase<T>, IDisposable
{
    private readonly int _parameterCount;
    private readonly int _threshold;
    private readonly FixedPointRingEncoding _encoding;
    private readonly ISecureAggregationPrg _prg;

    private P256KeyPair? _cipherKeys;
    private P256KeyPair? _maskingKeys;
    private byte[] _selfMaskSeed = Array.Empty<byte>();
    private byte[] _ownMaskingKeyShare = Array.Empty<byte>();
    private byte[] _ownSelfMaskShare = Array.Empty<byte>();
    private readonly Dictionary<int, SecureAggregationAdvertisedKeys> _u1 = new();
    private readonly Dictionary<int, byte[]> _receivedCiphertexts = new();
    private int[] _u2 = Array.Empty<int>();
    private bool _disposed;

    /// <summary>
    /// Creates client <paramref name="clientId"/> for one aggregation over <paramref name="parameterCount"/> values.
    /// </summary>
    /// <param name="clientId">The client identifier <c>u</c>; identifiers order the pairwise mask signs.</param>
    /// <param name="parameterCount">The input length <c>m</c>.</param>
    /// <param name="threshold">The Shamir threshold <c>t</c> (at least 2).</param>
    /// <param name="encoding">The fixed-point ring encoding shared by all participants.</param>
    /// <param name="prg">The PRG shared by all participants; <c>null</c> selects <see cref="AesCtrSecureAggregationPrg"/>.</param>
    public SecureAggregationClient(
        int clientId,
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

        ClientId = clientId;
        _parameterCount = parameterCount;
        _threshold = threshold;
        _encoding = encoding ?? throw new ArgumentNullException(nameof(encoding));
        _prg = prg ?? new AesCtrSecureAggregationPrg();
        Round = SecureAggregationRound.AdvertiseKeys;
    }

    /// <summary>Gets the client identifier <c>u</c>.</summary>
    public int ClientId { get; }

    /// <summary>Gets the next round this client expects to run.</summary>
    public SecureAggregationRound Round { get; private set; }

    /// <summary>Round 0 (AdvertiseKeys): generates both key pairs and returns the public keys.</summary>
    public SecureAggregationAdvertisedKeys AdvertiseKeys()
    {
        ExpectRound(SecureAggregationRound.AdvertiseKeys);
        _cipherKeys = P256KeyAgreement.Generate();
        _maskingKeys = P256KeyAgreement.Generate();
        Round = SecureAggregationRound.ShareKeys;
        return new SecureAggregationAdvertisedKeys(
            ClientId,
            (byte[])_cipherKeys.PublicKey.Clone(),
            (byte[])_maskingKeys.PublicKey.Clone());
    }

    /// <summary>
    /// Round 1 (ShareKeys): receives the server's broadcast of <c>U_1</c> and returns one encrypted share message for
    /// every other client in <c>U_1</c>.
    /// </summary>
    /// <exception cref="SecureAggregationAbortedException"><c>|U_1| &lt; t</c>, the list omits or alters this client's
    /// keys, or two clients advertised the same key.</exception>
    public IReadOnlyList<SecureAggregationEncryptedShare> ShareKeys(IReadOnlyList<SecureAggregationAdvertisedKeys> advertisedKeys)
    {
        ExpectRound(SecureAggregationRound.ShareKeys);
        if (advertisedKeys is null)
        {
            throw new ArgumentNullException(nameof(advertisedKeys));
        }

        var cipherKeys = RequireKeys(_cipherKeys);
        var maskingKeys = RequireKeys(_maskingKeys);
        var seenKeys = new HashSet<string>(StringComparer.Ordinal);
        foreach (var keys in advertisedKeys)
        {
            if (keys is null)
            {
                throw Abort(SecureAggregationRound.ShareKeys, "The advertised key list contains a null entry.");
            }

            if (_u1.ContainsKey(keys.ClientId))
            {
                throw Abort(SecureAggregationRound.ShareKeys, $"Client {keys.ClientId} appears twice in the advertised key list.");
            }

            P256KeyAgreement.ValidatePublicKey(keys.CipherPublicKey, nameof(advertisedKeys));
            P256KeyAgreement.ValidatePublicKey(keys.MaskingPublicKey, nameof(advertisedKeys));
            if (!seenKeys.Add(Convert.ToBase64String(keys.CipherPublicKey)) || !seenKeys.Add(Convert.ToBase64String(keys.MaskingPublicKey)))
            {
                throw Abort(SecureAggregationRound.ShareKeys, "Two advertised public keys are identical.");
            }

            _u1[keys.ClientId] = keys;
        }

        if (!_u1.TryGetValue(ClientId, out var own) ||
            !BytesEqual(own.CipherPublicKey, cipherKeys.PublicKey) ||
            !BytesEqual(own.MaskingPublicKey, maskingKeys.PublicKey))
        {
            throw Abort(SecureAggregationRound.ShareKeys, $"The advertised key list does not contain client {ClientId}'s own keys.");
        }

        if (_u1.Count < _threshold)
        {
            throw Abort(SecureAggregationRound.ShareKeys, $"Only {_u1.Count} clients advertised keys; the threshold is {_threshold}.");
        }

        var u1Sorted = _u1.Keys.OrderBy(id => id).ToArray();
        var x = SecureAggregationProtocol.XCoordinates(u1Sorted);

        _selfMaskSeed = new byte[SecureAggregationProtocol.SeedLength];
        using (var rng = RandomNumberGenerator.Create())
        {
            rng.GetBytes(_selfMaskSeed);
        }

        var maskingKeyShares = ShamirSecretSharing.SplitSecret(
            maskingKeys.PrivateScalar, x, _threshold, deterministicSeed: null, info: $"secagg:sk:{ClientId}");
        var selfMaskShares = ShamirSecretSharing.SplitSecret(
            _selfMaskSeed, x, _threshold, deterministicSeed: null, info: $"secagg:b:{ClientId}");

        var messages = new List<SecureAggregationEncryptedShare>(u1Sorted.Length - 1);
        foreach (var recipientId in u1Sorted)
        {
            if (recipientId == ClientId)
            {
                _ownMaskingKeyShare = maskingKeyShares[recipientId];
                _ownSelfMaskShare = selfMaskShares[recipientId];
                continue;
            }

            var key = SecureAggregationProtocol.DeriveShareCipherKey(
                cipherKeys.PrivateScalar, cipherKeys.PublicKey, _u1[recipientId].CipherPublicKey, ClientId, recipientId);
            var plaintext = SecureAggregationProtocol.EncodeSharePlaintext(
                ClientId, recipientId, maskingKeyShares[recipientId], selfMaskShares[recipientId]);
            try
            {
                messages.Add(new SecureAggregationEncryptedShare(
                    ClientId, recipientId, AuthenticatedShareCipher.Encrypt(key, plaintext)));
            }
            finally
            {
                Array.Clear(key, 0, key.Length);
                Array.Clear(plaintext, 0, plaintext.Length);
                Array.Clear(maskingKeyShares[recipientId], 0, ShamirSecretSharing.ShareByteLength);
                Array.Clear(selfMaskShares[recipientId], 0, ShamirSecretSharing.ShareByteLength);
            }
        }

        Round = SecureAggregationRound.MaskedInputCollection;
        return messages;
    }

    /// <summary>
    /// Round 2 (MaskedInputCollection): receives the ciphertexts addressed to this client and returns the masked input.
    /// </summary>
    /// <param name="input">The client's update <c>x_u</c>.</param>
    /// <param name="sharesForThisClient">The ciphertexts <c>e_{v,u}</c> the server forwarded; their senders and this
    /// client form <c>U_2</c>.</param>
    /// <param name="weight">A positive factor applied to the input before encoding (for example the client's share of
    /// the total sample count, so the aggregate is a weighted sum). The default 1 leaves the input unchanged.</param>
    /// <exception cref="SecureAggregationAbortedException"><c>|U_2| &lt; t</c> or a ciphertext is misaddressed.</exception>
    public SecureAggregationMaskedInput MaskInput(
        Vector<T> input,
        IReadOnlyList<SecureAggregationEncryptedShare> sharesForThisClient,
        double weight = 1.0)
    {
        ExpectRound(SecureAggregationRound.MaskedInputCollection);
        if (input is null)
        {
            throw new ArgumentNullException(nameof(input));
        }

        if (input.Length != _parameterCount)
        {
            throw new ArgumentException($"Expected {_parameterCount} parameters, got {input.Length}.", nameof(input));
        }

        if (double.IsNaN(weight) || double.IsInfinity(weight) || weight <= 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(weight), "Weight must be a positive finite number.");
        }

        if (sharesForThisClient is null)
        {
            throw new ArgumentNullException(nameof(sharesForThisClient));
        }

        foreach (var share in sharesForThisClient)
        {
            if (share is null || share.RecipientId != ClientId || share.SenderId == ClientId || !_u1.ContainsKey(share.SenderId))
            {
                throw Abort(SecureAggregationRound.MaskedInputCollection, "A forwarded share is misaddressed or from an unknown client.");
            }

            if (_receivedCiphertexts.ContainsKey(share.SenderId))
            {
                throw Abort(SecureAggregationRound.MaskedInputCollection, $"Two shares from client {share.SenderId} were forwarded.");
            }

            _receivedCiphertexts[share.SenderId] = share.Ciphertext;
        }

        _u2 = _receivedCiphertexts.Keys.Concat(new[] { ClientId }).OrderBy(id => id).ToArray();
        if (_u2.Length < _threshold)
        {
            throw Abort(SecureAggregationRound.MaskedInputCollection, $"Only {_u2.Length} clients shared keys; the threshold is {_threshold}.");
        }

        var maskingKeys = RequireKeys(_maskingKeys);
        var masked = new ulong[_parameterCount];
        for (int i = 0; i < _parameterCount; i++)
        {
            masked[i] = _encoding.Encode(NumOps.ToDouble(input[i]) * weight);
        }

        var scratch = new ulong[_parameterCount];
        SecureAggregationProtocol.ApplyMask(_prg, _encoding, _selfMaskSeed, masked, scratch, subtract: false);
        foreach (var otherId in _u2)
        {
            if (otherId == ClientId)
            {
                continue;
            }

            var seed = SecureAggregationProtocol.DerivePairwiseSeed(
                maskingKeys.PrivateScalar, maskingKeys.PublicKey, _u1[otherId].MaskingPublicKey, ClientId, otherId);
            try
            {
                SecureAggregationProtocol.ApplyMask(_prg, _encoding, seed, masked, scratch, subtract: ClientId > otherId);
            }
            finally
            {
                Array.Clear(seed, 0, seed.Length);
            }
        }

        Round = SecureAggregationRound.Unmasking;
        return new SecureAggregationMaskedInput(ClientId, masked);
    }

    /// <summary>
    /// Round 4 (Unmasking): receives <c>U_3</c>, the clients whose masked inputs the server holds, and returns this
    /// client's shares of the dropped clients' masking keys and of the survivors' self-mask seeds.
    /// </summary>
    /// <exception cref="SecureAggregationAbortedException"><c>|U_3| &lt; t</c>, <c>U_3</c> is not a subset of
    /// <c>U_2</c> containing this client, or a share fails authentication.</exception>
    public SecureAggregationUnmaskingResponse Unmask(IReadOnlyCollection<int> survivingClientIds)
    {
        ExpectRound(SecureAggregationRound.Unmasking);
        if (survivingClientIds is null)
        {
            throw new ArgumentNullException(nameof(survivingClientIds));
        }

        var u2 = new HashSet<int>(_u2);
        var u3 = new HashSet<int>(survivingClientIds);
        if (u3.Count != survivingClientIds.Count || !u3.IsSubsetOf(u2) || !u3.Contains(ClientId))
        {
            throw Abort(SecureAggregationRound.Unmasking, "The surviving-client list must be a duplicate-free subset of U_2 that contains this client.");
        }

        if (u3.Count < _threshold)
        {
            throw Abort(SecureAggregationRound.Unmasking, $"Only {u3.Count} clients sent masked inputs; the threshold is {_threshold}.");
        }

        var cipherKeys = RequireKeys(_cipherKeys);
        var maskingKeyShares = new Dictionary<int, byte[]>();
        var selfMaskShares = new Dictionary<int, byte[]>
        {
            [ClientId] = (byte[])_ownSelfMaskShare.Clone(),
        };

        foreach (var senderId in _u2)
        {
            if (senderId == ClientId)
            {
                continue;
            }

            var key = SecureAggregationProtocol.DeriveShareCipherKey(
                cipherKeys.PrivateScalar, cipherKeys.PublicKey, _u1[senderId].CipherPublicKey, ClientId, senderId);
            byte[] plaintext;
            bool authentic;
            try
            {
                authentic = AuthenticatedShareCipher.TryDecrypt(key, _receivedCiphertexts[senderId], out plaintext);
            }
            finally
            {
                Array.Clear(key, 0, key.Length);
            }

            if (!authentic ||
                !SecureAggregationProtocol.TryDecodeSharePlaintext(plaintext, out int from, out int to, out var maskingKeyShare, out var selfMaskShare) ||
                from != senderId || to != ClientId)
            {
                Array.Clear(plaintext, 0, plaintext.Length);
                throw Abort(SecureAggregationRound.Unmasking, $"The share from client {senderId} failed authentication.");
            }

            Array.Clear(plaintext, 0, plaintext.Length);
            if (u3.Contains(senderId))
            {
                selfMaskShares[senderId] = selfMaskShare;
                Array.Clear(maskingKeyShare, 0, maskingKeyShare.Length);
            }
            else
            {
                maskingKeyShares[senderId] = maskingKeyShare;
                Array.Clear(selfMaskShare, 0, selfMaskShare.Length);
            }
        }

        ClearSecrets();
        Round = SecureAggregationRound.Completed;
        return new SecureAggregationUnmaskingResponse(ClientId, maskingKeyShares, selfMaskShares);
    }

    /// <summary>Zeroes every secret this client holds and ends its participation.</summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        ClearSecrets();
        Round = SecureAggregationRound.Completed;
        _disposed = true;
    }

    private void ClearSecrets()
    {
        _cipherKeys?.Clear();
        _maskingKeys?.Clear();
        _cipherKeys = null;
        _maskingKeys = null;
        Array.Clear(_selfMaskSeed, 0, _selfMaskSeed.Length);
        Array.Clear(_ownMaskingKeyShare, 0, _ownMaskingKeyShare.Length);
        Array.Clear(_ownSelfMaskShare, 0, _ownSelfMaskShare.Length);
        _receivedCiphertexts.Clear();
    }

    private void ExpectRound(SecureAggregationRound expected)
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(SecureAggregationClient<T>));
        }

        if (Round != expected)
        {
            throw new InvalidOperationException($"Client {ClientId} expected round {Round}, not {expected}.");
        }
    }

    private SecureAggregationAbortedException Abort(SecureAggregationRound round, string message)
    {
        ClearSecrets();
        Round = SecureAggregationRound.Completed;
        return new SecureAggregationAbortedException(round, message);
    }

    private static P256KeyPair RequireKeys(P256KeyPair? keys)
        => keys ?? throw new InvalidOperationException("The client's keys are not available; run AdvertiseKeys first.");

    private static bool BytesEqual(byte[] left, byte[] right)
    {
        if (left.Length != right.Length)
        {
            return false;
        }

        int difference = 0;
        for (int i = 0; i < left.Length; i++)
        {
            difference |= left[i] ^ right[i];
        }

        return difference == 0;
    }
}
