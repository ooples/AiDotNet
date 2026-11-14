using System.Text;
using AiDotNet.FederatedLearning.Cryptography;
using AiDotNet.Models.Options;

namespace AiDotNet.FederatedLearning.Privacy;

/// <summary>
/// Primitives shared by <see cref="SecureAggregationClient{T}"/> and <see cref="SecureAggregationServer{T}"/>: key
/// derivation, share encoding, Shamir x-coordinates, mask expansion and threshold resolution.
/// </summary>
internal static class SecureAggregationProtocol
{
    internal const int SeedLength = 32;

    private const string Domain = "AiDotNet.SecAgg.Bonawitz2017.v1";
    private const int SharePlaintextLength = 8 + 2 * ShamirSecretSharing.ShareByteLength;
    private static readonly byte[] HkdfSalt = Encoding.UTF8.GetBytes(Domain);

    /// <summary>
    /// The pairwise mask seed <c>s_{u,v}</c>: HKDF-SHA256 of <c>KA.agree(s_u^SK, s_v^PK)</c>, bound to the unordered pair.
    /// </summary>
    internal static byte[] DerivePairwiseSeed(byte[] privateScalar, byte[] ownPublicKey, byte[] peerPublicKey, int a, int b)
    {
        var shared = P256KeyAgreement.Agree(privateScalar, ownPublicKey, peerPublicKey);
        try
        {
            return HkdfSha256.DeriveKey(shared, HkdfSalt, PairInfo("mask", a, b), SeedLength);
        }
        finally
        {
            Array.Clear(shared, 0, shared.Length);
        }
    }

    /// <summary>The share-encryption key <c>KA.agree(c_u^SK, c_v^PK)</c>, expanded with HKDF-SHA256.</summary>
    internal static byte[] DeriveShareCipherKey(byte[] privateScalar, byte[] ownPublicKey, byte[] peerPublicKey, int a, int b)
    {
        var shared = P256KeyAgreement.Agree(privateScalar, ownPublicKey, peerPublicKey);
        try
        {
            return HkdfSha256.DeriveKey(shared, HkdfSalt, PairInfo("enc", a, b), AuthenticatedShareCipher.KeyLength);
        }
        finally
        {
            Array.Clear(shared, 0, shared.Length);
        }
    }

    /// <summary>Shamir x-coordinates: the 1-based position of each client in the sorted set <c>U_1</c>.</summary>
    internal static Dictionary<int, int> XCoordinates(IReadOnlyList<int> sortedClientIds)
    {
        var x = new Dictionary<int, int>(sortedClientIds.Count);
        for (int i = 0; i < sortedClientIds.Count; i++)
        {
            x[sortedClientIds[i]] = i + 1;
        }

        return x;
    }

    /// <summary>Encodes the share plaintext <c>u || v || s_{u,v}^SK || b_{u,v}</c>.</summary>
    internal static byte[] EncodeSharePlaintext(int senderId, int recipientId, byte[] maskingKeyShare, byte[] selfMaskShare)
    {
        if (maskingKeyShare.Length != ShamirSecretSharing.ShareByteLength || selfMaskShare.Length != ShamirSecretSharing.ShareByteLength)
        {
            throw new ArgumentException("Shamir shares have an unexpected length.");
        }

        var plaintext = new byte[SharePlaintextLength];
        WriteInt32(plaintext, 0, senderId);
        WriteInt32(plaintext, 4, recipientId);
        Buffer.BlockCopy(maskingKeyShare, 0, plaintext, 8, ShamirSecretSharing.ShareByteLength);
        Buffer.BlockCopy(selfMaskShare, 0, plaintext, 8 + ShamirSecretSharing.ShareByteLength, ShamirSecretSharing.ShareByteLength);
        return plaintext;
    }

    /// <summary>Decodes a share plaintext; returns false when it is malformed.</summary>
    internal static bool TryDecodeSharePlaintext(
        byte[] plaintext,
        out int senderId,
        out int recipientId,
        out byte[] maskingKeyShare,
        out byte[] selfMaskShare)
    {
        senderId = 0;
        recipientId = 0;
        maskingKeyShare = Array.Empty<byte>();
        selfMaskShare = Array.Empty<byte>();
        if (plaintext.Length != SharePlaintextLength)
        {
            return false;
        }

        senderId = ReadInt32(plaintext, 0);
        recipientId = ReadInt32(plaintext, 4);
        maskingKeyShare = new byte[ShamirSecretSharing.ShareByteLength];
        selfMaskShare = new byte[ShamirSecretSharing.ShareByteLength];
        Buffer.BlockCopy(plaintext, 8, maskingKeyShare, 0, ShamirSecretSharing.ShareByteLength);
        Buffer.BlockCopy(plaintext, 8 + ShamirSecretSharing.ShareByteLength, selfMaskShare, 0, ShamirSecretSharing.ShareByteLength);
        return true;
    }

    /// <summary>
    /// Encodes <c>weight * x_i</c> for every element: one conversion of <typeparamref name="T"/> to <see cref="double"/>,
    /// then <see cref="FixedPointRingEncoding.Encode(double)"/>.
    /// </summary>
    [AiDotNet.Attributes.ScalarKernel("Fixed-point encoding into Z_(2^b) is per-element rounding and modular integer arithmetic; T appears only at this one conversion to and from double, for which there is no engine op.")]
    internal static ulong[] EncodeVector<T>(FixedPointRingEncoding encoding, Vector<T> input, double weight)
    {
        var numOps = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        var values = input.ToArray();
        var encoded = new ulong[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            encoded[i] = encoding.Encode(numOps.ToDouble(values[i]) * weight);
        }

        return encoded;
    }

    /// <summary>
    /// Decodes a ring vector: each element to <see cref="double"/>, divided by <paramref name="divisor"/> in
    /// <see cref="double"/>, then rounded once to <typeparamref name="T"/>.
    /// </summary>
    [AiDotNet.Attributes.ScalarKernel("Fixed-point encoding into Z_(2^b) is per-element rounding and modular integer arithmetic; T appears only at this one conversion to and from double, for which there is no engine op.")]
    internal static Vector<T> DecodeVector<T>(FixedPointRingEncoding encoding, ulong[] ringSum, double divisor)
    {
        var numOps = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        var values = new T[ringSum.Length];
        for (int i = 0; i < ringSum.Length; i++)
        {
            values[i] = numOps.FromDouble(encoding.Decode(ringSum[i]) / divisor);
        }

        return new Vector<T>(values);
    }

    /// <summary>
    /// Adds (or, when <paramref name="subtract"/> is set, subtracts) <c>PRG(seed)</c> into <paramref name="accumulator"/>
    /// modulo <c>R</c>. <paramref name="scratch"/> must have the accumulator's length.
    /// </summary>
    internal static void ApplyMask(
        ISecureAggregationPrg prg,
        FixedPointRingEncoding encoding,
        byte[] seed,
        ulong[] accumulator,
        ulong[] scratch,
        bool subtract)
    {
        prg.Expand(seed, encoding.BitWidth, scratch);
        if (subtract)
        {
            encoding.SubtractInPlace(accumulator, scratch);
        }
        else
        {
            encoding.AddInPlace(accumulator, scratch);
        }

        Array.Clear(scratch, 0, scratch.Length);
    }

    /// <summary>
    /// Resolves the threshold <c>t</c> for <paramref name="clientCount"/> clients.
    /// </summary>
    /// <remarks>
    /// <see cref="SecureAggregationMode.FullParticipation"/> uses <c>t = n</c>: any dropout aborts the round. In
    /// <see cref="SecureAggregationMode.ThresholdDropoutResilient"/> a configured threshold is used as given; otherwise
    /// the default is <c>floor(2n/3) + 1</c>, the bound <c>t &gt; 2n/3</c> under which Bonawitz et al. (2017), section 6,
    /// prove security even against an active adversary, which tolerates just under a third of the clients dropping.
    /// </remarks>
    internal static int ResolveThreshold(int clientCount, SecureAggregationMode mode, int configuredThreshold)
    {
        if (clientCount < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(clientCount), "Secure aggregation needs at least two clients.");
        }

        if (mode == SecureAggregationMode.FullParticipation)
        {
            return clientCount;
        }

        if (mode != SecureAggregationMode.ThresholdDropoutResilient)
        {
            throw new ArgumentOutOfRangeException(nameof(mode), $"Unknown secure aggregation mode '{mode}'.");
        }

        if (configuredThreshold > 0)
        {
            if (configuredThreshold < 2 || configuredThreshold > clientCount)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(configuredThreshold),
                    $"The reconstruction threshold must be between 2 and the number of clients ({clientCount}).");
            }

            return configuredThreshold;
        }

        int threshold = (2 * clientCount / 3) + 1;
        return Math.Max(2, Math.Min(clientCount, threshold));
    }

    private static byte[] PairInfo(string purpose, int a, int b)
    {
        int min = Math.Min(a, b);
        int max = Math.Max(a, b);
        return Encoding.UTF8.GetBytes($"{Domain}|{purpose}|{min}|{max}");
    }

    private static void WriteInt32(byte[] buffer, int offset, int value)
    {
        unchecked
        {
            buffer[offset] = (byte)value;
            buffer[offset + 1] = (byte)(value >> 8);
            buffer[offset + 2] = (byte)(value >> 16);
            buffer[offset + 3] = (byte)(value >> 24);
        }
    }

    private static int ReadInt32(byte[] buffer, int offset)
    {
        return buffer[offset]
            | (buffer[offset + 1] << 8)
            | (buffer[offset + 2] << 16)
            | (buffer[offset + 3] << 24);
    }
}
