using System.Numerics;
using System.Security.Cryptography;
using System.Text;

namespace AiDotNet.FederatedLearning.Cryptography;

/// <summary>
/// Internal Shamir secret sharing implementation used by dropout-resilient secure aggregation.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Secret sharing lets you split a secret into many pieces ("shares") so that:
/// - Any <c>threshold</c> number of shares can reconstruct the secret.
/// - Fewer than <c>threshold</c> shares reveal nothing useful about the secret.
/// </remarks>
internal static class ShamirSecretSharing
{
    private static readonly BigInteger Prime = (BigInteger.One << 521) - BigInteger.One;
    internal const int ShareByteLength = 66;

    internal static Dictionary<int, byte[]> SplitSecret(
        byte[] secret,
        IReadOnlyDictionary<int, int> xByRecipient,
        int threshold,
        int? deterministicSeed,
        string info)
    {
        if (secret == null)
        {
            throw new ArgumentNullException(nameof(secret));
        }

        if (secret.Length == 0)
        {
            throw new ArgumentException("Secret cannot be empty.", nameof(secret));
        }

        if (xByRecipient == null || xByRecipient.Count == 0)
        {
            throw new ArgumentException("Recipient mapping cannot be null or empty.", nameof(xByRecipient));
        }

        if (threshold < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(threshold), "Threshold must be at least 2.");
        }

        if (threshold > xByRecipient.Count)
        {
            throw new ArgumentOutOfRangeException(nameof(threshold), "Threshold cannot exceed the number of recipients.");
        }

        if (string.IsNullOrWhiteSpace(info))
        {
            throw new ArgumentException("Info cannot be null or empty.", nameof(info));
        }

        var secretValue = FromBigEndianUnsigned(secret);
        if (secretValue.Sign < 0 || secretValue >= Prime)
        {
            throw new ArgumentException("Secret is out of range for the selected field.", nameof(secret));
        }

        var coefficients = new BigInteger[threshold - 1];
        for (int i = 0; i < coefficients.Length; i++)
        {
            coefficients[i] = GetFieldElement(deterministicSeed, info, $"coeff:{i + 1}");
        }

        var shares = new Dictionary<int, byte[]>(xByRecipient.Count);
        foreach (var (recipientId, xCoord) in xByRecipient)
        {
            if (xCoord == 0)
            {
                throw new ArgumentException("x coordinates must be non-zero.", nameof(xByRecipient));
            }

            var x = new BigInteger(xCoord);
            var y = secretValue;

            var xPow = x;
            for (int i = 0; i < coefficients.Length; i++)
            {
                y = Mod(y + Mod(coefficients[i] * xPow));
                xPow = Mod(xPow * x);
            }

            shares[recipientId] = ToBigEndianFixedLength(y, ShareByteLength);
        }

        return shares;
    }

    internal static byte[] CombineShares(
        IReadOnlyDictionary<int, byte[]> sharesByRecipient,
        IReadOnlyDictionary<int, int> xByRecipient,
        int threshold,
        int secretLength)
    {
        if (sharesByRecipient == null || sharesByRecipient.Count == 0)
        {
            throw new ArgumentException("Shares cannot be null or empty.", nameof(sharesByRecipient));
        }

        if (xByRecipient == null || xByRecipient.Count == 0)
        {
            throw new ArgumentException("Recipient mapping cannot be null or empty.", nameof(xByRecipient));
        }

        if (threshold < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(threshold), "Threshold must be at least 2.");
        }

        if (secretLength <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(secretLength), "Secret length must be positive.");
        }

        if (sharesByRecipient.Count < threshold)
        {
            throw new ArgumentException($"Need at least {threshold} shares to reconstruct the secret.", nameof(sharesByRecipient));
        }

        var orderedRecipients = sharesByRecipient.Keys
            .Where(id => xByRecipient.ContainsKey(id))
            .OrderBy(id => id)
            .Take(threshold)
            .ToArray();

        if (orderedRecipients.Length < threshold)
        {
            throw new ArgumentException($"Need at least {threshold} shares with known x coordinates to reconstruct the secret.", nameof(sharesByRecipient));
        }

        var xs = new BigInteger[threshold];
        var ys = new BigInteger[threshold];
        for (int i = 0; i < threshold; i++)
        {
            int recipientId = orderedRecipients[i];
            int xCoord = xByRecipient[recipientId];
            if (xCoord == 0)
            {
                throw new ArgumentException("x coordinates must be non-zero.", nameof(xByRecipient));
            }

            xs[i] = new BigInteger(xCoord);
            ys[i] = FromBigEndianUnsigned(sharesByRecipient[recipientId]);
            ys[i] = Mod(ys[i]);
        }

        BigInteger secret = BigInteger.Zero;
        for (int i = 0; i < threshold; i++)
        {
            BigInteger numerator = BigInteger.One;
            BigInteger denominator = BigInteger.One;

            for (int j = 0; j < threshold; j++)
            {
                if (i == j)
                {
                    continue;
                }

                numerator = Mod(numerator * Mod(-xs[j]));
                denominator = Mod(denominator * Mod(xs[i] - xs[j]));
            }

            var lagrange = Mod(numerator * ModInverse(denominator));
            secret = Mod(secret + Mod(ys[i] * lagrange));
        }

        return ToBigEndianFixedLength(secret, secretLength);
    }

    private static BigInteger GetFieldElement(int? deterministicSeed, string info, string label)
    {
        if (deterministicSeed.HasValue)
        {
            var seedBytes = BitConverter.GetBytes(deterministicSeed.Value);
            var salt = Encoding.UTF8.GetBytes("AiDotNet.Shamir.v1");
            var infoBytes = Encoding.UTF8.GetBytes($"{info}:{label}");
            var bytes = HkdfSha256.DeriveKey(seedBytes, salt, infoBytes, length: ShareByteLength);
            return Mod(FromBigEndianUnsigned(bytes));
        }

        var random = new byte[ShareByteLength];
        using (var rng = RandomNumberGenerator.Create())
        {
            rng.GetBytes(random);
        }

        return Mod(FromBigEndianUnsigned(random));
    }

    private static BigInteger Mod(BigInteger value)
    {
        var mod = value % Prime;
        return mod.Sign < 0 ? mod + Prime : mod;
    }

    private static BigInteger ModInverse(BigInteger value)
    {
        var v = Mod(value);
        if (v.IsZero)
        {
            throw new ArgumentException("Cannot invert zero in a finite field.");
        }

        return BigInteger.ModPow(v, Prime - 2, Prime);
    }

    private static BigInteger FromBigEndianUnsigned(byte[] bigEndian)
    {
        if (bigEndian == null)
        {
            throw new ArgumentNullException(nameof(bigEndian));
        }

        if (bigEndian.Length == 0)
        {
            return BigInteger.Zero;
        }

        var little = new byte[bigEndian.Length + 1];
        for (int i = 0; i < bigEndian.Length; i++)
        {
            little[i] = bigEndian[bigEndian.Length - 1 - i];
        }

        little[bigEndian.Length] = 0;
        return new BigInteger(little);
    }

    private static byte[] ToBigEndianFixedLength(BigInteger value, int length)
    {
        if (length <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(length), "Length must be positive.");
        }

        if (value.Sign < 0)
        {
            throw new ArgumentException("Value must be non-negative.", nameof(value));
        }

        var little = value.ToByteArray();

        if (little.Length > 1 && little[little.Length - 1] == 0)
        {
            Array.Resize(ref little, little.Length - 1);
        }

        if (little.Length > length)
        {
            // This should not happen for well-formed field elements, but fail safe rather than truncating.
            throw new ArgumentException("Value does not fit into the requested length.", nameof(value));
        }

        var big = new byte[length];
        for (int i = 0; i < little.Length; i++)
        {
            big[length - 1 - i] = little[i];
        }

        return big;
    }
}

