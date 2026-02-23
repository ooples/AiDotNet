using System.Security.Cryptography;

namespace AiDotNet.FederatedLearning.MPC;

/// <summary>
/// Implements XOR-based boolean secret sharing for bitwise operations.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Boolean secret sharing splits a bit string into random XOR shares.
/// To share a bit b among n parties: generate n-1 random bits r1..r(n-1), and compute
/// the last share as b XOR r1 XOR r2 XOR ... XOR r(n-1). To reconstruct, XOR all shares
/// together.</para>
///
/// <para><b>Why both arithmetic AND boolean sharing?</b> Different operations are cheaper in
/// different representations:</para>
/// <list type="bullet">
/// <item><description><b>Boolean sharing:</b> XOR gates are free, AND gates need 1 OT per bit.</description></item>
/// <item><description><b>Arithmetic sharing:</b> Addition is free, multiplication needs 1 Beaver triple.</description></item>
/// </list>
///
/// <para>Hybrid MPC uses boolean sharing for comparisons and bit manipulations, and arithmetic
/// sharing for linear algebra. Converting between the two is called "share conversion".</para>
///
/// <para><b>Reference:</b> ABY framework (NDSS 2015) for arithmetic, boolean, and Yao sharing.</para>
/// </remarks>
public class BooleanSecretSharing
{
    private readonly int _numberOfParties;

    /// <summary>
    /// Initializes a new instance of <see cref="BooleanSecretSharing"/>.
    /// </summary>
    /// <param name="numberOfParties">The number of parties. Default is 2.</param>
    public BooleanSecretSharing(int numberOfParties = 2)
    {
        if (numberOfParties < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(numberOfParties), "Need at least 2 parties.");
        }

        _numberOfParties = numberOfParties;
    }

    /// <summary>
    /// Splits a byte array into XOR shares for the specified number of parties.
    /// </summary>
    /// <param name="secret">The plaintext bytes to share.</param>
    /// <returns>An array of byte arrays, one per party.</returns>
    public byte[][] Share(byte[] secret)
    {
        if (secret is null || secret.Length == 0)
        {
            throw new ArgumentException("Secret must not be null or empty.", nameof(secret));
        }

        var shares = new byte[_numberOfParties][];

        // Generate n-1 random shares
        using (var rng = RandomNumberGenerator.Create())
        {
            for (int p = 0; p < _numberOfParties - 1; p++)
            {
                shares[p] = new byte[secret.Length];
                rng.GetBytes(shares[p]);
            }
        }

        // Last share = secret XOR all other shares
        shares[_numberOfParties - 1] = new byte[secret.Length];
        Buffer.BlockCopy(secret, 0, shares[_numberOfParties - 1], 0, secret.Length);

        for (int p = 0; p < _numberOfParties - 1; p++)
        {
            XorInPlace(shares[_numberOfParties - 1], shares[p]);
        }

        return shares;
    }

    /// <summary>
    /// Reconstructs the secret by XORing all shares together.
    /// </summary>
    /// <param name="shares">The shares from each party.</param>
    /// <returns>The reconstructed plaintext bytes.</returns>
    public byte[] Reconstruct(byte[][] shares)
    {
        if (shares is null || shares.Length == 0)
        {
            throw new ArgumentException("Shares must not be null or empty.", nameof(shares));
        }

        int len = shares[0].Length;
        var result = new byte[len];
        Buffer.BlockCopy(shares[0], 0, result, 0, len);

        for (int p = 1; p < shares.Length; p++)
        {
            if (shares[p].Length != len)
            {
                throw new ArgumentException("All shares must have the same length.", nameof(shares));
            }

            XorInPlace(result, shares[p]);
        }

        return result;
    }

    /// <summary>
    /// Performs XOR on two sets of boolean shares (local operation — no communication).
    /// </summary>
    /// <param name="sharesA">Boolean shares of the first operand.</param>
    /// <param name="sharesB">Boolean shares of the second operand.</param>
    /// <returns>Boolean shares of A XOR B.</returns>
    public byte[][] SecureXor(byte[][] sharesA, byte[][] sharesB)
    {
        if (sharesA is null || sharesB is null)
        {
            throw new ArgumentNullException(sharesA is null ? nameof(sharesA) : nameof(sharesB));
        }

        if (sharesA.Length != sharesB.Length)
        {
            throw new ArgumentException("Share arrays must have the same length.");
        }

        var result = new byte[sharesA.Length][];
        for (int p = 0; p < sharesA.Length; p++)
        {
            result[p] = new byte[sharesA[p].Length];
            Buffer.BlockCopy(sharesA[p], 0, result[p], 0, sharesA[p].Length);
            XorInPlace(result[p], sharesB[p]);
        }

        return result;
    }

    /// <summary>
    /// Performs AND on two sets of boolean shares using pre-shared correlated randomness.
    /// </summary>
    /// <param name="sharesA">Boolean shares of the first operand.</param>
    /// <param name="sharesB">Boolean shares of the second operand.</param>
    /// <param name="andTriple">A pre-shared AND triple (u, v, w) where w = u AND v.</param>
    /// <returns>Boolean shares of A AND B.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> AND on boolean shares is like multiplication on arithmetic shares —
    /// it requires correlated randomness (an "AND triple"). The protocol opens masked versions
    /// of the inputs, then each party locally computes its share of the result.</para>
    /// </remarks>
    public byte[][] SecureAnd(byte[][] sharesA, byte[][] sharesB, BooleanTriple andTriple)
    {
        if (sharesA is null || sharesB is null || andTriple is null)
        {
            throw new ArgumentNullException(
                sharesA is null ? nameof(sharesA) :
                sharesB is null ? nameof(sharesB) : nameof(andTriple));
        }

        int n = sharesA.Length;
        int len = sharesA[0].Length;

        // Compute d = a XOR u and e = b XOR v (secret-shared)
        var sharesD = SecureXor(sharesA, andTriple.SharesU);
        var sharesE = SecureXor(sharesB, andTriple.SharesV);

        // Open d and e
        var d = Reconstruct(sharesD);
        var e = Reconstruct(sharesE);

        // Each party computes: w_p XOR (e AND u_p) XOR (d AND v_p) XOR (for party 0 only: d AND e)
        var result = new byte[n][];
        for (int p = 0; p < n; p++)
        {
            result[p] = new byte[len];
            Buffer.BlockCopy(andTriple.SharesW[p], 0, result[p], 0, len);

            // XOR with (e AND u_p)
            var eAndU = new byte[len];
            AndBytes(e, andTriple.SharesU[p], eAndU);
            XorInPlace(result[p], eAndU);

            // XOR with (d AND v_p)
            var dAndV = new byte[len];
            AndBytes(d, andTriple.SharesV[p], dAndV);
            XorInPlace(result[p], dAndV);

            // Only party 0 adds d AND e
            if (p == 0)
            {
                var dAndE = new byte[len];
                AndBytes(d, e, dAndE);
                XorInPlace(result[p], dAndE);
            }
        }

        return result;
    }

    /// <summary>
    /// Generates an AND triple (u, v, w) where w = u AND v, split into boolean shares.
    /// </summary>
    /// <param name="byteLength">The length of each share in bytes.</param>
    /// <returns>A <see cref="BooleanTriple"/> with correlated shares.</returns>
    public BooleanTriple GenerateAndTriple(int byteLength)
    {
        if (byteLength <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(byteLength));
        }

        // Generate random u, v and compute w = u AND v
        var u = new byte[byteLength];
        var v = new byte[byteLength];
        var w = new byte[byteLength];

        using (var rng = RandomNumberGenerator.Create())
        {
            rng.GetBytes(u);
            rng.GetBytes(v);
        }

        AndBytes(u, v, w);

        // Secret-share each
        var sharesU = Share(u);
        var sharesV = Share(v);
        var sharesW = Share(w);

        return new BooleanTriple
        {
            SharesU = sharesU,
            SharesV = sharesV,
            SharesW = sharesW
        };
    }

    private static void XorInPlace(byte[] target, byte[] source)
    {
        for (int i = 0; i < target.Length; i++)
        {
            target[i] ^= source[i];
        }
    }

    private static void AndBytes(byte[] a, byte[] b, byte[] result)
    {
        for (int i = 0; i < a.Length; i++)
        {
            result[i] = (byte)(a[i] & b[i]);
        }
    }
}

/// <summary>
/// Represents a pre-shared AND triple for boolean secret sharing.
/// </summary>
public class BooleanTriple
{
    /// <summary>Gets or sets the boolean shares of random value u.</summary>
    public byte[][] SharesU { get; set; } = Array.Empty<byte[]>();

    /// <summary>Gets or sets the boolean shares of random value v.</summary>
    public byte[][] SharesV { get; set; } = Array.Empty<byte[]>();

    /// <summary>Gets or sets the boolean shares of w = u AND v.</summary>
    public byte[][] SharesW { get; set; } = Array.Empty<byte[]>();
}
