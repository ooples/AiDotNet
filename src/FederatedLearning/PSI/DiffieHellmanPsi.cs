using System.Numerics;
using System.Security.Cryptography;
using System.Text;
using AiDotNet.Models.Options;

namespace AiDotNet.FederatedLearning.PSI;

/// <summary>
/// Implements Diffie-Hellman based Private Set Intersection using commutative encryption.
/// </summary>
/// <remarks>
/// <para>This protocol exploits the commutative property of exponentiation in a prime-order group:
/// H(x)^(a*b) = (H(x)^a)^b = (H(x)^b)^a. Both parties hash their elements, raise to their secret
/// exponents, exchange and re-exponentiate, then compare the doubly-encrypted values.</para>
///
/// <para><b>For Beginners:</b> Think of this like two people mixing paint colors:</para>
/// <list type="number">
/// <item><description>Each person has a secret color (their private key).</description></item>
/// <item><description>Each person starts with their IDs and mixes in their secret color.</description></item>
/// <item><description>They swap these mixed colors.</description></item>
/// <item><description>Each person mixes in their secret color again.</description></item>
/// <item><description>Now both have the same double-mixed colors for matching IDs, but can't extract the individual secret colors.</description></item>
/// </list>
///
/// <para><b>Complexity:</b> O(n+m) computation, O(n+m) communication where n and m are set sizes.</para>
///
/// <para><b>Security:</b> Secure against semi-honest adversaries under the Decisional Diffie-Hellman assumption.</para>
///
/// <para><b>Reference:</b> Meadows, "A More Efficient Cryptographic Matchmaking Protocol for Use in the
/// Absence of a Continuously Available Third Party", IEEE S&amp;P 1986.</para>
/// </remarks>
public class DiffieHellmanPsi : PsiBase
{
    // Use a safe prime for the group. This is a 256-bit prime where (p-1)/2 is also prime.
    // In production, a standardized group (e.g., RFC 3526) would be used.
    private static readonly BigInteger SafePrime = BigInteger.Parse(
        "115792089237316195423570985008687907853269984665640564039457584007908834671663");

    private static readonly BigInteger GroupOrder = (SafePrime - 1) / 2;

    /// <inheritdoc/>
    public override string ProtocolName => "DiffieHellman";

    /// <inheritdoc/>
    protected override PsiResult ComputeExactIntersection(
        IReadOnlyList<string> localIds, IReadOnlyList<string> remoteIds, PsiOptions options)
    {
        // Step 1: Generate private exponents for both parties
        var (localExponent, remoteExponent) = GenerateExponents(options.RandomSeed);

        // Step 2: Hash-to-group and exponentiate each party's elements
        // Party A computes: H(x)^a for each x in A's set
        var localSingleEncrypted = new BigInteger[localIds.Count];
        for (int i = 0; i < localIds.Count; i++)
        {
            var h = HashToGroup(localIds[i], options.SecurityParameter);
            localSingleEncrypted[i] = BigInteger.ModPow(h, localExponent, SafePrime);
        }

        // Party B computes: H(y)^b for each y in B's set
        var remoteSingleEncrypted = new BigInteger[remoteIds.Count];
        for (int i = 0; i < remoteIds.Count; i++)
        {
            var h = HashToGroup(remoteIds[i], options.SecurityParameter);
            remoteSingleEncrypted[i] = BigInteger.ModPow(h, remoteExponent, SafePrime);
        }

        // Step 3: Re-exponentiate received values
        // Party A re-encrypts B's values: (H(y)^b)^a = H(y)^(ab)
        var remoteDoubleEncrypted = new BigInteger[remoteIds.Count];
        for (int i = 0; i < remoteIds.Count; i++)
        {
            remoteDoubleEncrypted[i] = BigInteger.ModPow(remoteSingleEncrypted[i], localExponent, SafePrime);
        }

        // Party B re-encrypts A's values: (H(x)^a)^b = H(x)^(ab)
        var localDoubleEncrypted = new BigInteger[localIds.Count];
        for (int i = 0; i < localIds.Count; i++)
        {
            localDoubleEncrypted[i] = BigInteger.ModPow(localSingleEncrypted[i], remoteExponent, SafePrime);
        }

        // Step 4: Compare doubly-encrypted values to find intersection
        // H(x)^(ab) == H(y)^(ab) iff x == y (with overwhelming probability)
        var remoteSet = new Dictionary<BigInteger, int>(remoteIds.Count);
        for (int i = 0; i < remoteDoubleEncrypted.Length; i++)
        {
            if (!remoteSet.ContainsKey(remoteDoubleEncrypted[i]))
            {
                remoteSet[remoteDoubleEncrypted[i]] = i;
            }
        }

        var intersectionIds = new List<string>();
        var localToShared = new Dictionary<int, int>();
        var remoteToShared = new Dictionary<int, int>();
        int sharedIndex = 0;

        for (int i = 0; i < localDoubleEncrypted.Length; i++)
        {
            if (remoteSet.TryGetValue(localDoubleEncrypted[i], out int remoteIdx))
            {
                intersectionIds.Add(localIds[i]);
                localToShared[i] = sharedIndex;
                remoteToShared[remoteIdx] = sharedIndex;
                sharedIndex++;
            }
        }

        return new PsiResult
        {
            IntersectionIds = intersectionIds,
            IntersectionSize = intersectionIds.Count,
            LocalToSharedIndexMap = localToShared,
            RemoteToSharedIndexMap = remoteToShared
        };
    }

    /// <summary>
    /// Hashes a string identifier to a group element using hash-to-curve-like approach.
    /// </summary>
    private static BigInteger HashToGroup(string id, int securityParameter)
    {
        byte[] idBytes = Encoding.UTF8.GetBytes(id);
        byte[] hashBytes;
        if (securityParameter >= 256)
        {
            using var sha512 = SHA512.Create();
            hashBytes = sha512.ComputeHash(idBytes);
        }
        else
        {
            using var sha256 = SHA256.Create();
            hashBytes = sha256.ComputeHash(idBytes);
        }

        // Ensure positive and reduce modulo prime
        byte[] extended = new byte[hashBytes.Length + 1];
        Array.Copy(hashBytes, 0, extended, 0, hashBytes.Length);
        extended[hashBytes.Length] = 0; // Force positive

        var h = new BigInteger(extended);
        h = ((h % SafePrime) + SafePrime) % SafePrime;

        // Square to get into the quadratic residue subgroup of order (p-1)/2
        h = BigInteger.ModPow(h, 2, SafePrime);

        // Ensure non-identity
        if (h.IsZero || h.IsOne)
        {
            h = BigInteger.ModPow(new BigInteger(2), 2, SafePrime);
        }

        return h;
    }

    /// <summary>
    /// Generates private exponents for both parties.
    /// </summary>
    private static (BigInteger local, BigInteger remote) GenerateExponents(int? seed)
    {
        if (seed.HasValue)
        {
            var rng = Tensors.Helpers.RandomHelper.CreateSeededRandom(seed.Value);
            var localBytes = new byte[32];
            var remoteBytes = new byte[32];
            rng.NextBytes(localBytes);
            rng.NextBytes(remoteBytes);
            localBytes[31] &= 0x7F; // Force positive
            remoteBytes[31] &= 0x7F;
            var local = (new BigInteger(localBytes) % GroupOrder + GroupOrder) % GroupOrder;
            var remote = (new BigInteger(remoteBytes) % GroupOrder + GroupOrder) % GroupOrder;
            if (local.IsZero) local = BigInteger.One;
            if (remote.IsZero) remote = BigInteger.One;
            return (local, remote);
        }
        else
        {
            using var rng = RandomNumberGenerator.Create();
            var localBytes = new byte[33];
            var remoteBytes = new byte[33];
            rng.GetBytes(localBytes, 0, 32);
            rng.GetBytes(remoteBytes, 0, 32);
            localBytes[32] = 0; // Force positive
            remoteBytes[32] = 0;
            var local = (new BigInteger(localBytes) % GroupOrder + GroupOrder) % GroupOrder;
            var remote = (new BigInteger(remoteBytes) % GroupOrder + GroupOrder) % GroupOrder;
            if (local.IsZero) local = BigInteger.One;
            if (remote.IsZero) remote = BigInteger.One;
            return (local, remote);
        }
    }
}
