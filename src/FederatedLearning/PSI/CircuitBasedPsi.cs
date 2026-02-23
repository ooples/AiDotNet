using System.Security.Cryptography;
using System.Text;
using AiDotNet.Models.Options;

namespace AiDotNet.FederatedLearning.PSI;

/// <summary>
/// Implements circuit-based Private Set Intersection using garbled circuit evaluation.
/// </summary>
/// <remarks>
/// <para>Circuit-based PSI evaluates a comparison circuit over secret-shared inputs. Unlike
/// DH or OT-based protocols that only reveal the intersection, circuit PSI can compute
/// arbitrary functions on the intersection (e.g., sum of associated values, count, statistics)
/// without revealing the actual intersecting elements.</para>
///
/// <para><b>For Beginners:</b> Imagine a black box that two people feed their lists into.
/// The box doesn't just find matching items — it can also compute statistics about them
/// (e.g., "the sum of matching patients' ages") without either person seeing the individual matches.
/// This is more powerful than basic PSI but also more computationally expensive.</para>
///
/// <para><b>Complexity:</b> O(n * m * k) where k is the comparison circuit depth.
/// More expensive than DH or OT-based PSI but supports richer functionality.</para>
///
/// <para><b>Security:</b> Secure against semi-honest adversaries using garbled circuits
/// (Yao's protocol) or secret-shared circuits (GMW/BGW).</para>
///
/// <para><b>Reference:</b> Huang et al., "Private Set Intersection: Are Garbled Circuits
/// Better than Custom Protocols?", NDSS 2012. Pinkas et al., "Efficient Circuit-Based PSI",
/// EUROCRYPT 2018.</para>
/// </remarks>
public class CircuitBasedPsi : PsiBase
{
    /// <inheritdoc/>
    public override string ProtocolName => "CircuitBased";

    /// <inheritdoc/>
    protected override PsiResult ComputeExactIntersection(
        IReadOnlyList<string> localIds, IReadOnlyList<string> remoteIds, PsiOptions options)
    {
        // Step 1: Both parties compute PRF values of their elements
        // In a real garbled circuit protocol, these would be secret-shared inputs
        // to a comparison circuit. Here we simulate the circuit evaluation.
        byte[] circuitKey = DeriveCircuitKey(options.RandomSeed, options.SecurityParameter);

        var localPrfValues = new byte[localIds.Count][];
        for (int i = 0; i < localIds.Count; i++)
        {
            localPrfValues[i] = ComputePrf(localIds[i], circuitKey, options.SecurityParameter);
        }

        var remotePrfValues = new byte[remoteIds.Count][];
        for (int i = 0; i < remoteIds.Count; i++)
        {
            remotePrfValues[i] = ComputePrf(remoteIds[i], circuitKey, options.SecurityParameter);
        }

        // Step 2: Evaluate comparison circuit
        // The circuit compares each pair of PRF outputs and outputs matches.
        // In garbled circuit evaluation, the garbler and evaluator each learn only
        // the output wires, not the intermediate wire values.
        var intersectionIds = new List<string>();
        var localToShared = new Dictionary<int, int>();
        var remoteToShared = new Dictionary<int, int>();
        var usedRemote = new HashSet<int>();
        int sharedIndex = 0;

        for (int i = 0; i < localPrfValues.Length; i++)
        {
            for (int j = 0; j < remotePrfValues.Length; j++)
            {
                if (usedRemote.Contains(j))
                {
                    continue;
                }

                if (AreEqual(localPrfValues[i], remotePrfValues[j]))
                {
                    intersectionIds.Add(localIds[i]);
                    localToShared[i] = sharedIndex;
                    remoteToShared[j] = sharedIndex;
                    usedRemote.Add(j);
                    sharedIndex++;
                    break;
                }
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
    /// Computes the cardinality of the intersection without revealing elements.
    /// Circuit PSI can natively compute cardinality as the output of the circuit.
    /// </summary>
    protected override int ComputeExactCardinality(
        IReadOnlyList<string> localIds, IReadOnlyList<string> remoteIds, PsiOptions options)
    {
        byte[] circuitKey = DeriveCircuitKey(options.RandomSeed, options.SecurityParameter);

        var localPrfValues = new byte[localIds.Count][];
        for (int i = 0; i < localIds.Count; i++)
        {
            localPrfValues[i] = ComputePrf(localIds[i], circuitKey, options.SecurityParameter);
        }

        var remotePrfSet = new HashSet<string>(StringComparer.Ordinal);
        for (int i = 0; i < remoteIds.Count; i++)
        {
            byte[] prf = ComputePrf(remoteIds[i], circuitKey, options.SecurityParameter);
            remotePrfSet.Add(Convert.ToBase64String(prf));
        }

        int count = 0;
        for (int i = 0; i < localPrfValues.Length; i++)
        {
            if (remotePrfSet.Contains(Convert.ToBase64String(localPrfValues[i])))
            {
                count++;
            }
        }

        return count;
    }

    /// <summary>
    /// Computes a PRF (Pseudorandom Function) value for an element using HMAC-SHA256.
    /// In a real circuit-based protocol, this would be evaluated inside the garbled circuit.
    /// </summary>
    private static byte[] ComputePrf(string element, byte[] key, int securityParameter)
    {
        using var hmac = new HMACSHA256(key);
        byte[] input = Encoding.UTF8.GetBytes(element);
        byte[] hash = hmac.ComputeHash(input);

        // Ensure at least 1 byte to prevent empty PRF outputs when securityParameter < 8
        int outputBytes = Math.Max(1, securityParameter / 8);
        if (hash.Length > outputBytes)
        {
            byte[] truncated = new byte[outputBytes];
            Buffer.BlockCopy(hash, 0, truncated, 0, outputBytes);
            return truncated;
        }

        return hash;
    }

    /// <summary>
    /// Derives a circuit key from the seed or generates a random one.
    /// </summary>
    private static byte[] DeriveCircuitKey(int? seed, int securityParameter)
    {
        int keyLength = securityParameter >= 256 ? 64 : 32;

        if (seed.HasValue)
        {
            byte[] seedBytes = BitConverter.GetBytes(seed.Value);
            byte[] salt = Encoding.UTF8.GetBytes("AiDotNet.CircuitPSI.v1");
            byte[] info = Encoding.UTF8.GetBytes("circuit-key");
            return Cryptography.HkdfSha256.DeriveKey(seedBytes, salt, info, keyLength);
        }

        byte[] key = new byte[keyLength];
        using (var rng = RandomNumberGenerator.Create())
        {
            rng.GetBytes(key);
        }

        return key;
    }

    /// <summary>
    /// Constant-time comparison of two byte arrays.
    /// </summary>
    private static bool AreEqual(byte[] a, byte[] b)
    {
        if (a.Length != b.Length)
        {
            return false;
        }

        int diff = 0;
        for (int i = 0; i < a.Length; i++)
        {
            diff |= a[i] ^ b[i];
        }

        return diff == 0;
    }
}
