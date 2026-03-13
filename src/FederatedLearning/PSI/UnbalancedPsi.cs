using System.Security.Cryptography;
using System.Text;
using AiDotNet.Models.Options;

namespace AiDotNet.FederatedLearning.PSI;

/// <summary>
/// Implements PSI optimized for asymmetric (unbalanced) set sizes.
/// </summary>
/// <remarks>
/// <para>When one party's set is much larger than the other's (e.g., 10M records vs 1K),
/// standard PSI protocols waste computation on the larger set. Unbalanced PSI optimizes
/// by having the smaller party (client) do minimal work while the larger party (server)
/// builds a compressed data structure for efficient querying.</para>
///
/// <para><b>For Beginners:</b> Imagine looking up 10 phone numbers in a phonebook with
/// 10 million entries. It's much faster to search the phonebook for your 10 numbers
/// than to compare every pair. Unbalanced PSI works similarly: the small party's elements
/// are looked up in a compact representation of the large party's set.</para>
///
/// <para><b>Algorithm:</b></para>
/// <list type="number">
/// <item><description>The larger party (server) builds an encrypted Cuckoo filter of its set.</description></item>
/// <item><description>The smaller party (client) queries the filter using OPRF (Oblivious PRF).</description></item>
/// <item><description>Communication is proportional to the smaller set size, not the larger.</description></item>
/// </list>
///
/// <para><b>Complexity:</b> O(n_small * log(n_large)) computation, O(n_small) communication.
/// Huge savings when set sizes differ by orders of magnitude.</para>
///
/// <para><b>Reference:</b> Chen et al., "Labeled PSI from Fully Homomorphic Encryption with
/// Malicious Security", ACM CCS 2018. Kiss et al., "Private Set Intersection for
/// Unequal Set Sizes", PETS 2017.</para>
/// </remarks>
public class UnbalancedPsi : PsiBase
{
    /// <inheritdoc/>
    public override string ProtocolName => "Unbalanced";

    /// <inheritdoc/>
    protected override PsiResult ComputeExactIntersection(
        IReadOnlyList<string> localIds, IReadOnlyList<string> remoteIds, PsiOptions options)
    {
        // Determine which party is the server (larger set) and client (smaller set)
        bool localIsSmaller = localIds.Count <= remoteIds.Count;
        var smallerSet = localIsSmaller ? localIds : remoteIds;
        var largerSet = localIsSmaller ? remoteIds : localIds;

        // Step 1: Server builds a compact hash table from its larger set
        // In a real protocol, this would be an encrypted Cuckoo filter or polynomial.
        // Here we use a hash-based lookup for the algorithmic approach.
        byte[] prfKey = DeriveKey(options.RandomSeed, options.SecurityParameter);

        var serverTable = new Dictionary<string, int>(largerSet.Count, StringComparer.Ordinal);
        for (int i = 0; i < largerSet.Count; i++)
        {
            string tag = ComputeTag(largerSet[i], prfKey, options.SecurityParameter);
            if (!serverTable.ContainsKey(tag))
            {
                serverTable[tag] = i;
            }
        }

        // Step 2: Client queries the server's table using OPRF
        // In a real protocol, the client sends blinded queries and the server evaluates
        // the PRF without learning the queries. The client unblinds to get PRF values.
        var intersectionIds = new List<string>();
        var localToShared = new Dictionary<int, int>();
        var remoteToShared = new Dictionary<int, int>();
        int sharedIndex = 0;

        for (int i = 0; i < smallerSet.Count; i++)
        {
            string clientTag = ComputeTag(smallerSet[i], prfKey, options.SecurityParameter);

            if (serverTable.TryGetValue(clientTag, out int serverIdx))
            {
                string matchedId = smallerSet[i];
                intersectionIds.Add(matchedId);

                int localIdx = localIsSmaller ? i : serverIdx;
                int remoteIdx = localIsSmaller ? serverIdx : i;

                localToShared[localIdx] = sharedIndex;
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

    /// <inheritdoc/>
    protected override int ComputeExactCardinality(
        IReadOnlyList<string> localIds, IReadOnlyList<string> remoteIds, PsiOptions options)
    {
        bool localIsSmaller = localIds.Count <= remoteIds.Count;
        var smallerSet = localIsSmaller ? localIds : remoteIds;
        var largerSet = localIsSmaller ? remoteIds : localIds;

        byte[] prfKey = DeriveKey(options.RandomSeed, options.SecurityParameter);

        var serverTags = new HashSet<string>(StringComparer.Ordinal);
        for (int i = 0; i < largerSet.Count; i++)
        {
            serverTags.Add(ComputeTag(largerSet[i], prfKey, options.SecurityParameter));
        }

        int count = 0;
        for (int i = 0; i < smallerSet.Count; i++)
        {
            string clientTag = ComputeTag(smallerSet[i], prfKey, options.SecurityParameter);
            if (serverTags.Contains(clientTag))
            {
                count++;
            }
        }

        return count;
    }

    /// <summary>
    /// Computes a PRF tag for an element. In a real OPRF protocol, the server would
    /// evaluate this obliviously on the client's blinded input.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when <paramref name="securityParameter"/> is less than 8, which would produce
    /// tags too short to prevent collisions.
    /// </exception>
    private static string ComputeTag(string element, byte[] key, int securityParameter)
    {
        if (securityParameter < 8)
        {
            throw new ArgumentOutOfRangeException(nameof(securityParameter),
                $"Security parameter must be at least 8 (got {securityParameter}). " +
                "Values below 8 produce tags shorter than 1 byte, causing catastrophic collisions.");
        }

        using var hmac = new HMACSHA256(key);
        byte[] input = Encoding.UTF8.GetBytes(element);
        byte[] hash = hmac.ComputeHash(input);

        int tagBytes = securityParameter / 8;
        if (hash.Length > tagBytes)
        {
            byte[] truncated = new byte[tagBytes];
            Buffer.BlockCopy(hash, 0, truncated, 0, tagBytes);
            return Convert.ToBase64String(truncated);
        }

        return Convert.ToBase64String(hash);
    }

    /// <summary>
    /// Derives a PRF key from the seed or generates a random one.
    /// </summary>
    private static byte[] DeriveKey(int? seed, int securityParameter)
    {
        int keyLength = securityParameter >= 256 ? 64 : 32;

        if (seed.HasValue)
        {
            byte[] seedBytes = BitConverter.GetBytes(seed.Value);
            byte[] salt = Encoding.UTF8.GetBytes("AiDotNet.UnbalancedPSI.v1");
            byte[] info = Encoding.UTF8.GetBytes("oprf-key");
            return Cryptography.HkdfSha256.DeriveKey(seedBytes, salt, info, keyLength);
        }

        byte[] key = new byte[keyLength];
        using (var rng = RandomNumberGenerator.Create())
        {
            rng.GetBytes(key);
        }

        return key;
    }
}
