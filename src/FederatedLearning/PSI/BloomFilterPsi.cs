using System.Security.Cryptography;
using System.Text;
using AiDotNet.Models.Options;

namespace AiDotNet.FederatedLearning.PSI;

/// <summary>
/// Implements Bloom filter based probabilistic Private Set Intersection.
/// </summary>
/// <remarks>
/// <para>Bloom filter PSI uses a probabilistic data structure to represent one party's set.
/// The other party queries the Bloom filter for each of its elements. Matches indicate
/// probable intersection membership, with a configurable false-positive rate.</para>
///
/// <para><b>For Beginners:</b> A Bloom filter is like a very compact checklist that can answer
/// "is this item on the list?" with two possible answers:</para>
/// <list type="bullet">
/// <item><description>"Definitely NOT on the list" — always correct.</description></item>
/// <item><description>"Probably on the list" — correct most of the time, but occasionally wrong (false positive).</description></item>
/// </list>
///
/// <para>The false-positive rate is configurable. A rate of 0.001 means roughly 1 in 1000
/// non-matching elements may be incorrectly reported as matching. This is usually acceptable
/// when followed by a verification step.</para>
///
/// <para><b>Complexity:</b> O(n+m) time, O(n) space for the Bloom filter (much smaller than
/// storing the full set). Fastest PSI protocol but probabilistic.</para>
///
/// <para><b>Security note:</b> Standard Bloom filter PSI leaks membership information.
/// For privacy, the Bloom filter should be encrypted or transmitted via secure channel.
/// This implementation simulates the protocol assuming a secure channel.</para>
///
/// <para><b>Reference:</b> Dong et al., "When Private Set Intersection Meets Big Data",
/// ACM CCS 2013.</para>
/// </remarks>
public class BloomFilterPsi : PsiBase
{
    /// <inheritdoc/>
    public override string ProtocolName => "BloomFilter";

    /// <inheritdoc/>
    protected override PsiResult ComputeExactIntersection(
        IReadOnlyList<string> localIds, IReadOnlyList<string> remoteIds, PsiOptions options)
    {
        double falsePositiveRate = options.BloomFilterFalsePositiveRate > 0
            ? options.BloomFilterFalsePositiveRate
            : 0.001;

        // Step 1: Compute optimal Bloom filter parameters
        int filterSize = ComputeOptimalFilterSize(localIds.Count, falsePositiveRate);
        int hashCount = options.BloomFilterHashCount > 0
            ? options.BloomFilterHashCount
            : ComputeOptimalHashCount(filterSize, localIds.Count);

        var hashSeeds = GenerateHashSeeds(hashCount, options.RandomSeed);

        // Step 2: Party A builds Bloom filter from its set
        var filter = new bool[filterSize];
        for (int i = 0; i < localIds.Count; i++)
        {
            InsertIntoFilter(filter, localIds[i], hashSeeds);
        }

        // Step 3: Party B queries the Bloom filter for each of its elements
        var candidateMatches = new List<(int remoteIndex, string id)>();
        for (int i = 0; i < remoteIds.Count; i++)
        {
            if (QueryFilter(filter, remoteIds[i], hashSeeds))
            {
                candidateMatches.Add((i, remoteIds[i]));
            }
        }

        // Step 4: Verify candidates against the actual local set to eliminate false positives
        // In a real protocol, this verification would use a separate secure protocol.
        // Here we use exact comparison to produce the correct intersection.
        var localSet = new Dictionary<string, int>(localIds.Count, StringComparer.Ordinal);
        for (int i = 0; i < localIds.Count; i++)
        {
            if (!localSet.ContainsKey(localIds[i]))
            {
                localSet[localIds[i]] = i;
            }
        }

        var intersectionIds = new List<string>();
        var localToShared = new Dictionary<int, int>();
        var remoteToShared = new Dictionary<int, int>();
        int sharedIndex = 0;

        foreach (var (remoteIdx, id) in candidateMatches)
        {
            if (localSet.TryGetValue(id, out int localIdx))
            {
                intersectionIds.Add(id);
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
        double falsePositiveRate = options.BloomFilterFalsePositiveRate > 0
            ? options.BloomFilterFalsePositiveRate
            : 0.001;

        int filterSize = ComputeOptimalFilterSize(localIds.Count, falsePositiveRate);
        int hashCount = options.BloomFilterHashCount > 0
            ? options.BloomFilterHashCount
            : ComputeOptimalHashCount(filterSize, localIds.Count);

        var hashSeeds = GenerateHashSeeds(hashCount, options.RandomSeed);

        var filter = new bool[filterSize];
        for (int i = 0; i < localIds.Count; i++)
        {
            InsertIntoFilter(filter, localIds[i], hashSeeds);
        }

        // For cardinality, just count Bloom filter hits
        // Note: this may slightly overcount due to false positives
        int count = 0;
        for (int i = 0; i < remoteIds.Count; i++)
        {
            if (QueryFilter(filter, remoteIds[i], hashSeeds))
            {
                count++;
            }
        }

        return count;
    }

    /// <summary>
    /// Inserts an element into the Bloom filter by setting the bits at each hash position.
    /// </summary>
    private static void InsertIntoFilter(bool[] filter, string element, int[] hashSeeds)
    {
        for (int i = 0; i < hashSeeds.Length; i++)
        {
            int position = ComputeFilterHash(element, hashSeeds[i], filter.Length);
            filter[position] = true;
        }
    }

    /// <summary>
    /// Queries the Bloom filter for membership. Returns true if element is probably in the set.
    /// </summary>
    private static bool QueryFilter(bool[] filter, string element, int[] hashSeeds)
    {
        for (int i = 0; i < hashSeeds.Length; i++)
        {
            int position = ComputeFilterHash(element, hashSeeds[i], filter.Length);
            if (!filter[position])
            {
                return false; // Definitely not in set
            }
        }

        return true; // Probably in set
    }

    /// <summary>
    /// Computes optimal Bloom filter size: m = -n*ln(p) / (ln(2))^2
    /// </summary>
    private static int ComputeOptimalFilterSize(int elementCount, double falsePositiveRate)
    {
        if (elementCount <= 0)
        {
            return 64;
        }

        double ln2Squared = Math.Log(2) * Math.Log(2);
        int size = (int)Math.Ceiling(-elementCount * Math.Log(falsePositiveRate) / ln2Squared);
        return Math.Max(64, size);
    }

    /// <summary>
    /// Computes optimal number of hash functions: k = (m/n) * ln(2)
    /// </summary>
    private static int ComputeOptimalHashCount(int filterSize, int elementCount)
    {
        if (elementCount <= 0)
        {
            return 3;
        }

        int k = (int)Math.Round((double)filterSize / elementCount * Math.Log(2));
        return Math.Max(1, Math.Min(k, 20));
    }

    private static int ComputeFilterHash(string element, int seed, int filterSize)
    {
        using var sha = SHA256.Create();
        byte[] seedBytes = BitConverter.GetBytes(seed);
        byte[] elementBytes = Encoding.UTF8.GetBytes(element);
        byte[] combined = new byte[seedBytes.Length + elementBytes.Length];
        Buffer.BlockCopy(seedBytes, 0, combined, 0, seedBytes.Length);
        Buffer.BlockCopy(elementBytes, 0, combined, seedBytes.Length, elementBytes.Length);

        byte[] hash = sha.ComputeHash(combined);
        int value = Math.Abs(BitConverter.ToInt32(hash, 0));
        return value % filterSize;
    }

    private static int[] GenerateHashSeeds(int count, int? baseSeed)
    {
        var seeds = new int[count];
        if (baseSeed.HasValue)
        {
            var rng = Tensors.Helpers.RandomHelper.CreateSeededRandom(baseSeed.Value);
            for (int i = 0; i < count; i++)
            {
                seeds[i] = rng.Next();
            }
        }
        else
        {
            using var rng = RandomNumberGenerator.Create();
            var bytes = new byte[4];
            for (int i = 0; i < count; i++)
            {
                rng.GetBytes(bytes);
                seeds[i] = BitConverter.ToInt32(bytes, 0);
            }
        }

        return seeds;
    }
}
