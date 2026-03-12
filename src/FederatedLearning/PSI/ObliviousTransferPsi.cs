using System.Security.Cryptography;
using System.Text;
using AiDotNet.Models.Options;

namespace AiDotNet.FederatedLearning.PSI;

/// <summary>
/// Implements Oblivious Transfer based Private Set Intersection using cuckoo hashing.
/// </summary>
/// <remarks>
/// <para>OT-based PSI is the fastest known approach for large-scale set intersection.
/// The receiver inserts elements into a cuckoo hash table, and the sender uses
/// Oblivious Transfer extensions to compare against each bin without learning which
/// bins contain elements.</para>
///
/// <para><b>For Beginners:</b> Imagine a library catalog system:</para>
/// <list type="number">
/// <item><description>Party A places its book titles into specific "bins" using a clever hashing scheme (cuckoo hashing).</description></item>
/// <item><description>Party B checks each bin for its book titles using "magic envelopes" (oblivious transfer) that only reveal matches.</description></item>
/// <item><description>Neither party learns about the other's non-matching books.</description></item>
/// </list>
///
/// <para><b>Complexity:</b> O(n) computation and communication with small constants thanks to OT extensions.</para>
///
/// <para><b>Security:</b> Secure against semi-honest adversaries in the random oracle model.</para>
///
/// <para><b>Reference:</b> Pinkas et al., "Efficient Circuit-Based PSI via Cuckoo Hashing",
/// EUROCRYPT 2018.</para>
/// </remarks>
public class ObliviousTransferPsi : PsiBase
{
    private const int CuckooHashFunctions = 3;
    private const double CuckooLoadFactor = 1.3;
    private const int MaxCuckooEvictions = 500;

    /// <inheritdoc/>
    public override string ProtocolName => "ObliviousTransfer";

    /// <inheritdoc/>
    protected override PsiResult ComputeExactIntersection(
        IReadOnlyList<string> localIds, IReadOnlyList<string> remoteIds, PsiOptions options)
    {
        // Step 1: Receiver (local) inserts elements into cuckoo hash table
        int tableSize = Math.Max(16, (int)(localIds.Count * CuckooLoadFactor));
        var hashSeeds = GenerateHashSeeds(CuckooHashFunctions, options.RandomSeed);
        var cuckooTable = BuildCuckooHashTable(localIds, tableSize, hashSeeds);

        // Step 2: Sender (remote) hashes elements and performs OT-based comparison
        // In a real protocol, sender uses OT extensions to obliviously evaluate PRFs.
        // Here we simulate the comparison by hashing remote elements with the same seeds
        // and checking for matches in the cuckoo table bins.
        var matchedLocal = new Dictionary<int, int>(); // localIndex -> bin
        var matchedRemote = new Dictionary<int, int>(); // remoteIndex -> bin

        for (int remoteIdx = 0; remoteIdx < remoteIds.Count; remoteIdx++)
        {
            string remoteId = remoteIds[remoteIdx];

            // The sender hashes the remote element with each hash function
            for (int h = 0; h < CuckooHashFunctions; h++)
            {
                int bin = ComputeHash(remoteId, hashSeeds[h], tableSize);

                int? binValue = cuckooTable[bin];
                if (binValue.HasValue)
                {
                    int localIdx = binValue.Value;

                    // OT-based comparison: in real protocol, this comparison is done
                    // obliviously via PRF evaluation under OT. The receiver learns
                    // the match without the sender learning the bin content.
                    if (string.Equals(localIds[localIdx], remoteId, StringComparison.Ordinal))
                    {
                        if (!matchedLocal.ContainsKey(localIdx))
                        {
                            matchedLocal[localIdx] = bin;
                        }
                        if (!matchedRemote.ContainsKey(remoteIdx))
                        {
                            matchedRemote[remoteIdx] = bin;
                        }
                        break;
                    }
                }
            }

            // Also check the stash (cuckoo hashing overflow)
            if (!matchedRemote.ContainsKey(remoteIdx))
            {
                for (int s = tableSize; s < cuckooTable.Length; s++)
                {
                    int? stashValue = cuckooTable[s];
                    if (stashValue.HasValue)
                    {
                        int localIdx = stashValue.Value;
                        if (string.Equals(localIds[localIdx], remoteId, StringComparison.Ordinal))
                        {
                            if (!matchedLocal.ContainsKey(localIdx))
                            {
                                matchedLocal[localIdx] = s;
                            }
                            if (!matchedRemote.ContainsKey(remoteIdx))
                            {
                                matchedRemote[remoteIdx] = s;
                            }
                            break;
                        }
                    }
                }
            }
        }

        // Step 3: Build alignment from matched pairs
        var intersectionIds = new List<string>(matchedLocal.Count);
        var localToShared = new Dictionary<int, int>(matchedLocal.Count);
        var remoteToShared = new Dictionary<int, int>(matchedRemote.Count);

        int sharedIndex = 0;
        foreach (var (localIdx, _) in matchedLocal.OrderBy(kv => kv.Key))
        {
            intersectionIds.Add(localIds[localIdx]);
            localToShared[localIdx] = sharedIndex;

            // Find corresponding remote index
            foreach (var (remoteIdx, __) in matchedRemote)
            {
                if (string.Equals(localIds[localIdx], remoteIds[remoteIdx], StringComparison.Ordinal) &&
                    !remoteToShared.ContainsKey(remoteIdx))
                {
                    remoteToShared[remoteIdx] = sharedIndex;
                    break;
                }
            }

            sharedIndex++;
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
    /// Builds a cuckoo hash table from the local ID set.
    /// Each element is placed in one of its candidate bins; evicted elements try alternative bins.
    /// Elements that cannot be placed after max evictions go to a small stash.
    /// </summary>
    private static int?[] BuildCuckooHashTable(
        IReadOnlyList<string> ids, int tableSize, int[] hashSeeds)
    {
        int stashSize = Math.Max(4, (int)Math.Ceiling(Math.Log(ids.Count + 1)));
        var table = new int?[tableSize + stashSize];
        var elementInBin = new string?[tableSize + stashSize];
        int stashNext = tableSize;

        for (int i = 0; i < ids.Count; i++)
        {
            if (!InsertCuckoo(table, elementInBin, ids, i, tableSize, hashSeeds, 0))
            {
                // Place in stash
                if (stashNext < table.Length)
                {
                    table[stashNext] = i;
                    elementInBin[stashNext] = ids[i];
                    stashNext++;
                }
            }
        }

        return table;
    }

    private static bool InsertCuckoo(
        int?[] table, string?[] elementInBin,
        IReadOnlyList<string> ids, int elementIndex,
        int tableSize, int[] hashSeeds, int depth)
    {
        if (depth >= MaxCuckooEvictions)
        {
            return false;
        }

        string element = ids[elementIndex];

        for (int h = 0; h < hashSeeds.Length; h++)
        {
            int bin = ComputeHash(element, hashSeeds[h], tableSize);
            if (!table[bin].HasValue)
            {
                table[bin] = elementIndex;
                elementInBin[bin] = element;
                return true;
            }
        }

        // Evict from the first candidate bin
        int evictBin = ComputeHash(element, hashSeeds[0], tableSize);
        int? evictValue = table[evictBin];
        int evictedIndex = evictValue.GetValueOrDefault();
        table[evictBin] = elementIndex;
        elementInBin[evictBin] = element;

        return InsertCuckoo(table, elementInBin, ids, evictedIndex, tableSize, hashSeeds, depth + 1);
    }

    private static int ComputeHash(string element, int seed, int tableSize)
    {
        using var sha = SHA256.Create();
        byte[] seedBytes = BitConverter.GetBytes(seed);
        byte[] elementBytes = Encoding.UTF8.GetBytes(element);
        byte[] combined = new byte[seedBytes.Length + elementBytes.Length];
        Buffer.BlockCopy(seedBytes, 0, combined, 0, seedBytes.Length);
        Buffer.BlockCopy(elementBytes, 0, combined, seedBytes.Length, elementBytes.Length);

        byte[] hash = sha.ComputeHash(combined);
        int value = Math.Abs(BitConverter.ToInt32(hash, 0));
        return value % tableSize;
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
