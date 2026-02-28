using System.Security.Cryptography;

namespace AiDotNet.FederatedLearning.PSI;

/// <summary>
/// Implements Optimized Private Set Analytics (OPSA) beyond basic intersection.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Standard PSI (Private Set Intersection) tells you which items
/// two parties share in common. OPSA extends this to richer analytics: set union cardinality
/// (how many unique items total?), frequency estimation (how common is each item?), and threshold
/// queries (which items appear in at least k parties?). All operations are private — no party
/// learns the other parties' raw sets.</para>
///
/// <para>Supported operations:</para>
/// <list type="bullet">
/// <item>Cardinality estimation via HyperLogLog sketches</item>
/// <item>Frequency estimation via count-min sketches</item>
/// <item>Threshold queries via additive secret-shared count-min sketches</item>
/// <item>Intersection cardinality estimation via inclusion-exclusion</item>
/// </list>
///
/// <para>Reference: Optimized Private Set Analytics for Federated Learning (2025).</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
public class OptimizedPrivateSetAnalytics<T> : Infrastructure.FederatedLearningComponentBase<T>
{
    private readonly int _sketchWidth;
    private readonly int _sketchDepth;
    private readonly int _hllPrecision;
    private readonly int _hllRegisterCount;
    private readonly int _seed;

    /// <summary>
    /// Creates a new OPSA instance.
    /// </summary>
    /// <param name="sketchWidth">Width of count-min sketch. Must be a power of 2. Default: 1024.</param>
    /// <param name="sketchDepth">Depth (number of hash functions) of count-min sketch. Default: 5.</param>
    /// <param name="hllPrecision">HyperLogLog precision (p). Uses 2^p registers. Default: 14 (~1.6% error).</param>
    /// <param name="seed">Random seed for deterministic hashing. Default: 42.</param>
    public OptimizedPrivateSetAnalytics(int sketchWidth = 1024, int sketchDepth = 5, int hllPrecision = 14, int seed = 42)
    {
        if (sketchWidth <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(sketchWidth), "Sketch width must be positive.");
        }

        if ((sketchWidth & (sketchWidth - 1)) != 0)
        {
            throw new ArgumentOutOfRangeException(nameof(sketchWidth),
                $"Sketch width must be a power of 2 (got {sketchWidth}). ComputeBucketIndex uses bitwise AND for modulo.");
        }

        if (sketchDepth <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(sketchDepth), "Sketch depth must be positive.");
        }

        if (hllPrecision < 4 || hllPrecision > 18)
        {
            throw new ArgumentOutOfRangeException(nameof(hllPrecision), "HLL precision must be in [4, 18].");
        }

        _sketchWidth = sketchWidth;
        _sketchDepth = sketchDepth;
        _hllPrecision = hllPrecision;
        _hllRegisterCount = 1 << hllPrecision;
        _seed = seed;
    }

    // ---- Count-Min Sketch (Frequency Estimation) ----

    /// <summary>
    /// Creates a count-min sketch from a set of items.
    /// </summary>
    /// <param name="items">Set of items to sketch.</param>
    /// <returns>The sketch matrix (depth x width).</returns>
    public int[,] CreateSketch(IEnumerable<string> items)
    {
        var sketch = new int[_sketchDepth, _sketchWidth];
        foreach (var item in items)
        {
            int hash = GetStableHash(item);
            for (int d = 0; d < _sketchDepth; d++)
            {
                int h = ComputeBucketIndex(hash, d);
                sketch[d, h]++;
            }
        }

        return sketch;
    }

    /// <summary>
    /// Estimates the frequency of an item from merged sketches.
    /// </summary>
    /// <param name="mergedSketch">Merged sketch from multiple parties.</param>
    /// <param name="item">Item to query.</param>
    /// <returns>Estimated frequency (minimum across hash functions).</returns>
    public int EstimateFrequency(int[,] mergedSketch, string item)
    {
        int hash = GetStableHash(item);
        int minCount = int.MaxValue;

        for (int d = 0; d < _sketchDepth; d++)
        {
            int h = ComputeBucketIndex(hash, d);
            minCount = Math.Min(minCount, mergedSketch[d, h]);
        }

        return minCount;
    }

    /// <summary>
    /// Merges sketches from multiple parties (element-wise sum).
    /// </summary>
    /// <param name="sketches">Collection of sketches to merge.</param>
    /// <returns>Merged sketch.</returns>
    public int[,] MergeSketches(IReadOnlyList<int[,]> sketches)
    {
        if (sketches.Count == 0)
        {
            throw new ArgumentException("No sketches to merge.", nameof(sketches));
        }

        var merged = new int[_sketchDepth, _sketchWidth];
        foreach (var sketch in sketches)
        {
            for (int d = 0; d < _sketchDepth; d++)
            {
                for (int w = 0; w < _sketchWidth; w++)
                {
                    merged[d, w] += sketch[d, w];
                }
            }
        }

        return merged;
    }

    // ---- HyperLogLog (Cardinality Estimation) ----

    /// <summary>
    /// Creates a HyperLogLog register array from a set of items.
    /// </summary>
    /// <param name="items">Set of items to sketch.</param>
    /// <returns>HLL register array of length 2^precision.</returns>
    public byte[] CreateHLLRegisters(IEnumerable<string> items)
    {
        Guard.NotNull(items);
        var registers = new byte[_hllRegisterCount];

        foreach (var item in items)
        {
            uint hash = MurmurHash3(item);

            // First p bits determine the register index.
            int registerIdx = (int)(hash >> (32 - _hllPrecision));
            // Remaining bits: count leading zeros + 1.
            uint remainingBits = (hash << _hllPrecision) | ((uint)1 << (_hllPrecision - 1)); // ensure non-zero
            int rho = CountLeadingZeros(remainingBits) + 1;

            if (rho > registers[registerIdx])
            {
                registers[registerIdx] = (byte)rho;
            }
        }

        return registers;
    }

    /// <summary>
    /// Merges HLL registers from multiple parties (element-wise max).
    /// </summary>
    /// <param name="registerSets">HLL registers from each party.</param>
    /// <returns>Merged HLL registers.</returns>
    public byte[] MergeHLLRegisters(IReadOnlyList<byte[]> registerSets)
    {
        if (registerSets.Count == 0)
        {
            throw new ArgumentException("No register sets to merge.", nameof(registerSets));
        }

        var merged = new byte[_hllRegisterCount];
        foreach (var registers in registerSets)
        {
            for (int i = 0; i < Math.Min(registers.Length, _hllRegisterCount); i++)
            {
                if (registers[i] > merged[i])
                {
                    merged[i] = registers[i];
                }
            }
        }

        return merged;
    }

    /// <summary>
    /// Estimates the cardinality (number of distinct elements) from HLL registers.
    /// Uses the HyperLogLog algorithm with bias correction.
    /// </summary>
    /// <param name="registers">HLL register array.</param>
    /// <returns>Estimated cardinality.</returns>
    public double EstimateCardinality(byte[] registers)
    {
        Guard.NotNull(registers);
        if (registers.Length == 0)
        {
            return 0;
        }

        int m = _hllRegisterCount;

        // Compute harmonic mean of 2^(-register[j]).
        double harmonicSum = 0;
        int zeroCount = 0;

        for (int i = 0; i < m; i++)
        {
            harmonicSum += Math.Pow(2, -registers[i]);
            if (registers[i] == 0)
            {
                zeroCount++;
            }
        }

        // Alpha_m constant (bias correction factor).
        double alphaM;
        if (m == 16)
        {
            alphaM = 0.673;
        }
        else if (m == 32)
        {
            alphaM = 0.697;
        }
        else if (m == 64)
        {
            alphaM = 0.709;
        }
        else
        {
            alphaM = 0.7213 / (1.0 + 1.079 / m);
        }

        double estimate = alphaM * m * m / harmonicSum;

        // Small range correction: use linear counting if estimate is small.
        if (estimate <= 2.5 * m && zeroCount > 0)
        {
            estimate = m * Math.Log((double)m / zeroCount);
        }

        // Large range correction (for 32-bit hash).
        double twoTo32 = 4294967296.0; // 2^32
        if (estimate > twoTo32 / 30.0)
        {
            estimate = -twoTo32 * Math.Log(1.0 - estimate / twoTo32);
        }

        return estimate;
    }

    /// <summary>
    /// Estimates union cardinality by merging HLL registers from multiple parties.
    /// </summary>
    /// <param name="clientRegisters">HLL register arrays per client.</param>
    /// <returns>Estimated union cardinality.</returns>
    public double EstimateUnionCardinality(IReadOnlyList<byte[]> clientRegisters)
    {
        var merged = MergeHLLRegisters(clientRegisters);
        return EstimateCardinality(merged);
    }

    /// <summary>
    /// Estimates intersection cardinality of two parties using inclusion-exclusion:
    /// |A ∩ B| = |A| + |B| - |A ∪ B|.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> If you know how many unique items each party has and how
    /// many unique items exist in total (union), you can estimate how many items they share
    /// (intersection) using the formula: shared = partyA + partyB - total. This avoids
    /// revealing which specific items are shared.</para>
    /// </remarks>
    /// <param name="registersA">HLL registers from party A.</param>
    /// <param name="registersB">HLL registers from party B.</param>
    /// <returns>Estimated intersection cardinality (floored at 0 to handle estimation noise).</returns>
    public double EstimateIntersectionCardinality(byte[] registersA, byte[] registersB)
    {
        double cardA = EstimateCardinality(registersA);
        double cardB = EstimateCardinality(registersB);
        double cardUnion = EstimateUnionCardinality(new[] { registersA, registersB });

        // Inclusion-exclusion: |A ∩ B| = |A| + |B| - |A ∪ B|
        // Floor at 0 because estimation noise can make this slightly negative.
        return Math.Max(0, cardA + cardB - cardUnion);
    }

    /// <summary>
    /// Estimates the Jaccard similarity between two parties: |A ∩ B| / |A ∪ B|.
    /// </summary>
    /// <param name="registersA">HLL registers from party A.</param>
    /// <param name="registersB">HLL registers from party B.</param>
    /// <returns>Estimated Jaccard similarity in [0, 1].</returns>
    public double EstimateJaccardSimilarity(byte[] registersA, byte[] registersB)
    {
        double cardUnion = EstimateUnionCardinality(new[] { registersA, registersB });
        if (cardUnion < 1.0)
        {
            return 0;
        }

        double cardIntersection = EstimateIntersectionCardinality(registersA, registersB);
        return cardIntersection / cardUnion;
    }

    // ---- Threshold Queries (Additive Secret-Shared Count-Min Sketch) ----

    /// <summary>
    /// Creates additive secret shares of a count-min sketch for threshold queries.
    /// Each party's sketch is split into numShares random shares that sum to the original.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Secret sharing splits a value into random pieces so that
    /// no single piece reveals anything about the original. Only when all pieces are combined
    /// can you recover the true sketch. This uses a cryptographic random number generator
    /// for security — predictable randomness would compromise the privacy guarantee.</para>
    /// </remarks>
    /// <param name="sketch">The original sketch to share.</param>
    /// <param name="numShares">Number of shares to create. Default: 2.</param>
    /// <returns>List of share sketches that additively reconstruct the original.</returns>
    public List<int[,]> CreateSecretShares(int[,] sketch, int numShares = 2)
    {
        Guard.NotNull(sketch);
        if (numShares < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(numShares), "Must create at least 2 shares.");
        }

        var shares = new List<int[,]>();

        // Use cryptographic RNG — secret shares require unpredictable randomness.
        // A predictable PRNG (System.Random) would allow an adversary who knows the seed
        // to recover the original sketch from a single share.
        using var csprng = RandomNumberGenerator.Create();
        var buffer = new byte[4];

        // Create (numShares - 1) random shares.
        for (int s = 0; s < numShares - 1; s++)
        {
            var share = new int[_sketchDepth, _sketchWidth];
            for (int d = 0; d < _sketchDepth; d++)
            {
                for (int w = 0; w < _sketchWidth; w++)
                {
                    csprng.GetBytes(buffer);
                    share[d, w] = BitConverter.ToInt32(buffer, 0);
                }
            }

            shares.Add(share);
        }

        // Last share = original - sum of all other shares.
        var lastShare = new int[_sketchDepth, _sketchWidth];
        for (int d = 0; d < _sketchDepth; d++)
        {
            for (int w = 0; w < _sketchWidth; w++)
            {
                int shareSum = 0;
                for (int s = 0; s < numShares - 1; s++)
                {
                    shareSum += shares[s][d, w];
                }

                lastShare[d, w] = sketch[d, w] - shareSum;
            }
        }

        shares.Add(lastShare);
        return shares;
    }

    /// <summary>
    /// Finds items from a candidate set that appear at least <paramref name="threshold"/> times
    /// across all parties, using merged count-min sketches.
    /// </summary>
    /// <param name="mergedSketch">The merged sketch from all parties.</param>
    /// <param name="candidateItems">Items to check against the threshold.</param>
    /// <param name="threshold">Minimum frequency to include in results.</param>
    /// <returns>Items meeting the threshold with their estimated frequencies.</returns>
    public Dictionary<string, int> ThresholdQuery(int[,] mergedSketch, IEnumerable<string> candidateItems, int threshold)
    {
        Guard.NotNull(mergedSketch);
        Guard.NotNull(candidateItems);
        if (threshold <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(threshold), "Threshold must be positive.");
        }

        var results = new Dictionary<string, int>();

        foreach (var item in candidateItems)
        {
            int freq = EstimateFrequency(mergedSketch, item);
            if (freq >= threshold)
            {
                results[item] = freq;
            }
        }

        return results;
    }

    private int ComputeBucketIndex(int hash, int depth)
    {
        // Use a different hash function per depth level via multiplicative hashing.
        unchecked
        {
            int h = (int)((long)hash * (2654435761L + (long)depth * 2246822519L));
            return ((h >> 16) ^ h) & (_sketchWidth - 1);
        }
    }

    private static int GetStableHash(string s)
    {
        // Deterministic hash (string.GetHashCode is not stable across processes in .NET Core).
        unchecked
        {
            int hash = 17;
            foreach (char c in s)
            {
                hash = hash * 31 + c;
            }

            return hash;
        }
    }

    private static uint MurmurHash3(string key)
    {
        // MurmurHash3 32-bit (x86) for HyperLogLog.
        // Processes 4-byte (2-char) blocks per the reference implementation for correct distribution.
        unchecked
        {
            const uint c1 = 0xcc9e2d51;
            const uint c2 = 0x1b873593;
            const uint seed = 0x9747b28c;

            uint h1 = seed;
            int len = key.Length;
            int nblocks = len / 2; // 2 chars = 4 bytes per block

            // Body: process 4-byte blocks (2 chars each).
            for (int i = 0; i < nblocks; i++)
            {
                uint k1 = (uint)key[i * 2] | ((uint)key[i * 2 + 1] << 16);
                k1 *= c1;
                k1 = RotateLeft(k1, 15);
                k1 *= c2;

                h1 ^= k1;
                h1 = RotateLeft(h1, 13);
                h1 = h1 * 5 + 0xe6546b64;
            }

            // Tail: remaining char (if odd length).
            if (len % 2 != 0)
            {
                uint k1 = key[len - 1];
                k1 *= c1;
                k1 = RotateLeft(k1, 15);
                k1 *= c2;
                h1 ^= k1;
            }

            // Finalization: mix with byte length (chars * 2).
            h1 ^= (uint)(len * 2);
            h1 ^= h1 >> 16;
            h1 *= 0x85ebca6b;
            h1 ^= h1 >> 13;
            h1 *= 0xc2b2ae35;
            h1 ^= h1 >> 16;

            return h1;
        }
    }

    private static uint RotateLeft(uint value, int count)
    {
        return (value << count) | (value >> (32 - count));
    }

    private static int CountLeadingZeros(uint value)
    {
        if (value == 0)
        {
            return 32;
        }

        int n = 0;
        if ((value & 0xFFFF0000) == 0) { n += 16; value <<= 16; }
        if ((value & 0xFF000000) == 0) { n += 8; value <<= 8; }
        if ((value & 0xF0000000) == 0) { n += 4; value <<= 4; }
        if ((value & 0xC0000000) == 0) { n += 2; value <<= 2; }
        if ((value & 0x80000000) == 0) { n += 1; }

        return n;
    }

    /// <summary>Gets the sketch width.</summary>
    public int SketchWidth => _sketchWidth;

    /// <summary>Gets the sketch depth.</summary>
    public int SketchDepth => _sketchDepth;

    /// <summary>Gets the HyperLogLog precision parameter (p).</summary>
    public int HLLPrecision => _hllPrecision;

    /// <summary>Gets the number of HyperLogLog registers (2^p).</summary>
    public int HLLRegisterCount => _hllRegisterCount;
}
