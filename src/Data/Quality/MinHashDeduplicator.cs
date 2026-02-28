using AiDotNet.Helpers;

namespace AiDotNet.Data.Quality;

/// <summary>
/// Detects near-duplicate documents using MinHash with Locality-Sensitive Hashing (LSH).
/// </summary>
/// <remarks>
/// <para>
/// MinHash approximates Jaccard similarity between document shingle sets.
/// LSH banding reduces comparison count from O(n^2) to near-linear.
/// Commonly used for deduplicating web crawl data (e.g., C4, The Pile).
/// </para>
/// </remarks>
public class MinHashDeduplicator
{
    private readonly MinHashDeduplicatorOptions _options;
    private readonly int[] _hashA;
    private readonly int[] _hashB;
    private readonly Random _random;

    public MinHashDeduplicator(MinHashDeduplicatorOptions? options = null)
    {
        _options = options ?? new MinHashDeduplicatorOptions();
        _random = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();

        // Generate random hash function parameters
        _hashA = new int[_options.NumHashFunctions];
        _hashB = new int[_options.NumHashFunctions];
        for (int i = 0; i < _options.NumHashFunctions; i++)
        {
            _hashA[i] = _random.Next(1, int.MaxValue);
            _hashB[i] = _random.Next(0, int.MaxValue);
        }
    }

    /// <summary>
    /// Computes the MinHash signature for a document.
    /// </summary>
    /// <param name="text">The document text.</param>
    /// <returns>MinHash signature array of length NumHashFunctions.</returns>
    public int[] ComputeSignature(string text)
    {
        var shingles = GetShingles(text);
        var signature = new int[_options.NumHashFunctions];
        for (int s = 0; s < signature.Length; s++)
            signature[s] = int.MaxValue;

        foreach (int shingleHash in shingles)
        {
            for (int i = 0; i < _options.NumHashFunctions; i++)
            {
                int hashVal = unchecked(_hashA[i] * shingleHash + _hashB[i]);
                if (hashVal < signature[i])
                    signature[i] = hashVal;
            }
        }

        return signature;
    }

    /// <summary>
    /// Estimates Jaccard similarity between two documents from their MinHash signatures.
    /// </summary>
    public double EstimateSimilarity(int[] sig1, int[] sig2)
    {
        int matches = 0;
        for (int i = 0; i < sig1.Length; i++)
        {
            if (sig1[i] == sig2[i])
                matches++;
        }
        return (double)matches / sig1.Length;
    }

    /// <summary>
    /// Finds duplicate indices from a collection of documents using LSH banding.
    /// </summary>
    /// <param name="documents">The documents to deduplicate.</param>
    /// <returns>Set of indices that are duplicates (should be removed).</returns>
    public HashSet<int> FindDuplicates(IReadOnlyList<string> documents)
    {
        var signatures = new int[documents.Count][];
        for (int i = 0; i < documents.Count; i++)
            signatures[i] = ComputeSignature(documents[i]);

        int rowsPerBand = _options.NumHashFunctions / _options.NumBands;
        var duplicates = new HashSet<int>();
        var kept = new HashSet<int>();

        // LSH banding: hash each band and find candidate pairs
        for (int b = 0; b < _options.NumBands; b++)
        {
            var buckets = new Dictionary<long, List<int>>();
            int bandStart = b * rowsPerBand;

            for (int doc = 0; doc < documents.Count; doc++)
            {
                if (duplicates.Contains(doc)) continue;

                long bandHash = 0;
                for (int r = 0; r < rowsPerBand && bandStart + r < _options.NumHashFunctions; r++)
                {
                    bandHash = unchecked(bandHash * 31 + signatures[doc][bandStart + r]);
                }

                if (!buckets.TryGetValue(bandHash, out var bucket))
                {
                    bucket = new List<int>();
                    buckets[bandHash] = bucket;
                }

                // Check against existing items in bucket
                foreach (int existing in bucket)
                {
                    if (kept.Contains(existing) && !duplicates.Contains(doc))
                    {
                        double sim = EstimateSimilarity(signatures[existing], signatures[doc]);
                        if (sim >= _options.SimilarityThreshold)
                        {
                            duplicates.Add(doc);
                            break;
                        }
                    }
                }

                if (!duplicates.Contains(doc))
                {
                    kept.Add(doc);
                    bucket.Add(doc);
                }
            }
        }

        return duplicates;
    }

    private HashSet<int> GetShingles(string text)
    {
        var shingles = new HashSet<int>();
        string normalized = text.ToLowerInvariant();
        for (int i = 0; i <= normalized.Length - _options.ShingleSize; i++)
        {
            shingles.Add(normalized.Substring(i, _options.ShingleSize).GetHashCode());
        }
        return shingles;
    }
}
