using AiDotNet.Audio.Features;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.Fingerprinting;

/// <summary>
/// Chromaprint-style audio fingerprinter based on chroma features.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This fingerprinter uses chromagram analysis similar to the Chromaprint algorithm
/// used by AcoustID. It extracts chroma features and converts them to a compact
/// binary representation that is robust to tempo changes and transposition.
/// </para>
/// <para><b>For Beginners:</b> Chromaprint works by analyzing the musical notes
/// present in the audio. It groups all octaves of the same note together (C1, C2, C3
/// all become "C") and tracks how these change over time. This makes it good at
/// matching different recordings of the same song.
/// </para>
/// </remarks>
public class ChromaprintFingerprinter<T> : AudioFingerprinterBase<T>
{
    private readonly ChromaExtractor<T> _chromaExtractor;
    private readonly ChromaprintOptions _options;

    /// <summary>
    /// Gets the name of the fingerprinting algorithm.
    /// </summary>
    public override string Name => "Chromaprint";

    /// <summary>
    /// Creates a new Chromaprint fingerprinter.
    /// </summary>
    /// <param name="options">Configuration options.</param>
    public ChromaprintFingerprinter(ChromaprintOptions? options = null)
    {
        _options = options ?? new ChromaprintOptions();

        // Set base class properties
        SampleRate = _options.SampleRate;
        FingerprintLength = 32;

        _chromaExtractor = new ChromaExtractor<T>(new ChromaOptions
        {
            SampleRate = _options.SampleRate,
            FftSize = _options.FftSize,
            HopLength = _options.HopLength
        });
    }

    /// <summary>
    /// Generates a fingerprint from audio tensor.
    /// </summary>
    public override AudioFingerprint<T> Fingerprint(Tensor<T> audio)
    {
        // Extract chroma features
        var chroma = _chromaExtractor.Extract(audio);

        // Convert to fingerprint
        return ComputeFingerprint(chroma, audio.Length);
    }

    /// <summary>
    /// Generates a fingerprint from audio vector.
    /// </summary>
    public override AudioFingerprint<T> Fingerprint(Vector<T> audio)
    {
        // Extract chroma features
        var chroma = _chromaExtractor.Extract(audio);

        // Convert to fingerprint
        return ComputeFingerprintFromMatrix(chroma, audio.Length);
    }

    private AudioFingerprint<T> ComputeFingerprint(Tensor<T> chroma, int audioLength)
    {
        int numFrames = chroma.Shape[0];
        int numChroma = chroma.Shape[1];

        var hashes = new List<uint>();
        var fpData = new List<T>();

        // Process overlapping frames to create hash sequence
        for (int i = 0; i < numFrames - _options.ContextSize + 1; i += _options.HashStep)
        {
            uint hash = ComputeFrameHash(chroma, i, numChroma);
            hashes.Add(hash);
            fpData.Add(NumOps.FromDouble(hash));
        }

        return new AudioFingerprint<T>
        {
            Data = [.. fpData],
            Hash = [.. hashes],
            Duration = (double)audioLength / _options.SampleRate,
            SampleRate = _options.SampleRate,
            Algorithm = Name,
            FrameCount = numFrames
        };
    }

    private AudioFingerprint<T> ComputeFingerprintFromMatrix(Matrix<T> chroma, int audioLength)
    {
        int numFrames = chroma.Rows;
        int numChroma = chroma.Columns;

        var hashes = new List<uint>();
        var fpData = new List<T>();

        for (int i = 0; i < numFrames - _options.ContextSize + 1; i += _options.HashStep)
        {
            uint hash = ComputeFrameHashFromMatrix(chroma, i, numChroma);
            hashes.Add(hash);
            fpData.Add(NumOps.FromDouble(hash));
        }

        return new AudioFingerprint<T>
        {
            Data = [.. fpData],
            Hash = [.. hashes],
            Duration = (double)audioLength / _options.SampleRate,
            SampleRate = _options.SampleRate,
            Algorithm = Name,
            FrameCount = numFrames
        };
    }

    private uint ComputeFrameHash(Tensor<T> chroma, int startFrame, int numChroma)
    {
        // Collect context frames
        var context = new double[_options.ContextSize, numChroma];
        for (int f = 0; f < _options.ContextSize; f++)
        {
            for (int c = 0; c < numChroma; c++)
            {
                context[f, c] = NumOps.ToDouble(chroma[startFrame + f, c]);
            }
        }

        // Compute gray codes from chroma differences
        return ComputeGrayCode(context, numChroma);
    }

    private uint ComputeFrameHashFromMatrix(Matrix<T> chroma, int startFrame, int numChroma)
    {
        var context = new double[_options.ContextSize, numChroma];
        for (int f = 0; f < _options.ContextSize; f++)
        {
            for (int c = 0; c < numChroma; c++)
            {
                context[f, c] = NumOps.ToDouble(chroma[startFrame + f, c]);
            }
        }

        return ComputeGrayCode(context, numChroma);
    }

    private uint ComputeGrayCode(double[,] context, int numChroma)
    {
        uint hash = 0;
        int bitPos = 0;

        // Compare adjacent chroma bins across time
        for (int f = 0; f < context.GetLength(0) - 1 && bitPos < 32; f++)
        {
            for (int c = 0; c < numChroma - 1 && bitPos < 32; c++)
            {
                // Compare current bin to next bin
                double diff1 = context[f, c] - context[f, c + 1];
                // Compare current time to next time
                double diff2 = context[f, c] - context[f + 1, c];

                // Encode as bits
                if (diff1 > 0) hash |= (1u << bitPos);
                bitPos++;
                if (bitPos >= 32) break;

                if (diff2 > 0) hash |= (1u << bitPos);
                bitPos++;
            }
        }

        return hash;
    }

    /// <summary>
    /// Computes similarity between two fingerprints.
    /// </summary>
    public override double ComputeSimilarity(AudioFingerprint<T> fp1, AudioFingerprint<T> fp2)
    {
        if (fp1.Hash is null || fp2.Hash is null)
        {
            return ComputeDataSimilarity(fp1.Data, fp2.Data);
        }

        // Use hash-based similarity
        return ComputeHashSimilarity(fp1.Hash, fp2.Hash);
    }

    private double ComputeHashSimilarity(uint[] hash1, uint[] hash2)
    {
        // Find best alignment using cross-correlation of hash sequences
        int minLen = Math.Min(hash1.Length, hash2.Length);
        int maxOffset = Math.Max(hash1.Length, hash2.Length) - minLen;

        double bestSimilarity = 0;

        for (int offset = -maxOffset; offset <= maxOffset; offset++)
        {
            int matches = 0;
            int total = 0;

            for (int i = 0; i < minLen; i++)
            {
                int idx1 = i;
                int idx2 = i + offset;

                if (idx2 >= 0 && idx2 < hash2.Length)
                {
                    // Count matching bits
                    uint xor = hash1[idx1] ^ hash2[idx2];
                    int differentBits = BitCount(xor);
                    matches += 32 - differentBits;
                    total += 32;
                }
            }

            if (total > 0)
            {
                double similarity = (double)matches / total;
                bestSimilarity = Math.Max(bestSimilarity, similarity);
            }
        }

        return bestSimilarity;
    }

    private double ComputeDataSimilarity(T[] data1, T[] data2)
    {
        // Compute correlation
        int minLen = Math.Min(data1.Length, data2.Length);
        double sum = 0;
        double norm1 = 0;
        double norm2 = 0;

        for (int i = 0; i < minLen; i++)
        {
            double v1 = NumOps.ToDouble(data1[i]);
            double v2 = NumOps.ToDouble(data2[i]);
            sum += v1 * v2;
            norm1 += v1 * v1;
            norm2 += v2 * v2;
        }

        double denominator = Math.Sqrt(norm1 * norm2);
        if (denominator < 1e-10) return 0;

        return sum / denominator;
    }

    /// <summary>
    /// Finds matching segments between fingerprints.
    /// </summary>
    public override IReadOnlyList<FingerprintMatch> FindMatches(
        AudioFingerprint<T> query,
        AudioFingerprint<T> reference,
        int minMatchLength = 10)
    {
        var matches = new List<FingerprintMatch>();

        if (query.Hash is null || reference.Hash is null)
            return matches;

        // Build hash-to-position index for reference
        var hashIndex = new Dictionary<uint, List<int>>();
        for (int i = 0; i < reference.Hash.Length; i++)
        {
            uint h = reference.Hash[i];
            if (!hashIndex.TryGetValue(h, out var positions))
            {
                positions = [];
                hashIndex[h] = positions;
            }
            positions.Add(i);
        }

        // Find matching hash pairs and build offset histogram
        var offsetCounts = new Dictionary<int, List<(int queryPos, int refPos)>>();

        for (int q = 0; q < query.Hash.Length; q++)
        {
            uint qHash = query.Hash[q];

            // Allow approximate matches (1-2 bit differences)
            foreach (uint variant in GetHashVariants(qHash, _options.MaxBitDifference))
            {
                if (hashIndex.TryGetValue(variant, out var refPositions))
                {
                    foreach (int r in refPositions)
                    {
                        int offset = r - q;
                        if (!offsetCounts.TryGetValue(offset, out var pairs))
                        {
                            pairs = [];
                            offsetCounts[offset] = pairs;
                        }
                        pairs.Add((q, r));
                    }
                }
            }
        }

        // Find significant offset peaks
        double frameRate = _options.SampleRate / (double)_options.HopLength;

        foreach (var (offset, pairs) in offsetCounts)
        {
            if (pairs.Count < minMatchLength)
                continue;

            // Find contiguous segments
            var segments = FindContiguousSegments(pairs, minMatchLength);

            foreach (var (start, length, matchCount) in segments)
            {
                double queryStart = pairs[start].queryPos / frameRate;
                double refStart = pairs[start].refPos / frameRate;
                double duration = length / frameRate;

                matches.Add(new FingerprintMatch
                {
                    QueryStartTime = queryStart,
                    ReferenceStartTime = refStart,
                    Duration = duration,
                    Confidence = (double)matchCount / length,
                    MatchCount = matchCount
                });
            }
        }

        return matches.OrderByDescending(m => m.Confidence * m.Duration).ToList();
    }

    private static IEnumerable<uint> GetHashVariants(uint hash, int maxDiff)
    {
        yield return hash;

        if (maxDiff >= 1)
        {
            // Single bit flips
            for (int i = 0; i < 32; i++)
            {
                yield return hash ^ (1u << i);
            }
        }

        if (maxDiff >= 2)
        {
            // Two bit flips
            for (int i = 0; i < 31; i++)
            {
                for (int j = i + 1; j < 32; j++)
                {
                    yield return hash ^ (1u << i) ^ (1u << j);
                }
            }
        }
    }

    private static List<(int start, int length, int matchCount)> FindContiguousSegments(
        List<(int queryPos, int refPos)> pairs,
        int minLength)
    {
        var sorted = pairs.OrderBy(p => p.queryPos).ToList();
        var segments = new List<(int start, int length, int matchCount)>();

        int start = 0;
        int count = 1;

        for (int i = 1; i < sorted.Count; i++)
        {
            // Check if positions are contiguous (allowing small gaps)
            if (sorted[i].queryPos - sorted[i - 1].queryPos <= 3)
            {
                count++;
            }
            else
            {
                if (count >= minLength)
                {
                    segments.Add((start, sorted[i - 1].queryPos - sorted[start].queryPos + 1, count));
                }
                start = i;
                count = 1;
            }
        }

        // Check last segment
        if (count >= minLength)
        {
            segments.Add((start, sorted[^1].queryPos - sorted[start].queryPos + 1, count));
        }

        return segments;
    }

    private static int BitCount(uint n)
    {
        int count = 0;
        while (n != 0)
        {
            count++;
            n &= (n - 1);
        }
        return count;
    }
}
