using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.Fingerprinting;

/// <summary>
/// Base class for audio fingerprinting algorithms.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Audio fingerprinting creates compact identifiers that can recognize audio content
/// even after degradation (compression, noise, speed changes). Unlike neural network
/// approaches, fingerprinting typically uses signal processing techniques.
/// </para>
/// <para>
/// <b>For Beginners:</b> Audio fingerprinting is like Shazam - it can identify a song
/// from a short audio clip, even if the audio is noisy or compressed.
///
/// How it works:
/// 1. Audio is converted to a spectrogram
/// 2. Key features (peaks, patterns) are extracted
/// 3. Features are hashed into a compact fingerprint
/// 4. Fingerprints can be matched against a database
///
/// This base class provides:
/// - Hash computation utilities
/// - Hamming distance for comparison
/// - Time alignment for matching
/// </para>
/// </remarks>
public abstract class AudioFingerprinterBase<T> : IAudioFingerprinter<T>
{
    /// <summary>
    /// Operations for the numeric type T.
    /// </summary>
    protected readonly INumericOperations<T> NumOps;

    /// <summary>
    /// Gets the name of the fingerprinting algorithm.
    /// </summary>
    public abstract string Name { get; }

    /// <summary>
    /// Gets the expected sample rate for input audio.
    /// </summary>
    public int SampleRate { get; protected set; } = 22050;

    /// <summary>
    /// Gets the fingerprint length in bits or elements.
    /// </summary>
    public int FingerprintLength { get; protected set; } = 32;

    /// <summary>
    /// Gets or sets the duration of each fingerprint frame in seconds.
    /// </summary>
    public double FrameDuration { get; protected set; } = 0.37;

    /// <summary>
    /// Initializes a new instance of the AudioFingerprinterBase class.
    /// </summary>
    protected AudioFingerprinterBase()
    {
        NumOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Generates a fingerprint from audio data.
    /// </summary>
    /// <param name="audio">Audio samples as a tensor (mono audio).</param>
    /// <returns>The audio fingerprint.</returns>
    public abstract AudioFingerprint<T> Fingerprint(Tensor<T> audio);

    /// <summary>
    /// Generates a fingerprint from audio data.
    /// </summary>
    /// <param name="audio">Audio samples as a vector (mono audio).</param>
    /// <returns>The audio fingerprint.</returns>
    public virtual AudioFingerprint<T> Fingerprint(Vector<T> audio)
    {
        // Convert vector to tensor and delegate to main implementation
        var tensor = new Tensor<T>([audio.Length]);
        for (int i = 0; i < audio.Length; i++)
        {
            tensor[i] = audio[i];
        }
        return Fingerprint(tensor);
    }

    /// <summary>
    /// Computes the similarity between two fingerprints.
    /// </summary>
    /// <param name="fp1">First fingerprint.</param>
    /// <param name="fp2">Second fingerprint.</param>
    /// <returns>Similarity score (0-1, higher is more similar).</returns>
    public abstract double ComputeSimilarity(AudioFingerprint<T> fp1, AudioFingerprint<T> fp2);

    /// <summary>
    /// Finds matching segments between two fingerprints.
    /// </summary>
    /// <param name="query">The query fingerprint.</param>
    /// <param name="reference">The reference fingerprint to search in.</param>
    /// <param name="minMatchLength">Minimum length of matching segment.</param>
    /// <returns>List of matching segments with time offsets.</returns>
    public abstract IReadOnlyList<FingerprintMatch> FindMatches(
        AudioFingerprint<T> query,
        AudioFingerprint<T> reference,
        int minMatchLength = 10);

    /// <summary>
    /// Computes Hamming distance between two fingerprint hashes.
    /// </summary>
    /// <param name="fp1">First fingerprint hash.</param>
    /// <param name="fp2">Second fingerprint hash.</param>
    /// <returns>Number of differing bits.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Hamming distance counts how many bits are different
    /// between two fingerprints. Lower distance = more similar audio.
    /// </para>
    /// </remarks>
    protected int ComputeHammingDistance(uint fp1, uint fp2)
    {
        uint xor = fp1 ^ fp2;
        int distance = 0;

        while (xor != 0)
        {
            distance += (int)(xor & 1);
            xor >>= 1;
        }

        return distance;
    }

    /// <summary>
    /// Computes average Hamming distance between fingerprint hash sequences.
    /// </summary>
    /// <param name="hashes1">First fingerprint hash sequence.</param>
    /// <param name="hashes2">Second fingerprint hash sequence.</param>
    /// <returns>Average Hamming distance (0.0 to 1.0 normalized).</returns>
    protected double ComputeAverageHammingDistance(uint[] hashes1, uint[] hashes2)
    {
        int minLen = Math.Min(hashes1.Length, hashes2.Length);
        if (minLen == 0) return 1.0;

        int totalDistance = 0;
        for (int i = 0; i < minLen; i++)
        {
            totalDistance += ComputeHammingDistance(hashes1[i], hashes2[i]);
        }

        return (double)totalDistance / (minLen * FingerprintLength);
    }

    /// <summary>
    /// Finds the best alignment offset between two fingerprint hash sequences.
    /// </summary>
    /// <param name="queryHashes">Query fingerprint hashes.</param>
    /// <param name="referenceHashes">Reference fingerprint hashes.</param>
    /// <param name="maxOffset">Maximum offset to search.</param>
    /// <returns>Best offset and corresponding similarity score.</returns>
    protected (int Offset, double Similarity) FindBestAlignment(
        uint[] queryHashes,
        uint[] referenceHashes,
        int maxOffset = 100)
    {
        int bestOffset = 0;
        double bestSimilarity = 0;

        int searchRange = Math.Min(maxOffset, referenceHashes.Length - 1);

        for (int offset = -searchRange; offset <= searchRange; offset++)
        {
            double similarity = ComputeAlignedSimilarity(queryHashes, referenceHashes, offset);
            if (similarity > bestSimilarity)
            {
                bestSimilarity = similarity;
                bestOffset = offset;
            }
        }

        return (bestOffset, bestSimilarity);
    }

    /// <summary>
    /// Computes similarity at a specific alignment offset.
    /// </summary>
    /// <param name="queryHashes">Query fingerprint hashes.</param>
    /// <param name="referenceHashes">Reference fingerprint hashes.</param>
    /// <param name="offset">Alignment offset.</param>
    /// <returns>Similarity score (0.0 to 1.0).</returns>
    protected double ComputeAlignedSimilarity(uint[] queryHashes, uint[] referenceHashes, int offset)
    {
        int queryStart = offset < 0 ? -offset : 0;
        int refStart = offset > 0 ? offset : 0;

        int queryEnd = Math.Min(queryHashes.Length, referenceHashes.Length - offset);
        int refEnd = Math.Min(referenceHashes.Length, queryHashes.Length + offset);

        int overlapLen = Math.Min(queryEnd - queryStart, refEnd - refStart);
        if (overlapLen <= 0) return 0;

        int matches = 0;
        int totalBits = overlapLen * FingerprintLength;

        for (int i = 0; i < overlapLen; i++)
        {
            int distance = ComputeHammingDistance(queryHashes[queryStart + i], referenceHashes[refStart + i]);
            matches += FingerprintLength - distance;
        }

        return (double)matches / totalBits;
    }

    /// <summary>
    /// Converts a feature matrix to hash values.
    /// </summary>
    /// <param name="features">Feature matrix [rows, cols].</param>
    /// <param name="threshold">Threshold for binarization.</param>
    /// <returns>Array of hash values.</returns>
    protected uint[] ComputeHashes(T[,] features, T threshold)
    {
        int rows = features.GetLength(0);
        int cols = features.GetLength(1);
        int numHashes = rows;
        uint[] hashes = new uint[numHashes];

        for (int i = 0; i < rows; i++)
        {
            uint hash = 0;
            for (int j = 0; j < Math.Min(cols, 32); j++)
            {
                if (NumOps.GreaterThan(features[i, j], threshold))
                {
                    hash |= (1u << j);
                }
            }
            hashes[i] = hash;
        }

        return hashes;
    }

    /// <summary>
    /// Converts frame index to time in seconds.
    /// </summary>
    /// <param name="frameIndex">Frame index.</param>
    /// <returns>Time in seconds.</returns>
    protected double FrameToTime(int frameIndex)
    {
        return frameIndex * FrameDuration;
    }

    /// <summary>
    /// Converts time in seconds to frame index.
    /// </summary>
    /// <param name="time">Time in seconds.</param>
    /// <returns>Frame index.</returns>
    protected int TimeToFrame(double time)
    {
        return (int)(time / FrameDuration);
    }
}
