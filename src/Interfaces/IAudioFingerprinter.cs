using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for audio fingerprinting algorithms.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Audio fingerprinting creates compact representations of audio that can be
/// used for identification and similarity matching. Different algorithms
/// trade off between accuracy, speed, and robustness to transformations.
/// </para>
/// <para><b>For Beginners:</b> An audio fingerprint is like a "signature" for
/// a piece of audio. Just like human fingerprints identify individuals, audio
/// fingerprints identify songs or sound recordings. Services like Shazam use
/// fingerprinting to identify songs from short recordings.
/// </para>
/// </remarks>
public interface IAudioFingerprinter<T>
{
    /// <summary>
    /// Gets the name of the fingerprinting algorithm.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Gets the expected sample rate for input audio.
    /// </summary>
    int SampleRate { get; }

    /// <summary>
    /// Gets the fingerprint length in bits or elements.
    /// </summary>
    int FingerprintLength { get; }

    /// <summary>
    /// Generates a fingerprint from audio data.
    /// </summary>
    /// <param name="audio">Audio samples as a tensor (mono audio).</param>
    /// <returns>The audio fingerprint.</returns>
    AudioFingerprint<T> Fingerprint(Tensor<T> audio);

    /// <summary>
    /// Generates a fingerprint from audio data.
    /// </summary>
    /// <param name="audio">Audio samples as a vector (mono audio).</param>
    /// <returns>The audio fingerprint.</returns>
    AudioFingerprint<T> Fingerprint(Vector<T> audio);

    /// <summary>
    /// Computes the similarity between two fingerprints.
    /// </summary>
    /// <param name="fp1">First fingerprint.</param>
    /// <param name="fp2">Second fingerprint.</param>
    /// <returns>Similarity score (0-1, higher is more similar).</returns>
    double ComputeSimilarity(AudioFingerprint<T> fp1, AudioFingerprint<T> fp2);

    /// <summary>
    /// Finds matching segments between two fingerprints.
    /// </summary>
    /// <param name="query">The query fingerprint.</param>
    /// <param name="reference">The reference fingerprint to search in.</param>
    /// <param name="minMatchLength">Minimum length of matching segment.</param>
    /// <returns>List of matching segments with time offsets.</returns>
    IReadOnlyList<FingerprintMatch> FindMatches(
        AudioFingerprint<T> query,
        AudioFingerprint<T> reference,
        int minMatchLength = 10);
}

/// <summary>
/// Represents an audio fingerprint.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class AudioFingerprint<T>
{
    /// <summary>
    /// Gets or sets the fingerprint data.
    /// </summary>
    public required T[] Data { get; set; }

    /// <summary>
    /// Gets or sets the binary hash (for compact storage/comparison).
    /// </summary>
    public uint[]? Hash { get; set; }

    /// <summary>
    /// Gets or sets the duration of the fingerprinted audio in seconds.
    /// </summary>
    public double Duration { get; set; }

    /// <summary>
    /// Gets or sets the sample rate of the source audio.
    /// </summary>
    public int SampleRate { get; set; }

    /// <summary>
    /// Gets or sets the algorithm used to create the fingerprint.
    /// </summary>
    public string Algorithm { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the number of frames in the fingerprint.
    /// </summary>
    public int FrameCount { get; set; }

    /// <summary>
    /// Gets or sets optional metadata about the fingerprint.
    /// </summary>
    public Dictionary<string, object>? Metadata { get; set; }
}

/// <summary>
/// Represents a match between two fingerprints.
/// </summary>
public class FingerprintMatch
{
    /// <summary>
    /// Gets or sets the start time in the query (seconds).
    /// </summary>
    public double QueryStartTime { get; set; }

    /// <summary>
    /// Gets or sets the start time in the reference (seconds).
    /// </summary>
    public double ReferenceStartTime { get; set; }

    /// <summary>
    /// Gets or sets the duration of the match (seconds).
    /// </summary>
    public double Duration { get; set; }

    /// <summary>
    /// Gets or sets the confidence score (0-1).
    /// </summary>
    public double Confidence { get; set; }

    /// <summary>
    /// Gets or sets the number of matching hashes.
    /// </summary>
    public int MatchCount { get; set; }
}
