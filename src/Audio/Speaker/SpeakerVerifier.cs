using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.Speaker;

/// <summary>
/// Verifies speaker identity by comparing embeddings against enrolled speakers.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Speaker verification answers the question "Is this the person they claim to be?"
/// by comparing a test utterance against enrolled speaker embeddings.
/// </para>
/// <para><b>For Beginners:</b> Speaker verification is like voice-based password checking:
/// 1. First, you "enroll" a speaker by recording their voice samples
/// 2. Later, when someone claims to be that person, you record them and compare
/// 3. If the voices match closely enough, the identity is verified
///
/// Common applications:
/// - Voice-based authentication (banking apps)
/// - Access control systems
/// - Personalized voice assistants
///
/// Usage:
/// <code>
/// var verifier = new SpeakerVerifier&lt;float&gt;();
///
/// // Enroll a speaker
/// verifier.Enroll("user123", embedding1, embedding2, embedding3);
///
/// // Verify identity
/// var result = verifier.Verify("user123", testEmbedding);
/// if (result.IsVerified)
///     Console.WriteLine("Identity verified!");
/// </code>
/// </para>
/// </remarks>
public class SpeakerVerifier<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly SpeakerVerifierOptions _options;
    private readonly Dictionary<string, EnrolledSpeaker<T>> _enrolledSpeakers;

    /// <summary>
    /// Gets the number of enrolled speakers.
    /// </summary>
    public int EnrolledCount => _enrolledSpeakers.Count;

    /// <summary>
    /// Creates a new speaker verifier.
    /// </summary>
    /// <param name="options">Verifier options.</param>
    public SpeakerVerifier(SpeakerVerifierOptions? options = null)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _options = options ?? new SpeakerVerifierOptions();
        _enrolledSpeakers = new Dictionary<string, EnrolledSpeaker<T>>();
    }

    /// <summary>
    /// Enrolls a speaker with one or more embeddings.
    /// </summary>
    /// <param name="speakerId">Unique speaker identifier.</param>
    /// <param name="embeddings">One or more enrollment embeddings.</param>
    public void Enroll(string speakerId, params SpeakerEmbedding<T>[] embeddings)
    {
        if (embeddings.Length == 0)
            throw new ArgumentException("At least one embedding required for enrollment.", nameof(embeddings));

        var centroid = ComputeCentroid(embeddings);

        _enrolledSpeakers[speakerId] = new EnrolledSpeaker<T>
        {
            SpeakerId = speakerId,
            Centroid = centroid,
            Embeddings = [.. embeddings],
            EnrollmentTime = DateTime.UtcNow
        };
    }

    /// <summary>
    /// Adds additional embeddings to an existing enrollment.
    /// </summary>
    /// <param name="speakerId">Speaker identifier.</param>
    /// <param name="embeddings">Additional embeddings.</param>
    public void AddToEnrollment(string speakerId, params SpeakerEmbedding<T>[] embeddings)
    {
        if (!_enrolledSpeakers.TryGetValue(speakerId, out var speaker))
            throw new KeyNotFoundException($"Speaker '{speakerId}' not enrolled.");

        speaker.Embeddings.AddRange(embeddings);
        speaker.Centroid = ComputeCentroid(speaker.Embeddings);
    }

    /// <summary>
    /// Removes a speaker's enrollment.
    /// </summary>
    /// <param name="speakerId">Speaker identifier.</param>
    /// <returns>True if removed, false if not found.</returns>
    public bool Unenroll(string speakerId)
    {
        return _enrolledSpeakers.Remove(speakerId);
    }

    /// <summary>
    /// Verifies if a test embedding matches an enrolled speaker.
    /// </summary>
    /// <param name="speakerId">Claimed speaker identity.</param>
    /// <param name="testEmbedding">Embedding from test utterance.</param>
    /// <returns>Verification result.</returns>
    public VerificationResult Verify(string speakerId, SpeakerEmbedding<T> testEmbedding)
    {
        if (!_enrolledSpeakers.TryGetValue(speakerId, out var speaker))
        {
            return new VerificationResult
            {
                ClaimedSpeakerId = speakerId,
                IsVerified = false,
                Score = 0,
                Threshold = _options.VerificationThreshold,
                ErrorMessage = "Speaker not enrolled"
            };
        }

        // Compare against centroid
        double score = testEmbedding.CosineSimilarity(speaker.Centroid);

        return new VerificationResult
        {
            ClaimedSpeakerId = speakerId,
            IsVerified = score >= _options.VerificationThreshold,
            Score = score,
            Threshold = _options.VerificationThreshold
        };
    }

    /// <summary>
    /// Identifies the most likely speaker from enrolled set.
    /// </summary>
    /// <param name="testEmbedding">Embedding from test utterance.</param>
    /// <returns>Identification result with ranked matches.</returns>
    public IdentificationResult Identify(SpeakerEmbedding<T> testEmbedding)
    {
        var scores = new List<(string speakerId, double score)>();

        foreach (var (speakerId, speaker) in _enrolledSpeakers)
        {
            double score = testEmbedding.CosineSimilarity(speaker.Centroid);
            scores.Add((speakerId, score));
        }

        var ranked = scores.OrderByDescending(s => s.score).ToList();

        var result = new IdentificationResult
        {
            Matches = ranked.Select(s => new SpeakerMatch
            {
                SpeakerId = s.speakerId,
                Score = s.score
            }).ToList(),
            Threshold = _options.IdentificationThreshold
        };

        if (ranked.Count > 0 && ranked[0].score >= _options.IdentificationThreshold)
        {
            result.IdentifiedSpeakerId = ranked[0].speakerId;
            result.TopScore = ranked[0].score;
        }

        return result;
    }

    /// <summary>
    /// Checks if a speaker is enrolled.
    /// </summary>
    public bool IsEnrolled(string speakerId) => _enrolledSpeakers.ContainsKey(speakerId);

    /// <summary>
    /// Gets all enrolled speaker IDs.
    /// </summary>
    public IReadOnlyList<string> GetEnrolledSpeakers() => _enrolledSpeakers.Keys.ToList();

    private SpeakerEmbedding<T> ComputeCentroid(IReadOnlyList<SpeakerEmbedding<T>> embeddings)
    {
        if (embeddings.Count == 1)
            return embeddings[0];

        int dim = embeddings[0].Vector.Length;
        var centroid = new T[dim];
        double totalDuration = 0;
        int totalFrames = 0;

        // Average the embeddings
        for (int d = 0; d < dim; d++)
        {
            double sum = 0;
            foreach (var emb in embeddings)
            {
                sum += _numOps.ToDouble(emb.Vector[d]);
            }
            centroid[d] = _numOps.FromDouble(sum / embeddings.Count);
        }

        foreach (var emb in embeddings)
        {
            totalDuration += emb.Duration;
            totalFrames += emb.NumFrames;
        }

        // L2 normalize
        double norm = 0;
        for (int d = 0; d < dim; d++)
        {
            double v = _numOps.ToDouble(centroid[d]);
            norm += v * v;
        }
        norm = Math.Sqrt(norm);

        if (norm > 1e-10)
        {
            for (int d = 0; d < dim; d++)
            {
                double v = _numOps.ToDouble(centroid[d]) / norm;
                centroid[d] = _numOps.FromDouble(v);
            }
        }

        return new SpeakerEmbedding<T>
        {
            Vector = centroid,
            Duration = totalDuration,
            NumFrames = totalFrames
        };
    }
}

/// <summary>
/// Represents an enrolled speaker.
/// </summary>
internal class EnrolledSpeaker<T>
{
    public string SpeakerId { get; set; } = string.Empty;
    public required SpeakerEmbedding<T> Centroid { get; set; }
    public List<SpeakerEmbedding<T>> Embeddings { get; set; } = [];
    public DateTime EnrollmentTime { get; set; }
}

/// <summary>
/// Result of speaker verification.
/// </summary>
public class VerificationResult
{
    /// <summary>
    /// Gets or sets the claimed speaker ID.
    /// </summary>
    public string ClaimedSpeakerId { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets whether the verification succeeded.
    /// </summary>
    public bool IsVerified { get; set; }

    /// <summary>
    /// Gets or sets the similarity score.
    /// </summary>
    public double Score { get; set; }

    /// <summary>
    /// Gets or sets the threshold used for verification.
    /// </summary>
    public double Threshold { get; set; }

    /// <summary>
    /// Gets or sets any error message.
    /// </summary>
    public string? ErrorMessage { get; set; }
}

/// <summary>
/// Result of speaker identification.
/// </summary>
public class IdentificationResult
{
    /// <summary>
    /// Gets or sets the identified speaker ID (null if no match above threshold).
    /// </summary>
    public string? IdentifiedSpeakerId { get; set; }

    /// <summary>
    /// Gets or sets the top match score.
    /// </summary>
    public double TopScore { get; set; }

    /// <summary>
    /// Gets or sets the threshold used for identification.
    /// </summary>
    public double Threshold { get; set; }

    /// <summary>
    /// Gets or sets ranked matches for all enrolled speakers.
    /// </summary>
    public List<SpeakerMatch> Matches { get; set; } = [];
}

/// <summary>
/// A speaker match with score.
/// </summary>
public class SpeakerMatch
{
    /// <summary>
    /// Gets or sets the speaker ID.
    /// </summary>
    public string SpeakerId { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the similarity score.
    /// </summary>
    public double Score { get; set; }
}

/// <summary>
/// Configuration options for speaker verification.
/// </summary>
public class SpeakerVerifierOptions
{
    /// <summary>
    /// Gets or sets the threshold for verification (0-1).
    /// </summary>
    public double VerificationThreshold { get; set; } = 0.7;

    /// <summary>
    /// Gets or sets the threshold for identification (0-1).
    /// </summary>
    public double IdentificationThreshold { get; set; } = 0.6;
}
