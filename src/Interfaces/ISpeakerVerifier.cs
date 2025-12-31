namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for speaker verification models that determine if audio matches a claimed identity.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Speaker verification (also called speaker authentication) determines whether a speech
/// sample matches a claimed identity. It answers the question "Is this person who they
/// claim to be?" This is a 1-to-1 comparison task.
/// </para>
/// <para>
/// <b>For Beginners:</b> Speaker verification is like a voice-based password check.
///
/// How it works:
/// 1. User enrolls by providing voice samples
/// 2. System creates a voiceprint (speaker embedding) for that user
/// 3. Later, user provides a new voice sample
/// 4. System compares new sample to stored voiceprint
/// 5. Decision: Accept (same person) or Reject (different person)
///
/// Common use cases:
/// - Voice banking authentication
/// - Phone-based customer verification
/// - Smart speaker personalization
/// - Access control systems
///
/// Key metrics:
/// - False Accept Rate (FAR): How often imposters are wrongly accepted
/// - False Reject Rate (FRR): How often legitimate users are wrongly rejected
/// - Equal Error Rate (EER): When FAR = FRR (lower is better)
/// </para>
/// <para>
/// This interface extends <see cref="IFullModel{T, TInput, TOutput}"/> for Tensor-based audio processing.
/// </para>
/// </remarks>
public interface ISpeakerVerifier<T> : IFullModel<T, Tensor<T>, Tensor<T>>
{
    /// <summary>
    /// Gets the expected sample rate for input audio.
    /// </summary>
    int SampleRate { get; }

    /// <summary>
    /// Gets the default decision threshold for verification.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Scores above this threshold indicate same speaker.
    /// The optimal threshold depends on the security requirements:
    /// - Higher threshold: More secure (fewer false accepts) but more false rejects
    /// - Lower threshold: More convenient (fewer false rejects) but more false accepts
    /// </para>
    /// </remarks>
    T DefaultThreshold { get; }

    /// <summary>
    /// Gets the underlying speaker embedding extractor.
    /// </summary>
    ISpeakerEmbeddingExtractor<T> EmbeddingExtractor { get; }

    /// <summary>
    /// Gets whether this model is running in ONNX inference mode.
    /// </summary>
    bool IsOnnxMode { get; }

    /// <summary>
    /// Verifies if audio matches a reference speaker embedding.
    /// </summary>
    /// <param name="audio">Audio to verify.</param>
    /// <param name="referenceEmbedding">Pre-computed speaker embedding of the claimed identity.</param>
    /// <param name="threshold">Decision threshold. Uses default if null.</param>
    /// <returns>Verification result with decision and confidence score.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This checks if the voice in the audio matches a known voiceprint.
    /// - true = Same person (accept)
    /// - false = Different person (reject)
    /// </para>
    /// </remarks>
    SpeakerVerificationResult<T> Verify(Tensor<T> audio, Tensor<T> referenceEmbedding);

    /// <summary>
    /// Verifies if audio matches a reference speaker embedding with custom threshold.
    /// </summary>
    /// <param name="audio">Audio to verify.</param>
    /// <param name="referenceEmbedding">Pre-computed speaker embedding of the claimed identity.</param>
    /// <param name="threshold">Custom decision threshold.</param>
    /// <returns>Verification result with decision and confidence score.</returns>
    SpeakerVerificationResult<T> Verify(Tensor<T> audio, Tensor<T> referenceEmbedding, T threshold);

    /// <summary>
    /// Verifies if audio matches reference audio of a claimed speaker.
    /// </summary>
    /// <param name="audio">Audio to verify.</param>
    /// <param name="referenceAudio">Reference audio of the claimed identity.</param>
    /// <returns>Verification result with decision and confidence score.</returns>
    SpeakerVerificationResult<T> VerifyWithReferenceAudio(Tensor<T> audio, Tensor<T> referenceAudio);

    /// <summary>
    /// Verifies if audio matches a reference speaker embedding asynchronously.
    /// </summary>
    /// <param name="audio">Audio to verify.</param>
    /// <param name="referenceEmbedding">Pre-computed speaker embedding of the claimed identity.</param>
    /// <param name="cancellationToken">Cancellation token for async operation.</param>
    /// <returns>Verification result with decision and confidence score.</returns>
    Task<SpeakerVerificationResult<T>> VerifyAsync(
        Tensor<T> audio,
        Tensor<T> referenceEmbedding,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Enrolls a speaker by creating a reference embedding from audio samples.
    /// </summary>
    /// <param name="enrollmentAudio">Audio samples from the speaker.</param>
    /// <returns>Enrolled speaker profile containing the aggregated embedding.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Enrollment is like setting up a new voice password.
    /// The more audio you provide, the better the system can recognize the speaker.
    /// </para>
    /// </remarks>
    SpeakerProfile<T> Enroll(IReadOnlyList<Tensor<T>> enrollmentAudio);

    /// <summary>
    /// Enrolls a speaker by creating a reference embedding from a single audio sample.
    /// </summary>
    /// <param name="enrollmentAudio">Single audio sample from the speaker.</param>
    /// <returns>Enrolled speaker profile.</returns>
    SpeakerProfile<T> Enroll(Tensor<T> enrollmentAudio);

    /// <summary>
    /// Updates an existing speaker profile with additional audio.
    /// </summary>
    /// <param name="existingProfile">The existing speaker profile.</param>
    /// <param name="newAudio">New audio to incorporate.</param>
    /// <returns>Updated speaker profile.</returns>
    SpeakerProfile<T> UpdateProfile(SpeakerProfile<T> existingProfile, Tensor<T> newAudio);

    /// <summary>
    /// Computes the verification score between audio and a reference.
    /// </summary>
    /// <param name="audio">Audio to verify.</param>
    /// <param name="referenceEmbedding">Reference speaker embedding.</param>
    /// <returns>Verification score (higher = more likely same speaker).</returns>
    T ComputeScore(Tensor<T> audio, Tensor<T> referenceEmbedding);

    /// <summary>
    /// Gets the recommended threshold for a target false accept rate.
    /// </summary>
    /// <param name="targetFAR">Target false accept rate (e.g., 0.01 for 1%).</param>
    /// <returns>Recommended threshold value.</returns>
    T GetThresholdForFAR(double targetFAR);
}

/// <summary>
/// Result of a speaker verification attempt.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class SpeakerVerificationResult<T>
{
    /// <summary>
    /// Gets or sets whether the verification was accepted.
    /// </summary>
    public bool IsAccepted { get; set; }

    /// <summary>
    /// Gets or sets the verification score.
    /// </summary>
    public T Score { get; set; } = default!;

    /// <summary>
    /// Gets or sets the threshold used for the decision.
    /// </summary>
    public T Threshold { get; set; } = default!;

    /// <summary>
    /// Gets or sets the confidence level of the decision.
    /// </summary>
    public T Confidence { get; set; } = default!;
}

/// <summary>
/// Represents an enrolled speaker profile.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class SpeakerProfile<T>
{
    /// <summary>
    /// Gets or sets the unique identifier for this speaker.
    /// </summary>
    public string SpeakerId { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the speaker embedding.
    /// </summary>
    public Tensor<T> Embedding { get; set; } = default!;

    /// <summary>
    /// Gets or sets the number of audio samples used to create this profile.
    /// </summary>
    public int NumEnrollmentSamples { get; set; }

    /// <summary>
    /// Gets or sets the total duration of enrollment audio in seconds.
    /// </summary>
    public double TotalEnrollmentDuration { get; set; }

    /// <summary>
    /// Gets or sets when this profile was created.
    /// </summary>
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets when this profile was last updated.
    /// </summary>
    public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;
}
