namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for speaker diarization models that segment audio by speaker ("who spoke when").
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Speaker diarization partitions an audio stream into segments based on speaker identity.
/// It answers the question "Who spoke when?" without necessarily knowing who the speakers
/// are (unlike speaker identification which requires enrolled speakers).
/// </para>
/// <para>
/// <b>For Beginners:</b> Diarization is like labeling a transcript with "Speaker A said..."
/// "Speaker B said..." without knowing their names.
///
/// How it works:
/// 1. Audio is segmented into small chunks
/// 2. Speaker embeddings are extracted for each chunk
/// 3. Clustering groups similar embeddings together
/// 4. Each cluster represents a unique speaker
/// 5. Output: Timeline showing when each speaker talks
///
/// Common use cases:
/// - Meeting transcription (separating participants)
/// - Podcast/interview processing
/// - Call center analytics
/// - Medical dictation
///
/// Challenges:
/// - Overlapping speech (multiple people talking at once)
/// - Short turns (quick back-and-forth conversation)
/// - Similar voices (e.g., siblings)
/// - Background noise and music
/// </para>
/// <para>
/// This interface extends <see cref="IFullModel{T, TInput, TOutput}"/> for Tensor-based audio processing.
/// </para>
/// </remarks>
public interface ISpeakerDiarizer<T> : IFullModel<T, Tensor<T>, Tensor<T>>
{
    /// <summary>
    /// Gets the expected sample rate for input audio.
    /// </summary>
    int SampleRate { get; }

    /// <summary>
    /// Gets the minimum segment duration in seconds.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Segments shorter than this may not contain enough speech for reliable
    /// speaker assignment.
    /// </para>
    /// </remarks>
    double MinSegmentDuration { get; }

    /// <summary>
    /// Gets whether this model can detect overlapping speech.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Overlapping speech is when two or more people
    /// talk at the same time. Not all diarization systems can handle this.
    /// </para>
    /// </remarks>
    bool SupportsOverlapDetection { get; }

    /// <summary>
    /// Gets whether this model is running in ONNX inference mode.
    /// </summary>
    bool IsOnnxMode { get; }

    /// <summary>
    /// Performs speaker diarization on audio.
    /// </summary>
    /// <param name="audio">Audio waveform tensor [samples].</param>
    /// <param name="numSpeakers">Expected number of speakers. Auto-detected if null.</param>
    /// <param name="minSpeakers">Minimum number of speakers (for auto-detection).</param>
    /// <param name="maxSpeakers">Maximum number of speakers (for auto-detection).</param>
    /// <returns>Diarization result with speaker segments.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the main method for finding who spoke when.
    /// - Pass in audio of a conversation
    /// - Get back a timeline of speaker turns
    /// - Speakers are labeled as "Speaker_0", "Speaker_1", etc.
    /// </para>
    /// </remarks>
    DiarizationResult<T> Diarize(
        Tensor<T> audio,
        int? numSpeakers = null,
        int minSpeakers = 1,
        int maxSpeakers = 10);

    /// <summary>
    /// Performs speaker diarization asynchronously.
    /// </summary>
    /// <param name="audio">Audio waveform tensor [samples].</param>
    /// <param name="numSpeakers">Expected number of speakers. Auto-detected if null.</param>
    /// <param name="minSpeakers">Minimum number of speakers (for auto-detection).</param>
    /// <param name="maxSpeakers">Maximum number of speakers (for auto-detection).</param>
    /// <param name="cancellationToken">Cancellation token for async operation.</param>
    /// <returns>Diarization result with speaker segments.</returns>
    Task<DiarizationResult<T>> DiarizeAsync(
        Tensor<T> audio,
        int? numSpeakers = null,
        int minSpeakers = 1,
        int maxSpeakers = 10,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Performs diarization with known speaker profiles.
    /// </summary>
    /// <param name="audio">Audio waveform tensor [samples].</param>
    /// <param name="knownSpeakers">Known speaker profiles to match against.</param>
    /// <param name="allowUnknownSpeakers">Whether to create new labels for unknown speakers.</param>
    /// <returns>Diarization result with identified speaker segments.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> If you know who might be speaking, you can provide
    /// their voice profiles and the system will label segments with actual names
    /// instead of generic "Speaker_0" labels.
    /// </para>
    /// </remarks>
    DiarizationResult<T> DiarizeWithKnownSpeakers(
        Tensor<T> audio,
        IReadOnlyList<SpeakerProfile<T>> knownSpeakers,
        bool allowUnknownSpeakers = true);

    /// <summary>
    /// Gets speaker embeddings for each detected speaker.
    /// </summary>
    /// <param name="audio">Audio waveform tensor [samples].</param>
    /// <param name="diarizationResult">Previous diarization result.</param>
    /// <returns>Dictionary mapping speaker labels to their embeddings.</returns>
    IReadOnlyDictionary<string, Tensor<T>> ExtractSpeakerEmbeddings(
        Tensor<T> audio,
        DiarizationResult<T> diarizationResult);

    /// <summary>
    /// Refines diarization result by re-segmenting with different parameters.
    /// </summary>
    /// <param name="audio">Audio waveform tensor [samples].</param>
    /// <param name="previousResult">Previous diarization result to refine.</param>
    /// <param name="mergeThreshold">Threshold for merging similar speakers.</param>
    /// <returns>Refined diarization result.</returns>
    DiarizationResult<T> RefineDiarization(
        Tensor<T> audio,
        DiarizationResult<T> previousResult,
        T mergeThreshold);
}

/// <summary>
/// Result of speaker diarization.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class DiarizationResult<T>
{
    /// <summary>
    /// Gets or sets the detected speaker segments.
    /// </summary>
    public IReadOnlyList<SpeakerSegment<T>> Segments { get; set; } = Array.Empty<SpeakerSegment<T>>();

    /// <summary>
    /// Gets or sets the number of unique speakers detected.
    /// </summary>
    public int NumSpeakers { get; set; }

    /// <summary>
    /// Gets or sets the unique speaker labels.
    /// </summary>
    public IReadOnlyList<string> SpeakerLabels { get; set; } = Array.Empty<string>();

    /// <summary>
    /// Gets or sets the total audio duration in seconds.
    /// </summary>
    public double TotalDuration { get; set; }

    /// <summary>
    /// Gets or sets overlapping speech regions (if detected).
    /// </summary>
    public IReadOnlyList<OverlapRegion<T>> OverlapRegions { get; set; } = Array.Empty<OverlapRegion<T>>();

    /// <summary>
    /// Gets speaker statistics (speaking time, number of turns).
    /// </summary>
    public IReadOnlyDictionary<string, SpeakerStatistics<T>> SpeakerStats { get; set; } =
        new Dictionary<string, SpeakerStatistics<T>>();
}

/// <summary>
/// Represents a speaker segment in diarization output.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class SpeakerSegment<T>
{
    /// <summary>
    /// Gets or sets the speaker label (e.g., "Speaker_0" or actual name if known).
    /// </summary>
    public string Speaker { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the start time in seconds.
    /// </summary>
    public double StartTime { get; set; }

    /// <summary>
    /// Gets or sets the end time in seconds.
    /// </summary>
    public double EndTime { get; set; }

    /// <summary>
    /// Gets or sets the confidence score for this segment.
    /// </summary>
    public T Confidence { get; set; } = default!;

    /// <summary>
    /// Gets the duration of this segment in seconds.
    /// </summary>
    public double Duration => EndTime - StartTime;
}

/// <summary>
/// Represents a region where speakers overlap.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class OverlapRegion<T>
{
    /// <summary>
    /// Gets or sets the start time of overlap in seconds.
    /// </summary>
    public double StartTime { get; set; }

    /// <summary>
    /// Gets or sets the end time of overlap in seconds.
    /// </summary>
    public double EndTime { get; set; }

    /// <summary>
    /// Gets or sets the speakers involved in the overlap.
    /// </summary>
    public IReadOnlyList<string> Speakers { get; set; } = Array.Empty<string>();
}

/// <summary>
/// Statistics for a speaker in diarization output.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class SpeakerStatistics<T>
{
    /// <summary>
    /// Gets or sets the total speaking time in seconds.
    /// </summary>
    public double TotalSpeakingTime { get; set; }

    /// <summary>
    /// Gets or sets the number of speaking turns.
    /// </summary>
    public int NumTurns { get; set; }

    /// <summary>
    /// Gets or sets the average turn duration in seconds.
    /// </summary>
    public double AverageTurnDuration { get; set; }

    /// <summary>
    /// Gets or sets the percentage of total audio this speaker occupies.
    /// </summary>
    public double SpeakingPercentage { get; set; }
}
