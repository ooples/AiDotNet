using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// Represents an audio-visual event with temporal boundaries.
/// </summary>
public class AudioVisualEvent
{
    /// <summary>Start time in seconds.</summary>
    public double StartTime { get; set; }

    /// <summary>End time in seconds.</summary>
    public double EndTime { get; set; }

    /// <summary>Event label or description.</summary>
    public string Label { get; set; } = string.Empty;

    /// <summary>Confidence score.</summary>
    public double Confidence { get; set; }

    /// <summary>Whether the event is primarily audio, visual, or both.</summary>
    public string Modality { get; set; } = "both";

    /// <summary>Spatial location in video frame (if applicable).</summary>
    public (int X, int Y, int Width, int Height)? BoundingBox { get; set; }
}

/// <summary>
/// Defines the contract for audio-visual correspondence learning models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Audio-visual correspondence learning focuses on understanding the relationship
/// between what we see and what we hear. This enables tasks like finding the
/// source of a sound in a video, synchronizing audio and video, and understanding
/// audio-visual events.
/// </para>
/// <para><b>For Beginners:</b> Teaching AI to connect sounds with visuals!
///
/// Key capabilities:
/// - Sound source localization: Where in the image is the sound coming from?
/// - Audio-visual synchronization: Are the audio and video in sync?
/// - Cross-modal retrieval: Find images matching sounds and vice versa
/// - Audio-visual scene understanding: What's happening based on both modalities?
///
/// Examples:
/// - A dog barking → The model highlights the dog in the image
/// - Piano music → The model finds images of pianos
/// - Clapping sound → The model locates hands in the video
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("AudioVisualCorrespondenceModel")]
public interface IAudioVisualCorrespondenceModel<T>
{
    /// <summary>
    /// Gets the embedding dimension for audio-visual features.
    /// </summary>
    int EmbeddingDimension { get; }

    /// <summary>
    /// Gets the expected audio sample rate.
    /// </summary>
    int AudioSampleRate { get; }

    /// <summary>
    /// Gets the expected video frame rate.
    /// </summary>
    double VideoFrameRate { get; }

    /// <summary>
    /// Computes audio embedding from waveform.
    /// </summary>
    /// <param name="audioWaveform">Audio waveform tensor.</param>
    /// <param name="sampleRate">Sample rate of the audio.</param>
    /// <returns>Normalized audio embedding.</returns>
    Vector<T> GetAudioEmbedding(Tensor<T> audioWaveform, int sampleRate);

    /// <summary>
    /// Computes visual embedding from video frames.
    /// </summary>
    /// <param name="frames">Sequence of video frames.</param>
    /// <returns>Normalized visual embedding.</returns>
    Vector<T> GetVisualEmbedding(IEnumerable<Tensor<T>> frames);

    /// <summary>
    /// Computes audio-visual correspondence score.
    /// </summary>
    /// <param name="audioWaveform">Audio waveform.</param>
    /// <param name="frames">Video frames.</param>
    /// <returns>Correspondence score (higher = better match).</returns>
    T ComputeCorrespondence(Tensor<T> audioWaveform, IEnumerable<Tensor<T>> frames);

    /// <summary>
    /// Localizes sound sources in video frames.
    /// </summary>
    /// <param name="audioWaveform">Audio waveform.</param>
    /// <param name="frames">Video frames.</param>
    /// <returns>Attention maps showing sound source locations for each frame.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Find where sounds come from in images!
    ///
    /// Returns a "heat map" for each frame showing which regions
    /// are most likely producing the sound we hear.
    /// </para>
    /// </remarks>
    IEnumerable<Tensor<T>> LocalizeSoundSource(
        Tensor<T> audioWaveform,
        IEnumerable<Tensor<T>> frames);

    /// <summary>
    /// Checks audio-visual synchronization.
    /// </summary>
    /// <param name="audioWaveform">Audio waveform.</param>
    /// <param name="frames">Video frames.</param>
    /// <returns>Sync offset in seconds (positive = audio ahead, negative = audio behind) and confidence.</returns>
    (double OffsetSeconds, T Confidence) CheckSynchronization(
        Tensor<T> audioWaveform,
        IEnumerable<Tensor<T>> frames);

    /// <summary>
    /// Retrieves visual content matching audio.
    /// </summary>
    /// <param name="audioWaveform">Query audio.</param>
    /// <param name="visualDatabase">Database of visual embeddings.</param>
    /// <param name="topK">Number of results.</param>
    /// <returns>Indices and scores of matching visuals.</returns>
    IEnumerable<(int Index, T Score)> RetrieveVisualsFromAudio(
        Tensor<T> audioWaveform,
        IEnumerable<Vector<T>> visualDatabase,
        int topK = 10);

    /// <summary>
    /// Retrieves audio content matching visual input.
    /// </summary>
    /// <param name="frames">Query video frames.</param>
    /// <param name="audioDatabase">Database of audio embeddings.</param>
    /// <param name="topK">Number of results.</param>
    /// <returns>Indices and scores of matching audio.</returns>
    IEnumerable<(int Index, T Score)> RetrieveAudioFromVisuals(
        IEnumerable<Tensor<T>> frames,
        IEnumerable<Vector<T>> audioDatabase,
        int topK = 10);

    /// <summary>
    /// Separates audio sources based on visual guidance.
    /// </summary>
    /// <param name="mixedAudio">Mixed audio waveform.</param>
    /// <param name="targetVisual">Visual of the target sound source.</param>
    /// <returns>Separated audio for the target source.</returns>
    /// <remarks>
    /// <para>
    /// Uses visual information to guide audio source separation.
    /// For example, given a video of two people talking and pointing
    /// at one person, extracts just that person's voice.
    /// </para>
    /// </remarks>
    Tensor<T> SeparateAudioByVisual(
        Tensor<T> mixedAudio,
        Tensor<T> targetVisual);

    /// <summary>
    /// Generates audio description from visual content.
    /// </summary>
    /// <param name="frames">Video frames.</param>
    /// <returns>Description of expected sounds.</returns>
    string DescribeExpectedAudio(IEnumerable<Tensor<T>> frames);

    /// <summary>
    /// Classifies audio-visual scenes.
    /// </summary>
    /// <param name="audioWaveform">Audio waveform.</param>
    /// <param name="frames">Video frames.</param>
    /// <param name="sceneLabels">Possible scene labels.</param>
    /// <returns>Classification probabilities.</returns>
    Dictionary<string, T> ClassifyScene(
        Tensor<T> audioWaveform,
        IEnumerable<Tensor<T>> frames,
        IEnumerable<string> sceneLabels);

    /// <summary>
    /// Learns correspondence from paired audio-visual data.
    /// </summary>
    /// <param name="audioSamples">Audio samples.</param>
    /// <param name="visualSamples">Corresponding visual samples.</param>
    /// <param name="epochs">Training epochs.</param>
    void LearnCorrespondence(
        IEnumerable<Tensor<T>> audioSamples,
        IEnumerable<IEnumerable<Tensor<T>>> visualSamples,
        int epochs = 10);
}

/// <summary>
/// Defines the contract for audio-visual event localization models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Audio-visual event localization identifies WHEN and WHERE events occur
/// in video by jointly analyzing audio and visual streams. This goes beyond
/// simple detection to provide precise temporal boundaries and spatial locations.
/// </para>
/// <para><b>For Beginners:</b> Finding events in videos using sight AND sound!
///
/// Key capabilities:
/// - Temporal localization: When does the dog bark? (2.3s - 4.1s)
/// - Spatial localization: Where is the barking dog? (bounding box)
/// - Event classification: What kind of event is it? (animal sound)
/// - Multi-event detection: Find all events in a video
///
/// Use cases:
/// - Video surveillance: Detect glass breaking sounds and locate the window
/// - Sports analysis: Find and timestamp all goals using crowd cheering
/// - Content moderation: Detect and locate inappropriate audio-visual content
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("AudioVisualEventLocalizationModel")]
public interface IAudioVisualEventLocalizationModel<T>
{
    /// <summary>
    /// Gets the temporal resolution in seconds.
    /// </summary>
    double TemporalResolution { get; }

    /// <summary>
    /// Gets the supported event categories.
    /// </summary>
    IReadOnlyList<string> SupportedEventCategories { get; }

    /// <summary>
    /// Detects and localizes all audio-visual events in a video.
    /// </summary>
    /// <param name="audioWaveform">Audio waveform.</param>
    /// <param name="frames">Video frames.</param>
    /// <param name="frameRate">Video frame rate.</param>
    /// <returns>List of detected events with temporal and spatial localization.</returns>
    IEnumerable<AudioVisualEvent> DetectEvents(
        Tensor<T> audioWaveform,
        IEnumerable<Tensor<T>> frames,
        double frameRate);

    /// <summary>
    /// Detects events of specific categories.
    /// </summary>
    /// <param name="audioWaveform">Audio waveform.</param>
    /// <param name="frames">Video frames.</param>
    /// <param name="targetCategories">Categories to detect.</param>
    /// <param name="frameRate">Video frame rate.</param>
    /// <returns>Detected events matching the target categories.</returns>
    IEnumerable<AudioVisualEvent> DetectSpecificEvents(
        Tensor<T> audioWaveform,
        IEnumerable<Tensor<T>> frames,
        IEnumerable<string> targetCategories,
        double frameRate);

    /// <summary>
    /// Localizes a specific event described in text.
    /// </summary>
    /// <param name="audioWaveform">Audio waveform.</param>
    /// <param name="frames">Video frames.</param>
    /// <param name="eventDescription">Text description of the event.</param>
    /// <param name="frameRate">Video frame rate.</param>
    /// <returns>Temporal segments where the event occurs.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Find events using natural language!
    ///
    /// Example: "person laughing" → returns [(5.2s, 7.8s), (15.1s, 16.4s)]
    /// </para>
    /// </remarks>
    IEnumerable<(double StartTime, double EndTime, T Confidence)> LocalizeEventByDescription(
        Tensor<T> audioWaveform,
        IEnumerable<Tensor<T>> frames,
        string eventDescription,
        double frameRate);

    /// <summary>
    /// Generates temporal proposals for potential events.
    /// </summary>
    /// <param name="audioWaveform">Audio waveform.</param>
    /// <param name="frames">Video frames.</param>
    /// <param name="frameRate">Video frame rate.</param>
    /// <returns>Proposed time segments that may contain events.</returns>
    IEnumerable<(double StartTime, double EndTime, T EventnessScore)> GenerateProposals(
        Tensor<T> audioWaveform,
        IEnumerable<Tensor<T>> frames,
        double frameRate);

    /// <summary>
    /// Classifies a pre-segmented event.
    /// </summary>
    /// <param name="audioSegment">Audio segment for the event.</param>
    /// <param name="frameSegment">Video frames for the event.</param>
    /// <param name="candidateLabels">Possible event labels.</param>
    /// <returns>Classification probabilities.</returns>
    Dictionary<string, T> ClassifyEvent(
        Tensor<T> audioSegment,
        IEnumerable<Tensor<T>> frameSegment,
        IEnumerable<string> candidateLabels);

    /// <summary>
    /// Tracks an event across time.
    /// </summary>
    /// <param name="audioWaveform">Full audio waveform.</param>
    /// <param name="frames">All video frames.</param>
    /// <param name="initialEvent">Initial event detection.</param>
    /// <param name="frameRate">Video frame rate.</param>
    /// <returns>Event trajectory with updated temporal and spatial locations.</returns>
    IEnumerable<AudioVisualEvent> TrackEvent(
        Tensor<T> audioWaveform,
        IEnumerable<Tensor<T>> frames,
        AudioVisualEvent initialEvent,
        double frameRate);

    /// <summary>
    /// Detects audio-visual synchronization events (e.g., lip sync).
    /// </summary>
    /// <param name="audioWaveform">Audio waveform.</param>
    /// <param name="frames">Video frames.</param>
    /// <param name="frameRate">Video frame rate.</param>
    /// <returns>Sync events with quality scores.</returns>
    IEnumerable<(double StartTime, double EndTime, T SyncQuality, string Description)> DetectSyncEvents(
        Tensor<T> audioWaveform,
        IEnumerable<Tensor<T>> frames,
        double frameRate);

    /// <summary>
    /// Segments video into coherent audio-visual scenes.
    /// </summary>
    /// <param name="audioWaveform">Audio waveform.</param>
    /// <param name="frames">Video frames.</param>
    /// <param name="frameRate">Video frame rate.</param>
    /// <returns>Scene boundaries with descriptions.</returns>
    IEnumerable<(double StartTime, double EndTime, string SceneDescription)> SegmentScenes(
        Tensor<T> audioWaveform,
        IEnumerable<Tensor<T>> frames,
        double frameRate);

    /// <summary>
    /// Generates dense event captions for the entire video.
    /// </summary>
    /// <param name="audioWaveform">Audio waveform.</param>
    /// <param name="frames">Video frames.</param>
    /// <param name="frameRate">Video frame rate.</param>
    /// <returns>Time-stamped captions describing events.</returns>
    IEnumerable<(double Time, string Caption)> GenerateDenseCaptions(
        Tensor<T> audioWaveform,
        IEnumerable<Tensor<T>> frames,
        double frameRate);

    /// <summary>
    /// Answers questions about events in the video.
    /// </summary>
    /// <param name="audioWaveform">Audio waveform.</param>
    /// <param name="frames">Video frames.</param>
    /// <param name="question">Question about events.</param>
    /// <param name="frameRate">Video frame rate.</param>
    /// <returns>Answer with supporting temporal evidence.</returns>
    (string Answer, IEnumerable<(double StartTime, double EndTime)> Evidence) AnswerEventQuestion(
        Tensor<T> audioWaveform,
        IEnumerable<Tensor<T>> frames,
        string question,
        double frameRate);

    /// <summary>
    /// Detects anomalous events that don't match expected patterns.
    /// </summary>
    /// <param name="audioWaveform">Audio waveform.</param>
    /// <param name="frames">Video frames.</param>
    /// <param name="frameRate">Video frame rate.</param>
    /// <returns>Detected anomalies with anomaly scores.</returns>
    IEnumerable<(double StartTime, double EndTime, T AnomalyScore, string Description)> DetectAnomalies(
        Tensor<T> audioWaveform,
        IEnumerable<Tensor<T>> frames,
        double frameRate);

    /// <summary>
    /// Computes event-level audio-visual attention.
    /// </summary>
    /// <param name="audioSegment">Audio segment.</param>
    /// <param name="frameSegment">Video frame segment.</param>
    /// <returns>Cross-modal attention weights.</returns>
    (Tensor<T> AudioToVisualAttention, Tensor<T> VisualToAudioAttention) ComputeEventAttention(
        Tensor<T> audioSegment,
        IEnumerable<Tensor<T>> frameSegment);
}
