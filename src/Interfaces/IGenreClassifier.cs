namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for music genre classification models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Genre classification automatically categorizes music into genres like rock, jazz, classical,
/// hip-hop, etc. This is useful for music organization, recommendation, and discovery.
/// </para>
/// <para>
/// <b>For Beginners:</b> Genre classification is like having an expert listener label
/// what type of music a song is.
///
/// How it works:
/// 1. Audio features are extracted (mel-spectrograms, MFCCs, etc.)
/// 2. A neural network analyzes these features
/// 3. The network outputs probabilities for each genre
///
/// Common genres:
/// - Rock, Pop, Hip-Hop, R&B
/// - Jazz, Blues, Classical
/// - Electronic, EDM, House, Techno
/// - Country, Folk, Reggae
///
/// Use cases:
/// - Music library organization
/// - Streaming service recommendations
/// - Radio station automation
/// - Music analysis and research
///
/// Challenges:
/// - Genre boundaries are fuzzy (many songs blend genres)
/// - Genre definitions vary between cultures and time periods
/// - Sub-genres vs main genres (is "death metal" a genre or sub-genre?)
/// </para>
/// <para>
/// This interface extends <see cref="IFullModel{T, TInput, TOutput}"/> for Tensor-based audio processing.
/// </para>
/// </remarks>
public interface IGenreClassifier<T> : IFullModel<T, Tensor<T>, Tensor<T>>
{
    /// <summary>
    /// Gets the expected sample rate for input audio.
    /// </summary>
    int SampleRate { get; }

    /// <summary>
    /// Gets the list of genres this model can classify.
    /// </summary>
    IReadOnlyList<string> SupportedGenres { get; }

    /// <summary>
    /// Gets whether this model supports multi-label classification.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Multi-label means a song can belong to multiple genres
    /// at once (e.g., both "rock" and "electronic"). Single-label forces one choice.
    /// </para>
    /// </remarks>
    bool SupportsMultiLabel { get; }

    /// <summary>
    /// Gets whether this model is running in ONNX inference mode.
    /// </summary>
    bool IsOnnxMode { get; }

    /// <summary>
    /// Classifies the genre of audio.
    /// </summary>
    /// <param name="audio">Audio waveform tensor [samples] or [channels, samples].</param>
    /// <returns>Genre classification result.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the main method for classifying genre.
    /// - Pass in audio of a song
    /// - Get back the most likely genre(s)
    /// </para>
    /// </remarks>
    GenreClassificationResult<T> Classify(Tensor<T> audio);

    /// <summary>
    /// Classifies genre asynchronously.
    /// </summary>
    /// <param name="audio">Audio waveform tensor.</param>
    /// <param name="cancellationToken">Cancellation token for async operation.</param>
    /// <returns>Genre classification result.</returns>
    Task<GenreClassificationResult<T>> ClassifyAsync(Tensor<T> audio, CancellationToken cancellationToken = default);

    /// <summary>
    /// Gets genre probabilities for all supported genres.
    /// </summary>
    /// <param name="audio">Audio waveform tensor.</param>
    /// <returns>Dictionary mapping genre names to probability scores.</returns>
    IReadOnlyDictionary<string, T> GetGenreProbabilities(Tensor<T> audio);

    /// <summary>
    /// Gets top-K genre predictions.
    /// </summary>
    /// <param name="audio">Audio waveform tensor.</param>
    /// <param name="k">Number of top genres to return.</param>
    /// <returns>List of top genre predictions with probabilities.</returns>
    IReadOnlyList<GenrePrediction<T>> GetTopGenres(Tensor<T> audio, int k = 5);

    /// <summary>
    /// Tracks genre over time within a piece.
    /// </summary>
    /// <param name="audio">Audio waveform tensor.</param>
    /// <param name="segmentDuration">Duration of each analysis segment in seconds.</param>
    /// <returns>Genre tracking result showing genre over time.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Some songs change style during playback. This tracks
    /// the dominant genre at different points in the song.
    /// </para>
    /// </remarks>
    GenreTrackingResult<T> TrackGenreOverTime(Tensor<T> audio, double segmentDuration = 10.0);

    /// <summary>
    /// Extracts audio features used for classification.
    /// </summary>
    /// <param name="audio">Audio waveform tensor.</param>
    /// <returns>Feature tensor used by the classifier.</returns>
    Tensor<T> ExtractFeatures(Tensor<T> audio);
}

/// <summary>
/// Result of genre classification.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class GenreClassificationResult<T>
{
    /// <summary>
    /// Gets or sets the predicted genre (or primary genre if multi-label).
    /// </summary>
    public string PredictedGenre { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the confidence for the predicted genre.
    /// </summary>
    public T Confidence { get; set; } = default!;

    /// <summary>
    /// Gets or sets all predicted genres (for multi-label classification).
    /// </summary>
    public IReadOnlyList<GenrePrediction<T>> AllGenres { get; set; } = Array.Empty<GenrePrediction<T>>();

    /// <summary>
    /// Gets or sets whether this is a multi-label result.
    /// </summary>
    public bool IsMultiLabel { get; set; }

    /// <summary>
    /// Gets all probabilities as a dictionary (legacy API compatibility).
    /// </summary>
    public IReadOnlyDictionary<string, T> AllProbabilities =>
        AllGenres.ToDictionary(g => g.Genre, g => g.Probability);

    /// <summary>
    /// Gets top predictions as a list of tuples (legacy API compatibility).
    /// </summary>
    public IReadOnlyList<(string Genre, T Probability)> TopPredictions =>
        AllGenres.Take(5).Select(g => (g.Genre, g.Probability)).ToList();

    /// <summary>
    /// Gets or sets extracted features used for classification (legacy API compatibility).
    /// </summary>
    public GenreFeatures<T>? Features { get; set; }
}

/// <summary>
/// Features extracted for genre classification (generic version).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class GenreFeatures<T>
{
    /// <summary>
    /// Gets or sets the mean MFCC coefficients.
    /// </summary>
    public T[] MfccMean { get; set; } = Array.Empty<T>();

    /// <summary>
    /// Gets or sets the standard deviation of MFCC coefficients.
    /// </summary>
    public T[] MfccStd { get; set; } = Array.Empty<T>();

    /// <summary>
    /// Gets or sets the estimated tempo in BPM.
    /// </summary>
    public T Tempo { get; set; } = default!;
}

/// <summary>
/// A single genre prediction with confidence.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class GenrePrediction<T>
{
    /// <summary>
    /// Gets or sets the genre name.
    /// </summary>
    public string Genre { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the probability/confidence score.
    /// </summary>
    public T Probability { get; set; } = default!;

    /// <summary>
    /// Gets or sets the rank (1 = most likely).
    /// </summary>
    public int Rank { get; set; }
}

/// <summary>
/// Result of tracking genre over time.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class GenreTrackingResult<T>
{
    /// <summary>
    /// Gets or sets the overall dominant genre.
    /// </summary>
    public string DominantGenre { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets genre segments over time.
    /// </summary>
    public IReadOnlyList<GenreSegment<T>> Segments { get; set; } = Array.Empty<GenreSegment<T>>();

    /// <summary>
    /// Gets or sets whether the genre changes significantly.
    /// </summary>
    public bool HasGenreChanges { get; set; }

    /// <summary>
    /// Gets or sets genre distribution over the entire track.
    /// </summary>
    public IReadOnlyDictionary<string, double> GenreDistribution { get; set; } = new Dictionary<string, double>();
}

/// <summary>
/// A segment with genre information.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class GenreSegment<T>
{
    /// <summary>
    /// Gets or sets the start time in seconds.
    /// </summary>
    public double StartTime { get; set; }

    /// <summary>
    /// Gets or sets the end time in seconds.
    /// </summary>
    public double EndTime { get; set; }

    /// <summary>
    /// Gets or sets the dominant genre in this segment.
    /// </summary>
    public string Genre { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the confidence score.
    /// </summary>
    public T Confidence { get; set; } = default!;
}
