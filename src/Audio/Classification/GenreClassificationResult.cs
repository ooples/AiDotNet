namespace AiDotNet.Audio.Classification;

/// <summary>
/// Result of genre classification.
/// </summary>
public class GenreClassificationResult
{
    /// <summary>Most likely genre.</summary>
    public required string PredictedGenre { get; init; }

    /// <summary>Confidence score for predicted genre (0-1).</summary>
    public double Confidence { get; init; }

    /// <summary>Probabilities for all genres.</summary>
    public required Dictionary<string, double> AllProbabilities { get; init; }

    /// <summary>Top K predictions with probabilities.</summary>
    public required List<(string Genre, double Probability)> TopPredictions { get; init; }

    /// <summary>Extracted features used for classification.</summary>
    public required GenreFeatures Features { get; init; }
}
