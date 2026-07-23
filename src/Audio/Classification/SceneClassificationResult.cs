namespace AiDotNet.Audio.Classification;

/// <summary>
/// Result of acoustic scene classification.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> SceneClassificationResult provides AI safety functionality. Default values follow the original paper settings.</para>
/// </remarks>
public class SceneClassificationResult
{
    /// <summary>Most likely scene.</summary>
    public required string PredictedScene { get; init; }

    /// <summary>Scene category (indoor, outdoor_urban, outdoor_nature, transportation).</summary>
    public required string Category { get; init; }

    /// <summary>Confidence score for predicted scene (0-1).</summary>
    public double Confidence { get; init; }

    /// <summary>Probabilities for all scenes.</summary>
    public required Dictionary<string, double> AllProbabilities { get; init; }

    /// <summary>Top K predictions with probabilities.</summary>
    public required List<(string Scene, double Probability)> TopPredictions { get; init; }

    /// <summary>Extracted features used for classification.</summary>
    public required SceneFeatures Features { get; init; }
}
