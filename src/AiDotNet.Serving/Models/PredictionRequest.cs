namespace AiDotNet.Serving.Models;

/// <summary>
/// Represents a prediction request containing input features.
/// </summary>
public class PredictionRequest
{
    /// <summary>
    /// Gets or sets the input features for prediction.
    /// Can be a single array for one prediction or a 2D array for batch predictions.
    /// </summary>
    public double[][] Features { get; set; } = Array.Empty<double[]>();

    /// <summary>
    /// Gets or sets an optional request ID for tracking purposes.
    /// </summary>
    public string? RequestId { get; set; }
}

