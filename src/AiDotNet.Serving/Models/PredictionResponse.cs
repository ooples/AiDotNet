namespace AiDotNet.Serving.Models;

/// <summary>
/// Represents a prediction response containing the model's predictions.
/// </summary>
public class PredictionResponse
{
    /// <summary>
    /// Gets or sets the predictions.
    /// Each array represents the output for one input sample.
    /// </summary>
    public double[][] Predictions { get; set; } = Array.Empty<double[]>();

    /// <summary>
    /// Gets or sets the request ID that was provided in the request.
    /// </summary>
    public string? RequestId { get; set; }

    /// <summary>
    /// Gets or sets the time taken to process the request in milliseconds.
    /// </summary>
    public long ProcessingTimeMs { get; set; }

    /// <summary>
    /// Gets or sets the batch size used for this prediction.
    /// </summary>
    public int BatchSize { get; set; }
}

