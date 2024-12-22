namespace AiDotNet.Models;

public class ShapleyValueFitDetectorOptions
{
    /// <summary>
    /// The threshold for cumulative importance to determine significant features.
    /// </summary>
    public double ImportanceThreshold { get; set; } = 0.8;

    /// <summary>
    /// The number of Monte Carlo samples to use when calculating Shapley values.
    /// </summary>
    public int MonteCarloSamples { get; set; } = 1000;

    /// <summary>
    /// The threshold for the ratio of important features to total features,
    /// below which the model is considered to be overfitting.
    /// </summary>
    public double OverfitThreshold { get; set; } = 0.2;

    /// <summary>
    /// The threshold for the ratio of important features to total features,
    /// above which the model is considered to be underfitting.
    /// </summary>
    public double UnderfitThreshold { get; set; } = 0.8;
}