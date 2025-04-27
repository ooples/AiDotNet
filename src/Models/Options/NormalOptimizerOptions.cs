namespace AiDotNet.Models.Options;

/// <summary>
/// Options for the normal optimizer.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> These are the settings that control how the optimizer works,
/// like how many features to consider, how much to adjust parameters, and how these
/// settings can adapt over time.
/// </remarks>
public class NormalOptimizerOptions<T, TInput, TOutput> : OptimizationAlgorithmOptions<T, TInput, TOutput>
{
    // Feature selection settings

    /// <summary>
    /// Gets or sets the absolute maximum number of features that can be used.
    /// </summary>
    public int AbsoluteMaximumFeatures { get; set; } = 100;

    /// <summary>
    /// Gets or sets the probability of applying feature selection in each iteration.
    /// </summary>
    public double FeatureSelectionProbability { get; set; } = 0.7;

    /// <summary>
    /// Gets or sets the minimum allowed value for feature selection probability.
    /// </summary>
    public double MinFeatureSelectionProbability { get; set; } = 0.2;

    /// <summary>
    /// Gets or sets the maximum allowed value for feature selection probability.
    /// </summary>
    public double MaxFeatureSelectionProbability { get; set; } = 0.9;

    // Parameter adjustment settings

    /// <summary>
    /// Gets or sets the scale of parameter adjustments.
    /// </summary>
    public double ParameterAdjustmentScale { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the minimum allowed value for parameter adjustment scale.
    /// </summary>
    public double MinParameterAdjustmentScale { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the maximum allowed value for parameter adjustment scale.
    /// </summary>
    public double MaxParameterAdjustmentScale { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the probability of flipping a parameter's sign.
    /// </summary>
    public double SignFlipProbability { get; set; } = 0.05;

    /// <summary>
    /// Gets or sets the minimum allowed value for sign flip probability.
    /// </summary>
    public double MinSignFlipProbability { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the maximum allowed value for sign flip probability.
    /// </summary>
    public double MaxSignFlipProbability { get; set; } = 0.2;

    /// <summary>
    /// Gets or sets the probability of applying parameter adjustments in each iteration.
    /// </summary>
    public double ParameterAdjustmentProbability { get; set; } = 0.8;

    /// <summary>
    /// Gets or sets the minimum allowed value for parameter adjustment probability.
    /// </summary>
    public double MinParameterAdjustmentProbability { get; set; } = 0.3;

    /// <summary>
    /// Gets or sets the maximum allowed value for parameter adjustment probability.
    /// </summary>
    public double MaxParameterAdjustmentProbability { get; set; } = 0.95;
}