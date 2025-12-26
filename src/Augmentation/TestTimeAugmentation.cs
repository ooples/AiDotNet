namespace AiDotNet.Augmentation;

/// <summary>
/// Contains the result of a Test-Time Augmentation prediction, including individual and aggregated predictions.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This class gives you both:
/// 1. The final combined prediction (what you usually want)
/// 2. All the individual predictions (for debugging or analysis)
///
/// You also get uncertainty information - if all 5 predictions were similar, you can be
/// confident in the result. If they varied wildly, you might want to be more cautious.
/// </para>
/// </remarks>
/// <typeparam name="TOutput">The type of prediction output (e.g., Vector for class probabilities).</typeparam>
public class TestTimeAugmentationResult<TOutput>
{
    /// <summary>
    /// Gets the final combined prediction after aggregating all augmented predictions.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the main result you'll use. It's the combination
    /// of all individual predictions based on the aggregation method (mean, median, vote, etc.).</para>
    /// </remarks>
    public TOutput AggregatedPrediction { get; }

    /// <summary>
    /// Gets all the individual predictions from each augmented version.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This shows what the model predicted for each variation.
    /// Useful for:
    /// - Debugging: See if one augmentation is causing problems
    /// - Analysis: Understand how much predictions vary
    /// - Visualization: Show uncertainty in results
    /// </para>
    /// </remarks>
    public IReadOnlyList<TOutput> IndividualPredictions { get; }

    /// <summary>
    /// Gets the confidence score of the aggregated prediction (if available).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> A value between 0 and 1 indicating how confident the
    /// model is in its prediction. Higher = more confident. May be null if the model
    /// doesn't provide confidence scores.</para>
    /// </remarks>
    public double? Confidence { get; }

    /// <summary>
    /// Gets the standard deviation of predictions, measuring uncertainty.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This tells you how much the predictions varied:
    /// - Low standard deviation: Predictions were consistent (good!)
    /// - High standard deviation: Predictions varied a lot (less reliable)
    ///
    /// Use this to decide how much to trust the result.
    /// </para>
    /// </remarks>
    public double? StandardDeviation { get; }

    /// <summary>
    /// Creates a new Test-Time Augmentation result.
    /// </summary>
    /// <param name="aggregated">The combined prediction.</param>
    /// <param name="individual">All individual predictions.</param>
    /// <param name="confidence">Optional confidence score.</param>
    /// <param name="standardDeviation">Optional standard deviation of predictions.</param>
    public TestTimeAugmentationResult(
        TOutput aggregated,
        IReadOnlyList<TOutput> individual,
        double? confidence = null,
        double? standardDeviation = null)
    {
        AggregatedPrediction = aggregated;
        IndividualPredictions = individual;
        Confidence = confidence;
        StandardDeviation = standardDeviation;
    }
}
