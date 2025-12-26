namespace AiDotNet.Augmentation;

/// <summary>
/// Specifies how to combine predictions from multiple augmented versions of the same input.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> When you make predictions on several variations of an image
/// (flipped, rotated, etc.), you need a way to combine those predictions into one final answer.
/// This enum controls how that combination happens.
///
/// Think of it like asking 5 friends to estimate the price of a used car:
/// - <b>Mean:</b> Take the average of all estimates ($15K + $18K + $16K + $14K + $17K) / 5 = $16K
/// - <b>Median:</b> Take the middle value when sorted ($14K, $15K, [$16K], $17K, $18K) = $16K
/// - <b>Vote:</b> If 3 friends say "buy" and 2 say "don't buy", go with "buy"
/// </para>
/// </remarks>
public enum PredictionAggregationMethod
{
    /// <summary>
    /// Average all predictions together. Best for regression and probability outputs.
    /// </summary>
    /// <remarks>
    /// This is the most common choice. Works well for:
    /// - Predicting prices, temperatures, or any continuous value
    /// - Classification confidence scores (e.g., "80% confident this is a cat")
    /// </remarks>
    Mean,

    /// <summary>
    /// Take the middle prediction when sorted. More robust when some predictions are outliers.
    /// </summary>
    /// <remarks>
    /// Use this when you suspect some augmentations might give bad predictions.
    /// For example, if 4 predictions are around 100 but one is 1000, median ignores the outlier.
    /// </remarks>
    Median,

    /// <summary>
    /// Take the highest prediction. Useful for object detection confidence scores.
    /// </summary>
    /// <remarks>
    /// Use this when you want to capture the "best case" prediction.
    /// Common in object detection where you want the most confident bounding box.
    /// </remarks>
    Max,

    /// <summary>
    /// Take the lowest prediction. Useful for conservative estimates.
    /// </summary>
    /// <remarks>
    /// Use this when you want the most conservative prediction.
    /// For example, when estimating costs where you'd rather underestimate.
    /// </remarks>
    Min,

    /// <summary>
    /// Count votes from each prediction and pick the winner. Best for classification.
    /// </summary>
    /// <remarks>
    /// Use this for classification tasks. Each augmented prediction "votes" for a class,
    /// and the class with the most votes wins. Like a democratic election for predictions.
    /// </remarks>
    Vote,

    /// <summary>
    /// Like Mean, but gives more weight to higher-confidence predictions.
    /// </summary>
    /// <remarks>
    /// Use this when some augmentations produce more reliable predictions than others.
    /// Predictions with higher confidence scores contribute more to the final answer.
    /// </remarks>
    WeightedMean,

    /// <summary>
    /// Multiply predictions together (then take the Nth root). Best for probability products.
    /// </summary>
    /// <remarks>
    /// Use this when predictions represent independent probabilities that should be combined.
    /// Less common but useful in specialized scenarios like combining class probabilities.
    /// </remarks>
    GeometricMean
}
