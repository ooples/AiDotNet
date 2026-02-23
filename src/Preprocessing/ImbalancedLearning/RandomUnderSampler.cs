using AiDotNet.LinearAlgebra;

namespace AiDotNet.Preprocessing.ImbalancedLearning;

/// <summary>
/// Implements random undersampling for handling imbalanced datasets.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Random undersampling randomly removes samples from the majority class until
/// the desired balance is achieved. It's the simplest undersampling method.
/// </para>
/// <para>
/// <b>For Beginners:</b> If you have 1000 "normal" samples and 100 "fraud" samples,
/// random undersampling might randomly select 100-200 of the "normal" samples to keep,
/// discarding the rest.
///
/// Advantages:
/// - Very simple and fast
/// - Good baseline to compare against
/// - Works well with large datasets where losing data isn't critical
///
/// Disadvantages:
/// - Randomly discards potentially useful information
/// - May remove important samples near decision boundary
/// - Results can vary significantly based on random selection
///
/// When to use:
/// - Large datasets where you can afford to lose data
/// - As a quick baseline before trying more sophisticated methods
/// - When you need fast results and perfect balance
///
/// References:
/// - Kotsiantis et al. (2006). "Handling imbalanced datasets: A review"
/// </para>
/// </remarks>
public class RandomUnderSampler<T> : UndersamplingBase<T>
{
    /// <summary>
    /// Whether to replace samples when selecting (sampling with replacement).
    /// </summary>
    private readonly bool _replacement;

    /// <summary>
    /// Gets the name of this undersampling strategy.
    /// </summary>
    public override string Name => "RandomUnderSampler";

    /// <summary>
    /// Initializes a new instance of the RandomUnderSampler class.
    /// </summary>
    /// <param name="samplingStrategy">Target ratio of minority to majority (1.0 for balanced). Default is 1.0.</param>
    /// <param name="replacement">Whether to sample with replacement. Default is false.</param>
    /// <param name="seed">Random seed for reproducibility.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Example usage:
    ///
    /// <code>
    /// // Default: balance classes by randomly removing majority samples
    /// var undersampler = new RandomUnderSampler&lt;double&gt;();
    ///
    /// // Custom: keep majority at 2x minority size
    /// var undersampler = new RandomUnderSampler&lt;double&gt;(samplingStrategy: 0.5);
    ///
    /// // Apply to your data
    /// var (newX, newY) = undersampler.Resample(trainX, trainY);
    /// </code>
    ///
    /// The replacement parameter:
    /// - false (default): Each sample can only be selected once
    /// - true: Same sample can be selected multiple times (rare use case)
    /// </para>
    /// </remarks>
    public RandomUnderSampler(double samplingStrategy = 1.0, bool replacement = false, int? seed = null)
        : base(samplingStrategy, seed)
    {
        _replacement = replacement;
    }

    /// <summary>
    /// Selects which majority samples to keep using random selection.
    /// </summary>
    /// <param name="x">The full feature matrix.</param>
    /// <param name="y">The class labels.</param>
    /// <param name="majorityIndices">Indices of majority class samples.</param>
    /// <param name="minorityIndices">Indices of minority class samples.</param>
    /// <param name="targetCount">Number of majority samples to keep.</param>
    /// <returns>Indices of majority samples to keep.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method simply shuffles the majority indices
    /// and takes the first targetCount samples. It's like drawing names from a hat.
    /// </para>
    /// </remarks>
    protected override List<int> SelectSamplesToKeep(
        Matrix<T> x,
        Vector<T> y,
        List<int> majorityIndices,
        List<int> minorityIndices,
        int targetCount)
    {
        if (_replacement)
        {
            // Sample with replacement - same index can appear multiple times
            var selected = new List<int>();
            for (int i = 0; i < targetCount; i++)
            {
                selected.Add(majorityIndices[Random.Next(majorityIndices.Count)]);
            }
            return selected;
        }
        else
        {
            // Sample without replacement - shuffle and take first n
            var shuffled = majorityIndices.OrderBy(_ => Random.Next()).ToList();
            return shuffled.Take(targetCount).ToList();
        }
    }
}
