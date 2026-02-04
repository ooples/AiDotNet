using AiDotNet.LinearAlgebra;

namespace AiDotNet.Preprocessing.ImbalancedLearning;

/// <summary>
/// Implements random oversampling for handling imbalanced datasets.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Random oversampling duplicates random samples from the minority class until
/// the desired balance is achieved. It's the simplest oversampling method.
/// </para>
/// <para>
/// <b>For Beginners:</b> If you have 1000 "normal" samples and 100 "fraud" samples,
/// random oversampling randomly duplicates "fraud" samples until you have enough.
///
/// Advantages:
/// - Very simple and fast
/// - Good baseline to compare against
/// - Preserves original data characteristics
///
/// Disadvantages:
/// - Creates exact duplicates (no diversity)
/// - Can lead to overfitting on duplicated samples
/// - Model may "memorize" rather than "learn"
///
/// Comparison with SMOTE:
/// - RandomOverSampler: Duplicates existing samples
/// - SMOTE: Creates new synthetic samples between existing ones
/// - SMOTE usually performs better, but RandomOverSampler is simpler
///
/// When to use:
/// - As a baseline for comparison
/// - When synthetic samples might introduce artifacts
/// - When you have very few minority samples (SMOTE needs at least k+1)
///
/// References:
/// - Kotsiantis et al. (2006). "Handling imbalanced datasets: A review"
/// </para>
/// </remarks>
public class RandomOverSampler<T> : OversamplingBase<T>
{
    /// <summary>
    /// Gets the name of this oversampling strategy.
    /// </summary>
    public override string Name => "RandomOverSampler";

    /// <summary>
    /// Initializes a new instance of the RandomOverSampler class.
    /// </summary>
    /// <param name="samplingStrategy">Target ratio of minority to majority (1.0 for balanced). Default is 1.0.</param>
    /// <param name="seed">Random seed for reproducibility.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Example usage:
    ///
    /// <code>
    /// // Default: balance classes by duplicating minority samples
    /// var oversampler = new RandomOverSampler&lt;double&gt;();
    ///
    /// // Custom: minority at 50% of majority size
    /// var oversampler = new RandomOverSampler&lt;double&gt;(samplingStrategy: 0.5);
    ///
    /// // Apply to your data
    /// var (newX, newY) = oversampler.Resample(trainX, trainY);
    /// </code>
    /// </para>
    /// </remarks>
    public RandomOverSampler(double samplingStrategy = 1.0, int? seed = null)
        : base(samplingStrategy, kNeighbors: 1, seed)  // kNeighbors not used but required by base
    {
    }

    /// <summary>
    /// Generates synthetic samples by duplicating existing minority samples.
    /// </summary>
    /// <param name="x">The full feature matrix.</param>
    /// <param name="classIndices">Indices of samples belonging to the minority class.</param>
    /// <param name="numSamples">Number of samples to generate (duplicates).</param>
    /// <returns>List of duplicated sample vectors.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method simply:
    /// 1. Picks a random minority sample
    /// 2. Creates an exact copy of it
    /// 3. Repeats until we have enough samples
    ///
    /// Unlike SMOTE, these are exact copies, not interpolations.
    /// </para>
    /// </remarks>
    protected override List<Vector<T>> GenerateSyntheticSamples(Matrix<T> x, List<int> classIndices, int numSamples)
    {
        var syntheticSamples = new List<Vector<T>>();

        if (classIndices.Count == 0)
        {
            return syntheticSamples;
        }

        for (int i = 0; i < numSamples; i++)
        {
            // Select a random minority sample
            int idx = classIndices[Random.Next(classIndices.Count)];
            var original = x.GetRow(idx);

            // Create an exact copy
            var copy = new Vector<T>(original.Length);
            for (int j = 0; j < original.Length; j++)
            {
                copy[j] = original[j];
            }

            syntheticSamples.Add(copy);
        }

        return syntheticSamples;
    }
}
