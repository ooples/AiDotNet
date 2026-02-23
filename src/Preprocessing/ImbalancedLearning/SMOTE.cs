using AiDotNet.LinearAlgebra;

namespace AiDotNet.Preprocessing.ImbalancedLearning;

/// <summary>
/// Implements SMOTE (Synthetic Minority Over-sampling Technique) for handling imbalanced datasets.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// SMOTE creates synthetic minority class samples by interpolating between existing minority samples
/// and their nearest neighbors. It was introduced by Chawla et al. in 2002 and is one of the most
/// widely used techniques for handling imbalanced data.
/// </para>
/// <para>
/// <b>For Beginners:</b> SMOTE works by creating "fake" minority class samples that are similar to
/// real ones. Here's how it works:
///
/// 1. Pick a minority class sample
/// 2. Find its k nearest neighbors (other minority samples that are similar)
/// 3. Randomly select one of these neighbors
/// 4. Create a new sample somewhere on the line between the original and the neighbor
///
/// Imagine you have two fraud examples at positions [1, 2] and [3, 4].
/// SMOTE might create a new sample at [2, 3] - right in the middle!
/// Or at [1.5, 2.5] - one quarter of the way between them.
///
/// This is better than just duplicating existing samples because:
/// - It creates diverse samples that help the model generalize
/// - The synthetic samples are realistic (they're combinations of real ones)
/// - It fills in the feature space around minority samples
///
/// When to use SMOTE:
/// - Binary or multi-class classification with imbalanced data
/// - When you have enough minority samples to find meaningful neighbors (at least k+1)
/// - When features are numeric (SMOTE doesn't work well with categorical features)
///
/// References:
/// - Chawla et al. (2002). "SMOTE: Synthetic Minority Over-sampling Technique"
/// </para>
/// </remarks>
public class SMOTE<T> : OversamplingBase<T>
{
    /// <summary>
    /// Gets the name of this oversampling strategy.
    /// </summary>
    public override string Name => "SMOTE";

    /// <summary>
    /// Initializes a new instance of the SMOTE class.
    /// </summary>
    /// <param name="samplingStrategy">Target ratio of minority to majority (1.0 for balanced). Default is 1.0.</param>
    /// <param name="kNeighbors">Number of nearest neighbors for synthesis. Default is 5.</param>
    /// <param name="seed">Random seed for reproducibility. If null, uses cryptographically secure random.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Example usage:
    ///
    /// <code>
    /// // Default: balance classes using 5 nearest neighbors
    /// var smote = new SMOTE&lt;double&gt;();
    ///
    /// // Custom: partial balance (minority = 50% of majority), 3 neighbors
    /// var smote = new SMOTE&lt;double&gt;(samplingStrategy: 0.5, kNeighbors: 3);
    ///
    /// // With reproducibility
    /// var smote = new SMOTE&lt;double&gt;(seed: 42);
    ///
    /// // Apply to your data
    /// var (newX, newY) = smote.Resample(trainX, trainY);
    /// </code>
    /// </para>
    /// </remarks>
    public SMOTE(double samplingStrategy = 1.0, int kNeighbors = 5, int? seed = null)
        : base(samplingStrategy, kNeighbors, seed)
    {
    }

    /// <summary>
    /// Generates synthetic samples using SMOTE interpolation.
    /// </summary>
    /// <param name="x">The full feature matrix.</param>
    /// <param name="classIndices">Indices of samples belonging to the minority class.</param>
    /// <param name="numSamples">Number of synthetic samples to generate.</param>
    /// <returns>List of synthetic sample vectors.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> For each synthetic sample:
    /// 1. Pick a random minority sample as the "base"
    /// 2. Find its k nearest neighbors (other minority samples)
    /// 3. Pick one neighbor randomly
    /// 4. Generate a point on the line segment between base and neighbor
    ///
    /// The interpolation formula is:
    /// synthetic = base + random(0,1) × (neighbor - base)
    ///
    /// This creates samples anywhere along the line between two minority samples.
    /// </para>
    /// </remarks>
    protected override List<Vector<T>> GenerateSyntheticSamples(Matrix<T> x, List<int> classIndices, int numSamples)
    {
        var syntheticSamples = new List<Vector<T>>();

        if (classIndices.Count < 2)
        {
            // Not enough samples to interpolate
            return syntheticSamples;
        }

        int effectiveK = Math.Min(KNeighbors, classIndices.Count - 1);

        // Precompute nearest neighbors for each minority sample
        var neighborCache = new Dictionary<int, List<int>>();
        foreach (int idx in classIndices)
        {
            neighborCache[idx] = FindKNearestNeighbors(x, idx, classIndices, effectiveK);
        }

        // Generate synthetic samples
        for (int i = 0; i < numSamples; i++)
        {
            // Select a random minority sample as the base
            int baseIdx = classIndices[Random.Next(classIndices.Count)];
            var baseSample = x.GetRow(baseIdx);

            // Get its nearest neighbors
            var neighbors = neighborCache[baseIdx];
            if (neighbors.Count == 0) continue;

            // Select a random neighbor
            int neighborIdx = neighbors[Random.Next(neighbors.Count)];
            var neighborSample = x.GetRow(neighborIdx);

            // Generate synthetic sample by interpolation
            var synthetic = InterpolateSamples(baseSample, neighborSample);
            syntheticSamples.Add(synthetic);
        }

        return syntheticSamples;
    }

    /// <summary>
    /// Creates a synthetic sample by interpolating between two samples.
    /// </summary>
    /// <param name="sample1">The first sample (base).</param>
    /// <param name="sample2">The second sample (neighbor).</param>
    /// <returns>A new synthetic sample on the line segment between the two samples.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This creates a point somewhere on the line between two samples.
    /// If sample1 = [1, 2] and sample2 = [5, 6], and we randomly pick t = 0.25, then:
    /// synthetic = [1, 2] + 0.25 × ([5, 6] - [1, 2])
    ///           = [1, 2] + 0.25 × [4, 4]
    ///           = [1, 2] + [1, 1]
    ///           = [2, 3]
    ///
    /// The synthetic sample is 25% of the way from sample1 to sample2.
    /// </para>
    /// </remarks>
    protected virtual Vector<T> InterpolateSamples(Vector<T> sample1, Vector<T> sample2)
    {
        T t = NumOps.FromDouble(Random.NextDouble());
        var synthetic = new Vector<T>(sample1.Length);

        for (int j = 0; j < sample1.Length; j++)
        {
            T diff = NumOps.Subtract(sample2[j], sample1[j]);
            synthetic[j] = NumOps.Add(sample1[j], NumOps.Multiply(t, diff));
        }

        return synthetic;
    }
}
