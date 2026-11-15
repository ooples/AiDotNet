using AiDotNet.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.ImbalancedLearning;

/// <summary>
/// Implements ADASYN (Adaptive Synthetic Sampling) for handling imbalanced datasets.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// ADASYN is an extension of SMOTE that adaptively generates synthetic samples based on the
/// local density of minority samples. It creates more synthetic samples in regions where
/// the minority class is harder to learn (i.e., surrounded by majority class samples).
/// </para>
/// <para>
/// <b>For Beginners:</b> ADASYN improves on SMOTE by being smarter about WHERE to create
/// synthetic samples:
///
/// 1. For each minority sample, look at its k nearest neighbors (from ALL classes)
/// 2. Count how many of those neighbors are majority class samples
/// 3. Minority samples with MORE majority neighbors are "harder to learn"
/// 4. Create MORE synthetic samples near the "hard" minority samples
///
/// Example scenario:
/// - Minority sample A has 4 minority neighbors, 1 majority neighbor → Easy, few synthetics
/// - Minority sample B has 1 minority neighbor, 4 majority neighbors → Hard, many synthetics
///
/// This is better than regular SMOTE because:
/// - Focuses sampling effort where it's needed most
/// - Helps the model learn the difficult boundary cases
/// - Reduces risk of overfitting in easy regions
///
/// When to use ADASYN:
/// - When the boundary between classes is complex
/// - When some minority samples are "islands" in majority territory
/// - When you want adaptive, data-driven sampling
///
/// References:
/// - He et al. (2008). "ADASYN: Adaptive Synthetic Sampling Approach for Imbalanced Learning"
/// </para>
/// </remarks>
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
public class ADASYN<T> : OversamplingBase<T>
{
    /// <summary>
    /// Gets the name of this oversampling strategy.
    /// </summary>
    public override string Name => "ADASYN";

    /// <summary>
    /// Initializes a new instance of the ADASYN class.
    /// </summary>
    /// <param name="samplingStrategy">Target ratio of minority to majority (1.0 for balanced). Default is 1.0.</param>
    /// <param name="kNeighbors">Number of nearest neighbors for synthesis and density estimation. Default is 5.</param>
    /// <param name="seed">Random seed for reproducibility.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Example usage:
    ///
    /// <code>
    /// var trainX = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 }, { 7.0, 8.0 } });
    /// var trainY = new Vector&lt;double&gt;(new double[] { 0.0, 1.0, 0.0, 1.0 });
    /// // Default: balance classes adaptively
    /// var adasyn = new ADASYN&lt;double&gt;();
    ///
    /// // Custom settings
    /// var adasyn2 = new ADASYN&lt;double&gt;(samplingStrategy: 0.8, kNeighbors: 7);
    ///
    /// // Apply to your data
    /// var (newX, newY) = adasyn2.Resample(trainX, trainY);
    /// </code>
    /// </para>
    /// </remarks>
    public ADASYN(double samplingStrategy = 1.0, int kNeighbors = 5, int? seed = null)
        : base(samplingStrategy, kNeighbors, seed)
    {
    }

    /// <summary>
    /// Generates synthetic samples using ADASYN's adaptive approach.
    /// </summary>
    /// <param name="x">The full feature matrix.</param>
    /// <param name="classIndices">Indices of samples belonging to the minority class.</param>
    /// <param name="numSamples">Number of synthetic samples to generate.</param>
    /// <returns>List of synthetic sample vectors.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> ADASYN's process:
    ///
    /// 1. For each minority sample, find k nearest neighbors from ALL classes
    /// 2. Calculate "difficulty ratio" = (# majority neighbors) / k
    /// 3. Normalize ratios so they sum to 1 (probability distribution)
    /// 4. Each minority sample gets round(ratio * numSamples) synthetic samples (half to even, as numpy's rint
    ///    in imbalanced-learn); the rounded counts are not re-balanced, so their total can differ slightly
    /// 5. Generate synthetic samples using SMOTE-style interpolation
    ///
    /// Minority samples near the decision boundary (high difficulty) get more synthetic samples.
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">
    /// No minority sample has a neighbour of another class, or the rounded counts are all zero (as in
    /// imbalanced-learn).
    /// </exception>
    protected override List<Vector<T>> GenerateSyntheticSamples(Matrix<T> x, List<int> classIndices, int numSamples)
    {
        var syntheticSamples = new List<Vector<T>>();

        if (classIndices.Count < 2)
        {
            return syntheticSamples;
        }

        var data = Tensor<T>.FromMatrix(x);
        int effectiveK = Math.Min(KNeighbors, x.Rows - 1);
        var inClass = new HashSet<int>(classIndices);
        var allIndices = Enumerable.Range(0, x.Rows).ToArray();

        // Difficulty ratio of each minority sample: the share of its k nearest neighbours (from ALL samples)
        // that belong to another class.
        var ratios = classIndices
            .Select(idx => (double)ResamplingTensorOps.Nearest(data, idx, allIndices, effectiveK)
                .Count(neighbor => !inClass.Contains(neighbor)) / effectiveK)
            .ToArray();
        double sumRatios = ratios.Sum();

        // imbalanced-learn ADASYN (He et al., 2008): with no neighbour of another class the density
        // distribution is undefined (division by zero), so ADASYN is not applicable.
        if (sumRatios <= 0)
        {
            throw new InvalidOperationException(
                "Not any neighbours belong to the majority class. This case will induce a NaN case with a " +
                "division by zero. ADASYN is not suited for this specific dataset. Use SMOTE instead.");
        }

        // g_i = rint(r_i / sum(r) * G); like imbalanced-learn, the rounded counts are used as they are
        // (their total may differ slightly from G).
        var samplesPerMinority = ratios
            .Select(ratio => (int)Math.Round(ratio / sumRatios * numSamples, MidpointRounding.ToEven))
            .ToArray();
        if (samplesPerMinority.Sum() == 0)
        {
            throw new InvalidOperationException("No samples will be generated with the provided ratio settings.");
        }

        // Generate synthetic samples
        int minorityK = Math.Min(KNeighbors, classIndices.Count - 1);

        for (int i = 0; i < classIndices.Count; i++)
        {
            int baseIdx = classIndices[i];
            var baseSample = x.GetRow(baseIdx);

            // Find nearest minority neighbors for interpolation
            var minorityNeighbors = ResamplingTensorOps.Nearest(data, baseIdx, classIndices, minorityK);

            if (minorityNeighbors.Count == 0) continue;

            // Generate the assigned number of synthetic samples for this minority sample
            for (int j = 0; j < samplesPerMinority[i]; j++)
            {
                // Select random neighbor
                int neighborIdx = minorityNeighbors[Random.Next(minorityNeighbors.Count)];
                var neighborSample = x.GetRow(neighborIdx);

                // Interpolate
                var synthetic = InterpolateSamples(baseSample, neighborSample);
                syntheticSamples.Add(synthetic);
            }
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
    /// <b>For Beginners:</b> Same interpolation as SMOTE - creates a point somewhere
    /// on the line between two minority samples.
    /// </para>
    /// </remarks>
    private Vector<T> InterpolateSamples(Vector<T> sample1, Vector<T> sample2)
    {
        T t = NumOps.FromDouble(Random.NextDouble());
        return ResamplingTensorOps.Interpolate(sample1, sample2, t);
    }
}
