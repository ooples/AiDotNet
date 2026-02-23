using AiDotNet.LinearAlgebra;

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
    /// // Default: balance classes adaptively
    /// var adasyn = new ADASYN&lt;double&gt;();
    ///
    /// // Custom settings
    /// var adasyn = new ADASYN&lt;double&gt;(samplingStrategy: 0.8, kNeighbors: 7);
    ///
    /// // Apply to your data
    /// var (newX, newY) = adasyn.Resample(trainX, trainY);
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
    /// 4. Each minority sample gets a number of synthetic samples proportional to its ratio
    /// 5. Generate synthetic samples using SMOTE-style interpolation
    ///
    /// Minority samples near the decision boundary (high difficulty) get more synthetic samples.
    /// </para>
    /// </remarks>
    protected override List<Vector<T>> GenerateSyntheticSamples(Matrix<T> x, List<int> classIndices, int numSamples)
    {
        var syntheticSamples = new List<Vector<T>>();

        if (classIndices.Count < 2)
        {
            return syntheticSamples;
        }

        int effectiveK = Math.Min(KNeighbors, x.Rows - 1);

        // Calculate difficulty ratio for each minority sample
        var ratios = new double[classIndices.Count];
        double sumRatios = 0;

        for (int i = 0; i < classIndices.Count; i++)
        {
            int idx = classIndices[i];

            // Find k nearest neighbors from ALL samples
            var allIndices = Enumerable.Range(0, x.Rows).ToList();
            var neighbors = FindKNearestNeighbors(x, idx, allIndices, effectiveK);

            // Count majority neighbors (neighbors not in minority class)
            int majorityCount = 0;
            foreach (int neighborIdx in neighbors)
            {
                if (!classIndices.Contains(neighborIdx))
                {
                    majorityCount++;
                }
            }

            // Difficulty ratio = proportion of majority neighbors
            ratios[i] = (double)majorityCount / effectiveK;
            sumRatios += ratios[i];
        }

        // Normalize ratios
        if (sumRatios > 0)
        {
            for (int i = 0; i < ratios.Length; i++)
            {
                ratios[i] /= sumRatios;
            }
        }
        else
        {
            // All samples have no majority neighbors - use uniform distribution
            double uniform = 1.0 / classIndices.Count;
            for (int i = 0; i < ratios.Length; i++)
            {
                ratios[i] = uniform;
            }
        }

        // Calculate how many synthetic samples each minority sample should generate
        var samplesPerMinority = new int[classIndices.Count];
        int assignedSamples = 0;

        for (int i = 0; i < classIndices.Count; i++)
        {
            samplesPerMinority[i] = (int)Math.Round(ratios[i] * numSamples);
            assignedSamples += samplesPerMinority[i];
        }

        // Adjust for rounding errors
        while (assignedSamples < numSamples)
        {
            int maxIdx = Array.IndexOf(ratios, ratios.Max());
            samplesPerMinority[maxIdx]++;
            assignedSamples++;
        }
        while (assignedSamples > numSamples)
        {
            int maxIdx = 0;
            for (int i = 1; i < samplesPerMinority.Length; i++)
            {
                if (samplesPerMinority[i] > samplesPerMinority[maxIdx])
                {
                    maxIdx = i;
                }
            }
            if (samplesPerMinority[maxIdx] > 0)
            {
                samplesPerMinority[maxIdx]--;
                assignedSamples--;
            }
            else
            {
                break;
            }
        }

        // Generate synthetic samples
        int minorityK = Math.Min(KNeighbors, classIndices.Count - 1);

        for (int i = 0; i < classIndices.Count; i++)
        {
            int baseIdx = classIndices[i];
            var baseSample = x.GetRow(baseIdx);

            // Find nearest minority neighbors for interpolation
            var minorityNeighbors = FindKNearestNeighbors(x, baseIdx, classIndices, minorityK);

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
        var synthetic = new Vector<T>(sample1.Length);

        for (int j = 0; j < sample1.Length; j++)
        {
            T diff = NumOps.Subtract(sample2[j], sample1[j]);
            synthetic[j] = NumOps.Add(sample1[j], NumOps.Multiply(t, diff));
        }

        return synthetic;
    }
}
