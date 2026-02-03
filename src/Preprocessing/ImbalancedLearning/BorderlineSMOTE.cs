using AiDotNet.LinearAlgebra;

namespace AiDotNet.Preprocessing.ImbalancedLearning;

/// <summary>
/// Implements Borderline-SMOTE for handling imbalanced datasets.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Borderline-SMOTE focuses on minority samples that are near the decision boundary
/// (borderline samples), as these are the most informative for classification.
/// </para>
/// <para>
/// <b>For Beginners:</b> Regular SMOTE creates synthetic samples uniformly among all
/// minority samples. Borderline-SMOTE is smarter - it identifies which minority samples
/// are "borderline" (near majority class samples) and only creates synthetics from those.
///
/// How it works:
/// 1. For each minority sample, count its majority neighbors
/// 2. Classify each minority sample as:
///    - SAFE: Mostly minority neighbors (not borderline, skip)
///    - DANGER: Mix of minority and majority neighbors (borderline, use for synthesis)
///    - NOISE: Mostly majority neighbors (outlier, skip)
/// 3. Only generate synthetic samples from DANGER samples
///
/// Why this is better:
/// - Creates samples where they're needed most (at the boundary)
/// - Avoids wasting effort on samples deep in minority territory
/// - Reduces noise by not synthesizing from outliers
///
/// References:
/// - Han et al. (2005). "Borderline-SMOTE: A New Over-Sampling Method in Imbalanced Data Sets Learning"
/// </para>
/// </remarks>
public class BorderlineSMOTE<T> : OversamplingBase<T>
{
    /// <summary>
    /// The variant of Borderline-SMOTE to use.
    /// </summary>
    private readonly BorderlineSMOTEKind _kind;

    /// <summary>
    /// Gets the name of this oversampling strategy.
    /// </summary>
    public override string Name => $"BorderlineSMOTE-{_kind}";

    /// <summary>
    /// Initializes a new instance of the BorderlineSMOTE class.
    /// </summary>
    /// <param name="samplingStrategy">Target ratio of minority to majority (1.0 for balanced). Default is 1.0.</param>
    /// <param name="kNeighbors">Number of nearest neighbors for synthesis. Default is 5.</param>
    /// <param name="mNeighbors">Number of nearest neighbors for borderline detection. Default is 10.</param>
    /// <param name="kind">Variant of Borderline-SMOTE to use. Default is BorderlineSMOTE-1.</param>
    /// <param name="seed">Random seed for reproducibility.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The parameters are:
    /// - kNeighbors: Used when creating synthetic samples (like regular SMOTE)
    /// - mNeighbors: Used to determine if a sample is "borderline" (larger value = more context)
    /// - kind:
    ///   - Kind1: Only interpolate between borderline and minority neighbors
    ///   - Kind2: Can also interpolate with majority neighbors (more diverse but riskier)
    /// </para>
    /// </remarks>
    public BorderlineSMOTE(
        double samplingStrategy = 1.0,
        int kNeighbors = 5,
        int mNeighbors = 10,
        BorderlineSMOTEKind kind = BorderlineSMOTEKind.Kind1,
        int? seed = null)
        : base(samplingStrategy, kNeighbors, seed)
    {
        MNeighbors = mNeighbors;
        _kind = kind;
    }

    /// <summary>
    /// Number of neighbors used for borderline detection.
    /// </summary>
    protected int MNeighbors { get; }

    /// <summary>
    /// Generates synthetic samples using Borderline-SMOTE.
    /// </summary>
    /// <param name="x">The full feature matrix.</param>
    /// <param name="classIndices">Indices of samples belonging to the minority class.</param>
    /// <param name="numSamples">Number of synthetic samples to generate.</param>
    /// <returns>List of synthetic sample vectors.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The algorithm:
    ///
    /// 1. For each minority sample, find m nearest neighbors from ALL classes
    /// 2. Count how many neighbors are majority class
    /// 3. If half or more are majority → sample is in DANGER zone (borderline)
    /// 4. If ALL are majority → sample is NOISE (outlier), skip it
    /// 5. Generate synthetic samples only from DANGER samples
    ///
    /// For Kind1: Interpolate only with minority neighbors
    /// For Kind2: Can interpolate with any neighbor (including majority)
    /// </para>
    /// </remarks>
    protected override List<Vector<T>> GenerateSyntheticSamples(Matrix<T> x, List<int> classIndices, int numSamples)
    {
        var syntheticSamples = new List<Vector<T>>();

        if (classIndices.Count < 2)
        {
            return syntheticSamples;
        }

        int effectiveM = Math.Min(MNeighbors, x.Rows - 1);

        // Find borderline (DANGER) samples
        var borderlineIndices = new List<int>();
        var allIndices = Enumerable.Range(0, x.Rows).ToList();

        foreach (int idx in classIndices)
        {
            // Find m nearest neighbors from ALL samples
            var neighbors = FindKNearestNeighbors(x, idx, allIndices, effectiveM);

            // Count majority neighbors
            int majorityCount = 0;
            foreach (int neighborIdx in neighbors)
            {
                if (!classIndices.Contains(neighborIdx))
                {
                    majorityCount++;
                }
            }

            // Classify the sample
            // DANGER: m/2 <= majorityCount < m (borderline)
            // NOISE: majorityCount == m (all majority neighbors)
            // SAFE: majorityCount < m/2 (mostly minority neighbors)
            if (majorityCount >= effectiveM / 2 && majorityCount < effectiveM)
            {
                borderlineIndices.Add(idx);
            }
        }

        if (borderlineIndices.Count == 0)
        {
            // No borderline samples found, fall back to regular SMOTE behavior
            return GenerateRegularSMOTE(x, classIndices, numSamples);
        }

        // Generate synthetic samples from borderline samples
        int effectiveK = Math.Min(KNeighbors, classIndices.Count - 1);

        for (int i = 0; i < numSamples; i++)
        {
            // Select a random borderline sample
            int baseIdx = borderlineIndices[Random.Next(borderlineIndices.Count)];
            var baseSample = x.GetRow(baseIdx);

            List<int> neighborPool;
            if (_kind == BorderlineSMOTEKind.Kind1)
            {
                // Kind1: Only use minority neighbors for interpolation
                neighborPool = FindKNearestNeighbors(x, baseIdx, classIndices, effectiveK);
            }
            else
            {
                // Kind2: Can use any neighbor (including majority)
                neighborPool = FindKNearestNeighbors(x, baseIdx, allIndices, effectiveK);
            }

            if (neighborPool.Count == 0) continue;

            // Select random neighbor and interpolate
            int neighborIdx = neighborPool[Random.Next(neighborPool.Count)];
            var neighborSample = x.GetRow(neighborIdx);

            var synthetic = InterpolateSamples(baseSample, neighborSample);
            syntheticSamples.Add(synthetic);
        }

        return syntheticSamples;
    }

    /// <summary>
    /// Fallback to regular SMOTE when no borderline samples are found.
    /// </summary>
    private List<Vector<T>> GenerateRegularSMOTE(Matrix<T> x, List<int> classIndices, int numSamples)
    {
        var syntheticSamples = new List<Vector<T>>();
        int effectiveK = Math.Min(KNeighbors, classIndices.Count - 1);

        for (int i = 0; i < numSamples; i++)
        {
            int baseIdx = classIndices[Random.Next(classIndices.Count)];
            var baseSample = x.GetRow(baseIdx);

            var neighbors = FindKNearestNeighbors(x, baseIdx, classIndices, effectiveK);
            if (neighbors.Count == 0) continue;

            int neighborIdx = neighbors[Random.Next(neighbors.Count)];
            var neighborSample = x.GetRow(neighborIdx);

            var synthetic = InterpolateSamples(baseSample, neighborSample);
            syntheticSamples.Add(synthetic);
        }

        return syntheticSamples;
    }

    /// <summary>
    /// Creates a synthetic sample by interpolating between two samples.
    /// </summary>
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

/// <summary>
/// Specifies the variant of Borderline-SMOTE to use.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b>
/// - Kind1: Safer, only creates samples between minority class points
/// - Kind2: More diverse, can create samples between minority and majority points
/// </para>
/// </remarks>
public enum BorderlineSMOTEKind
{
    /// <summary>
    /// Interpolates only between borderline samples and their minority class neighbors.
    /// </summary>
    Kind1,

    /// <summary>
    /// Interpolates between borderline samples and any neighbor (including majority class).
    /// </summary>
    Kind2
}
