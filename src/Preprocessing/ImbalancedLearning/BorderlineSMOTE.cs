using AiDotNet.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

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
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
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
    /// 5. Generate synthetic samples only from DANGER samples (none when there are no DANGER samples)
    ///
    /// For Kind1: Interpolate only with minority neighbors
    /// For Kind2: Can interpolate with any neighbor; towards a majority neighbor the gap is drawn from [0, 0.5)
    /// </para>
    /// </remarks>
    protected override List<Vector<T>> GenerateSyntheticSamples(Matrix<T> x, List<int> classIndices, int numSamples)
    {
        var syntheticSamples = new List<Vector<T>>();

        if (classIndices.Count < 2)
        {
            return syntheticSamples;
        }

        var data = Tensor<T>.FromMatrix(x);
        int effectiveM = Math.Min(MNeighbors, x.Rows - 1);
        var inClass = new HashSet<int>(classIndices);
        var allIndices = Enumerable.Range(0, x.Rows).ToArray();

        // DANGER samples (Han et al., 2005; imbalanced-learn's _in_danger_noise): m/2 <= m' < m, where m' is
        // the number of the m nearest neighbours (from ALL samples) that belong to another class. NOISE
        // (m' == m) and SAFE (m' < m/2) samples are not used as bases.
        var borderlineIndices = classIndices
            .Where(idx =>
            {
                int majorityCount = ResamplingTensorOps.Nearest(data, idx, allIndices, effectiveM)
                    .Count(neighbor => !inClass.Contains(neighbor));
                return 2 * majorityCount >= effectiveM && majorityCount < effectiveM;
            })
            .ToList();

        if (borderlineIndices.Count == 0)
        {
            // No borderline samples: like imbalanced-learn, nothing is generated for this class
            return syntheticSamples;
        }

        // Generate synthetic samples from borderline samples
        int effectiveK = Math.Min(KNeighbors, classIndices.Count - 1);
        var neighborCache = new Dictionary<int, List<int>>();

        for (int i = 0; i < numSamples; i++)
        {
            // Select a random borderline sample
            int baseIdx = borderlineIndices[Random.Next(borderlineIndices.Count)];
            var baseSample = x.GetRow(baseIdx);

            if (!neighborCache.TryGetValue(baseIdx, out var neighborPool))
            {
                // Kind1: only minority neighbours; Kind2: neighbours from all classes
                neighborPool = _kind == BorderlineSMOTEKind.Kind1
                    ? ResamplingTensorOps.Nearest(data, baseIdx, classIndices, effectiveK)
                    : ResamplingTensorOps.Nearest(data, baseIdx, allIndices, effectiveK);
                neighborCache[baseIdx] = neighborPool;
            }

            if (neighborPool.Count == 0) continue;

            // Select random neighbor and interpolate. Borderline-SMOTE2 (Han et al., 2005) moves towards a
            // neighbour of another class by a random gap in [0, 0.5) so the sample stays nearer the minority.
            int neighborIdx = neighborPool[Random.Next(neighborPool.Count)];
            var neighborSample = x.GetRow(neighborIdx);

            double stepSize = inClass.Contains(neighborIdx) ? 1.0 : 0.5;
            var synthetic = InterpolateSamples(baseSample, neighborSample, stepSize);
            syntheticSamples.Add(synthetic);
        }

        return syntheticSamples;
    }

    /// <summary>
    /// The synthetic sample <c>sample1 + t (sample2 - sample1)</c> with <c>t</c> drawn uniformly from
    /// [0, <paramref name="stepSize"/>).
    /// </summary>
    private Vector<T> InterpolateSamples(Vector<T> sample1, Vector<T> sample2, double stepSize)
    {
        T t = NumOps.FromDouble(stepSize * Random.NextDouble());
        return ResamplingTensorOps.Interpolate(sample1, sample2, t);
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
    /// Interpolates between borderline samples and any neighbor (including majority class); towards a
    /// neighbor of another class the random gap is drawn from [0, 0.5) (Han et al., 2005).
    /// </summary>
    Kind2
}
