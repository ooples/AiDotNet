using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Augmentation.Tabular;

/// <summary>
/// Implements Borderline-SMOTE for imbalanced datasets, focusing on samples near the decision boundary.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Borderline-SMOTE is an improvement over standard SMOTE that only
/// generates synthetic samples from minority instances that are near the decision boundary
/// (i.e., in "danger" zones where they have majority class neighbors). This focuses synthetic
/// data generation where it matters most.</para>
///
/// <para><b>How it works:</b>
/// <list type="number">
/// <item>For each minority sample, find its k nearest neighbors from BOTH classes</item>
/// <item>Classify each minority sample as:
///   <list type="bullet">
///     <item>SAFE: Most neighbors are minority class (not used for synthesis)</item>
///     <item>DANGER: Half to most neighbors are majority class (used for synthesis)</item>
///     <item>NOISE: All neighbors are majority class (ignored)</item>
///   </list>
/// </item>
/// <item>Only generate synthetic samples from DANGER samples</item>
/// </list>
/// </para>
///
/// <para><b>Borderline-SMOTE Variants:</b>
/// <list type="bullet">
/// <item>Borderline-SMOTE1: Synthetic samples interpolate only between DANGER samples and minority neighbors</item>
/// <item>Borderline-SMOTE2: Synthetic samples can also interpolate toward majority neighbors</item>
/// </list>
/// This implementation uses Borderline-SMOTE1 by default, configurable via <see cref="UseBorderline2"/>.
/// </para>
///
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>When standard SMOTE generates too many samples in easy regions</item>
/// <item>When you want to focus on the challenging boundary between classes</item>
/// <item>When minority samples near the boundary are most important</item>
/// </list>
/// </para>
///
/// <para><b>Reference:</b> Han et al., "Borderline-SMOTE: A New Over-Sampling Method in Imbalanced Data Sets Learning" (2005)</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class BorderlineSmoteAugmenter<T> : TabularAugmenterBase<T>
{
    /// <summary>
    /// Sample classification for Borderline-SMOTE.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> These categories determine which minority samples are used
    /// to generate synthetic data.</para>
    /// </remarks>
    private enum SampleType
    {
        /// <summary>Sample has mostly minority neighbors - not used for synthesis.</summary>
        Safe,
        /// <summary>Sample has mix of neighbors - USED for synthesis.</summary>
        Danger,
        /// <summary>Sample has only majority neighbors - considered noise, not used.</summary>
        Noise
    }

    /// <summary>
    /// Gets the number of nearest neighbors to consider.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Controls how many neighbors are examined to classify each
    /// minority sample as SAFE, DANGER, or NOISE.</para>
    /// <para>Default: 5</para>
    /// </remarks>
    public int KNeighbors { get; }

    /// <summary>
    /// Gets the number of minority neighbors to use for interpolation.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> After identifying DANGER samples, this parameter controls
    /// how many minority neighbors are considered for generating synthetic samples.</para>
    /// <para>Default: 5</para>
    /// </remarks>
    public int MNeighbors { get; }

    /// <summary>
    /// Gets the sampling ratio for synthetic sample generation.
    /// </summary>
    /// <remarks>
    /// <para>Default: 1.0 (generate as many synthetic samples as DANGER samples)</para>
    /// </remarks>
    public double SamplingRatio { get; }

    /// <summary>
    /// Gets whether to use Borderline-SMOTE2 (can interpolate toward majority).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Borderline-SMOTE2 can create synthetic samples that
    /// interpolate toward majority class neighbors, potentially pushing the boundary further.
    /// Use with caution as it may create samples in majority territory.</para>
    /// <para>Default: false (use Borderline-SMOTE1)</para>
    /// </remarks>
    public bool UseBorderline2 { get; }

    /// <summary>
    /// Creates a new Borderline-SMOTE augmenter.
    /// </summary>
    /// <param name="kNeighbors">Number of neighbors for danger classification (default: 5).</param>
    /// <param name="mNeighbors">Number of minority neighbors for interpolation (default: 5).</param>
    /// <param name="samplingRatio">Ratio of synthetic samples to generate (default: 1.0).</param>
    /// <param name="useBorderline2">Use Borderline-SMOTE2 variant (default: false).</param>
    /// <param name="probability">Probability of applying this augmentation (default: 1.0).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> The default parameters work well for most cases. Try
    /// Borderline-SMOTE2 if SMOTE1 doesn't improve boundary classification enough.</para>
    /// </remarks>
    public BorderlineSmoteAugmenter(
        int kNeighbors = 5,
        int mNeighbors = 5,
        double samplingRatio = 1.0,
        bool useBorderline2 = false,
        double probability = 1.0) : base(probability)
    {
        if (kNeighbors < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(kNeighbors), "K neighbors must be at least 1.");
        }

        if (mNeighbors < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(mNeighbors), "M neighbors must be at least 1.");
        }

        if (samplingRatio <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(samplingRatio), "Sampling ratio must be positive.");
        }

        KNeighbors = kNeighbors;
        MNeighbors = mNeighbors;
        SamplingRatio = samplingRatio;
        UseBorderline2 = useBorderline2;
    }

    /// <inheritdoc />
    protected override Matrix<T> ApplyAugmentation(Matrix<T> data, AugmentationContext<T> context)
    {
        // Without class labels, treat all as minority and use standard SMOTE behavior
        return GenerateSyntheticSamples(data, null, context);
    }

    /// <summary>
    /// Applies Borderline-SMOTE to generate synthetic samples for the minority class.
    /// </summary>
    /// <param name="minorityData">Matrix containing only minority class samples.</param>
    /// <param name="majorityData">Matrix containing only majority class samples.</param>
    /// <param name="context">The augmentation context.</param>
    /// <returns>Matrix containing synthetic samples (original data is NOT included).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The majority data is used to identify which minority samples
    /// are in the DANGER zone. Without majority data, all samples are treated as DANGER.</para>
    /// </remarks>
    public Matrix<T> GenerateSyntheticSamples(
        Matrix<T> minorityData,
        Matrix<T>? majorityData,
        AugmentationContext<T> context)
    {
        int minorityCount = GetSampleCount(minorityData);
        int cols = GetFeatureCount(minorityData);

        if (minorityCount < 2)
        {
            return new Matrix<T>(0, cols);
        }

        // Classify all minority samples
        var sampleTypes = ClassifyMinoritySamples(minorityData, majorityData);

        // Get indices of DANGER samples
        var dangerIndices = new List<int>();
        for (int i = 0; i < minorityCount; i++)
        {
            if (sampleTypes[i] == SampleType.Danger)
            {
                dangerIndices.Add(i);
            }
        }

        if (dangerIndices.Count == 0)
        {
            // No danger samples - fall back to all minority samples
            for (int i = 0; i < minorityCount; i++)
            {
                if (sampleTypes[i] != SampleType.Noise)
                {
                    dangerIndices.Add(i);
                }
            }
        }

        if (dangerIndices.Count == 0)
        {
            return new Matrix<T>(0, cols);
        }

        // Calculate how many synthetic samples to generate
        int numSynthetic = (int)Math.Ceiling(dangerIndices.Count * SamplingRatio);
        var syntheticSamples = new Matrix<T>(numSynthetic, cols);

        // Precompute minority distances for neighbor finding
        var minorityDistances = ComputeDistanceMatrix(minorityData);
        int effectiveM = Math.Min(MNeighbors, minorityCount - 1);

        int syntheticIndex = 0;
        while (syntheticIndex < numSynthetic)
        {
            // Select a random DANGER sample
            int dangerIdx = dangerIndices[context.Random.Next(dangerIndices.Count)];

            // Find m nearest minority neighbors
            var minorityNeighbors = GetKNearestNeighbors(minorityDistances, dangerIdx, effectiveM);

            int neighborIdx;
            if (UseBorderline2 && majorityData is not null && context.Random.NextDouble() < 0.5)
            {
                // Borderline-SMOTE2: sometimes use a majority neighbor
                neighborIdx = GetNearestMajorityNeighbor(minorityData, dangerIdx, majorityData, context);
            }
            else
            {
                // Borderline-SMOTE1: always use minority neighbor
                neighborIdx = minorityNeighbors[context.Random.Next(minorityNeighbors.Length)];
            }

            // Generate synthetic sample by interpolating
            double gap = context.Random.NextDouble();

            // For SMOTE2 with majority neighbor, only interpolate half way toward majority
            bool isMajorityNeighbor = UseBorderline2 && majorityData is not null && neighborIdx >= minorityCount;
            if (isMajorityNeighbor)
            {
                gap *= 0.5;
            }

            for (int c = 0; c < cols; c++)
            {
                double val1 = NumOps.ToDouble(minorityData[dangerIdx, c]);
                double val2;

                if (isMajorityNeighbor && majorityData is not null)
                {
                    // neighborIdx is in majority data
                    val2 = NumOps.ToDouble(majorityData[neighborIdx - minorityCount, c]);
                }
                else
                {
                    val2 = NumOps.ToDouble(minorityData[neighborIdx, c]);
                }

                double synthetic = val1 + gap * (val2 - val1);
                syntheticSamples[syntheticIndex, c] = NumOps.FromDouble(synthetic);
            }

            syntheticIndex++;
        }

        return syntheticSamples;
    }

    /// <summary>
    /// Classifies each minority sample as SAFE, DANGER, or NOISE.
    /// </summary>
    /// <param name="minorityData">The minority class samples.</param>
    /// <param name="majorityData">The majority class samples (optional).</param>
    /// <returns>Array of sample types for each minority sample.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// <list type="bullet">
    /// <item>SAFE: Less than half of k-neighbors are majority class</item>
    /// <item>DANGER: Half or more (but not all) of k-neighbors are majority class</item>
    /// <item>NOISE: All k-neighbors are majority class</item>
    /// </list>
    /// </para>
    /// </remarks>
    private SampleType[] ClassifyMinoritySamples(Matrix<T> minorityData, Matrix<T>? majorityData)
    {
        int minorityCount = GetSampleCount(minorityData);
        var types = new SampleType[minorityCount];

        if (majorityData is null || GetSampleCount(majorityData) == 0)
        {
            // Without majority data, all samples are DANGER
            for (int i = 0; i < minorityCount; i++)
            {
                types[i] = SampleType.Danger;
            }
            return types;
        }

        int majorityCount = GetSampleCount(majorityData);
        int cols = GetFeatureCount(minorityData);

        for (int i = 0; i < minorityCount; i++)
        {
            var allDistances = new List<(double Distance, bool IsMajority)>();

            // Distances to other minority samples
            for (int j = 0; j < minorityCount; j++)
            {
                if (i != j)
                {
                    double dist = ComputeDistance(minorityData, i, minorityData, j, cols);
                    allDistances.Add((dist, false));
                }
            }

            // Distances to majority samples
            for (int j = 0; j < majorityCount; j++)
            {
                double dist = ComputeDistance(minorityData, i, majorityData, j, cols);
                allDistances.Add((dist, true));
            }

            // Get k nearest neighbors
            var kNearest = allDistances
                .OrderBy(x => x.Distance)
                .Take(KNeighbors)
                .ToList();

            int majorityNeighbors = kNearest.Count(x => x.IsMajority);

            // Classify based on majority neighbor ratio
            if (majorityNeighbors == KNeighbors)
            {
                types[i] = SampleType.Noise;
            }
            else if (majorityNeighbors >= KNeighbors / 2)
            {
                types[i] = SampleType.Danger;
            }
            else
            {
                types[i] = SampleType.Safe;
            }
        }

        return types;
    }

    /// <summary>
    /// Gets the nearest majority neighbor for Borderline-SMOTE2.
    /// </summary>
    private int GetNearestMajorityNeighbor(
        Matrix<T> minorityData,
        int minorityIdx,
        Matrix<T> majorityData,
        AugmentationContext<T> context)
    {
        int majorityCount = GetSampleCount(majorityData);
        int cols = GetFeatureCount(minorityData);
        int minorityCount = GetSampleCount(minorityData);

        double minDist = double.MaxValue;
        int nearestIdx = 0;

        for (int j = 0; j < majorityCount; j++)
        {
            double dist = ComputeDistance(minorityData, minorityIdx, majorityData, j, cols);
            if (dist < minDist)
            {
                minDist = dist;
                nearestIdx = j;
            }
        }

        // Return index offset by minority count to indicate it's a majority sample
        return minorityCount + nearestIdx;
    }

    /// <summary>
    /// Computes the Euclidean distance between two samples.
    /// </summary>
    private double ComputeDistance(Matrix<T> data1, int idx1, Matrix<T> data2, int idx2, int cols)
    {
        double dist = 0;
        for (int c = 0; c < cols; c++)
        {
            double diff = NumOps.ToDouble(data1[idx1, c]) - NumOps.ToDouble(data2[idx2, c]);
            dist += diff * diff;
        }
        return Math.Sqrt(dist);
    }

    /// <summary>
    /// Computes the distance matrix for all pairs of samples.
    /// </summary>
    private double[,] ComputeDistanceMatrix(Matrix<T> data)
    {
        int rows = GetSampleCount(data);
        int cols = GetFeatureCount(data);
        var distances = new double[rows, rows];

        for (int i = 0; i < rows; i++)
        {
            for (int j = i + 1; j < rows; j++)
            {
                double dist = ComputeDistance(data, i, data, j, cols);
                distances[i, j] = dist;
                distances[j, i] = dist;
            }
        }

        return distances;
    }

    /// <summary>
    /// Gets the k nearest neighbors for a given sample.
    /// </summary>
    private int[] GetKNearestNeighbors(double[,] distances, int sampleIdx, int k)
    {
        int rows = distances.GetLength(0);
        var indexedDistances = new List<(int Index, double Distance)>();

        for (int i = 0; i < rows; i++)
        {
            if (i != sampleIdx)
            {
                indexedDistances.Add((i, distances[sampleIdx, i]));
            }
        }

        return indexedDistances
            .OrderBy(x => x.Distance)
            .Take(k)
            .Select(x => x.Index)
            .ToArray();
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var parameters = base.GetParameters();
        parameters["kNeighbors"] = KNeighbors;
        parameters["mNeighbors"] = MNeighbors;
        parameters["samplingRatio"] = SamplingRatio;
        parameters["useBorderline2"] = UseBorderline2;
        return parameters;
    }
}
