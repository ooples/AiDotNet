using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Augmentation.Tabular;

/// <summary>
/// Implements ADASYN (Adaptive Synthetic Sampling) for imbalanced datasets.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> ADASYN is an extension of SMOTE that adaptively generates more
/// synthetic samples in regions where the minority class is harder to learn (i.e., where there
/// are more majority class neighbors). This focuses the synthetic data where it's most needed.</para>
///
/// <para><b>How ADASYN differs from SMOTE:</b>
/// <list type="bullet">
/// <item>SMOTE: Generates the same number of synthetic samples for each minority instance</item>
/// <item>ADASYN: Generates more synthetic samples for minority instances that have more majority neighbors</item>
/// </list>
/// This adaptive approach helps the classifier focus on the hardest-to-learn examples.
/// </para>
///
/// <para><b>Algorithm:</b>
/// <list type="number">
/// <item>For each minority sample, calculate the ratio of majority neighbors to total neighbors</item>
/// <item>Normalize these ratios to get a distribution</item>
/// <item>Generate synthetic samples proportionally - more for samples with more majority neighbors</item>
/// </list>
/// </para>
///
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>When the decision boundary is complex and irregular</item>
/// <item>When minority samples near the boundary need more representation</item>
/// <item>When standard SMOTE doesn't improve minority class recall enough</item>
/// </list>
/// </para>
///
/// <para><b>Reference:</b> He et al., "ADASYN: Adaptive Synthetic Sampling Approach for Imbalanced Learning" (2008)</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class AdasynAugmenter<T> : TabularAugmenterBase<T>
{
    /// <summary>
    /// Gets the number of nearest neighbors to consider.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This parameter controls how many neighbors are examined to
    /// determine if a minority sample is in a "hard" region (surrounded by majority samples).</para>
    /// <para>Default: 5</para>
    /// </remarks>
    public int KNeighbors { get; }

    /// <summary>
    /// Gets the target balance ratio between minority and majority classes.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> A value of 1.0 means try to achieve equal numbers of
    /// minority and majority samples. Values less than 1.0 result in fewer synthetic samples.</para>
    /// <para>Default: 1.0</para>
    /// </remarks>
    public double Beta { get; }

    /// <summary>
    /// Creates a new ADASYN augmenter.
    /// </summary>
    /// <param name="kNeighbors">Number of nearest neighbors to use (default: 5).</param>
    /// <param name="beta">Target balance ratio (default: 1.0 for full balance).</param>
    /// <param name="probability">Probability of applying this augmentation (default: 1.0).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> The default parameters work well for most cases. Increase
    /// kNeighbors if you have a large minority class; decrease it if you have very few samples.</para>
    /// </remarks>
    public AdasynAugmenter(
        int kNeighbors = 5,
        double beta = 1.0,
        double probability = 1.0) : base(probability)
    {
        if (kNeighbors < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(kNeighbors), "K neighbors must be at least 1.");
        }

        if (beta <= 0 || beta > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(beta), "Beta must be in range (0, 1].");
        }

        KNeighbors = kNeighbors;
        Beta = beta;
    }

    /// <inheritdoc />
    protected override Matrix<T> ApplyAugmentation(Matrix<T> data, AugmentationContext<T> context)
    {
        // ADASYN on raw data without labels - generate synthetic samples for all rows
        return GenerateSyntheticSamples(data, null, context);
    }

    /// <summary>
    /// Applies ADASYN to generate synthetic samples for the minority class.
    /// </summary>
    /// <param name="minorityData">Matrix containing only minority class samples.</param>
    /// <param name="majorityData">Matrix containing only majority class samples (optional, improves adaptation).</param>
    /// <param name="context">The augmentation context.</param>
    /// <returns>Matrix containing synthetic samples (original data is NOT included).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Providing majorityData allows ADASYN to determine which
    /// minority samples are in "hard" regions. Without it, all samples are treated equally (like SMOTE).</para>
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

        // Calculate number of synthetic samples to generate
        int majorityCount = majorityData is not null ? GetSampleCount(majorityData) : minorityCount;
        int numToGenerate = (int)Math.Ceiling(Beta * (majorityCount - minorityCount));

        if (numToGenerate <= 0)
        {
            return new Matrix<T>(0, cols);
        }

        // Calculate difficulty ratio for each minority sample
        var difficultyRatios = CalculateDifficultyRatios(minorityData, majorityData, context);

        // Normalize ratios to form a distribution
        double sumRatios = difficultyRatios.Sum();
        if (sumRatios < 1e-10)
        {
            // All samples have zero difficulty - use uniform distribution
            for (int i = 0; i < difficultyRatios.Length; i++)
            {
                difficultyRatios[i] = 1.0 / difficultyRatios.Length;
            }
            sumRatios = 1.0;
        }

        // Calculate number of synthetic samples per minority instance
        var syntheticCountPerSample = new int[minorityCount];
        int totalAssigned = 0;

        for (int i = 0; i < minorityCount; i++)
        {
            syntheticCountPerSample[i] = (int)Math.Round(numToGenerate * difficultyRatios[i] / sumRatios);
            totalAssigned += syntheticCountPerSample[i];
        }

        // Adjust to exactly match numToGenerate
        while (totalAssigned < numToGenerate)
        {
            int idx = context.Random.Next(minorityCount);
            syntheticCountPerSample[idx]++;
            totalAssigned++;
        }
        while (totalAssigned > numToGenerate)
        {
            int idx = context.Random.Next(minorityCount);
            if (syntheticCountPerSample[idx] > 0)
            {
                syntheticCountPerSample[idx]--;
                totalAssigned--;
            }
        }

        // Generate synthetic samples
        var syntheticSamples = new Matrix<T>(numToGenerate, cols);
        int syntheticIndex = 0;

        var distances = ComputeDistanceMatrix(minorityData);
        int effectiveK = Math.Min(KNeighbors, minorityCount - 1);

        for (int i = 0; i < minorityCount && syntheticIndex < numToGenerate; i++)
        {
            int numForThisSample = syntheticCountPerSample[i];
            if (numForThisSample == 0) continue;

            var neighbors = GetKNearestNeighbors(distances, i, effectiveK);

            for (int n = 0; n < numForThisSample && syntheticIndex < numToGenerate; n++)
            {
                int neighborIdx = neighbors[context.Random.Next(neighbors.Length)];
                double gap = context.Random.NextDouble();

                for (int c = 0; c < cols; c++)
                {
                    double val1 = NumOps.ToDouble(minorityData[i, c]);
                    double val2 = NumOps.ToDouble(minorityData[neighborIdx, c]);
                    double synthetic = val1 + gap * (val2 - val1);
                    syntheticSamples[syntheticIndex, c] = NumOps.FromDouble(synthetic);
                }

                syntheticIndex++;
            }
        }

        return syntheticSamples;
    }

    /// <summary>
    /// Calculates the difficulty ratio for each minority sample.
    /// </summary>
    /// <param name="minorityData">The minority class samples.</param>
    /// <param name="majorityData">The majority class samples (optional).</param>
    /// <param name="context">The augmentation context.</param>
    /// <returns>Array of difficulty ratios (higher = harder to learn).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> A sample has a high difficulty ratio if most of its k nearest
    /// neighbors are from the majority class. This indicates it's in a "hard" region where the
    /// classes overlap, and more synthetic samples are needed there.</para>
    /// </remarks>
    private double[] CalculateDifficultyRatios(
        Matrix<T> minorityData,
        Matrix<T>? majorityData,
        AugmentationContext<T> context)
    {
        int minorityCount = GetSampleCount(minorityData);
        var ratios = new double[minorityCount];

        if (majorityData is null || GetSampleCount(majorityData) == 0)
        {
            // Without majority data, use uniform distribution
            for (int i = 0; i < minorityCount; i++)
            {
                ratios[i] = 1.0;
            }
            return ratios;
        }

        int majorityCount = GetSampleCount(majorityData);
        int cols = GetFeatureCount(minorityData);

        for (int i = 0; i < minorityCount; i++)
        {
            // Find k nearest neighbors from BOTH classes
            var allDistances = new List<(int Index, double Distance, bool IsMajority)>();

            // Distances to minority samples
            for (int j = 0; j < minorityCount; j++)
            {
                if (i != j)
                {
                    double dist = ComputeDistance(minorityData, i, minorityData, j, cols);
                    allDistances.Add((j, dist, false));
                }
            }

            // Distances to majority samples
            for (int j = 0; j < majorityCount; j++)
            {
                double dist = ComputeDistance(minorityData, i, majorityData, j, cols);
                allDistances.Add((j, dist, true));
            }

            // Get k nearest neighbors
            var kNearest = allDistances
                .OrderBy(x => x.Distance)
                .Take(KNeighbors)
                .ToList();

            // Count majority neighbors
            int majorityNeighbors = kNearest.Count(x => x.IsMajority);

            // Difficulty ratio = proportion of majority neighbors
            ratios[i] = (double)majorityNeighbors / KNeighbors;
        }

        return ratios;
    }

    /// <summary>
    /// Computes the Euclidean distance between two samples.
    /// </summary>
    /// <param name="data1">First data matrix.</param>
    /// <param name="idx1">Row index in first matrix.</param>
    /// <param name="data2">Second data matrix.</param>
    /// <param name="idx2">Row index in second matrix.</param>
    /// <param name="cols">Number of columns (features).</param>
    /// <returns>The Euclidean distance.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Euclidean distance is the "straight line" distance between
    /// two points in feature space. Closer samples are more similar.</para>
    /// </remarks>
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
    /// Computes the distance matrix for all pairs of minority samples.
    /// </summary>
    /// <param name="data">The data matrix.</param>
    /// <returns>A 2D array of pairwise distances.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This precomputes all distances between samples to avoid
    /// redundant calculations when finding nearest neighbors for each sample.</para>
    /// </remarks>
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
    /// <param name="distances">The precomputed distance matrix.</param>
    /// <param name="sampleIdx">The index of the sample.</param>
    /// <param name="k">The number of neighbors to return.</param>
    /// <returns>Array of neighbor indices.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Finds the k samples closest to the given sample based on
    /// the precomputed distances. These neighbors are used for interpolation.</para>
    /// </remarks>
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
        parameters["beta"] = Beta;
        return parameters;
    }
}
