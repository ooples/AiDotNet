using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Augmentation.Tabular;

/// <summary>
/// Implements SMOTE (Synthetic Minority Over-sampling Technique) for imbalanced datasets.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> SMOTE creates new synthetic samples for the minority class
/// by interpolating between existing minority samples and their nearest neighbors.
/// This helps balance imbalanced datasets where one class has far fewer samples than others.</para>
/// <para><b>How it works:</b>
/// <list type="number">
/// <item>For each minority sample, find its k nearest neighbors (also from minority class)</item>
/// <item>Randomly select one of these neighbors</item>
/// <item>Create a new sample along the line between the original and the neighbor</item>
/// </list>
/// </para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>Classification with severe class imbalance (e.g., fraud detection, rare disease)</item>
/// <item>When the minority class has too few samples to learn from</item>
/// <item>When undersampling the majority class would lose too much information</item>
/// </list>
/// </para>
/// <para><b>When NOT to use:</b>
/// <list type="bullet">
/// <item>When classes are already balanced</item>
/// <item>For regression tasks (use other techniques)</item>
/// <item>When features are highly categorical (use SMOTE-NC instead)</item>
/// </list>
/// </para>
/// <para><b>Reference:</b> Chawla et al., "SMOTE: Synthetic Minority Over-sampling Technique" (2002)</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class SmoteAugmenter<T> : TabularAugmenterBase<T>
{
    /// <summary>
    /// Gets the number of nearest neighbors to consider.
    /// </summary>
    /// <remarks>
    /// <para>Default: 5</para>
    /// <para>Higher values create more diverse synthetic samples but require more minority samples.</para>
    /// </remarks>
    public int KNeighbors { get; }

    /// <summary>
    /// Gets the sampling ratio for synthetic sample generation.
    /// </summary>
    /// <remarks>
    /// <para>Default: 1.0 (generate as many synthetic samples as original minority samples)</para>
    /// <para>Values > 1.0 create more synthetic samples; values &lt; 1.0 create fewer.</para>
    /// </remarks>
    public double SamplingRatio { get; }

    /// <summary>
    /// Creates a new SMOTE augmenter.
    /// </summary>
    /// <param name="kNeighbors">Number of nearest neighbors to use (default: 5).</param>
    /// <param name="samplingRatio">Ratio of synthetic samples to generate (default: 1.0).</param>
    /// <param name="probability">Probability of applying this augmentation (default: 1.0).</param>
    public SmoteAugmenter(
        int kNeighbors = 5,
        double samplingRatio = 1.0,
        double probability = 1.0) : base(probability)
    {
        if (kNeighbors < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(kNeighbors), "K neighbors must be at least 1.");
        }

        if (samplingRatio <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(samplingRatio), "Sampling ratio must be positive.");
        }

        KNeighbors = kNeighbors;
        SamplingRatio = samplingRatio;
    }

    /// <inheritdoc />
    protected override Matrix<T> ApplyAugmentation(Matrix<T> data, AugmentationContext<T> context)
    {
        // SMOTE on raw data without labels - generate synthetic samples for all rows
        return GenerateSyntheticSamples(data, context);
    }

    /// <summary>
    /// Applies SMOTE to generate synthetic samples for the minority class.
    /// </summary>
    /// <param name="minorityData">Matrix containing only minority class samples.</param>
    /// <param name="context">The augmentation context.</param>
    /// <returns>Matrix containing synthetic samples (original data is NOT included).</returns>
    public Matrix<T> GenerateSyntheticSamples(Matrix<T> minorityData, AugmentationContext<T> context)
    {
        int rows = GetSampleCount(minorityData);
        int cols = GetFeatureCount(minorityData);

        if (rows < 2)
        {
            // Need at least 2 samples to interpolate
            return new Matrix<T>(0, cols);
        }

        // Adjust k if we don't have enough samples
        int effectiveK = Math.Min(KNeighbors, rows - 1);

        // Calculate how many synthetic samples to generate
        int numSynthetic = (int)Math.Ceiling(rows * SamplingRatio);

        var syntheticSamples = new Matrix<T>(numSynthetic, cols);
        int syntheticIndex = 0;

        // Compute pairwise distances for k-NN
        var distances = ComputeDistanceMatrix(minorityData);

        while (syntheticIndex < numSynthetic)
        {
            // Randomly select a minority sample
            int sampleIdx = context.Random.Next(rows);

            // Find k nearest neighbors
            var neighbors = GetKNearestNeighbors(distances, sampleIdx, effectiveK);

            // Randomly select one neighbor
            int neighborIdx = neighbors[context.Random.Next(neighbors.Length)];

            // Generate synthetic sample by interpolating
            double gap = context.Random.NextDouble();

            for (int c = 0; c < cols; c++)
            {
                double val1 = NumOps.ToDouble(minorityData[sampleIdx, c]);
                double val2 = NumOps.ToDouble(minorityData[neighborIdx, c]);
                double synthetic = val1 + gap * (val2 - val1);
                syntheticSamples[syntheticIndex, c] = NumOps.FromDouble(synthetic);
            }

            syntheticIndex++;
        }

        return syntheticSamples;
    }

    /// <summary>
    /// Applies SMOTE and returns combined original and synthetic data.
    /// </summary>
    /// <param name="minorityData">Matrix containing only minority class samples.</param>
    /// <param name="minorityLabels">Labels for the minority class.</param>
    /// <param name="context">The augmentation context.</param>
    /// <returns>Tuple of (combined data, combined labels) including both original and synthetic samples.</returns>
    public (Matrix<T> Data, Vector<T> Labels) ApplySmoteWithLabels(
        Matrix<T> minorityData,
        Vector<T> minorityLabels,
        AugmentationContext<T> context)
    {
        var syntheticData = GenerateSyntheticSamples(minorityData, context);
        int numSynthetic = GetSampleCount(syntheticData);

        if (numSynthetic == 0)
        {
            return (minorityData.Clone(), minorityLabels.Clone());
        }

        int originalRows = GetSampleCount(minorityData);
        int cols = GetFeatureCount(minorityData);

        // Combine original and synthetic data
        var combinedData = new Matrix<T>(originalRows + numSynthetic, cols);
        var combinedLabels = new Vector<T>(originalRows + numSynthetic);

        // Copy original data
        for (int i = 0; i < originalRows; i++)
        {
            for (int c = 0; c < cols; c++)
            {
                combinedData[i, c] = minorityData[i, c];
            }
            combinedLabels[i] = minorityLabels[i];
        }

        // Copy synthetic data with same label as minority class
        T minorityLabel = minorityLabels[0];

        // Validate that all minority labels have the same value
        for (int i = 1; i < minorityLabels.Length; i++)
        {
            if (!NumOps.ToDouble(minorityLabels[i]).Equals(NumOps.ToDouble(minorityLabel)))
            {
                throw new ArgumentException("All minority labels must have the same value.", nameof(minorityLabels));
            }
        }

        for (int i = 0; i < numSynthetic; i++)
        {
            for (int c = 0; c < cols; c++)
            {
                combinedData[originalRows + i, c] = syntheticData[i, c];
            }
            combinedLabels[originalRows + i] = minorityLabel;
        }

        return (combinedData, combinedLabels);
    }

    private double[,] ComputeDistanceMatrix(Matrix<T> data)
    {
        int rows = GetSampleCount(data);
        int cols = GetFeatureCount(data);
        var distances = new double[rows, rows];

        for (int i = 0; i < rows; i++)
        {
            for (int j = i + 1; j < rows; j++)
            {
                double dist = 0;
                for (int c = 0; c < cols; c++)
                {
                    double diff = NumOps.ToDouble(data[i, c]) - NumOps.ToDouble(data[j, c]);
                    dist += diff * diff;
                }
                dist = Math.Sqrt(dist);
                distances[i, j] = dist;
                distances[j, i] = dist;
            }
        }

        return distances;
    }

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
        parameters["samplingRatio"] = SamplingRatio;
        return parameters;
    }
}
