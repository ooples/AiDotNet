using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Augmentation.Tabular;

/// <summary>
/// Implements SVM-SMOTE for imbalanced datasets, using SVM decision boundary to identify borderline samples.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> SVM-SMOTE is an enhancement over standard SMOTE that uses Support Vector Machine
/// classification to identify the most informative samples near the decision boundary. It generates synthetic
/// samples from support vectors (the most difficult cases), focusing oversampling effort where it matters most.</para>
///
/// <para><b>How it works:</b>
/// <list type="number">
/// <item>Train an SVM classifier on the data to identify support vectors</item>
/// <item>Identify minority samples that are support vectors (near the decision boundary)</item>
/// <item>Generate synthetic samples by interpolating between support vector minority samples and their neighbors</item>
/// </list>
/// </para>
///
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>When the decision boundary is critical for classification</item>
/// <item>When you want synthetic samples focused on difficult cases</item>
/// <item>When standard SMOTE generates too many samples in easy regions</item>
/// </list>
/// </para>
///
/// <para><b>Reference:</b> Nguyen et al., "Borderline Over-Sampling for Imbalanced Data Classification" (2011)</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class SvmSmoteAugmenter<T> : TabularAugmenterBase<T>
{
    /// <summary>
    /// Gets the number of nearest neighbors to use for interpolation.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This controls how many neighbors are considered when
    /// creating synthetic samples. More neighbors = more diversity but potentially more noise.</para>
    /// </remarks>
    public int KNeighbors { get; }

    /// <summary>
    /// Gets the sampling ratio for synthetic sample generation.
    /// </summary>
    public double SamplingRatio { get; }

    /// <summary>
    /// Gets the SVM regularization parameter (C).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The C parameter controls how strictly the SVM tries to
    /// classify all training points correctly. Higher C = stricter boundary, more support vectors.</para>
    /// </remarks>
    public double SvmC { get; }

    /// <summary>
    /// Gets the maximum iterations for SVM training.
    /// </summary>
    public int MaxIterations { get; }

    /// <summary>
    /// Creates a new SVM-SMOTE augmenter.
    /// </summary>
    /// <param name="kNeighbors">Number of neighbors for interpolation (default: 5).</param>
    /// <param name="samplingRatio">Ratio of synthetic samples to generate (default: 1.0).</param>
    /// <param name="svmC">SVM regularization parameter (default: 1.0).</param>
    /// <param name="maxIterations">Maximum SVM training iterations (default: 100).</param>
    /// <param name="probability">Probability of applying this augmentation (default: 1.0).</param>
    public SvmSmoteAugmenter(
        int kNeighbors = 5,
        double samplingRatio = 1.0,
        double svmC = 1.0,
        int maxIterations = 100,
        double probability = 1.0) : base(probability)
    {
        if (kNeighbors < 1)
            throw new ArgumentOutOfRangeException(nameof(kNeighbors), "K neighbors must be at least 1.");
        if (samplingRatio <= 0)
            throw new ArgumentOutOfRangeException(nameof(samplingRatio), "Sampling ratio must be positive.");
        if (svmC <= 0)
            throw new ArgumentOutOfRangeException(nameof(svmC), "SVM C parameter must be positive.");

        KNeighbors = kNeighbors;
        SamplingRatio = samplingRatio;
        SvmC = svmC;
        MaxIterations = maxIterations;
    }

    /// <inheritdoc />
    protected override Matrix<T> ApplyAugmentation(Matrix<T> data, AugmentationContext<T> context)
    {
        return GenerateSyntheticSamples(data, null, context);
    }

    /// <summary>
    /// Applies SVM-SMOTE to generate synthetic samples for the minority class.
    /// </summary>
    /// <param name="minorityData">Matrix containing only minority class samples.</param>
    /// <param name="majorityData">Matrix containing only majority class samples.</param>
    /// <param name="context">The augmentation context.</param>
    /// <returns>Matrix containing synthetic samples (original data is NOT included).</returns>
    public Matrix<T> GenerateSyntheticSamples(
        Matrix<T> minorityData,
        Matrix<T>? majorityData,
        AugmentationContext<T> context)
    {
        int minorityCount = GetSampleCount(minorityData);
        int cols = GetFeatureCount(minorityData);

        if (minorityCount < 2)
            return new Matrix<T>(0, cols);

        // Identify support vector indices using simplified SVM
        var supportVectorIndices = IdentifySupportVectors(minorityData, majorityData, context);

        if (supportVectorIndices.Count == 0)
        {
            // Fall back to all minority samples if no support vectors found
            for (int i = 0; i < minorityCount; i++)
                supportVectorIndices.Add(i);
        }

        // Calculate how many synthetic samples to generate
        int numSynthetic = (int)Math.Ceiling(supportVectorIndices.Count * SamplingRatio);
        var syntheticSamples = new Matrix<T>(numSynthetic, cols);

        // Precompute minority distances for neighbor finding
        var minorityDistances = ComputeDistanceMatrix(minorityData);
        int effectiveK = Math.Min(KNeighbors, minorityCount - 1);

        int syntheticIndex = 0;
        while (syntheticIndex < numSynthetic)
        {
            // Select a random support vector sample
            int svIdx = supportVectorIndices[context.Random.Next(supportVectorIndices.Count)];

            // Find k nearest minority neighbors
            var neighbors = GetKNearestNeighbors(minorityDistances, svIdx, effectiveK);
            int neighborIdx = neighbors[context.Random.Next(neighbors.Length)];

            // Generate synthetic sample by interpolating
            double gap = context.Random.NextDouble();

            for (int c = 0; c < cols; c++)
            {
                double val1 = NumOps.ToDouble(minorityData[svIdx, c]);
                double val2 = NumOps.ToDouble(minorityData[neighborIdx, c]);
                double synthetic = val1 + gap * (val2 - val1);
                syntheticSamples[syntheticIndex, c] = NumOps.FromDouble(synthetic);
            }

            syntheticIndex++;
        }

        return syntheticSamples;
    }

    /// <summary>
    /// Identifies support vectors using a simplified linear SVM approach.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Support vectors are the data points closest to the decision
    /// boundary. They're the "hardest" cases and define where the boundary should be.
    /// We identify them by checking which points have small margins after SVM training.</para>
    /// </remarks>
    private List<int> IdentifySupportVectors(
        Matrix<T> minorityData,
        Matrix<T>? majorityData,
        AugmentationContext<T> context)
    {
        var supportVectorIndices = new List<int>();
        int minorityCount = GetSampleCount(minorityData);
        int cols = GetFeatureCount(minorityData);

        if (majorityData is null || GetSampleCount(majorityData) == 0)
        {
            // Without majority data, use distance to centroid as proxy for "borderline"
            var centroid = new double[cols];
            for (int i = 0; i < minorityCount; i++)
            {
                for (int c = 0; c < cols; c++)
                    centroid[c] += NumOps.ToDouble(minorityData[i, c]);
            }
            for (int c = 0; c < cols; c++)
                centroid[c] /= minorityCount;

            // Calculate distances to centroid
            var distances = new List<(int Index, double Distance)>();
            for (int i = 0; i < minorityCount; i++)
            {
                double dist = 0;
                for (int c = 0; c < cols; c++)
                {
                    double diff = NumOps.ToDouble(minorityData[i, c]) - centroid[c];
                    dist += diff * diff;
                }
                distances.Add((i, Math.Sqrt(dist)));
            }

            // Select top 50% as "borderline" (furthest from centroid)
            distances.Sort((a, b) => b.Distance.CompareTo(a.Distance));
            int numBorderline = Math.Max(1, minorityCount / 2);
            for (int i = 0; i < numBorderline; i++)
                supportVectorIndices.Add(distances[i].Index);

            return supportVectorIndices;
        }

        // Train simplified linear SVM to identify support vectors
        int majorityCount = GetSampleCount(majorityData);
        int totalSamples = minorityCount + majorityCount;

        // Initialize weights
        var weights = new double[cols];
        double bias = 0;
        double learningRate = 0.01;

        // SGD training for linear SVM
        for (int iter = 0; iter < MaxIterations; iter++)
        {
            for (int i = 0; i < totalSamples; i++)
            {
                int actualIdx;
                int label;
                Matrix<T> data;

                if (i < minorityCount)
                {
                    actualIdx = i;
                    label = 1;
                    data = minorityData;
                }
                else
                {
                    actualIdx = i - minorityCount;
                    label = -1;
                    data = majorityData;
                }

                // Compute decision value
                double decision = bias;
                for (int c = 0; c < cols; c++)
                    decision += weights[c] * NumOps.ToDouble(data[actualIdx, c]);

                // Hinge loss gradient
                if (label * decision < 1)
                {
                    for (int c = 0; c < cols; c++)
                        weights[c] += learningRate * (label * NumOps.ToDouble(data[actualIdx, c]) - weights[c] / SvmC);
                    bias += learningRate * label;
                }
                else
                {
                    for (int c = 0; c < cols; c++)
                        weights[c] -= learningRate * weights[c] / SvmC;
                }
            }
        }

        // Identify support vectors from minority class (points with margin < 1)
        for (int i = 0; i < minorityCount; i++)
        {
            double decision = bias;
            for (int c = 0; c < cols; c++)
                decision += weights[c] * NumOps.ToDouble(minorityData[i, c]);

            // Support vectors have margin close to 1 or violate the margin
            double margin = 1 * decision; // label = 1 for minority
            if (margin <= 1.5) // Within or slightly beyond margin
                supportVectorIndices.Add(i);
        }

        return supportVectorIndices;
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
                indexedDistances.Add((i, distances[sampleIdx, i]));
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
        parameters["svmC"] = SvmC;
        parameters["maxIterations"] = MaxIterations;
        return parameters;
    }
}
