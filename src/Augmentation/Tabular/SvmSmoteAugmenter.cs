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
        var sources = new int[numSynthetic];
        var targets = new int[numSynthetic];
        var gaps = new double[numSynthetic];

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

            sources[syntheticIndex] = svIdx;
            targets[syntheticIndex] = neighborIdx;
            gaps[syntheticIndex] = gap;

            syntheticIndex++;
        }

        return TabularEngineOps<T>.InterpolateRows(minorityData, sources, targets, gaps, numSynthetic);
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
        var engine = Engine;

        var minorityValues = Tensor<T>.FromMatrix(minorityData);

        if (majorityData is null || GetSampleCount(majorityData) == 0)
        {
            // Without majority data, use distance to centroid as proxy for "borderline"
            var centroid = engine.TensorDivideScalar(engine.ReduceSum(minorityValues, [0], true), NumOps.FromDouble(minorityCount));

            // Calculate distances to centroid
            var offsets = engine.TensorSubtract(minorityValues, engine.TensorBroadcastTo(centroid, [minorityCount, cols]));
            var centroidDistances = engine.TensorSqrt(engine.ReduceSum(engine.TensorMultiply(offsets, offsets), [1], false));

            // Select top 50% as "borderline" (furthest from centroid; ties by sample order)
            var furthestFirst = TabularEngineOps<T>.StableOrder(centroidDistances, descending: true);
            int numBorderline = Math.Max(1, minorityCount / 2);
            supportVectorIndices.AddRange(furthestFirst.Take(numBorderline));

            return supportVectorIndices;
        }

        // Train simplified linear SVM to identify support vectors
        int majorityCount = GetSampleCount(majorityData);
        int totalSamples = minorityCount + majorityCount;
        var samples = TabularEngineOps<T>.ConcatenateRows(minorityData, majorityData);

        // Initialize weights; the hyper-parameters are converted to T once
        var weights = new Vector<T>(cols);
        T bias = NumOps.Zero;
        T learningRate = NumOps.FromDouble(0.01);
        T svmC = NumOps.FromDouble(SvmC);
        T one = NumOps.One;
        T negativeOne = NumOps.Negate(NumOps.One);

        // SGD training for linear SVM: each update depends on the previous one, so samples are visited
        // in order and only the taken branch is evaluated
        for (int iter = 0; iter < MaxIterations; iter++)
        {
            for (int i = 0; i < totalSamples; i++)
            {
                T label = i < minorityCount ? one : negativeOne;
                var sample = samples.GetRow(i);
                T decision = NumOps.Add(bias, engine.DotProduct(weights, sample));

                // Hinge loss gradient: w += lr * (label * x - w / C); otherwise w -= lr * w / C
                if (NumOps.LessThan(NumOps.Multiply(label, decision), one))
                {
                    weights = engine.Add(weights, engine.Multiply(
                        engine.Subtract(engine.Multiply(sample, label), engine.Divide(weights, svmC)), learningRate));
                    bias = NumOps.Add(bias, NumOps.Multiply(learningRate, label));
                }
                else
                {
                    weights = engine.Subtract(weights, engine.Divide(engine.Multiply(weights, learningRate), svmC));
                }
            }
        }

        // Identify support vectors from minority class: margin = 1 * decision (label = 1 for minority);
        // within or slightly beyond the margin means margin <= 1.5
        var margins = engine.TensorAddScalar(engine.MatrixVectorMultiply(minorityValues.ToMatrix(), weights).AsTensor(), bias);
        var withinMargin = engine.ScalarMinusTensor(one, engine.TensorGreaterThan(margins, NumOps.FromDouble(1.5)));
        supportVectorIndices.AddRange(engine.TensorNonzero(withinMargin).ToArray());

        return supportVectorIndices;
    }

    /// <summary>
    /// Computes the distance matrix for all pairs of samples.
    /// </summary>
    private static Tensor<T> ComputeDistanceMatrix(Matrix<T> data)
    {
        return TabularEngineOps<T>.PairwiseDistances(data);
    }

    /// <summary>
    /// Gets the k nearest neighbors for a given sample.
    /// </summary>
    private static int[] GetKNearestNeighbors(Tensor<T> distances, int sampleIdx, int k)
    {
        return TabularEngineOps<T>.NearestNeighbors(distances, sampleIdx, k);
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
