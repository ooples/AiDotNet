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

        var sources = new int[numSynthetic];
        var targets = new int[numSynthetic];
        var gaps = new Vector<T>(numSynthetic);
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

            sources[syntheticIndex] = sampleIdx;
            targets[syntheticIndex] = neighborIdx;
            gaps[syntheticIndex] = NumOps.FromDouble(gap);

            syntheticIndex++;
        }

        return TabularEngineOps<T>.InterpolateRows(minorityData, sources, targets, gaps, numSynthetic);
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

        // Copy synthetic data with same label as minority class
        T minorityLabel = minorityLabels[0];

        // Validate that all minority labels have the same value
        var engine = Engine;
        var labelTensor = minorityLabels.AsTensor();
        int matching = (int)NumOps.ToDouble(engine.TensorSum(engine.TensorEquals(labelTensor, minorityLabel)));
        if (matching != minorityLabels.Length)
        {
            throw new ArgumentException("All minority labels must have the same value.", nameof(minorityLabels));
        }

        // Combine original and synthetic data
        var combinedData = TabularEngineOps<T>.ConcatenateRows(minorityData, syntheticData);
        var combinedLabels = TabularEngineOps<T>.ConcatenateLabels(
            minorityLabels, originalRows, TabularEngineOps<T>.Filled(numSynthetic, minorityLabel), numSynthetic);

        return (combinedData, combinedLabels);
    }

    private static Tensor<T> ComputeDistanceMatrix(Matrix<T> data)
    {
        return TabularEngineOps<T>.PairwiseDistances(data);
    }

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
        return parameters;
    }
}
