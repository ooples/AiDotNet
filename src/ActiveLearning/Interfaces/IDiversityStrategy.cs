using AiDotNet.Tensors.Engines;

namespace AiDotNet.ActiveLearning.Interfaces;

/// <summary>
/// Interface for diversity-based sampling strategies in active learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input type for samples.</typeparam>
/// <typeparam name="TOutput">The output type for samples.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Diversity strategies select samples that are diverse or
/// representative of the data distribution. This helps ensure the model learns from
/// a variety of examples rather than similar ones.</para>
///
/// <para><b>Common Diversity Strategies:</b></para>
/// <list type="bullet">
/// <item><description><b>K-Center/CoreSet:</b> Selects samples that maximize coverage of the feature space</description></item>
/// <item><description><b>K-Means:</b> Clusters samples and selects from each cluster</description></item>
/// <item><description><b>Random:</b> Simple random sampling as a baseline</description></item>
/// <item><description><b>Density-Weighted:</b> Samples from high-density regions</description></item>
/// </list>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("DiversityStrategy")]
public interface IDiversityStrategy<T, TInput, TOutput>
{
    /// <summary>
    /// Gets the name of the diversity strategy.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Selects diverse samples from the unlabeled pool.
    /// </summary>
    /// <param name="unlabeledData">The pool of unlabeled samples to select from.</param>
    /// <param name="labeledData">The currently labeled samples (for comparison).</param>
    /// <param name="numSamples">The number of samples to select.</param>
    /// <returns>Indices of the selected samples from the unlabeled pool.</returns>
    int[] SelectDiverseSamples(
        IDataset<T, TInput, TOutput> unlabeledData,
        IDataset<T, TInput, TOutput>? labeledData,
        int numSamples);

    /// <summary>
    /// Computes diversity scores for samples in the unlabeled pool.
    /// </summary>
    /// <param name="unlabeledData">The pool of unlabeled samples.</param>
    /// <param name="labeledData">The currently labeled samples (for comparison).</param>
    /// <returns>Diversity score for each sample (higher = more diverse).</returns>
    Vector<T> ComputeDiversityScores(
        IDataset<T, TInput, TOutput> unlabeledData,
        IDataset<T, TInput, TOutput>? labeledData);

    /// <summary>
    /// Computes pairwise distances between samples.
    /// </summary>
    /// <param name="dataset">The dataset containing samples.</param>
    /// <returns>Distance matrix where element [i,j] is the distance between samples i and j.</returns>
    Matrix<T> ComputeDistanceMatrix(IDataset<T, TInput, TOutput> dataset);

    /// <summary>
    /// Gets the feature representation for a sample.
    /// </summary>
    /// <param name="input">The input sample.</param>
    /// <returns>Feature vector representation of the sample.</returns>
    Vector<T> GetFeatureRepresentation(TInput input);
}

/// <summary>
/// Base class for diversity strategies with common functionality.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input type for samples.</typeparam>
/// <typeparam name="TOutput">The output type for samples.</typeparam>
public abstract class DiversityStrategyBase<T, TInput, TOutput> : IDiversityStrategy<T, TInput, TOutput>
{
    /// <summary>
    /// Gets the hardware-accelerated computation engine for vectorized operations.
    /// </summary>
    protected IEngine Engine => AiDotNetEngine.Current;

    /// <summary>
    /// Gets numeric operations helper.
    /// </summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <inheritdoc />
    public abstract string Name { get; }

    /// <inheritdoc />
    public abstract int[] SelectDiverseSamples(
        IDataset<T, TInput, TOutput> unlabeledData,
        IDataset<T, TInput, TOutput>? labeledData,
        int numSamples);

    /// <inheritdoc />
    public abstract Vector<T> ComputeDiversityScores(
        IDataset<T, TInput, TOutput> unlabeledData,
        IDataset<T, TInput, TOutput>? labeledData);

    /// <inheritdoc />
    /// <remarks>
    /// Features are extracted once per sample. When <see cref="ComputeDistance"/> is not overridden
    /// and every sample has the same feature length, the Euclidean distances are computed blockwise
    /// with engine ops (exact differences, no Gram-matrix shortcut). Otherwise each pair goes through
    /// <see cref="ComputeDistance"/>, so subclasses that override it keep their metric.
    /// </remarks>
    public virtual Matrix<T> ComputeDistanceMatrix(IDataset<T, TInput, TOutput> dataset)
    {
        int n = dataset.Count;
        var features = new List<Vector<T>>(n);
        for (int i = 0; i < n; i++)
        {
            features.Add(GetFeatureRepresentation(dataset.GetInput(i)));
        }

        if (n == 0)
        {
            return new Matrix<T>(0, 0);
        }

        int d = features[0].Length;
        if (!OverridesComputeDistance() && features.TrueForAll(f => f.Length == d))
        {
            return ComputeUniformDistanceMatrix(features, d);
        }

        var distances = new Matrix<T>(n, n);
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                var dist = ComputeDistance(features[i], features[j]);
                distances[i, j] = dist;
                distances[j, i] = dist;
            }
        }

        return distances;
    }

    /// <summary>
    /// True when the runtime type overrides <see cref="ComputeDistance"/>, in which case the batched
    /// Euclidean matrix would not match the subclass's metric.
    /// </summary>
    private bool OverridesComputeDistance()
    {
        var method = GetType().GetMethod(
            nameof(ComputeDistance),
            System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.NonPublic,
            null,
            new[] { typeof(Vector<T>), typeof(Vector<T>) },
            null);
        return method is not null && method.DeclaringType != typeof(DiversityStrategyBase<T, TInput, TOutput>);
    }

    /// <summary>
    /// Pairwise Euclidean distances between equal-length feature vectors, computed blockwise.
    /// </summary>
    private Matrix<T> ComputeUniformDistanceMatrix(List<Vector<T>> features, int featureSize)
    {
        int n = features.Count;
        var rows = new Tensor<T>[n];
        for (int i = 0; i < n; i++)
        {
            rows[i] = Tensor<T>.FromVector(features[i]);
        }

        var stacked = Engine.TensorStack(rows, 0);
        var blockRows = AiDotNet.ActiveLearning.ActiveLearningEngineOps.PairwiseBlockRows(n, featureSize);
        var blocks = new List<Tensor<T>>();
        for (int start = 0; start < n; start += blockRows)
        {
            int count = Math.Min(blockRows, n - start);
            blocks.Add(AiDotNet.ActiveLearning.ActiveLearningEngineOps.DistanceBlock(stacked, start, count));
        }

        var all = Engine.TensorConcatenate(blocks.ToArray(), 0);
        return Matrix<T>.FromMemory(all.ToArray(), n, n);
    }

    /// <inheritdoc />
    public abstract Vector<T> GetFeatureRepresentation(TInput input);

    /// <summary>
    /// Computes Euclidean distance between two feature vectors.
    /// </summary>
    protected virtual T ComputeDistance(Vector<T> a, Vector<T> b)
    {
        int minLen = Math.Min(a.Length, b.Length);
        if (minLen == 0)
        {
            return NumOps.Zero;
        }

        var diff = Engine.Subtract(a.GetSubVector(0, minLen), b.GetSubVector(0, minLen));
        return NumOps.Sqrt(Engine.DotProduct(diff, diff));
    }
}
