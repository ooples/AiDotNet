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
    public virtual Matrix<T> ComputeDistanceMatrix(IDataset<T, TInput, TOutput> dataset)
    {
        int n = dataset.Count;
        var distances = new Matrix<T>(n, n);

        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                var fi = GetFeatureRepresentation(dataset.GetInput(i));
                var fj = GetFeatureRepresentation(dataset.GetInput(j));
                var dist = ComputeDistance(fi, fj);
                distances[i, j] = dist;
                distances[j, i] = dist;
            }
        }

        return distances;
    }

    /// <inheritdoc />
    public abstract Vector<T> GetFeatureRepresentation(TInput input);

    /// <summary>
    /// Computes Euclidean distance between two feature vectors.
    /// </summary>
    protected virtual T ComputeDistance(Vector<T> a, Vector<T> b)
    {
        T sum = NumOps.Zero;
        int minLen = Math.Min(a.Length, b.Length);

        for (int i = 0; i < minLen; i++)
        {
            var diff = NumOps.Subtract(a[i], b[i]);
            sum = NumOps.Add(sum, NumOps.Multiply(diff, diff));
        }

        return NumOps.Sqrt(sum);
    }
}
