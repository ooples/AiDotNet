namespace AiDotNet.ActiveLearning.Interfaces;

/// <summary>
/// Interface for batch selection strategies in active learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> When selecting multiple samples for labeling at once (batch mode),
/// we need to ensure diversity among the selected samples. Simply taking the top-N by score
/// might select very similar samples, which wastes labeling budget.</para>
///
/// <para><b>Batch Selection Approaches:</b></para>
/// <list type="bullet">
/// <item><description><b>Ranked Batch:</b> Select top-k by score, then filter for diversity</description></item>
/// <item><description><b>Clustered:</b> Cluster samples, select most informative from each cluster</description></item>
/// <item><description><b>Submodular:</b> Optimize a submodular function for diversity + informativeness</description></item>
/// </list>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("BatchStrategy")]
public interface IBatchStrategy<T, TInput, TOutput>
{
    /// <summary>
    /// Gets the name of the batch selection strategy.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Selects a diverse batch of samples from candidates.
    /// </summary>
    /// <param name="candidateIndices">Indices of candidate samples (pre-filtered by informativeness).</param>
    /// <param name="scores">Informativeness scores for the candidates.</param>
    /// <param name="unlabeledPool">The full unlabeled pool for feature access.</param>
    /// <param name="batchSize">Number of samples to select.</param>
    /// <returns>Indices of the selected samples from the original unlabeled pool.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method takes candidates that are already ranked by
    /// informativeness and selects a subset that is both informative AND diverse.</para>
    /// </remarks>
    int[] SelectBatch(
        int[] candidateIndices,
        Vector<T> scores,
        IDataset<T, TInput, TOutput> unlabeledPool,
        int batchSize);

    /// <summary>
    /// Computes pairwise diversity between samples.
    /// </summary>
    /// <param name="sample1">First sample.</param>
    /// <param name="sample2">Second sample.</param>
    /// <returns>Diversity score (higher = more diverse).</returns>
    T ComputeDiversity(TInput sample1, TInput sample2);

    /// <summary>
    /// Gets or sets the trade-off parameter between informativeness and diversity.
    /// </summary>
    /// <remarks>
    /// <para>Higher values favor diversity over informativeness.</para>
    /// </remarks>
    T DiversityTradeoff { get; set; }
}

/// <summary>
/// Interface for clustering-based batch selection.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
[AiDotNet.Configuration.YamlConfigurable("ClusteringBatchStrategy")]
public interface IClusteringBatchStrategy<T, TInput, TOutput> : IBatchStrategy<T, TInput, TOutput>
{
    /// <summary>
    /// Gets the number of clusters to use.
    /// </summary>
    int NumClusters { get; }

    /// <summary>
    /// Clusters the samples in the unlabeled pool.
    /// </summary>
    /// <param name="unlabeledPool">The pool of unlabeled samples.</param>
    /// <returns>Cluster assignments for each sample.</returns>
    int[] ClusterSamples(IDataset<T, TInput, TOutput> unlabeledPool);

    /// <summary>
    /// Selects the most informative sample from each cluster.
    /// </summary>
    /// <param name="clusterAssignments">Cluster assignments for each sample.</param>
    /// <param name="scores">Informativeness scores for each sample.</param>
    /// <param name="batchSize">Number of samples to select.</param>
    /// <returns>Indices of selected samples.</returns>
    int[] SelectFromClusters(int[] clusterAssignments, Vector<T> scores, int batchSize);
}

/// <summary>
/// Interface for submodular batch selection strategies.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Submodular functions have a "diminishing returns" property:
/// adding more similar samples provides less benefit. This naturally encourages diversity.</para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("SubmodularBatchStrategy")]
public interface ISubmodularBatchStrategy<T, TInput, TOutput> : IBatchStrategy<T, TInput, TOutput>
{
    /// <summary>
    /// Computes the marginal gain of adding a sample to the current selection.
    /// </summary>
    /// <param name="currentSelection">Currently selected sample indices.</param>
    /// <param name="candidateIndex">Index of the candidate sample to add.</param>
    /// <param name="unlabeledPool">The unlabeled pool.</param>
    /// <returns>The marginal gain of adding the candidate.</returns>
    T ComputeMarginalGain(
        IReadOnlyList<int> currentSelection,
        int candidateIndex,
        IDataset<T, TInput, TOutput> unlabeledPool);

    /// <summary>
    /// Performs greedy submodular maximization to select a batch.
    /// </summary>
    /// <param name="candidateIndices">Candidate sample indices.</param>
    /// <param name="unlabeledPool">The unlabeled pool.</param>
    /// <param name="batchSize">Number of samples to select.</param>
    /// <returns>Indices of the selected samples.</returns>
    int[] GreedyMaximization(
        int[] candidateIndices,
        IDataset<T, TInput, TOutput> unlabeledPool,
        int batchSize);
}

/// <summary>
/// Interface for gradient-based batch selection (e.g., BADGE).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> BADGE (Batch Active learning by Diverse Gradient Embeddings)
/// uses gradient embeddings to represent samples, then selects a diverse set using k-means++.</para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("GradientBatchStrategy")]
public interface IGradientBatchStrategy<T, TInput, TOutput> : IBatchStrategy<T, TInput, TOutput>
{
    /// <summary>
    /// Computes gradient embeddings for samples.
    /// </summary>
    /// <param name="model">The current model.</param>
    /// <param name="samples">The samples to embed.</param>
    /// <returns>Gradient embeddings for each sample.</returns>
    Matrix<T> ComputeGradientEmbeddings(
        IFullModel<T, TInput, TOutput> model,
        IDataset<T, TInput, TOutput> samples);

    /// <summary>
    /// Selects samples using k-means++ initialization on gradient embeddings.
    /// </summary>
    /// <param name="embeddings">Gradient embeddings for samples.</param>
    /// <param name="batchSize">Number of samples to select.</param>
    /// <returns>Indices of selected samples.</returns>
    int[] KMeansPlusPlusSelection(Matrix<T> embeddings, int batchSize);
}
