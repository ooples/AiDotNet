using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DataPreparation.Splitting.Specialized;

/// <summary>
/// Cluster-based splitter that divides data by similarity clusters.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This splitter groups similar samples together using clustering,
/// then assigns entire clusters to train or test. This ensures the test set contains
/// samples that are meaningfully different from training.
/// </para>
/// <para>
/// <b>When to Use:</b>
/// - When you want to test generalization to truly different data
/// - To avoid having very similar samples in both train and test
/// - For harder, more realistic model evaluation
/// </para>
/// <para>
/// <b>Note:</b> This splitter expects cluster assignments to be provided.
/// Run clustering (K-means, DBSCAN, etc.) beforehand and pass the cluster labels.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class ClusterBasedSplitter<T> : DataSplitterBase<T>
{
    private readonly int[] _clusterLabels;
    private readonly double _testSize;

    /// <summary>
    /// Creates a cluster-based splitter.
    /// </summary>
    /// <param name="clusterLabels">Cluster assignment for each sample.</param>
    /// <param name="testSize">Proportion of clusters to assign to test. Default is 0.2 (20%).</param>
    /// <param name="shuffle">Whether to shuffle clusters before assignment. Default is true.</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public ClusterBasedSplitter(int[] clusterLabels, double testSize = 0.2, bool shuffle = true, int randomSeed = 42)
        : base(shuffle, randomSeed)
    {
        if (clusterLabels is null || clusterLabels.Length == 0)
        {
            throw new ArgumentNullException(nameof(clusterLabels), "Cluster labels cannot be null or empty.");
        }

        if (testSize <= 0 || testSize >= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(testSize), "Test size must be between 0 and 1.");
        }

        _clusterLabels = clusterLabels;
        _testSize = testSize;
    }

    /// <inheritdoc/>
    public override string Description => $"Cluster-Based split ({_testSize * 100:F0}% test clusters)";

    /// <inheritdoc/>
    public override DataSplitResult<T> Split(Matrix<T> X, Vector<T>? y = null)
    {
        ValidateInputs(X, y);

        if (X.Rows != _clusterLabels.Length)
        {
            throw new ArgumentException(
                $"Cluster labels length ({_clusterLabels.Length}) must match number of samples ({X.Rows}).");
        }

        // Group samples by cluster
        var clusterGroups = new Dictionary<int, List<int>>();
        for (int i = 0; i < _clusterLabels.Length; i++)
        {
            if (!clusterGroups.TryGetValue(_clusterLabels[i], out var list))
            {
                list = new List<int>();
                clusterGroups[_clusterLabels[i]] = list;
            }
            list.Add(i);
        }

        var clusterIds = clusterGroups.Keys.ToArray();
        if (_shuffle)
        {
            ShuffleIndices(clusterIds);
        }

        int numTestClusters = Math.Max(1, (int)(clusterIds.Length * _testSize));

        var testClusters = new HashSet<int>(clusterIds.Take(numTestClusters));

        var trainIndices = new List<int>();
        var testIndices = new List<int>();

        for (int i = 0; i < _clusterLabels.Length; i++)
        {
            if (testClusters.Contains(_clusterLabels[i]))
            {
                testIndices.Add(i);
            }
            else
            {
                trainIndices.Add(i);
            }
        }

        return BuildResult(X, y, trainIndices.ToArray(), testIndices.ToArray());
    }
}
