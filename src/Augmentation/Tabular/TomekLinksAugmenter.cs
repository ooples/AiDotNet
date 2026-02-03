using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Augmentation.Tabular;

/// <summary>
/// Implements Tomek Links removal for cleaning decision boundaries in imbalanced datasets.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> A Tomek link is a pair of samples from different classes that
/// are each other's nearest neighbor. These pairs are typically noisy or borderline samples.
/// Removing Tomek links cleans the decision boundary, making classification easier.</para>
///
/// <para><b>What is a Tomek Link:</b>
/// Two samples (a, b) form a Tomek link if:
/// <list type="number">
/// <item>a and b belong to different classes</item>
/// <item>a is b's nearest neighbor</item>
/// <item>b is a's nearest neighbor</item>
/// </list>
/// These samples are ambiguous because they're very close to samples of the opposite class.
/// </para>
///
/// <para><b>Removal Strategies:</b>
/// <list type="bullet">
/// <item>RemoveBoth: Remove both samples in the Tomek link (aggressive cleaning)</item>
/// <item>RemoveMajority: Only remove the majority class sample (preserves minority)</item>
/// <item>RemoveMinority: Only remove the minority class sample (rarely used)</item>
/// </list>
/// </para>
///
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>After SMOTE to clean noisy synthetic samples (SMOTE-Tomek)</item>
/// <item>Before training to clean overlapping regions</item>
/// <item>When precision is more important than recall</item>
/// </list>
/// </para>
///
/// <para><b>Reference:</b> Tomek, "Two Modifications of CNN" (1976)</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class TomekLinksAugmenter<T> : TabularAugmenterBase<T>
{
    /// <summary>
    /// Strategy for removing samples in Tomek links.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Determines which sample(s) to remove when a Tomek link is found.</para>
    /// </remarks>
    public enum RemovalStrategy
    {
        /// <summary>Remove both samples in the Tomek link.</summary>
        RemoveBoth,
        /// <summary>Only remove the majority class sample (recommended for imbalanced data).</summary>
        RemoveMajority,
        /// <summary>Only remove the minority class sample (rarely used).</summary>
        RemoveMinority
    }

    /// <summary>
    /// Gets the removal strategy to use.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> RemoveMajority is recommended because it cleans the boundary
    /// while preserving precious minority class samples.</para>
    /// <para>Default: RemoveMajority</para>
    /// </remarks>
    public RemovalStrategy Strategy { get; }

    /// <summary>
    /// Creates a new Tomek Links augmenter.
    /// </summary>
    /// <param name="strategy">The removal strategy (default: RemoveMajority).</param>
    /// <param name="probability">Probability of applying this augmentation (default: 1.0).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use RemoveMajority for imbalanced datasets to avoid losing
    /// minority samples. Use RemoveBoth for more aggressive boundary cleaning.</para>
    /// </remarks>
    public TomekLinksAugmenter(
        RemovalStrategy strategy = RemovalStrategy.RemoveMajority,
        double probability = 1.0) : base(probability)
    {
        Strategy = strategy;
    }

    /// <inheritdoc />
    protected override Matrix<T> ApplyAugmentation(Matrix<T> data, AugmentationContext<T> context)
    {
        // Without labels, cannot identify Tomek links - return unchanged
        return data;
    }

    /// <summary>
    /// Removes samples that form Tomek links between classes.
    /// </summary>
    /// <param name="data">The full dataset with both classes.</param>
    /// <param name="labels">Class labels for each sample.</param>
    /// <param name="minorityLabel">The label value for the minority class.</param>
    /// <returns>Tuple of (cleaned data, cleaned labels) with Tomek links removed.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method identifies all Tomek links in the dataset and
    /// removes samples according to the configured strategy. The result has a cleaner boundary
    /// between classes.</para>
    /// </remarks>
    public (Matrix<T> Data, Vector<T> Labels) RemoveTomekLinks(
        Matrix<T> data,
        Vector<T> labels,
        T minorityLabel)
    {
        int rows = GetSampleCount(data);
        int cols = GetFeatureCount(data);

        if (rows < 2)
        {
            return (data.Clone(), labels.Clone());
        }

        // Find Tomek links
        var tomekIndices = FindTomekLinkIndices(data, labels, minorityLabel);

        if (tomekIndices.Count == 0)
        {
            return (data.Clone(), labels.Clone());
        }

        // Create mask of samples to keep
        var keepMask = new bool[rows];
        for (int i = 0; i < rows; i++)
        {
            keepMask[i] = true;
        }

        foreach (int idx in tomekIndices)
        {
            keepMask[idx] = false;
        }

        // Count samples to keep
        int keepCount = keepMask.Count(x => x);

        // Create filtered data
        var filteredData = new Matrix<T>(keepCount, cols);
        var filteredLabels = new Vector<T>(keepCount);

        int newIdx = 0;
        for (int i = 0; i < rows; i++)
        {
            if (keepMask[i])
            {
                for (int c = 0; c < cols; c++)
                {
                    filteredData[newIdx, c] = data[i, c];
                }
                filteredLabels[newIdx] = labels[i];
                newIdx++;
            }
        }

        return (filteredData, filteredLabels);
    }

    /// <summary>
    /// Finds indices of samples that form Tomek links.
    /// </summary>
    /// <param name="data">The full dataset.</param>
    /// <param name="labels">Class labels.</param>
    /// <param name="minorityLabel">The minority class label.</param>
    /// <returns>Set of indices to remove based on the strategy.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This scans all pairs to find mutual nearest neighbors of
    /// different classes. These form Tomek links, and some or all are marked for removal.</para>
    /// </remarks>
    private HashSet<int> FindTomekLinkIndices(Matrix<T> data, Vector<T> labels, T minorityLabel)
    {
        int rows = GetSampleCount(data);
        var indicesToRemove = new HashSet<int>();

        // Find nearest neighbor for each sample
        var nearestNeighbors = new int[rows];
        for (int i = 0; i < rows; i++)
        {
            nearestNeighbors[i] = FindNearestNeighbor(data, i);
        }

        // Find Tomek links
        for (int i = 0; i < rows; i++)
        {
            int nn = nearestNeighbors[i];
            if (nn < 0) continue;

            // Check if they are mutual nearest neighbors
            if (nearestNeighbors[nn] == i)
            {
                // Check if they belong to different classes
                bool iIsMinority = NumOps.ToDouble(labels[i]).Equals(NumOps.ToDouble(minorityLabel));
                bool nnIsMinority = NumOps.ToDouble(labels[nn]).Equals(NumOps.ToDouble(minorityLabel));

                if (iIsMinority != nnIsMinority)
                {
                    // This is a Tomek link!
                    switch (Strategy)
                    {
                        case RemovalStrategy.RemoveBoth:
                            indicesToRemove.Add(i);
                            indicesToRemove.Add(nn);
                            break;
                        case RemovalStrategy.RemoveMajority:
                            if (!iIsMinority)
                                indicesToRemove.Add(i);
                            else
                                indicesToRemove.Add(nn);
                            break;
                        case RemovalStrategy.RemoveMinority:
                            if (iIsMinority)
                                indicesToRemove.Add(i);
                            else
                                indicesToRemove.Add(nn);
                            break;
                    }
                }
            }
        }

        return indicesToRemove;
    }

    /// <summary>
    /// Finds the nearest neighbor for a given sample.
    /// </summary>
    /// <param name="data">The data matrix.</param>
    /// <param name="sampleIdx">The index of the sample.</param>
    /// <returns>Index of the nearest neighbor, or -1 if none found.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Finds the single closest sample to the given sample using
    /// Euclidean distance.</para>
    /// </remarks>
    private int FindNearestNeighbor(Matrix<T> data, int sampleIdx)
    {
        int rows = GetSampleCount(data);
        int cols = GetFeatureCount(data);

        double minDist = double.MaxValue;
        int nearestIdx = -1;

        for (int i = 0; i < rows; i++)
        {
            if (i == sampleIdx) continue;

            double dist = 0;
            for (int c = 0; c < cols; c++)
            {
                double diff = NumOps.ToDouble(data[sampleIdx, c]) - NumOps.ToDouble(data[i, c]);
                dist += diff * diff;
            }
            dist = Math.Sqrt(dist);

            if (dist < minDist)
            {
                minDist = dist;
                nearestIdx = i;
            }
        }

        return nearestIdx;
    }

    /// <summary>
    /// Gets the number of Tomek links in the dataset.
    /// </summary>
    /// <param name="data">The full dataset.</param>
    /// <param name="labels">Class labels.</param>
    /// <param name="minorityLabel">The minority class label.</param>
    /// <returns>Number of Tomek links found.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this to check how many Tomek links exist before deciding
    /// whether to apply cleaning. A high count suggests significant class overlap.</para>
    /// </remarks>
    public int CountTomekLinks(Matrix<T> data, Vector<T> labels, T minorityLabel)
    {
        int rows = GetSampleCount(data);
        var tomekLinks = new HashSet<(int, int)>();

        // Find nearest neighbor for each sample
        var nearestNeighbors = new int[rows];
        for (int i = 0; i < rows; i++)
        {
            nearestNeighbors[i] = FindNearestNeighbor(data, i);
        }

        // Find Tomek links
        for (int i = 0; i < rows; i++)
        {
            int nn = nearestNeighbors[i];
            if (nn < 0) continue;

            // Check if mutual nearest neighbors
            if (nearestNeighbors[nn] == i && i < nn)  // i < nn to avoid counting twice
            {
                bool iIsMinority = NumOps.ToDouble(labels[i]).Equals(NumOps.ToDouble(minorityLabel));
                bool nnIsMinority = NumOps.ToDouble(labels[nn]).Equals(NumOps.ToDouble(minorityLabel));

                if (iIsMinority != nnIsMinority)
                {
                    tomekLinks.Add((i, nn));
                }
            }
        }

        return tomekLinks.Count;
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var parameters = base.GetParameters();
        parameters["strategy"] = Strategy.ToString();
        return parameters;
    }
}
