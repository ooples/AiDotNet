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

        // Keep every other sample, in the original order.
        var keep = Enumerable.Range(0, rows).Where(i => !tomekIndices.Contains(i)).ToArray();
        return TabularResamplingHelper<T>.Select(data, labels, keep);
    }

    /// <summary>
    /// Finds indices of samples that form Tomek links.
    /// </summary>
    /// <param name="data">The full dataset.</param>
    /// <param name="labels">Class labels.</param>
    /// <param name="minorityLabel">The minority class label.</param>
    /// <returns>Set of indices to remove based on the strategy.</returns>
    /// <remarks>
    /// <para>
    /// A Tomek link is a pair of mutual nearest neighbours with different labels (Tomek 1976;
    /// imbalanced-learn <c>TomekLinks</c>), for any two classes, not only minority versus the rest.
    /// <see cref="RemovalStrategy.RemoveMajority"/> removes every linked sample outside the minority class
    /// (imbalanced-learn's default 'auto' strategy), <see cref="RemovalStrategy.RemoveMinority"/> removes the
    /// linked minority samples, and <see cref="RemovalStrategy.RemoveBoth"/> removes both ends of every link.
    /// </para>
    /// </remarks>
    private HashSet<int> FindTomekLinkIndices(Matrix<T> data, Vector<T> labels, T minorityLabel)
    {
        var indicesToRemove = new HashSet<int>();
        foreach (var (i, nn) in FindLinks(data, labels))
        {
            foreach (int member in new[] { i, nn })
            {
                bool isMinority = IsMinority(labels, member, minorityLabel);
                bool remove = Strategy switch
                {
                    RemovalStrategy.RemoveBoth => true,
                    RemovalStrategy.RemoveMajority => !isMinority,
                    RemovalStrategy.RemoveMinority => isMinority,
                    _ => false
                };
                if (remove)
                {
                    indicesToRemove.Add(member);
                }
            }
        }

        return indicesToRemove;
    }

    /// <summary>
    /// The Tomek links (i, nn) with i &lt; nn: mutual nearest neighbours whose labels differ.
    /// </summary>
    private List<(int, int)> FindLinks(Matrix<T> data, Vector<T> labels)
    {
        var links = new List<(int, int)>();
        int rows = GetSampleCount(data);
        if (rows < 2)
        {
            return links;
        }

        var nearest = TabularResamplingHelper<T>.NearestNeighbors(data);
        var engine = AiDotNetEngine.Current;
        var neighborLabels = engine.TensorIndexSelect(
            labels.AsTensor(), new Tensor<int>((int[])nearest.Clone(), new[] { rows }), 0);
        var differs = engine.TensorNotEquals(labels.AsTensor(), neighborLabels).Cast<int>().ToArray();

        for (int i = 0; i < rows; i++)
        {
            int nn = nearest[i];
            if (nn > i && nearest[nn] == i && differs[i] != 0)
            {
                links.Add((i, nn));
            }
        }

        return links;
    }

    private static bool IsMinority(Vector<T> labels, int index, T minorityLabel)
    {
        return NumOps.Compare(labels[index], minorityLabel) == 0;
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
        return FindLinks(data, labels).Count;
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var parameters = base.GetParameters();
        parameters["strategy"] = Strategy.ToString();
        return parameters;
    }
}
