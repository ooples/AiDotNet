using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DataPreparation.Splitting.ActiveLearning;

/// <summary>
/// Pool-based active learning splitter that maintains labeled and unlabeled pools.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> In pool-based active learning, we start with a small labeled dataset
/// and a large pool of unlabeled data. The model queries samples from the unlabeled pool
/// that it is most uncertain about, and an oracle (human expert) provides labels.
/// </para>
/// <para>
/// <b>How It Works:</b>
/// 1. Start with a small initial labeled set (seed)
/// 2. The rest becomes the unlabeled pool
/// 3. Model queries uncertain samples from the pool
/// 4. Oracle labels them, moving them to the labeled set
/// </para>
/// <para>
/// <b>When to Use:</b>
/// - Limited labeling budget
/// - Expensive manual annotation
/// - Interactive machine learning
/// - Iterative model improvement
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class PoolBasedSplitter<T> : DataSplitterBase<T>
{
    private readonly double _initialLabeledRatio;
    private readonly int _minLabeledSamples;
    private readonly SelectionStrategy _strategy;

    /// <summary>
    /// Strategy for selecting initial labeled samples.
    /// </summary>
    public enum SelectionStrategy
    {
        /// <summary>Select randomly from the pool.</summary>
        Random,
        /// <summary>Select diverse samples using clustering.</summary>
        Diverse,
        /// <summary>Ensure class balance in initial set (requires labels).</summary>
        Stratified
    }

    /// <summary>
    /// Creates a new pool-based active learning splitter.
    /// </summary>
    /// <param name="initialLabeledRatio">Ratio of samples to start as labeled. Default is 0.1 (10%).</param>
    /// <param name="minLabeledSamples">Minimum number of labeled samples. Default is 10.</param>
    /// <param name="strategy">Strategy for selecting initial labeled samples. Default is Random.</param>
    /// <param name="shuffle">Whether to shuffle before splitting. Default is true.</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public PoolBasedSplitter(
        double initialLabeledRatio = 0.1,
        int minLabeledSamples = 10,
        SelectionStrategy strategy = SelectionStrategy.Random,
        bool shuffle = true,
        int randomSeed = 42)
        : base(shuffle, randomSeed)
    {
        if (initialLabeledRatio <= 0 || initialLabeledRatio >= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(initialLabeledRatio),
                "Initial labeled ratio must be between 0 and 1.");
        }

        if (minLabeledSamples < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(minLabeledSamples),
                "Minimum labeled samples must be at least 1.");
        }

        _initialLabeledRatio = initialLabeledRatio;
        _minLabeledSamples = minLabeledSamples;
        _strategy = strategy;
    }

    /// <inheritdoc/>
    public override string Description => $"Pool-Based Active Learning ({_strategy}, {_initialLabeledRatio * 100:F0}% initial)";

    /// <inheritdoc/>
    public override bool RequiresLabels => _strategy == SelectionStrategy.Stratified;

    /// <inheritdoc/>
    public override DataSplitResult<T> Split(Matrix<T> X, Vector<T>? y = null)
    {
        ValidateInputs(X, y);

        int nSamples = X.Rows;
        int labeledSize = Math.Max(_minLabeledSamples, (int)(nSamples * _initialLabeledRatio));
        labeledSize = Math.Min(labeledSize, nSamples - 1); // Keep at least 1 for unlabeled pool

        int[] labeledIndices;
        int[] unlabeledIndices;

        switch (_strategy)
        {
            case SelectionStrategy.Stratified when y != null:
                (labeledIndices, unlabeledIndices) = SelectStratified(y, nSamples, labeledSize);
                break;
            case SelectionStrategy.Diverse:
                (labeledIndices, unlabeledIndices) = SelectDiverse(X, nSamples, labeledSize);
                break;
            default:
                (labeledIndices, unlabeledIndices) = SelectRandom(nSamples, labeledSize);
                break;
        }

        // Labeled = Train, Unlabeled = Test (pool for querying)
        return BuildResult(X, y, labeledIndices, unlabeledIndices);
    }

    private (int[] labeled, int[] unlabeled) SelectRandom(int nSamples, int labeledSize)
    {
        var indices = GetShuffledIndices(nSamples);
        var labeled = indices.Take(labeledSize).ToArray();
        var unlabeled = indices.Skip(labeledSize).ToArray();
        return (labeled, unlabeled);
    }

    private (int[] labeled, int[] unlabeled) SelectStratified(Vector<T> y, int nSamples, int labeledSize)
    {
        var groups = GroupByLabel(y);
        var labeled = new List<int>();
        var unlabeled = new List<int>();

        int nClasses = groups.Count;
        int perClass = Math.Max(1, labeledSize / nClasses);

        foreach (var kvp in groups)
        {
            var classIndices = kvp.Value.ToArray();
            if (_shuffle)
            {
                ShuffleIndices(classIndices);
            }

            int take = Math.Min(perClass, classIndices.Length);
            for (int i = 0; i < take; i++)
            {
                labeled.Add(classIndices[i]);
            }
            for (int i = take; i < classIndices.Length; i++)
            {
                unlabeled.Add(classIndices[i]);
            }
        }

        return (labeled.ToArray(), unlabeled.ToArray());
    }

    private (int[] labeled, int[] unlabeled) SelectDiverse(Matrix<T> X, int nSamples, int labeledSize)
    {
        // Simple diverse selection using max-min distance
        var labeled = new List<int>();
        var unlabeled = new HashSet<int>(Enumerable.Range(0, nSamples));

        // Start with a random sample
        int first = _random.Next(nSamples);
        labeled.Add(first);
        unlabeled.Remove(first);

        // Greedily select farthest points
        while (labeled.Count < labeledSize && unlabeled.Count > 0)
        {
            int bestCandidate = -1;
            double bestDistance = -1;

            foreach (int candidate in unlabeled)
            {
                double minDist = double.MaxValue;
                foreach (int selected in labeled)
                {
                    double dist = ComputeDistance(X, candidate, selected);
                    minDist = Math.Min(minDist, dist);
                }

                if (minDist > bestDistance)
                {
                    bestDistance = minDist;
                    bestCandidate = candidate;
                }
            }

            if (bestCandidate >= 0)
            {
                labeled.Add(bestCandidate);
                unlabeled.Remove(bestCandidate);
            }
        }

        return (labeled.ToArray(), unlabeled.ToArray());
    }

    private double ComputeDistance(Matrix<T> X, int i, int j)
    {
        double sum = 0;
        for (int k = 0; k < X.Columns; k++)
        {
            double diff = Convert.ToDouble(X[i, k]) - Convert.ToDouble(X[j, k]);
            sum += diff * diff;
        }
        return Math.Sqrt(sum);
    }
}
