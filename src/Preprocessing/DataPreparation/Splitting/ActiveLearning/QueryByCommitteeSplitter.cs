using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DataPreparation.Splitting.ActiveLearning;

/// <summary>
/// Query-by-Committee (QBC) splitter for ensemble-based active learning.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Query-by-Committee maintains multiple models (a "committee")
/// that are trained on different subsets of the labeled data. Samples where the committee
/// disagrees most are the most informative and should be labeled next.
/// </para>
/// <para>
/// <b>How It Works:</b>
/// 1. Create multiple subsets of the initial labeled data (one per committee member)
/// 2. Each subset can be used to train a different model
/// 3. Unlabeled samples with highest disagreement are prioritized for labeling
/// </para>
/// <para>
/// <b>When to Use:</b>
/// - When you can train multiple models
/// - Ensemble-based approaches
/// - Maximizing information gain from labeling
/// - Reducing labeling costs through disagreement sampling
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class QueryByCommitteeSplitter<T> : DataSplitterBase<T>
{
    private readonly int _committeeSize;
    private readonly double _subsampleRatio;
    private readonly double _initialLabeledRatio;

    /// <summary>
    /// Creates a new Query-by-Committee splitter.
    /// </summary>
    /// <param name="committeeSize">Number of committee members (subsets to create). Default is 5.</param>
    /// <param name="subsampleRatio">Ratio of labeled data for each committee member. Default is 0.7 (70%).</param>
    /// <param name="initialLabeledRatio">Ratio of total data to start as labeled. Default is 0.1 (10%).</param>
    /// <param name="shuffle">Whether to shuffle before splitting. Default is true.</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public QueryByCommitteeSplitter(
        int committeeSize = 5,
        double subsampleRatio = 0.7,
        double initialLabeledRatio = 0.1,
        bool shuffle = true,
        int randomSeed = 42)
        : base(shuffle, randomSeed)
    {
        if (committeeSize < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(committeeSize),
                "Committee size must be at least 2.");
        }

        if (subsampleRatio <= 0 || subsampleRatio >= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(subsampleRatio),
                "Subsample ratio must be between 0 and 1.");
        }

        if (initialLabeledRatio <= 0 || initialLabeledRatio >= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(initialLabeledRatio),
                "Initial labeled ratio must be between 0 and 1.");
        }

        _committeeSize = committeeSize;
        _subsampleRatio = subsampleRatio;
        _initialLabeledRatio = initialLabeledRatio;
    }

    /// <inheritdoc/>
    public override int NumSplits => _committeeSize;

    /// <inheritdoc/>
    public override string Description => $"Query-by-Committee ({_committeeSize} members, {_initialLabeledRatio * 100:F0}% initial)";

    /// <inheritdoc/>
    public override DataSplitResult<T> Split(Matrix<T> X, Vector<T>? y = null)
    {
        // Return first committee member's split
        ValidateInputs(X, y);

        var splits = GetSplits(X, y).ToList();
        return splits.Count > 0 ? splits[0] : CreateEmptySplit(X, y);
    }

    /// <inheritdoc/>
    public override IEnumerable<DataSplitResult<T>> GetSplits(Matrix<T> X, Vector<T>? y = null)
    {
        ValidateInputs(X, y);

        int nSamples = X.Rows;
        int labeledSize = Math.Max(2, (int)(nSamples * _initialLabeledRatio));

        // First, split into labeled and unlabeled pools
        var allIndices = GetShuffledIndices(nSamples);
        var labeledPool = allIndices.Take(labeledSize).ToArray();
        var unlabeledPool = allIndices.Skip(labeledSize).ToArray();

        // Create subsets for each committee member
        int subsetSize = Math.Max(1, (int)(labeledPool.Length * _subsampleRatio));

        for (int c = 0; c < _committeeSize; c++)
        {
            // Bootstrap subsample from labeled pool
            var memberIndices = new int[subsetSize];
            for (int i = 0; i < subsetSize; i++)
            {
                memberIndices[i] = labeledPool[_random.Next(labeledPool.Length)];
            }

            // The "test" set is the unlabeled pool (same for all committee members)
            yield return BuildResult(X, y, memberIndices, unlabeledPool,
                foldIndex: c, totalFolds: _committeeSize);
        }
    }

    private DataSplitResult<T> CreateEmptySplit(Matrix<T> X, Vector<T>? y)
    {
        var emptyIndices = Array.Empty<int>();
        var allIndices = GetIndices(X.Rows);
        return BuildResult(X, y, emptyIndices, allIndices);
    }
}
