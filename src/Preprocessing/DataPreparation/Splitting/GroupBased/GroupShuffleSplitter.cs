using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DataPreparation.Splitting.GroupBased;

/// <summary>
/// Random group-based train/test splits (Monte Carlo with groups).
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This is like ShuffleSplitter, but respects group boundaries.
/// Groups are randomly assigned to train or test, keeping all samples from the same
/// group together.
/// </para>
/// <para>
/// <b>When to Use:</b>
/// - When you want multiple random evaluations
/// - But need to keep groups together
/// - Good for getting variance estimates of group-based evaluation
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class GroupShuffleSplitter<T> : DataSplitterBase<T>
{
    private readonly int[] _groups;
    private readonly int _nSplits;
    private readonly double _testSize;

    /// <summary>
    /// Creates a new Group Shuffle splitter.
    /// </summary>
    /// <param name="groups">Array indicating group membership for each sample.</param>
    /// <param name="nSplits">Number of random splits. Default is 10.</param>
    /// <param name="testSize">Proportion of groups for test. Default is 0.2 (20%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public GroupShuffleSplitter(int[] groups, int nSplits = 10, double testSize = 0.2, int randomSeed = 42)
        : base(shuffle: true, randomSeed)
    {
        if (groups is null || groups.Length == 0)
        {
            throw new ArgumentNullException(nameof(groups), "Groups array cannot be null or empty.");
        }

        if (nSplits < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(nSplits), "Number of splits must be at least 1.");
        }

        if (testSize <= 0 || testSize >= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(testSize), "Test size must be between 0 and 1.");
        }

        _groups = groups;
        _nSplits = nSplits;
        _testSize = testSize;
    }

    /// <inheritdoc/>
    public override int NumSplits => _nSplits;

    /// <inheritdoc/>
    public override string Description => $"Group Shuffle-Split ({_nSplits} splits, {_testSize * 100:F0}% test groups)";

    /// <inheritdoc/>
    public override DataSplitResult<T> Split(Matrix<T> X, Vector<T>? y = null)
    {
        return GetSplits(X, y).First();
    }

    /// <inheritdoc/>
    public override IEnumerable<DataSplitResult<T>> GetSplits(Matrix<T> X, Vector<T>? y = null)
    {
        ValidateInputs(X, y);

        if (X.Rows != _groups.Length)
        {
            throw new ArgumentException(
                $"Groups array length ({_groups.Length}) must match number of samples ({X.Rows}).");
        }

        // Build group -> indices mapping
        var groupIndices = new Dictionary<int, List<int>>();
        for (int i = 0; i < _groups.Length; i++)
        {
            if (!groupIndices.TryGetValue(_groups[i], out var list))
            {
                list = new List<int>();
                groupIndices[_groups[i]] = list;
            }
            list.Add(i);
        }

        var groupIds = groupIndices.Keys.ToArray();
        int numTestGroups = Math.Max(1, (int)(groupIds.Length * _testSize));

        for (int split = 0; split < _nSplits; split++)
        {
            // Shuffle groups
            var shuffledGroups = (int[])groupIds.Clone();
            ShuffleIndices(shuffledGroups);

            // Select test groups
            var testGroupSet = new HashSet<int>(shuffledGroups.Take(numTestGroups));

            var trainIndices = new List<int>();
            var testIndices = new List<int>();

            for (int i = 0; i < _groups.Length; i++)
            {
                if (testGroupSet.Contains(_groups[i]))
                {
                    testIndices.Add(i);
                }
                else
                {
                    trainIndices.Add(i);
                }
            }

            yield return BuildResult(X, y, trainIndices.ToArray(), testIndices.ToArray(),
                foldIndex: split, totalFolds: _nSplits);
        }
    }
}
