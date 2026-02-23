using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DataPreparation.Splitting.GroupBased;

/// <summary>
/// K-Fold cross-validation that keeps samples from the same group together.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Sometimes your data has natural groups that should stay together.
/// For example, if you have multiple measurements from the same patient, you don't want
/// patient A's Monday measurement in training and their Tuesday measurement in test -
/// that would be data leakage!
/// </para>
/// <para>
/// <b>How It Works:</b>
/// Instead of splitting by samples, we split by groups:
/// <code>
/// Fold 1: Groups [A,B,C] → Train, Group [D] → Test
/// Fold 2: Groups [A,B,D] → Train, Group [C] → Test
/// Fold 3: Groups [A,C,D] → Train, Group [B] → Test
/// Fold 4: Groups [B,C,D] → Train, Group [A] → Test
/// </code>
/// </para>
/// <para>
/// <b>Common Use Cases:</b>
/// - Medical studies: Multiple measurements per patient
/// - User studies: Multiple sessions per user
/// - Multi-site studies: Multiple samples per location
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class GroupKFoldSplitter<T> : DataSplitterBase<T>
{
    private readonly int _k;
    private readonly int[] _groups;

    /// <summary>
    /// Creates a new Group K-Fold splitter.
    /// </summary>
    /// <param name="groups">Array indicating group membership for each sample.</param>
    /// <param name="k">Number of folds. If null, uses number of unique groups.</param>
    /// <param name="shuffle">Whether to shuffle groups before splitting. Default is true.</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public GroupKFoldSplitter(int[] groups, int? k = null, bool shuffle = true, int randomSeed = 42)
        : base(shuffle, randomSeed)
    {
        if (groups is null || groups.Length == 0)
        {
            throw new ArgumentNullException(nameof(groups), "Groups array cannot be null or empty.");
        }

        _groups = groups;

        int numUniqueGroups = groups.Distinct().Count();
        _k = k ?? numUniqueGroups;

        if (_k < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(k), "Number of folds must be at least 2.");
        }

        if (_k > numUniqueGroups)
        {
            throw new ArgumentException(
                $"Cannot have more folds ({_k}) than unique groups ({numUniqueGroups}).");
        }
    }

    /// <inheritdoc/>
    public override int NumSplits => _k;

    /// <inheritdoc/>
    public override string Description => $"Group {_k}-Fold cross-validation";

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

        // Get unique groups and their sample indices
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

        // Get group IDs and optionally shuffle
        var groupIds = groupIndices.Keys.ToArray();
        if (_shuffle)
        {
            ShuffleIndices(groupIds);
        }

        // Assign groups to folds
        int numGroups = groupIds.Length;
        int baseGroupsPerFold = numGroups / _k;
        int remainder = numGroups % _k;

        var foldGroups = new List<List<int>>();
        int groupIndex = 0;

        for (int fold = 0; fold < _k; fold++)
        {
            int groupsInFold = baseGroupsPerFold + (fold < remainder ? 1 : 0);
            var groups = new List<int>();

            for (int g = 0; g < groupsInFold && groupIndex < numGroups; g++, groupIndex++)
            {
                groups.Add(groupIds[groupIndex]);
            }

            foldGroups.Add(groups);
        }

        // Generate splits
        for (int fold = 0; fold < _k; fold++)
        {
            var testGroups = new HashSet<int>(foldGroups[fold]);

            var trainIndices = new List<int>();
            var testIndices = new List<int>();

            for (int i = 0; i < _groups.Length; i++)
            {
                if (testGroups.Contains(_groups[i]))
                {
                    testIndices.Add(i);
                }
                else
                {
                    trainIndices.Add(i);
                }
            }

            yield return BuildResult(X, y, trainIndices.ToArray(), testIndices.ToArray(),
                foldIndex: fold, totalFolds: _k);
        }
    }
}
