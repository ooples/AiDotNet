using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DataPreparation.Splitting.GroupBased;

/// <summary>
/// Leave-P-Groups-Out cross-validation that uses all combinations of P groups as test sets.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This is similar to Leave-One-Group-Out, but instead of using
/// one group at a time for testing, it uses all combinations of P groups.
/// </para>
/// <para>
/// <b>Example:</b>
/// With groups A, B, C, D and p=2:
/// - Split 1: Test on A,B; Train on C,D
/// - Split 2: Test on A,C; Train on B,D
/// - Split 3: Test on A,D; Train on B,C
/// - Split 4: Test on B,C; Train on A,D
/// - Split 5: Test on B,D; Train on A,C
/// - Split 6: Test on C,D; Train on A,B
/// </para>
/// <para>
/// <b>When to Use:</b>
/// - When you want more test set variety than Leave-One-Group-Out
/// - When you have enough groups to afford larger test sets
/// - When evaluating model robustness to different group combinations
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class LeavePGroupsOutSplitter<T> : DataSplitterBase<T>
{
    private readonly int _p;
    private int[]? _groups;

    /// <summary>
    /// Creates a new Leave-P-Groups-Out splitter.
    /// </summary>
    /// <param name="p">Number of groups to leave out for testing in each split. Default is 2.</param>
    /// <param name="groups">Optional array assigning each sample to a group.</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public LeavePGroupsOutSplitter(int p = 2, int[]? groups = null, int randomSeed = 42)
        : base(shuffle: false, randomSeed)
    {
        if (p < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(p), "P must be at least 1.");
        }

        _p = p;
        _groups = groups;
    }

    /// <summary>
    /// Sets the group assignments for samples.
    /// </summary>
    /// <param name="groups">Array where groups[i] indicates the group ID for sample i.</param>
    public LeavePGroupsOutSplitter<T> WithGroups(int[] groups)
    {
        _groups = groups;
        return this;
    }

    /// <inheritdoc/>
    public override int NumSplits
    {
        get
        {
            if (_groups == null) return 0;
            int nGroups = _groups.Distinct().Count();
            return Combinations(nGroups, _p);
        }
    }

    /// <inheritdoc/>
    public override string Description => $"Leave-{_p}-Groups-Out cross-validation";

    /// <inheritdoc/>
    public override DataSplitResult<T> Split(Matrix<T> X, Vector<T>? y = null)
    {
        return GetSplits(X, y).First();
    }

    /// <inheritdoc/>
    public override IEnumerable<DataSplitResult<T>> GetSplits(Matrix<T> X, Vector<T>? y = null)
    {
        ValidateInputs(X, y);

        int nSamples = X.Rows;

        // Use provided groups or create default (each sample is its own group)
        int[] groups = _groups ?? Enumerable.Range(0, nSamples).ToArray();

        if (groups.Length != nSamples)
        {
            throw new ArgumentException(
                $"Groups array length ({groups.Length}) must match number of samples ({nSamples}).");
        }

        // Group samples by their group ID
        var groupIndices = new Dictionary<int, List<int>>();
        for (int i = 0; i < nSamples; i++)
        {
            int groupId = groups[i];
            if (!groupIndices.TryGetValue(groupId, out var list))
            {
                list = new List<int>();
                groupIndices[groupId] = list;
            }
            list.Add(i);
        }

        var uniqueGroups = groupIndices.Keys.ToArray();
        int nGroups = uniqueGroups.Length;

        if (_p >= nGroups)
        {
            throw new ArgumentException(
                $"P ({_p}) must be less than the number of groups ({nGroups}).");
        }

        // Generate all combinations of P groups
        var combinations = GetCombinations(uniqueGroups, _p);

        int splitIndex = 0;
        foreach (var testGroupIds in combinations)
        {
            var testGroupSet = new HashSet<int>(testGroupIds);
            var trainIndices = new List<int>();
            var testIndices = new List<int>();

            foreach (var kvp in groupIndices)
            {
                if (testGroupSet.Contains(kvp.Key))
                {
                    testIndices.AddRange(kvp.Value);
                }
                else
                {
                    trainIndices.AddRange(kvp.Value);
                }
            }

            yield return BuildResult(X, y, trainIndices.ToArray(), testIndices.ToArray(),
                foldIndex: splitIndex, totalFolds: NumSplits);

            splitIndex++;
        }
    }

    private static int Combinations(int n, int k)
    {
        if (k > n) return 0;
        if (k == 0 || k == n) return 1;

        int result = 1;
        for (int i = 0; i < k; i++)
        {
            result = result * (n - i) / (i + 1);
        }
        return result;
    }

    private static IEnumerable<int[]> GetCombinations(int[] elements, int k)
    {
        if (k == 0)
        {
            yield return Array.Empty<int>();
            yield break;
        }

        if (elements.Length == k)
        {
            yield return elements.ToArray();
            yield break;
        }

        for (int i = 0; i <= elements.Length - k; i++)
        {
            int head = elements[i];
            int[] tail = elements.Skip(i + 1).ToArray();

            foreach (var combination in GetCombinations(tail, k - 1))
            {
                yield return new[] { head }.Concat(combination).ToArray();
            }
        }
    }
}
