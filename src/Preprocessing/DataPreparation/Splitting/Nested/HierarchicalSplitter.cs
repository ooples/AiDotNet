using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DataPreparation.Splitting.Nested;

/// <summary>
/// Hierarchical splitter for multi-level nested data structures.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Some data has natural hierarchical structure, like:
/// - Patients → Hospitals → Regions
/// - Students → Classes → Schools
/// - Products → Categories → Departments
/// </para>
/// <para>
/// This splitter respects the hierarchy, ensuring that splits occur at the
/// appropriate level to avoid data leakage.
/// </para>
/// <para>
/// <b>When to Use:</b>
/// - Clinical trials with patients nested in hospitals
/// - Educational data with students in schools
/// - Any multi-level grouped data
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class HierarchicalSplitter<T> : DataSplitterBase<T>
{
    private readonly double _testSize;
    private readonly int _splitLevel;
    private int[][]? _levelAssignments;

    /// <summary>
    /// Creates a new hierarchical splitter.
    /// </summary>
    /// <param name="testSize">Proportion for test set. Default is 0.2 (20%).</param>
    /// <param name="splitLevel">Level at which to perform the split (0 = highest). Default is 0.</param>
    /// <param name="shuffle">Whether to shuffle groups before splitting. Default is true.</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public HierarchicalSplitter(
        double testSize = 0.2,
        int splitLevel = 0,
        bool shuffle = true,
        int randomSeed = 42)
        : base(shuffle, randomSeed)
    {
        if (testSize <= 0 || testSize >= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(testSize), "Test size must be between 0 and 1.");
        }

        if (splitLevel < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(splitLevel), "Split level must be non-negative.");
        }

        _testSize = testSize;
        _splitLevel = splitLevel;
    }

    /// <summary>
    /// Sets the level assignments for samples.
    /// </summary>
    /// <param name="levelAssignments">Array of arrays where levelAssignments[level][sample] gives the group ID at that level.</param>
    public HierarchicalSplitter<T> WithLevelAssignments(int[][] levelAssignments)
    {
        _levelAssignments = levelAssignments;
        return this;
    }

    /// <inheritdoc/>
    public override string Description => $"Hierarchical split (level {_splitLevel}, {_testSize * 100:F0}% test)";

    /// <inheritdoc/>
    public override DataSplitResult<T> Split(Matrix<T> X, Vector<T>? y = null)
    {
        ValidateInputs(X, y);

        int nSamples = X.Rows;

        if (_levelAssignments == null || _levelAssignments.Length == 0)
        {
            throw new InvalidOperationException(
                "Level assignments must be set for hierarchical splitting. Use WithLevelAssignments() method.");
        }

        int nLevels = _levelAssignments.Length;
        int effectiveLevel = Math.Min(_splitLevel, nLevels - 1);

        // Validate level assignments
        foreach (var level in _levelAssignments)
        {
            if (level.Length != nSamples)
            {
                throw new ArgumentException(
                    $"All level assignments must have {nSamples} elements.");
            }
        }

        // Get groups at the split level
        var splitLevelGroups = _levelAssignments[effectiveLevel];
        var groupIndices = new Dictionary<int, List<int>>();

        for (int i = 0; i < nSamples; i++)
        {
            int groupId = splitLevelGroups[i];
            if (!groupIndices.TryGetValue(groupId, out var list))
            {
                list = new List<int>();
                groupIndices[groupId] = list;
            }
            list.Add(i);
        }

        var uniqueGroups = groupIndices.Keys.ToArray();
        int nGroups = uniqueGroups.Length;

        if (nGroups < 2)
        {
            throw new ArgumentException("Need at least 2 groups at the split level.");
        }

        if (_shuffle)
        {
            ShuffleIndices(uniqueGroups);
        }

        int targetTestGroups = Math.Max(1, (int)(nGroups * _testSize));

        var trainIndices = new List<int>();
        var testIndices = new List<int>();

        for (int i = 0; i < nGroups; i++)
        {
            int groupId = uniqueGroups[i];
            if (i < targetTestGroups)
            {
                testIndices.AddRange(groupIndices[groupId]);
            }
            else
            {
                trainIndices.AddRange(groupIndices[groupId]);
            }
        }

        return BuildResult(X, y, trainIndices.ToArray(), testIndices.ToArray());
    }
}
