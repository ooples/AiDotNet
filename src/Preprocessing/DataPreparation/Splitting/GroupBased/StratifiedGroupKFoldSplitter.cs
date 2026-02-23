using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DataPreparation.Splitting.GroupBased;

/// <summary>
/// Stratified Group K-Fold cross-validation that keeps groups together while preserving class distribution.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This combines two important concepts:
/// 1. Group K-Fold: Ensures all samples from the same group stay together
/// 2. Stratification: Maintains the proportion of each class in train/test sets
/// </para>
/// <para>
/// <b>When to Use:</b>
/// - Medical studies where patients have multiple measurements AND classes are imbalanced
/// - Customer data where purchases from same customer should stay together AND you have rare categories
/// - Any grouped data with classification targets where class balance matters
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class StratifiedGroupKFoldSplitter<T> : DataSplitterBase<T>
{
    private readonly int _k;
    private int[]? _groups;

    /// <summary>
    /// Creates a new Stratified Group K-Fold splitter.
    /// </summary>
    /// <param name="k">Number of folds. Default is 5.</param>
    /// <param name="groups">Optional array assigning each sample to a group.</param>
    /// <param name="shuffle">Whether to shuffle groups before splitting. Default is true.</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public StratifiedGroupKFoldSplitter(int k = 5, int[]? groups = null, bool shuffle = true, int randomSeed = 42)
        : base(shuffle, randomSeed)
    {
        if (k < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(k), "Number of folds (k) must be at least 2.");
        }

        _k = k;
        _groups = groups;
    }

    /// <summary>
    /// Sets the group assignments for samples.
    /// </summary>
    /// <param name="groups">Array where groups[i] indicates the group ID for sample i.</param>
    public StratifiedGroupKFoldSplitter<T> WithGroups(int[] groups)
    {
        _groups = groups;
        return this;
    }

    /// <inheritdoc/>
    public override int NumSplits => _k;

    /// <inheritdoc/>
    public override bool RequiresLabels => true;

    /// <inheritdoc/>
    public override string Description => $"Stratified Group {_k}-Fold cross-validation";

    /// <inheritdoc/>
    public override DataSplitResult<T> Split(Matrix<T> X, Vector<T>? y = null)
    {
        return GetSplits(X, y).First();
    }

    /// <inheritdoc/>
    public override IEnumerable<DataSplitResult<T>> GetSplits(Matrix<T> X, Vector<T>? y = null)
    {
        ValidateInputs(X, y);

        if (y is null)
        {
            throw new ArgumentNullException(nameof(y), "Stratified Group K-Fold requires target labels (y).");
        }

        int nSamples = X.Rows;

        // Use provided groups or create default (each sample is its own group)
        int[] groups = _groups ?? Enumerable.Range(0, nSamples).ToArray();

        if (groups.Length != nSamples)
        {
            throw new ArgumentException(
                $"Groups array length ({groups.Length}) must match number of samples ({nSamples}).");
        }

        // Group samples by their group ID and calculate class distribution per group
        var groupInfo = new Dictionary<int, (List<int> indices, Dictionary<double, int> classCounts)>();

        for (int i = 0; i < nSamples; i++)
        {
            int groupId = groups[i];
            double label = Convert.ToDouble(y[i]);

            if (!groupInfo.TryGetValue(groupId, out var info))
            {
                info = (new List<int>(), new Dictionary<double, int>());
                groupInfo[groupId] = info;
            }

            info.indices.Add(i);
            info.classCounts.TryGetValue(label, out int count);
            info.classCounts[label] = count + 1;
        }

        var uniqueGroups = groupInfo.Keys.ToList();
        int nGroups = uniqueGroups.Count;

        if (_k > nGroups)
        {
            throw new ArgumentException(
                $"Cannot have more folds ({_k}) than groups ({nGroups}).");
        }

        // Sort groups by their dominant class to help with stratification
        var groupsByClass = new Dictionary<double, List<int>>();
        foreach (var kvp in groupInfo)
        {
            var dominantClass = kvp.Value.classCounts.OrderByDescending(c => c.Value).First().Key;
            if (!groupsByClass.TryGetValue(dominantClass, out var list))
            {
                list = new List<int>();
                groupsByClass[dominantClass] = list;
            }
            list.Add(kvp.Key);
        }

        if (_shuffle)
        {
            foreach (var list in groupsByClass.Values)
            {
                ShuffleList(list);
            }
        }

        // Assign groups to folds trying to balance class distribution
        var foldAssignments = new int[nGroups];
        var foldClassCounts = new Dictionary<double, int>[_k];
        for (int i = 0; i < _k; i++)
        {
            foldClassCounts[i] = new Dictionary<double, int>();
        }

        int groupIndex = 0;
        foreach (var kvp in groupsByClass)
        {
            foreach (int groupId in kvp.Value)
            {
                // Assign to fold with lowest count of this class
                int bestFold = 0;
                int minCount = int.MaxValue;
                for (int f = 0; f < _k; f++)
                {
                    foldClassCounts[f].TryGetValue(kvp.Key, out int count);
                    if (count < minCount)
                    {
                        minCount = count;
                        bestFold = f;
                    }
                }

                foldAssignments[groupIndex] = bestFold;

                // Update fold class counts
                foreach (var classCount in groupInfo[groupId].classCounts)
                {
                    foldClassCounts[bestFold].TryGetValue(classCount.Key, out int current);
                    foldClassCounts[bestFold][classCount.Key] = current + classCount.Value;
                }

                groupIndex++;
            }
        }

        // Create mapping from group ID to fold
        var groupToFold = new Dictionary<int, int>();
        groupIndex = 0;
        foreach (var kvp in groupsByClass)
        {
            foreach (int groupId in kvp.Value)
            {
                groupToFold[groupId] = foldAssignments[groupIndex];
                groupIndex++;
            }
        }

        // Generate splits
        for (int fold = 0; fold < _k; fold++)
        {
            var trainIndices = new List<int>();
            var testIndices = new List<int>();

            foreach (var kvp in groupInfo)
            {
                if (groupToFold[kvp.Key] == fold)
                {
                    testIndices.AddRange(kvp.Value.indices);
                }
                else
                {
                    trainIndices.AddRange(kvp.Value.indices);
                }
            }

            yield return BuildResult(X, y, trainIndices.ToArray(), testIndices.ToArray(),
                foldIndex: fold, totalFolds: _k);
        }
    }

    private void ShuffleList(List<int> list)
    {
        for (int i = list.Count - 1; i > 0; i--)
        {
            int j = _random.Next(i + 1);
            (list[i], list[j]) = (list[j], list[i]);
        }
    }
}
