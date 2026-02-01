using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.CrossValidation;

/// <summary>
/// Stratified Group K-Fold: combines stratification and group constraints.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This strategy ensures both:
/// <list type="bullet">
/// <item>Class distribution is preserved in each fold (stratification)</item>
/// <item>Groups are kept together (no leakage between train/validation)</item>
/// </list>
/// </para>
/// <para><b>Example use case:</b> Medical study where:
/// <list type="bullet">
/// <item>Multiple samples from the same patient (group = patient ID)</item>
/// <item>Classes are imbalanced (disease vs. healthy)</item>
/// <item>Need both group integrity and class balance</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class StratifiedGroupKFoldStrategy<T> : ICrossValidationStrategy<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly int _nFolds;
    private readonly int? _randomSeed;
    private int[]? _groups;

    public string Name => $"Stratified Group {_nFolds}-Fold";
    public string Description => "K-Fold with both stratification and group constraints.";
    public int NumSplits => _nFolds;

    /// <summary>
    /// Initializes Stratified Group K-Fold.
    /// </summary>
    /// <param name="nFolds">Number of folds. Default: 5.</param>
    /// <param name="groups">Group identifiers for each sample.</param>
    /// <param name="randomSeed">Random seed for reproducibility.</param>
    public StratifiedGroupKFoldStrategy(int nFolds = 5, int[]? groups = null, int? randomSeed = null)
    {
        if (nFolds < 2)
            throw new ArgumentException("Number of folds must be at least 2.");
        _nFolds = nFolds;
        _groups = groups;
        _randomSeed = randomSeed;
    }

    /// <summary>
    /// Sets the group identifiers.
    /// </summary>
    public void SetGroups(int[] groups)
    {
        _groups = groups;
    }

    public IEnumerable<(int[] TrainIndices, int[] ValidationIndices)> Split(int dataSize, ReadOnlySpan<T> labels = default)
    {
        if (_groups == null)
            throw new InvalidOperationException("Groups must be set before splitting.");
        if (_groups.Length != dataSize)
            throw new ArgumentException("Groups array must match data size.");

        // Convert span to array for internal processing
        return SplitInternal(dataSize, labels.IsEmpty ? null : labels.ToArray());
    }

    private IEnumerable<(int[] TrainIndices, int[] ValidationIndices)> SplitInternal(int dataSize, T[]? labelsArray)
    {
        var random = _randomSeed.HasValue
            ? RandomHelper.CreateSeededRandom(_randomSeed.Value)
            : new Random();

        // Get unique groups and their class distributions
        var uniqueGroups = _groups!.Distinct().ToList();
        var groupClasses = new Dictionary<int, int>();
        var groupSamples = new Dictionary<int, List<int>>();

        foreach (var group in uniqueGroups)
        {
            groupSamples[group] = new List<int>();
        }

        for (int i = 0; i < dataSize; i++)
        {
            int group = _groups![i];
            groupSamples[group].Add(i);
        }

        // Assign majority class to each group
        if (labelsArray != null)
        {
            foreach (var group in uniqueGroups)
            {
                var samples = groupSamples[group];
                var classVotes = new Dictionary<int, int>();

                foreach (var idx in samples)
                {
                    int cls = NumOps.ToDouble(labelsArray[idx]) >= 0.5 ? 1 : 0;
                    classVotes[cls] = classVotes.GetValueOrDefault(cls, 0) + 1;
                }

                groupClasses[group] = classVotes.OrderByDescending(kv => kv.Value).First().Key;
            }
        }
        else
        {
            foreach (var group in uniqueGroups)
            {
                groupClasses[group] = 0; // Default class if no labels
            }
        }

        // Separate groups by class
        var class0Groups = uniqueGroups.Where(g => groupClasses[g] == 0).OrderBy(_ => random.Next()).ToList();
        var class1Groups = uniqueGroups.Where(g => groupClasses[g] == 1).OrderBy(_ => random.Next()).ToList();

        // Assign groups to folds while maintaining stratification
        var foldAssignments = new int[uniqueGroups.Count];
        var groupToFold = new Dictionary<int, int>();

        // Distribute class 0 groups
        for (int i = 0; i < class0Groups.Count; i++)
        {
            groupToFold[class0Groups[i]] = i % _nFolds;
        }

        // Distribute class 1 groups
        for (int i = 0; i < class1Groups.Count; i++)
        {
            groupToFold[class1Groups[i]] = i % _nFolds;
        }

        // Generate folds
        for (int fold = 0; fold < _nFolds; fold++)
        {
            var trainIndices = new List<int>();
            var valIndices = new List<int>();

            foreach (var group in uniqueGroups)
            {
                if (groupToFold[group] == fold)
                    valIndices.AddRange(groupSamples[group]);
                else
                    trainIndices.AddRange(groupSamples[group]);
            }

            yield return (trainIndices.ToArray(), valIndices.ToArray());
        }
    }
}
