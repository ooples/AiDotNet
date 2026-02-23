using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.Evaluation.CrossValidation;

/// <summary>
/// Group K-Fold: K-Fold that keeps related samples (groups) together in the same fold.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Group K-Fold ensures that samples from the same group are never
/// split between training and validation:
/// <list type="bullet">
/// <item>Prevents data leakage when samples are related (e.g., multiple measurements from same patient)</item>
/// <item>Groups could be: patients, subjects, time periods, locations, etc.</item>
/// <item>Essential when independence assumption would be violated by standard K-Fold</item>
/// </list>
/// </para>
/// <para>
/// <b>Example:</b> In medical data with multiple scans per patient, you want all scans from
/// a patient in the same fold. Otherwise, the model might "memorize" patient characteristics
/// from training and appear to perform well on validation (but fail on new patients).
/// </para>
/// </remarks>
public class GroupKFoldStrategy<T> : ICrossValidationStrategy<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly int _k;
    private readonly int[] _groups;

    /// <summary>
    /// Initializes Group K-Fold cross-validation.
    /// </summary>
    /// <param name="groups">Array mapping each sample index to its group ID.</param>
    /// <param name="k">Number of folds. If null or greater than number of unique groups, uses number of unique groups.</param>
    public GroupKFoldStrategy(int[] groups, int? k = null)
    {
        Guard.NotNull(groups);
        _groups = groups;
        var uniqueGroups = groups.Distinct().Count();
        _k = k.HasValue ? Math.Min(k.Value, uniqueGroups) : Math.Min(5, uniqueGroups);

        if (_k < 2) throw new ArgumentException("Must have at least 2 groups for Group K-Fold.");
    }

    public string Name => $"Group {_k}-Fold";
    public int NumSplits => _k;
    public string Description => $"Group {_k}-fold cross-validation keeping related samples together.";

    public IEnumerable<(int[] TrainIndices, int[] ValidationIndices)> Split(int dataSize, ReadOnlySpan<T> labels = default)
    {
        if (dataSize != _groups.Length)
            throw new ArgumentException("Groups array length must match data size.", nameof(dataSize));

        // Get unique groups and their sample indices
        var groupToIndices = new Dictionary<int, List<int>>();
        for (int i = 0; i < dataSize; i++)
        {
            int group = _groups[i];
            if (!groupToIndices.ContainsKey(group))
                groupToIndices[group] = new List<int>();
            groupToIndices[group].Add(i);
        }

        var uniqueGroups = groupToIndices.Keys.ToList();
        int numGroups = uniqueGroups.Count;

        // Assign groups to folds
        int baseGroupsPerFold = numGroups / _k;
        int remainder = numGroups % _k;

        int groupIdx = 0;
        var foldGroups = new List<int>[_k];
        for (int fold = 0; fold < _k; fold++)
        {
            int groupsInThisFold = baseGroupsPerFold + (fold < remainder ? 1 : 0);
            foldGroups[fold] = new List<int>();
            for (int g = 0; g < groupsInThisFold && groupIdx < numGroups; g++)
            {
                foldGroups[fold].Add(uniqueGroups[groupIdx++]);
            }
        }

        // Generate train/validation splits
        for (int fold = 0; fold < _k; fold++)
        {
            var validationIndices = new List<int>();
            var trainIndices = new List<int>();

            for (int f = 0; f < _k; f++)
            {
                foreach (var group in foldGroups[f])
                {
                    var indices = groupToIndices[group];
                    if (f == fold)
                        validationIndices.AddRange(indices);
                    else
                        trainIndices.AddRange(indices);
                }
            }

            yield return (trainIndices.ToArray(), validationIndices.ToArray());
        }
    }
}
