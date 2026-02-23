using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DataPreparation.Splitting.GroupBased;

/// <summary>
/// Leave-One-Group-Out cross-validation where each group is the test set once.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This is like Leave-One-Out, but for groups instead of samples.
/// Each unique group becomes the test set while all other groups form the training set.
/// </para>
/// <para>
/// <b>Example - Patient Study:</b>
/// <code>
/// Fold 1: Patient A → Test, Patients B,C,D → Train
/// Fold 2: Patient B → Test, Patients A,C,D → Train
/// Fold 3: Patient C → Test, Patients A,B,D → Train
/// Fold 4: Patient D → Test, Patients A,B,C → Train
/// </code>
/// </para>
/// <para>
/// <b>When to Use:</b>
/// - Cross-subject validation in user studies
/// - Cross-patient validation in medical research
/// - When you want to test generalization to new groups
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class LeaveOneGroupOutSplitter<T> : DataSplitterBase<T>
{
    private readonly int[] _groups;
    private int _numGroups;

    /// <summary>
    /// Creates a new Leave-One-Group-Out splitter.
    /// </summary>
    /// <param name="groups">Array indicating group membership for each sample.</param>
    public LeaveOneGroupOutSplitter(int[] groups) : base(shuffle: false, randomSeed: 42)
    {
        if (groups is null || groups.Length == 0)
        {
            throw new ArgumentNullException(nameof(groups), "Groups array cannot be null or empty.");
        }

        _groups = groups;
        _numGroups = groups.Distinct().Count();
    }

    /// <inheritdoc/>
    public override int NumSplits => _numGroups;

    /// <inheritdoc/>
    public override string Description => "Leave-One-Group-Out cross-validation";

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

        var uniqueGroups = _groups.Distinct().OrderBy(g => g).ToArray();
        _numGroups = uniqueGroups.Length;

        int fold = 0;
        foreach (int testGroup in uniqueGroups)
        {
            var trainIndices = new List<int>();
            var testIndices = new List<int>();

            for (int i = 0; i < _groups.Length; i++)
            {
                if (_groups[i] == testGroup)
                {
                    testIndices.Add(i);
                }
                else
                {
                    trainIndices.Add(i);
                }
            }

            yield return BuildResult(X, y, trainIndices.ToArray(), testIndices.ToArray(),
                foldIndex: fold++, totalFolds: _numGroups);
        }
    }
}
