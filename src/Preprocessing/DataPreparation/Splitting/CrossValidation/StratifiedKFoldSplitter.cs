using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DataPreparation.Splitting.CrossValidation;

/// <summary>
/// Stratified K-Fold cross-validation that preserves class distribution in each fold.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Stratified K-Fold is like regular K-Fold, but it ensures that
/// each fold has approximately the same proportion of each class as the original dataset.
/// </para>
/// <para>
/// <b>Why Stratification Matters:</b>
/// Imagine you have 90% cats and 10% dogs. With regular K-Fold, one fold might randomly
/// get no dogs at all! Stratification prevents this by ensuring each fold has ~90% cats and ~10% dogs.
/// </para>
/// <para>
/// <b>When to Use:</b>
/// - Classification problems (required for labels)
/// - Imbalanced datasets (more of one class than another)
/// - Any classification task - it's the industry standard!
/// </para>
/// <para>
/// <b>Industry Standard:</b> For classification tasks, ALWAYS prefer StratifiedKFold over regular KFold.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class StratifiedKFoldSplitter<T> : DataSplitterBase<T>
{
    private readonly int _k;

    /// <summary>
    /// Creates a new Stratified K-Fold cross-validation splitter.
    /// </summary>
    /// <param name="k">Number of folds. Default is 5.</param>
    /// <param name="shuffle">Whether to shuffle within each class before splitting. Default is true.</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public StratifiedKFoldSplitter(int k = 5, bool shuffle = true, int randomSeed = 42)
        : base(shuffle, randomSeed)
    {
        if (k < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(k),
                "Number of folds (k) must be at least 2.");
        }

        _k = k;
    }

    /// <inheritdoc/>
    public override int NumSplits => _k;

    /// <inheritdoc/>
    public override bool RequiresLabels => true;

    /// <inheritdoc/>
    public override string Description => $"Stratified {_k}-Fold cross-validation";

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
            throw new ArgumentNullException(nameof(y),
                "Stratified K-Fold requires target labels (y) to preserve class distribution.");
        }

        int nSamples = X.Rows;

        // Group indices by class label
        var labelGroups = GroupByLabel(y);

        // Validate each class has enough samples
        foreach (var group in labelGroups)
        {
            if (group.Value.Count < _k)
            {
                throw new ArgumentException(
                    $"Class with label {group.Key} has only {group.Value.Count} samples, " +
                    $"but {_k} folds require at least {_k} samples per class. " +
                    $"Reduce k or use regular KFoldSplitter.");
            }
        }

        // Shuffle within each class if enabled
        if (_shuffle)
        {
            foreach (var group in labelGroups.Values)
            {
                ShuffleList(group);
            }
        }

        // Assign each sample to a fold, maintaining stratification
        var foldAssignments = new int[nSamples];
        foreach (var group in labelGroups.Values)
        {
            AssignToFolds(group, foldAssignments);
        }

        // Generate folds
        for (int fold = 0; fold < _k; fold++)
        {
            var testIndices = new List<int>();
            var trainIndices = new List<int>();

            for (int i = 0; i < nSamples; i++)
            {
                if (foldAssignments[i] == fold)
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

    private void AssignToFolds(List<int> indices, int[] foldAssignments)
    {
        // Distribute indices across folds as evenly as possible
        for (int i = 0; i < indices.Count; i++)
        {
            foldAssignments[indices[i]] = i % _k;
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
