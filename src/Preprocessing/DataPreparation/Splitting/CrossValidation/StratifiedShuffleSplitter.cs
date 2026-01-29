using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DataPreparation.Splitting.CrossValidation;

/// <summary>
/// Stratified Monte Carlo cross-validation that preserves class distribution in random splits.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This combines:
/// - <b>Shuffle-Split:</b> Random train/test splits repeated multiple times
/// - <b>Stratification:</b> Each split preserves the original class distribution
/// </para>
/// <para>
/// <b>When to Use:</b>
/// - Classification problems with imbalanced classes
/// - When you want multiple random evaluations (like Monte Carlo CV)
/// - But also need to ensure each split has proper class representation
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class StratifiedShuffleSplitter<T> : DataSplitterBase<T>
{
    private readonly int _nSplits;
    private readonly double _testSize;

    /// <summary>
    /// Creates a new Stratified Shuffle splitter.
    /// </summary>
    /// <param name="nSplits">Number of random splits to generate. Default is 10.</param>
    /// <param name="testSize">Proportion for test set. Default is 0.2 (20%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public StratifiedShuffleSplitter(int nSplits = 10, double testSize = 0.2, int randomSeed = 42)
        : base(shuffle: true, randomSeed)
    {
        if (nSplits < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(nSplits), "Number of splits must be at least 1.");
        }

        if (testSize <= 0 || testSize >= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(testSize), "Test size must be between 0 and 1.");
        }

        _nSplits = nSplits;
        _testSize = testSize;
    }

    /// <inheritdoc/>
    public override int NumSplits => _nSplits;

    /// <inheritdoc/>
    public override bool RequiresLabels => true;

    /// <inheritdoc/>
    public override string Description => $"Stratified Shuffle-Split ({_nSplits} splits, {_testSize * 100:F0}% test)";

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
                "Stratified Shuffle-Split requires target labels (y).");
        }

        int nSamples = X.Rows;

        // Group indices by class
        var labelGroups = GroupByLabel(y);

        for (int split = 0; split < _nSplits; split++)
        {
            var trainIndices = new List<int>();
            var testIndices = new List<int>();

            // For each class, sample proportionally
            foreach (var group in labelGroups)
            {
                var classIndices = group.Value.ToArray();
                ShuffleIndices(classIndices);

                int classTestSize = Math.Max(1, (int)(classIndices.Length * _testSize));
                int classTrainSize = classIndices.Length - classTestSize;

                // Add to train
                for (int i = 0; i < classTrainSize; i++)
                {
                    trainIndices.Add(classIndices[i]);
                }

                // Add to test
                for (int i = classTrainSize; i < classIndices.Length; i++)
                {
                    testIndices.Add(classIndices[i]);
                }
            }

            yield return BuildResult(X, y, trainIndices.ToArray(), testIndices.ToArray(),
                foldIndex: split, totalFolds: _nSplits);
        }
    }
}
