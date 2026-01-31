using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DataPreparation.Splitting.Bootstrap;

/// <summary>
/// Stratified bootstrap sampling that preserves class distribution.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This combines bootstrap sampling with stratification.
/// Within each class, we sample with replacement, ensuring the bootstrap sample
/// maintains approximately the same class proportions as the original data.
/// </para>
/// <para>
/// <b>When to Use:</b>
/// - Bootstrap with imbalanced classes
/// - When you need class proportions preserved in each bootstrap sample
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class StratifiedBootstrapSplitter<T> : DataSplitterBase<T>
{
    private readonly int _nIterations;

    /// <summary>
    /// Creates a new stratified bootstrap splitter.
    /// </summary>
    /// <param name="nIterations">Number of bootstrap iterations. Default is 100.</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public StratifiedBootstrapSplitter(int nIterations = 100, int randomSeed = 42)
        : base(shuffle: true, randomSeed)
    {
        if (nIterations < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(nIterations), "Number of iterations must be at least 1.");
        }

        _nIterations = nIterations;
    }

    /// <inheritdoc/>
    public override int NumSplits => _nIterations;

    /// <inheritdoc/>
    public override bool RequiresLabels => true;

    /// <inheritdoc/>
    public override string Description => $"Stratified Bootstrap ({_nIterations} iterations)";

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
            throw new ArgumentNullException(nameof(y), "Stratified bootstrap requires target labels (y).");
        }

        var labelGroups = GroupByLabel(y);

        for (int iter = 0; iter < _nIterations; iter++)
        {
            var trainIndices = new List<int>();
            var selectedSet = new HashSet<int>();

            // Bootstrap within each class
            foreach (var group in labelGroups.Values)
            {
                var classIndices = group.ToArray();
                int classSize = classIndices.Length;

                // Sample with replacement within this class
                for (int i = 0; i < classSize; i++)
                {
                    int idx = classIndices[_random.Next(classSize)];
                    trainIndices.Add(idx);
                    selectedSet.Add(idx);
                }
            }

            // OOB samples
            var testIndices = new List<int>();
            for (int i = 0; i < X.Rows; i++)
            {
                if (!selectedSet.Contains(i))
                {
                    testIndices.Add(i);
                }
            }

            if (testIndices.Count == 0)
            {
                continue;
            }

            yield return BuildResult(X, y, trainIndices.ToArray(), testIndices.ToArray(),
                foldIndex: iter, totalFolds: _nIterations);
        }
    }
}
