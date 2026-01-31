using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DataPreparation.Splitting.Bootstrap;

/// <summary>
/// Bootstrap sampling with out-of-bag (OOB) samples as the test set.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Bootstrap is a resampling technique that creates training sets
/// by randomly sampling WITH replacement from your data.
/// </para>
/// <para>
/// <b>How It Works:</b>
/// 1. Randomly select n samples from n total (with replacement)
/// 2. Some samples will be picked multiple times, others not at all
/// 3. The samples NOT picked (~36.8% on average) form the "out-of-bag" (OOB) test set
/// </para>
/// <para>
/// <b>Key Property:</b>
/// Each sample has a ~63.2% chance of being in the training set and ~36.8% chance
/// of being in the OOB test set. This is because P(not selected) = (1-1/n)^n ≈ 1/e.
/// </para>
/// <para>
/// <b>When to Use:</b>
/// - Error estimation with variance estimates
/// - When you want to use all data for training
/// - Foundation for bagging and random forests
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class BootstrapSplitter<T> : DataSplitterBase<T>
{
    private readonly int _nIterations;

    /// <summary>
    /// Creates a new bootstrap splitter.
    /// </summary>
    /// <param name="nIterations">Number of bootstrap iterations. Default is 100.</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public BootstrapSplitter(int nIterations = 100, int randomSeed = 42)
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
    public override string Description => $"Bootstrap ({_nIterations} iterations)";

    /// <inheritdoc/>
    public override DataSplitResult<T> Split(Matrix<T> X, Vector<T>? y = null)
    {
        return GetSplits(X, y).First();
    }

    /// <inheritdoc/>
    public override IEnumerable<DataSplitResult<T>> GetSplits(Matrix<T> X, Vector<T>? y = null)
    {
        ValidateInputs(X, y);

        int n = X.Rows;

        for (int iter = 0; iter < _nIterations; iter++)
        {
            // Sample with replacement
            var trainIndices = new List<int>();
            var selectedSet = new HashSet<int>();

            for (int i = 0; i < n; i++)
            {
                int idx = _random.Next(n);
                trainIndices.Add(idx);
                selectedSet.Add(idx);
            }

            // OOB samples are those not selected
            var testIndices = new List<int>();
            for (int i = 0; i < n; i++)
            {
                if (!selectedSet.Contains(i))
                {
                    testIndices.Add(i);
                }
            }

            // Skip iteration if no OOB samples (rare but possible)
            if (testIndices.Count == 0)
            {
                continue;
            }

            yield return BuildResult(X, y, trainIndices.ToArray(), testIndices.ToArray(),
                foldIndex: iter, totalFolds: _nIterations);
        }
    }
}
