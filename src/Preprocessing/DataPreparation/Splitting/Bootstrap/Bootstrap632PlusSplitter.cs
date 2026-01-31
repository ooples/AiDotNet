using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DataPreparation.Splitting.Bootstrap;

/// <summary>
/// .632+ Bootstrap splitter that improves upon .632 Bootstrap for high-variance scenarios.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The .632 bootstrap can still be biased when the model overfits.
/// The .632+ method adaptively adjusts the weighting based on how much the model overfits,
/// providing more reliable estimates.
/// </para>
/// <para>
/// <b>How It Works:</b>
/// The .632+ error is calculated as:
/// error_632+ = (1-w)*error_train + w*error_test
/// where w varies between 0.632 and 1 based on overfitting degree.
/// </para>
/// <para>
/// <b>Mathematical Details:</b>
/// - R = (error_test - error_train) / (gamma - error_train)
/// - gamma = error under the null model (random predictions)
/// - w = 0.632 / (1 - 0.368 * R)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class Bootstrap632PlusSplitter<T> : DataSplitterBase<T>
{
    private readonly int _nIterations;

    /// <summary>
    /// Creates a new .632+ Bootstrap splitter.
    /// </summary>
    /// <param name="nIterations">Number of bootstrap iterations. Default is 100.</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public Bootstrap632PlusSplitter(int nIterations = 100, int randomSeed = 42)
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
    public override string Description => $".632+ Bootstrap ({_nIterations} iterations)";

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

        for (int iteration = 0; iteration < _nIterations; iteration++)
        {
            // Sample with replacement for training
            var trainIndices = new int[nSamples];
            var inBag = new bool[nSamples];

            for (int i = 0; i < nSamples; i++)
            {
                int idx = _random.Next(nSamples);
                trainIndices[i] = idx;
                inBag[idx] = true;
            }

            // Out-of-bag samples for testing
            var testIndices = new List<int>();
            for (int i = 0; i < nSamples; i++)
            {
                if (!inBag[i])
                {
                    testIndices.Add(i);
                }
            }

            // If no OOB samples (rare but possible), resample
            if (testIndices.Count == 0)
            {
                testIndices.Add(_random.Next(nSamples));
            }

            // Note: The actual .632+ error calculation happens during model evaluation,
            // not during splitting. This splitter provides the bootstrap samples needed.
            yield return BuildResult(X, y, trainIndices, testIndices.ToArray(),
                foldIndex: iteration, totalFolds: _nIterations);
        }
    }
}
