using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DataPreparation.Splitting.TimeSeries;

/// <summary>
/// Rolling origin evaluation for multi-step forecasting.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Rolling origin is designed for evaluating forecasts at multiple
/// horizons (1-step, 2-step, etc. ahead).
/// </para>
/// <para>
/// <b>How It Works:</b>
/// <code>
/// Origin 100: Train[1-100] → Test[101-105] (horizon of 5)
/// Origin 101: Train[1-101] → Test[102-106]
/// Origin 102: Train[1-102] → Test[103-107]
/// </code>
/// The "origin" is the last training point, and you forecast H steps ahead.
/// </para>
/// <para>
/// <b>When to Use:</b>
/// - Multi-step forecasting evaluation
/// - When you need to test different forecast horizons
/// - Standard time series forecasting evaluation
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class RollingOriginSplitter<T> : DataSplitterBase<T>
{
    private readonly int _initialOrigin;
    private readonly int _horizon;
    private readonly int _step;
    private int _nSplits;

    /// <summary>
    /// Creates a new rolling origin splitter.
    /// </summary>
    /// <param name="initialOrigin">First origin point (last training index for first split).</param>
    /// <param name="horizon">Forecast horizon (number of steps to predict).</param>
    /// <param name="step">How much to advance the origin each iteration. Default is 1.</param>
    public RollingOriginSplitter(int initialOrigin, int horizon, int step = 1)
        : base(shuffle: false, randomSeed: 42)
    {
        if (initialOrigin < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(initialOrigin), "Initial origin must be at least 1.");
        }

        if (horizon < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(horizon), "Horizon must be at least 1.");
        }

        if (step < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(step), "Step must be at least 1.");
        }

        _initialOrigin = initialOrigin;
        _horizon = horizon;
        _step = step;
    }

    /// <inheritdoc/>
    public override int NumSplits => _nSplits;

    /// <inheritdoc/>
    public override string Description => $"Rolling Origin (start={_initialOrigin}, horizon={_horizon}, step={_step})";

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
        _nSplits = 0;

        int origin = _initialOrigin;
        int split = 0;

        while (origin + _horizon <= n)
        {
            // Train from 0 to origin (inclusive)
            var trainIndices = Enumerable.Range(0, origin).ToArray();

            // Test from origin to origin + horizon
            var testIndices = Enumerable.Range(origin, _horizon).ToArray();

            _nSplits++;

            yield return BuildResult(X, y, trainIndices, testIndices,
                foldIndex: split++, totalFolds: _nSplits);

            origin += _step;
        }
    }
}
