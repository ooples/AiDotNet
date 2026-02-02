namespace AiDotNet.Preprocessing.TimeSeries;

/// <summary>
/// Provides time-series cross-validation with expanding or sliding window splits.
/// </summary>
/// <remarks>
/// <para>
/// Time series data requires special cross-validation that respects temporal ordering.
/// Unlike standard k-fold cross-validation, time series splits ensure that:
/// - Training data always comes before validation data
/// - No future information "leaks" into the training set
/// </para>
/// <para><b>For Beginners:</b> When working with time series data (like stock prices or weather),
/// you can't randomly split the data like you would for regular machine learning.
///
/// Why? Because predicting the future using future data is cheating! If you train on
/// data from 2023 to predict 2022, you're using information you wouldn't have had.
///
/// TimeSeriesSplit creates splits like this:
/// <code>
/// Split 1: Train=[Jan-Mar], Test=[Apr]
/// Split 2: Train=[Jan-Jun], Test=[Jul]
/// Split 3: Train=[Jan-Sep], Test=[Oct]
/// </code>
///
/// Each training set includes all previous data, and the test set is always "in the future"
/// relative to the training data.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var split = new TimeSeriesSplit(nSplits: 5, maxTrainSize: 1000);
///
/// foreach (var (trainIndices, testIndices) in split.Split(data.Length))
/// {
///     // trainIndices contains indices for training data
///     // testIndices contains indices for test/validation data
/// }
/// </code>
/// </example>
public class TimeSeriesSplit
{
    #region Properties

    /// <summary>
    /// Gets the number of splits to generate.
    /// </summary>
    public int NSplits { get; }

    /// <summary>
    /// Gets the maximum training set size. If null, the training set grows with each split.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When null, this uses "expanding window" mode where
    /// the training set includes all data up to the test set.
    ///
    /// When set to a value, this uses "sliding window" mode where the training set
    /// is limited to the most recent N observations.
    ///
    /// Sliding window is useful when:
    /// - You believe only recent data is relevant (e.g., market conditions change)
    /// - You want consistent training set sizes for fair comparison
    /// - Memory constraints prevent using all historical data
    /// </para>
    /// </remarks>
    public int? MaxTrainSize { get; }

    /// <summary>
    /// Gets the test set size for each split. If null, defaults to n_samples / (n_splits + 1).
    /// </summary>
    public int? TestSize { get; }

    /// <summary>
    /// Gets the gap between train and test sets, useful to avoid data leakage.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Sometimes you need a gap between training and test data.
    ///
    /// For example, if you're predicting stock prices 5 days ahead, you need at least
    /// a 5-day gap so your predictions don't use data from the prediction period.
    ///
    /// Common use cases:
    /// - Forecasting with a prediction horizon (gap = horizon)
    /// - Avoiding look-ahead bias when features use lagged values
    /// - Accounting for data that becomes available with a delay
    /// </para>
    /// </remarks>
    public int Gap { get; }

    #endregion

    #region Constructor

    /// <summary>
    /// Creates a new TimeSeriesSplit with the specified configuration.
    /// </summary>
    /// <param name="nSplits">Number of splits (must be at least 2).</param>
    /// <param name="maxTrainSize">Maximum training set size, or null for expanding window.</param>
    /// <param name="testSize">Test set size per split, or null for automatic sizing.</param>
    /// <param name="gap">Gap between train and test sets (default 0).</param>
    /// <exception cref="ArgumentException">If nSplits is less than 2.</exception>
    public TimeSeriesSplit(int nSplits = 5, int? maxTrainSize = null, int? testSize = null, int gap = 0)
    {
        if (nSplits < 2)
        {
            throw new ArgumentException("Number of splits must be at least 2.", nameof(nSplits));
        }

        if (maxTrainSize.HasValue && maxTrainSize.Value < 1)
        {
            throw new ArgumentException("Max train size must be positive.", nameof(maxTrainSize));
        }

        if (testSize.HasValue && testSize.Value < 1)
        {
            throw new ArgumentException("Test size must be positive.", nameof(testSize));
        }

        if (gap < 0)
        {
            throw new ArgumentException("Gap cannot be negative.", nameof(gap));
        }

        NSplits = nSplits;
        MaxTrainSize = maxTrainSize;
        TestSize = testSize;
        Gap = gap;
    }

    #endregion

    #region Split Methods

    /// <summary>
    /// Generates train/test index splits for the given number of samples.
    /// </summary>
    /// <param name="nSamples">The total number of samples in the dataset.</param>
    /// <returns>Enumerable of (trainIndices, testIndices) tuples.</returns>
    /// <exception cref="ArgumentException">If there isn't enough data for the splits.</exception>
    /// <example>
    /// <code>
    /// var splitter = new TimeSeriesSplit(nSplits: 3);
    /// var data = LoadTimeSeriesData(); // 100 samples
    ///
    /// foreach (var (train, test) in splitter.Split(data.Length))
    /// {
    ///     var trainData = data.GetRows(train);
    ///     var testData = data.GetRows(test);
    ///
    ///     model.Fit(trainData);
    ///     var predictions = model.Predict(testData);
    ///     var score = Evaluate(predictions, testData.Labels);
    /// }
    /// </code>
    /// </example>
    public IEnumerable<(int[] TrainIndices, int[] TestIndices)> Split(int nSamples)
    {
        // Calculate test size
        int testSize = TestSize ?? nSamples / (NSplits + 1);

        if (testSize < 1)
        {
            testSize = 1;
        }

        // Calculate the minimum amount of data needed
        int minRequired = testSize * NSplits + Gap;
        if (nSamples < minRequired)
        {
            throw new ArgumentException(
                $"Not enough samples ({nSamples}) for {NSplits} splits with test size {testSize} and gap {Gap}. " +
                $"Minimum required: {minRequired}.");
        }

        // Generate splits
        for (int i = 0; i < NSplits; i++)
        {
            // Test indices
            int testEnd = nSamples - (NSplits - i - 1) * testSize;
            int testStart = testEnd - testSize;

            // Train indices (everything before the gap)
            int trainEnd = testStart - Gap;
            int trainStart = 0;

            // Apply max train size if specified (sliding window)
            if (MaxTrainSize.HasValue && trainEnd - trainStart > MaxTrainSize.Value)
            {
                trainStart = trainEnd - MaxTrainSize.Value;
            }

            // Ensure we have at least some training data
            if (trainEnd <= trainStart)
            {
                continue; // Skip this split if there's no training data
            }

            var trainIndices = Enumerable.Range(trainStart, trainEnd - trainStart).ToArray();
            var testIndices = Enumerable.Range(testStart, testSize).ToArray();

            yield return (trainIndices, testIndices);
        }
    }

    /// <summary>
    /// Gets the number of splits that will be generated for the given number of samples.
    /// </summary>
    /// <param name="nSamples">The total number of samples.</param>
    /// <returns>The number of valid splits.</returns>
    public int GetNSplits(int nSamples)
    {
        return Split(nSamples).Count();
    }

    #endregion

    #region Utility Methods

    /// <summary>
    /// Performs cross-validation on time series data using the specified evaluation function.
    /// </summary>
    /// <typeparam name="TData">The type of data elements.</typeparam>
    /// <param name="data">The time series data array.</param>
    /// <param name="evaluator">Function that takes (trainData, testData) and returns a score.</param>
    /// <returns>Array of scores from each split.</returns>
    /// <example>
    /// <code>
    /// var scores = splitter.CrossValidate(
    ///     data,
    ///     (train, test) =>
    ///     {
    ///         model.Fit(train);
    ///         var predictions = model.Predict(test);
    ///         return CalculateMSE(predictions, test);
    ///     });
    ///
    /// Console.WriteLine($"Mean Score: {scores.Average():F4}");
    /// Console.WriteLine($"Std Dev: {StandardDeviation(scores):F4}");
    /// </code>
    /// </example>
    public double[] CrossValidate<TData>(TData[] data, Func<TData[], TData[], double> evaluator)
    {
        var scores = new List<double>();

        foreach (var (trainIndices, testIndices) in Split(data.Length))
        {
            var trainData = trainIndices.Select(i => data[i]).ToArray();
            var testData = testIndices.Select(i => data[i]).ToArray();

            double score = evaluator(trainData, testData);
            scores.Add(score);
        }

        return [.. scores];
    }

    /// <summary>
    /// Creates a summary of the split configuration for diagnostics.
    /// </summary>
    /// <param name="nSamples">The number of samples to show splits for.</param>
    /// <returns>A string describing all splits.</returns>
    public string GetSplitSummary(int nSamples)
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine($"TimeSeriesSplit Configuration:");
        sb.AppendLine($"  N Splits: {NSplits}");
        sb.AppendLine($"  Max Train Size: {(MaxTrainSize.HasValue ? MaxTrainSize.Value.ToString() : "None (expanding window)")}");
        sb.AppendLine($"  Test Size: {(TestSize.HasValue ? TestSize.Value.ToString() : "Auto")}");
        sb.AppendLine($"  Gap: {Gap}");
        sb.AppendLine($"  Total Samples: {nSamples}");
        sb.AppendLine();
        sb.AppendLine("Splits:");

        int splitNum = 0;
        foreach (var (train, test) in Split(nSamples))
        {
            sb.AppendLine($"  Split {splitNum++}:");
            sb.AppendLine($"    Train: [{train.First()}..{train.Last()}] ({train.Length} samples)");
            sb.AppendLine($"    Test:  [{test.First()}..{test.Last()}] ({test.Length} samples)");
        }

        return sb.ToString();
    }

    #endregion
}

/// <summary>
/// Provides specialized time series cross-validation strategies beyond basic TimeSeriesSplit.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Different time series problems need different validation strategies:
///
/// <b>Blocked Time Series Split:</b> Like TimeSeriesSplit but with additional purging around
/// the test set to prevent data leakage from overlapping features.
///
/// <b>Walk-Forward Validation:</b> Simulates real-world deployment by retraining the model
/// at each step with all available data up to that point.
///
/// <b>Purged Group Time Series Split:</b> Groups data by time period and ensures no group
/// is split across train/test sets.
/// </para>
/// </remarks>
public static class TimeSeriesValidation
{
    /// <summary>
    /// Creates a blocked time series split with purging around test sets.
    /// </summary>
    /// <param name="nSamples">Total number of samples.</param>
    /// <param name="nSplits">Number of splits.</param>
    /// <param name="embargoPct">Percentage of data to embargo after test set (default 1%).</param>
    /// <returns>Enumerable of (trainIndices, testIndices) tuples.</returns>
    /// <remarks>
    /// <para>
    /// The embargo period removes data after the test set from future training sets,
    /// which is important when features have look-ahead bias (e.g., rolling windows
    /// that include future values).
    /// </para>
    /// </remarks>
    public static IEnumerable<(int[] TrainIndices, int[] TestIndices)> BlockedTimeSeriesSplit(
        int nSamples,
        int nSplits = 5,
        double embargoPct = 0.01)
    {
        int embargoSize = (int)(nSamples * embargoPct);
        int testSize = nSamples / (nSplits + 1);

        for (int i = 0; i < nSplits; i++)
        {
            int testEnd = nSamples - (nSplits - i - 1) * testSize;
            int testStart = testEnd - testSize;
            int trainEnd = testStart;

            var testIndices = Enumerable.Range(testStart, testSize).ToArray();

            // Apply embargo by excluding indices immediately before the test set
            // This prevents data leakage from features with look-ahead (e.g., rolling windows)
            int effectiveTrainEnd = embargoSize > 0 ? Math.Max(0, trainEnd - embargoSize) : trainEnd;
            var trainIndices = Enumerable.Range(0, effectiveTrainEnd).ToArray();

            yield return (trainIndices, testIndices);
        }
    }

    /// <summary>
    /// Generates walk-forward validation splits.
    /// </summary>
    /// <param name="nSamples">Total number of samples.</param>
    /// <param name="trainSize">Size of each training window.</param>
    /// <param name="testSize">Size of each test window.</param>
    /// <param name="step">Step size between windows (default: testSize).</param>
    /// <returns>Enumerable of (trainIndices, testIndices) tuples.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Walk-forward validation simulates how you would actually
    /// use a model in production:
    ///
    /// 1. Train on window [0, trainSize)
    /// 2. Predict on [trainSize, trainSize + testSize)
    /// 3. Move forward by 'step' samples
    /// 4. Retrain and repeat
    ///
    /// This gives you a realistic estimate of how well your model would perform
    /// if you retrained it periodically with new data.
    /// </para>
    /// </remarks>
    public static IEnumerable<(int[] TrainIndices, int[] TestIndices)> WalkForward(
        int nSamples,
        int trainSize,
        int testSize,
        int? step = null)
    {
        if (nSamples < 1)
            throw new ArgumentOutOfRangeException(nameof(nSamples), "Number of samples must be at least 1.");
        if (trainSize < 1)
            throw new ArgumentOutOfRangeException(nameof(trainSize), "Training size must be at least 1.");
        if (testSize < 1)
            throw new ArgumentOutOfRangeException(nameof(testSize), "Test size must be at least 1.");

        int actualStep = step ?? testSize;
        if (actualStep < 1)
            throw new ArgumentOutOfRangeException(nameof(step), "Step must be at least 1.");

        int start = 0;
        while (start + trainSize + testSize <= nSamples)
        {
            var trainIndices = Enumerable.Range(start, trainSize).ToArray();
            var testIndices = Enumerable.Range(start + trainSize, testSize).ToArray();

            yield return (trainIndices, testIndices);

            start += actualStep;
        }
    }
}
