using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.DecompositionMethods.TimeSeriesDecomposition;

/// <summary>
/// Implements the Beveridge-Nelson decomposition method for time series analysis.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float)</typeparam>
/// <remarks>
/// <b>For Beginners:</b> The Beveridge-Nelson decomposition separates a time series into two components:
/// 1. A permanent component (trend) - the long-term path the data would follow if there were no temporary fluctuations
/// 2. A temporary component (cycle) - short-term fluctuations that eventually fade away
/// 
/// This is useful for understanding which changes in your data (like stock prices or economic indicators)
/// are likely to persist versus which are temporary fluctuations.
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.TimeSeriesModel)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.DimensionalityReduction)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Vector<>), typeof(Vector<>))]
[ResearchPaper("A New Approach to Decomposition of Economic Time Series into Permanent and Transitory Components", "https://doi.org/10.1016/0304-3932(81)90040-4", Year = 1981, Authors = "Stephen Beveridge, Charles R. Nelson")]
public partial class BeveridgeNelsonDecomposition<T> : TimeSeriesDecompositionBase<T>
{
    private readonly BeveridgeNelsonAlgorithmType _algorithm;
    private readonly ARIMAOptions<T> _arimaOptions;
    private readonly int _forecastHorizon;
    [AiDotNet.Attributes.TrainableParameter]
    private readonly Matrix<T> _multivariateSeries;

    /// <summary>
    /// Initializes a new instance of the Beveridge-Nelson decomposition.
    /// </summary>
    /// <param name="timeSeries">The time series data to decompose</param>
    /// <param name="algorithm">The algorithm type to use for decomposition</param>
    /// <param name="arimaOptions">Options for the ARIMA model if using ARIMA-based decomposition</param>
    /// <param name="forecastHorizon">Number of periods to forecast for trend calculation</param>
    /// <param name="multivariateSeries">Additional time series data for multivariate decomposition</param>
    /// <remarks>
    /// <b>For Beginners:</b> This constructor sets up the decomposition with your time series data.
    /// - Standard algorithm: Simple approach that works well for most cases
    /// - ARIMA algorithm: More advanced approach that can handle more complex patterns
    /// - Multivariate algorithm: Used when you have multiple related time series
    /// </remarks>
    public BeveridgeNelsonDecomposition(Vector<T> timeSeries,
        BeveridgeNelsonAlgorithmType algorithm = BeveridgeNelsonAlgorithmType.Standard,
        ARIMAOptions<T>? arimaOptions = null,
        int forecastHorizon = 100,
        Matrix<T>? multivariateSeries = null)
        : base(timeSeries)
    {
        _algorithm = algorithm;
        _arimaOptions = arimaOptions ?? new ARIMAOptions<T> { P = 1, D = 1, Q = 1 };
        _forecastHorizon = forecastHorizon;
        _multivariateSeries = multivariateSeries ?? new Matrix<T>(TimeSeries.Length, 1);
        DecomposeInternal();
    }

    /// <summary>
    /// Performs the decomposition based on the selected algorithm.
    /// </summary>
    /// <remarks>
    /// This method selects and executes the appropriate decomposition algorithm.
    /// </remarks>
    protected override void Decompose() => DecomposeInternal();

    private void DecomposeInternal()
    {
        switch (_algorithm)
        {
            case BeveridgeNelsonAlgorithmType.Standard:
                DecomposeStandard();
                break;
            case BeveridgeNelsonAlgorithmType.ARIMA:
                DecomposeARIMA();
                break;
            case BeveridgeNelsonAlgorithmType.Multivariate:
                DecomposeMultivariate();
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(_algorithm), _algorithm, $"Beveridge-Nelson decomposition algorithm {_algorithm} is not supported.");
        }
    }

    /// <summary>
    /// Performs the standard Beveridge-Nelson decomposition.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This method uses a straightforward approach to separate your data into:
    /// - Trend: The long-term direction your data is moving
    /// - Cycle: Short-term ups and downs around the trend
    /// </remarks>
    private void DecomposeStandard()
    {
        // Implement standard Beveridge-Nelson decomposition
        Vector<T> trend = CalculateStandardTrend();
        Vector<T> cycle = CalculateStandardCycle(trend);

        AddComponent(DecompositionComponentType.Trend, trend);
        AddComponent(DecompositionComponentType.Cycle, cycle);
    }

    /// <summary>
    /// Performs the ARIMA-based Beveridge-Nelson decomposition.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This method uses a more sophisticated statistical model (ARIMA) to:
    /// 1. Learn patterns in your data (like how values depend on previous values)
    /// 2. Use these patterns to separate the permanent trend from temporary fluctuations
    /// 
    /// ARIMA stands for "AutoRegressive Integrated Moving Average" - it's a powerful
    /// forecasting method that can capture complex patterns in time series data.
    /// </remarks>
    private void DecomposeARIMA()
    {
        // Implement ARIMA-based Beveridge-Nelson decomposition
        var arimaModel = new ARIMAModel<T>(_arimaOptions);
        arimaModel.Train(new Matrix<T>(TimeSeries.Length, 1), TimeSeries);

        Vector<T> trend = CalculateARIMATrend(arimaModel);
        Vector<T> cycle = CalculateARIMACycle(trend);

        AddComponent(DecompositionComponentType.Trend, trend);
        AddComponent(DecompositionComponentType.Cycle, cycle);
    }

    /// <summary>
    /// Performs the multivariate Beveridge-Nelson decomposition.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This method is used when you have multiple related time series.
    /// For example, if you're analyzing both unemployment rates and GDP growth,
    /// this method can find connections between them and provide a more accurate
    /// decomposition by considering how they influence each other.
    /// 
    /// It uses a Vector AutoRegression (VAR) model, which is like ARIMA but for
    /// multiple time series at once.
    /// </remarks>
    private void DecomposeMultivariate()
    {
        int n = _multivariateSeries.Rows;
        int m = _multivariateSeries.Columns;

        // Step 1: Estimate VAR model
        var varOptions = new VARModelOptions<T> { Lag = 1, OutputDimension = m }; // Assuming first-order VAR
        var varModel = new VectorAutoRegressionModel<T>(varOptions);
        varModel.Train(_multivariateSeries, Vector<T>.Empty()); // Passing an empty vector as y since it's not used in VAR

        // Step 2: Calculate long-run impact matrix
        Matrix<T> longRunImpact = CalculateLongRunImpactMatrix(varModel, varOptions);

        // Step 3: Calculate permanent component
        Matrix<T> permanent = new Matrix<T>(n, m);
        for (int i = 0; i < n; i++)
        {
            Vector<T> cumSum = new Vector<T>(m);
            for (int j = 0; j <= i; j++)
            {
                cumSum = cumSum.Add(longRunImpact.Multiply(_multivariateSeries.GetRow(j)));
            }

            permanent.SetRow(i, cumSum);
        }

        // Step 4: Calculate transitory component
        Matrix<T> transitory = _multivariateSeries.Subtract(permanent);

        // Add components
        AddComponent(DecompositionComponentType.Trend, permanent.GetColumn(0));
        AddComponent(DecompositionComponentType.Cycle, transitory.GetColumn(0));

        // If there are additional series, add them as separate components
        for (int i = 1; i < m; i++)
        {
            AddComponent((DecompositionComponentType)((int)DecompositionComponentType.Trend + i), permanent.GetColumn(i));
            AddComponent((DecompositionComponentType)((int)DecompositionComponentType.Cycle + i), transitory.GetColumn(i));
        }
    }

    /// <summary>
    /// Calculates the long-run impact matrix for multivariate decomposition.
    /// </summary>
    /// <param name="varModel">The trained Vector AutoRegression model</param>
    /// <param name="varOptions">Options used for the VAR model</param>
    /// <returns>The long-run impact matrix</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This matrix shows how changes in one variable permanently affect
    /// all variables in the system. For example, how a permanent change in interest rates
    /// might affect GDP, inflation, and unemployment in the long run.
    /// </remarks>
    private Matrix<T> CalculateLongRunImpactMatrix(VectorAutoRegressionModel<T> varModel, VARModelOptions<T> varOptions)
    {
        int m = _multivariateSeries.Columns;
        Matrix<T> identity = Matrix<T>.CreateIdentity(m);
        Matrix<T> coeffSum = new Matrix<T>(m, m);

        // Assuming Coefficients is a single matrix representing all lags
        Matrix<T> coefficients = varModel.Coefficients;

        // Sum up the coefficient matrices for each lag
        for (int i = 0; i < varOptions.Lag; i++)
        {
            int startRow = i * m;
            Matrix<T> lagCoeff = coefficients.Slice(startRow, m);
            coeffSum = coeffSum.Add(lagCoeff);
        }

        return identity.Subtract(coeffSum).Inverse();
    }

    /// <summary>
    /// Calculates the trend component using the standard Beveridge-Nelson method.
    /// </summary>
    /// <returns>A vector containing the trend component of the time series</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method extracts the long-term trend from your data by:
    /// 1. Finding how your data changes from one point to the next (differences)
    /// 2. Calculating the average change and how these changes relate to each other over time
    /// 3. Using this information to separate what's likely to be permanent (trend) from what's temporary
    /// 
    /// The trend represents the "permanent" part of your data - the path it would follow
    /// if there were no temporary fluctuations.
    /// </remarks>
    private Vector<T> CalculateStandardTrend()
    {
        int n = TimeSeries.Length;
        if (n <= 1)
        {
            // No first differences: the whole series is trend.
            return TimeSeries.Clone();
        }

        // Calculate first differences
        Vector<T> differenced = Engine.Subtract(TimeSeries.GetSubVector(1, n - 1), TimeSeries.GetSubVector(0, n - 1));

        // Calculate mean of differenced series
        T meanDiff = StatisticsHelper<T>.CalculateMean(differenced);

        int m = n - 1;

        // Autocovariance of the differenced series for every lag k in [0, m): the zero-padded
        // self-correlation of the centred series puts lag k at output position m - 1 - k.
        var centered = Engine.TensorSubtractScalar(Tensor<T>.FromVector(differenced), meanDiff);
        var signal = centered.Reshape(new[] { 1, 1, m });
        var correlation = Engine.Conv1D(signal, signal, 1, m - 1).Reshape(new[] { 2 * m - 1 });
        var lagSums = Engine.TensorFlip(Engine.TensorSlice(correlation, new[] { 0 }, new[] { m }), new[] { 0 });
        var pairCounts = Engine.ScalarMinusTensor(NumOps.FromDouble(m), Engine.TensorArange<T>(0, m, 1));
        var autocovariance = DivideElementwise(lagSums, pairCounts);

        // Long-run variance = gamma_0 + 2 * sum_{k >= 1} gamma_k
        T gamma0 = Engine.TensorSum(Engine.TensorSlice(autocovariance, new[] { 0 }, new[] { 1 }));
        if (NumOps.Equals(gamma0, NumOps.Zero))
        {
            // Constant first differences (a linear or constant series, or any two-point series):
            // there is no transitory variance to separate, so the series is all trend.
            return TimeSeries.Clone();
        }

        T longRunVariance = NumOps.Subtract(
            NumOps.Multiply(NumOps.FromDouble(2), Engine.TensorSum(autocovariance)), gamma0);

        // trend[i] = x[i] + i * (longRunVariance / gamma_0) * meanDiff
        var steps = Engine.TensorArange<T>(0, n, 1);
        var adjustment = Engine.TensorMultiplyScalar(
            Engine.TensorMultiplyScalar(steps, NumOps.Divide(longRunVariance, gamma0)), meanDiff);
        return Engine.TensorAdd(Tensor<T>.FromVector(TimeSeries), adjustment).ToVector();
    }

    /// <summary>
    /// Calculates the cyclical component using the standard Beveridge-Nelson method.
    /// </summary>
    /// <param name="trend">The previously calculated trend component</param>
    /// <returns>A vector containing the cyclical component of the time series</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method extracts the temporary ups and downs in your data.
    /// It simply subtracts the trend (long-term path) from your original data,
    /// leaving only the short-term fluctuations that tend to disappear over time.
    /// 
    /// These fluctuations are called the "cycle" because they often show patterns
    /// of ups and downs around the trend line.
    /// </remarks>
    private Vector<T> CalculateStandardCycle(Vector<T> trend)
    {
        // Calculate cycle as the difference between the time series and the trend
        return TimeSeries.Subtract(trend);
    }

    /// <summary>
    /// Calculates the trend component using an ARIMA model-based Beveridge-Nelson method.
    /// </summary>
    /// <param name="model">The trained ARIMA model</param>
    /// <returns>A vector containing the trend component of the time series</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method uses a statistical model called ARIMA to find patterns in your data.
    /// ARIMA (AutoRegressive Integrated Moving Average) is like a smart pattern-finder that:
    /// 
    /// 1. Looks at how past values influence current values (AutoRegressive part)
    /// 2. Handles data that needs to be "differenced" to become stable (Integrated part)
    /// 3. Considers how past random fluctuations affect current values (Moving Average part)
    /// 
    /// Using these patterns, the method calculates what your data would look like in the
    /// very long run - this becomes your trend component.
    /// </remarks>
    private Vector<T> CalculateARIMATrend(ARIMAModel<T> model)
    {
        int n = TimeSeries.Length;
        Vector<T> trend = new Vector<T>(n);

        // Calculate input dimension based on ARIMA order
        int inputDimension = Math.Max(_arimaOptions.P, _arimaOptions.Q + 1);
        if (_arimaOptions.SeasonalPeriod > 0)
        {
            inputDimension = Math.Max(inputDimension,
                Math.Max(_arimaOptions.P + _arimaOptions.SeasonalP * _arimaOptions.SeasonalPeriod,
                         _arimaOptions.Q + _arimaOptions.SeasonalQ * _arimaOptions.SeasonalPeriod + 1));
        }

        if (n == 0)
        {
            return trend;
        }

        // Series with inputDimension - 1 leading zeros: the input row for time i is the reversed
        // window padded[i .. i + inputDimension - 1] = (x[i], x[i - 1], ..., 0 for negative lags).
        var padded = Engine.TensorConstantPad(Tensor<T>.FromVector(TimeSeries), new[] { inputDimension - 1, 0 }, NumOps.Zero);
        var trendValues = new Tensor<T>[n];

        // Calculate the long-run forecast for each time point
        for (int i = 0; i < n; i++)
        {
            var window = Engine.TensorSlice(padded, new[] { i }, new[] { inputDimension });
            Matrix<T> input = Engine.TensorFlip(window, new[] { 0 }).Reshape(new[] { 1, inputDimension }).ToMatrix();

            // Predict future values
            var forecast = Tensor<T>.FromVector(model.Predict(input));

            // Long-run forecast (Beveridge-Nelson trend): f[0] + sum_k (f[k] - f[k - 1])
            var first = Engine.TensorSlice(forecast, new[] { 0 }, new[] { 1 });
            trendValues[i] = forecast.Length > 1
                ? Engine.TensorAddScalar(first, Engine.TensorSum(Engine.TensorDiff(forecast)))
                : first;
        }

        return Engine.TensorConcatenate(trendValues, 0).ToVector();
    }

    /// <summary>
    /// Calculates the cyclical component using the ARIMA-based Beveridge-Nelson method.
    /// </summary>
    /// <param name="trend">The previously calculated trend component</param>
    /// <returns>A vector containing the cyclical component of the time series</returns>
    /// <remarks>
    /// <b>For Beginners:</b> After finding the long-term trend using the ARIMA model,
    /// this method simply subtracts that trend from your original data.
    /// 
    /// What remains is the cyclical component - the temporary ups and downs
    /// that don't persist in the long run. These might represent short-term
    /// economic cycles, seasonal patterns, or other temporary influences on your data.
    /// </remarks>
    private Vector<T> CalculateARIMACycle(Vector<T> trend)
    {
        // Calculate cycle as the difference between the time series and the trend
        return TimeSeries.Subtract(trend);
    }
}
