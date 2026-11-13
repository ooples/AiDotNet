using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.DecompositionMethods.TimeSeriesDecomposition;

/// <summary>
/// Performs multiplicative decomposition of time series data into trend, seasonal, and residual components.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Multiplicative decomposition is used when the seasonal variations in your data increase 
/// or decrease proportionally with the level of the time series. In this model, the components are multiplied 
/// together (Original = Trend × Seasonal × Residual) rather than added. This is often appropriate for economic 
/// or financial data where percentage changes are more meaningful than absolute changes.
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.TimeSeriesModel)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.DimensionalityReduction)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Vector<>), typeof(Vector<>))]
    [ResearchPaper("Time Series Analysis: Forecasting and Control", "https://doi.org/10.1002/9781118619193")]
public class MultiplicativeDecomposition<T> : TimeSeriesDecompositionBase<T>
{
    private readonly MultiplicativeAlgorithmType _algorithm;
    private readonly int _seasonalPeriod;

    /// <summary>
    /// Initializes a new instance of the <see cref="MultiplicativeDecomposition{T}"/> class.
    /// </summary>
    /// <param name="timeSeries">The time series data to decompose.</param>
    /// <param name="algorithm">The algorithm to use for decomposition.</param>
    /// <param name="seasonalPeriod">The number of observations in one seasonal cycle (e.g., 12 for monthly data with yearly seasonality).</param>
    public MultiplicativeDecomposition(Vector<T> timeSeries, MultiplicativeAlgorithmType algorithm = MultiplicativeAlgorithmType.GeometricMovingAverage, int seasonalPeriod = 12)
        : base(timeSeries)
    {
        _algorithm = algorithm;
        _seasonalPeriod = seasonalPeriod;
        Decompose();
    }

    /// <summary>
    /// Performs the time series decomposition using the selected algorithm.
    /// </summary>
    protected override void Decompose()
    {
        switch (_algorithm)
        {
            case MultiplicativeAlgorithmType.GeometricMovingAverage:
                DecomposeGeometricMovingAverage();
                break;
            case MultiplicativeAlgorithmType.MultiplicativeExponentialSmoothing:
                DecomposeMultiplicativeExponentialSmoothing();
                break;
            case MultiplicativeAlgorithmType.LogTransformedSTL:
                DecomposeLogTransformedSTL();
                break;
            default:
                throw new ArgumentException("Unsupported Multiplicative decomposition algorithm.");
        }
    }

    /// <summary>
    /// Decomposes the time series using a geometric moving average approach.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This method extracts the trend by calculating the geometric mean (multiplying values 
    /// and taking the nth root) of data points within a moving window. The geometric mean is used instead 
    /// of the arithmetic mean (simple average) because we're working with multiplicative relationships.
    /// Think of it like calculating compound growth rates rather than simple averages.
    /// </remarks>
    private void DecomposeGeometricMovingAverage()
    {
        int n = TimeSeries.Length;
        if (n == 0)
        {
            AddComponent(DecompositionComponentType.Trend, new Vector<T>(0));
            AddComponent(DecompositionComponentType.Seasonal, new Vector<T>(0));
            AddComponent(DecompositionComponentType.Residual, new Vector<T>(0));
            return;
        }

        // Calculate trend using geometric moving average: (prod of the window)^(1 / count), with
        // the window product expressed as sign * exp(sum log|x|) so it reduces to window sums.
        int halfWindow = _seasonalPeriod / 2;
        var x = Tensor<T>.FromVector(TimeSeries);
        var logMagnitude = Engine.TensorLog(Engine.TensorAbs(x));
        var isNegative = Engine.TensorLessThan(x, NumOps.Zero);
        var isZero = Engine.TensorEquals(x, NumOps.Zero);
        var window = BoxKernel(halfWindow);

        var logSum = WindowSums(logMagnitude, window);
        var negativeCount = WindowSums(isNegative, window);
        var zeroCount = WindowSums(isZero, window);
        var count = WindowSums(Engine.TensorOnesLike(x), window);

        // A zero factor gives 0; a negative product has no real root (NaN, as Power gives);
        // otherwise exp(mean log|x|).
        var geometricMean = Engine.TensorExp(DivideElementwise(logSum, count));
        var oddNegatives = Engine.TensorRemainder(negativeCount, Engine.TensorFullLike(negativeCount, NumOps.FromDouble(2)));
        var trendTensor = Engine.TensorWhere(
            Engine.TensorGreaterThan(oddNegatives, NumOps.FromDouble(0.5)),
            Engine.TensorFullLike(x, NumOps.FromDouble(double.NaN)),
            geometricMean);
        trendTensor = Engine.TensorWhere(
            Engine.TensorGreaterThan(zeroCount, NumOps.FromDouble(0.5)), Engine.TensorZerosLike(x), trendTensor);
        Vector<T> trend = trendTensor.ToVector();

        // Calculate seasonal component
        var seasonalTensor = DivideElementwise(x, trendTensor);

        // Normalize seasonal component by the mean of its phase
        Vector<T> averageSeasonal = PhaseMeans(seasonalTensor.ToVector(), _seasonalPeriod);
        var phaseAverage = Tensor<T>.FromVector(RepeatPeriodic(averageSeasonal, n));
        seasonalTensor = DivideElementwise(seasonalTensor, phaseAverage);
        Vector<T> seasonal = seasonalTensor.ToVector();

        // Calculate residual
        Vector<T> residual = DivideElementwise(x, Engine.TensorMultiply(trendTensor, seasonalTensor)).ToVector();

        AddComponent(DecompositionComponentType.Trend, trend);
        AddComponent(DecompositionComponentType.Seasonal, seasonal);
        AddComponent(DecompositionComponentType.Residual, residual);
    }

    /// <summary>Sums of <paramref name="values"/> over the centred, series-truncated window of <paramref name="kernel"/>.</summary>
    private Tensor<T> WindowSums(Tensor<T> values, Tensor<T> kernel)
    {
        int n = values.Length;
        int width = kernel.Length;
        var sums = Engine.Conv1D(values.Reshape(new[] { 1, 1, n }), kernel.Reshape(new[] { 1, 1, width }), 1, width / 2);
        return sums.Reshape(new[] { n });
    }

    /// <summary>
    /// Decomposes the time series using multiplicative exponential smoothing.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Exponential smoothing is like calculating a weighted average where recent values 
    /// have more influence than older values. The "multiplicative" part means we're working with ratios 
    /// and products rather than differences and sums. This method uses three smoothing factors (alpha, beta, gamma) 
    /// that control how quickly the model adapts to changes in level, trend, and seasonality.
    /// </remarks>
    private void DecomposeMultiplicativeExponentialSmoothing()
    {
        int n = TimeSeries.Length;
        Vector<T> trend = new Vector<T>(n);
        Vector<T> seasonal = new Vector<T>(n);
        Vector<T> residual = new Vector<T>(n);

        T alpha = NumOps.FromDouble(0.2); // Smoothing factor for level
        T beta = NumOps.FromDouble(0.1);  // Smoothing factor for trend
        T gamma = NumOps.FromDouble(0.3); // Smoothing factor for seasonal

        T level = TimeSeries[0];
        T trendComponent = NumOps.One;

        // Use a separate rotating buffer for seasonal factors
        T[] seasonalFactors = new T[_seasonalPeriod];
        for (int i = 0; i < _seasonalPeriod; i++)
        {
            seasonalFactors[i] = NumOps.Divide(TimeSeries[i], level);
        }

        for (int t = 0; t < n; t++)
        {
            T value = TimeSeries[t];
            T lastLevel = level;
            T lastTrend = trendComponent;
            T lastSeasonal = seasonalFactors[t % _seasonalPeriod];

            // Update level
            level = NumOps.Add(
                NumOps.Multiply(alpha, NumOps.Divide(value, lastSeasonal)),
                NumOps.Multiply(NumOps.Subtract(NumOps.One, alpha), NumOps.Multiply(lastLevel, lastTrend))
            );

            // Update trend
            trendComponent = NumOps.Add(
                NumOps.Multiply(beta, NumOps.Divide(level, lastLevel)),
                NumOps.Multiply(NumOps.Subtract(NumOps.One, beta), lastTrend)
            );

            // Update seasonal factor
            seasonalFactors[t % _seasonalPeriod] = NumOps.Add(
                NumOps.Multiply(gamma, NumOps.Divide(value, level)),
                NumOps.Multiply(NumOps.Subtract(NumOps.One, gamma), lastSeasonal)
            );

            // Store per-timestep seasonal for correct reconstruction: data = trend * seasonal * residual
            trend[t] = NumOps.Multiply(level, trendComponent);
            seasonal[t] = seasonalFactors[t % _seasonalPeriod];
            residual[t] = NumOps.Divide(value, NumOps.Multiply(trend[t], seasonal[t]));
        }

        AddComponent(DecompositionComponentType.Trend, trend);
        AddComponent(DecompositionComponentType.Seasonal, seasonal);
        AddComponent(DecompositionComponentType.Residual, residual);
    }

    /// <summary>
    /// Decomposes the time series using a log-transformed STL (Seasonal-Trend decomposition using LOESS) approach.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This method transforms multiplicative relationships into additive ones by taking 
    /// the logarithm of the data. After this transformation, we can use STL (Seasonal-Trend decomposition 
    /// using LOESS), which is a powerful technique that works well with additive patterns. LOESS stands for 
    /// "LOcally Estimated Scatterplot Smoothing" - it's a way to find patterns in data by looking at small, 
    /// overlapping chunks. After decomposition, we transform the components back to the original scale using 
    /// the exponential function (the opposite of logarithm).
    /// </remarks>
    private void DecomposeLogTransformedSTL()
    {
        int n = TimeSeries.Length;

        // Log transform the time series
        Vector<T> logTimeSeries = Engine.TensorLog(Tensor<T>.FromVector(TimeSeries)).ToVector();

        // Perform STL decomposition on log-transformed data
        var stlOptions = new STLDecompositionOptions<T>
        {
            SeasonalPeriod = _seasonalPeriod,
            RobustIterations = 1,
            InnerLoopPasses = 2,
            SeasonalDegree = 1,
            TrendDegree = 1,
            SeasonalJump = 1,
            TrendJump = 1,
            SeasonalBandwidth = 0.75,
            TrendBandwidth = 0.75,
            LowPassBandwidth = 0.75
        };

        var stlDecomposition = new STLDecomposition<T>(stlOptions);
        stlDecomposition.Train(new Matrix<T>(n, 1), logTimeSeries);

        Vector<T> logTrend = stlDecomposition.GetTrend();
        Vector<T> logSeasonal = stlDecomposition.GetSeasonal();
        Vector<T> logResidual = stlDecomposition.GetResidual();

        // Transform components back to original scale
        Vector<T> trend = Engine.TensorExp(Tensor<T>.FromVector(logTrend)).ToVector();
        Vector<T> seasonal = Engine.TensorExp(Tensor<T>.FromVector(logSeasonal)).ToVector();
        Vector<T> residual = Engine.TensorExp(Tensor<T>.FromVector(logResidual)).ToVector();

        AddComponent(DecompositionComponentType.Trend, trend);
        AddComponent(DecompositionComponentType.Seasonal, seasonal);
        AddComponent(DecompositionComponentType.Residual, residual);
    }
}
