using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.DecompositionMethods.TimeSeriesDecomposition;

/// <summary>
/// Implements additive time series decomposition, breaking a time series into trend, seasonal, and residual components.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Time series decomposition is like breaking down a complex signal (like sales data over time) 
/// into simpler parts that are easier to understand. The additive model assumes these components add up to form 
/// the original data: Original = Trend + Seasonal + Residual.
/// </para>
/// <para>
/// - Trend: The long-term progression (going up, down, or staying flat over time)
/// - Seasonal: Repeating patterns at fixed intervals (like higher sales every December)
/// - Residual: What's left over after removing trend and seasonal components (the "noise")
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.TimeSeriesModel)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.DimensionalityReduction)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Vector<>), typeof(Vector<>))]
    [ResearchPaper("Time Series Analysis: Forecasting and Control", "https://doi.org/10.1002/9781118619193")]
public class AdditiveDecomposition<T> : TimeSeriesDecompositionBase<T>
{
    /// <summary>
    /// The algorithm type used for decomposition
    /// </summary>
    private readonly AdditiveDecompositionAlgorithmType _algorithm;

    /// <summary>
    /// Creates a new instance of the AdditiveDecomposition class
    /// </summary>
    /// <param name="timeSeries">The time series data to decompose</param>
    /// <param name="algorithm">The algorithm to use for decomposition (defaults to MovingAverage)</param>
    /// <remarks>
    /// <b>For Beginners:</b> When you create this object, it automatically decomposes your time series data
    /// using the algorithm you specify (or MovingAverage if you don't specify one).
    /// </remarks>
    public AdditiveDecomposition(Vector<T> timeSeries, AdditiveDecompositionAlgorithmType algorithm = AdditiveDecompositionAlgorithmType.MovingAverage)
        : base(timeSeries)
    {
        _algorithm = algorithm;
        Decompose();
    }

    /// <summary>
    /// Performs the decomposition based on the selected algorithm
    /// </summary>
    protected override void Decompose()
    {
        switch (_algorithm)
        {
            case AdditiveDecompositionAlgorithmType.MovingAverage:
                DecomposeMovingAverage();
                break;
            case AdditiveDecompositionAlgorithmType.ExponentialSmoothing:
                DecomposeExponentialSmoothing();
                break;
            case AdditiveDecompositionAlgorithmType.STL:
                DecomposeSTL();
                break;
            default:
                throw new ArgumentException("Unsupported Additive decomposition algorithm.");
        }
    }

    /// <summary>
    /// Decomposes the time series using the Moving Average method
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Moving Average is like taking the average of a sliding window of data points.
    /// It helps smooth out short-term fluctuations and highlight longer-term trends.
    /// </remarks>
    private void DecomposeMovingAverage()
    {
        // Implementation of Moving Average decomposition
        Vector<T> trend = CalculateTrendMovingAverage();
        Vector<T> seasonal = CalculateSeasonalMovingAverage(trend);
        Vector<T> residual = CalculateResidual(trend, seasonal);

        AddComponent(DecompositionComponentType.Trend, trend);
        AddComponent(DecompositionComponentType.Seasonal, seasonal);
        AddComponent(DecompositionComponentType.Residual, residual);
    }

    /// <summary>
    /// Decomposes the time series using the Exponential Smoothing method
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Exponential Smoothing gives more weight to recent observations and less weight
    /// to older observations. It's like having a better memory of recent events compared to distant ones.
    /// </remarks>
    private void DecomposeExponentialSmoothing()
    {
        // Implementation of Exponential Smoothing decomposition
        Vector<T> trend = CalculateTrendExponentialSmoothing();
        Vector<T> seasonal = CalculateSeasonalExponentialSmoothing(trend);
        Vector<T> residual = CalculateResidual(trend, seasonal);

        AddComponent(DecompositionComponentType.Trend, trend);
        AddComponent(DecompositionComponentType.Seasonal, seasonal);
        AddComponent(DecompositionComponentType.Residual, residual);
    }

    /// <summary>
    /// Decomposes the time series using the Seasonal and Trend decomposition using Loess (STL) method
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> STL is a sophisticated method that can handle complex seasonal patterns that change over time.
    /// It uses a technique called "Loess" (locally estimated scatterplot smoothing) which fits simple models
    /// to small chunks of data to build a more complex and accurate overall model.
    /// </remarks>
    private void DecomposeSTL()
    {
        // Implementation of Seasonal and Trend decomposition using Loess (STL)
        (Vector<T> trend, Vector<T> seasonal) = PerformSTLDecomposition();
        Vector<T> residual = CalculateResidual(trend, seasonal);

        AddComponent(DecompositionComponentType.Trend, trend);
        AddComponent(DecompositionComponentType.Seasonal, seasonal);
        AddComponent(DecompositionComponentType.Residual, residual);
    }

    /// <summary>
    /// Performs the STL decomposition algorithm
    /// </summary>
    /// <returns>A tuple containing the trend and seasonal components</returns>
    private (Vector<T>, Vector<T>) PerformSTLDecomposition()
    {
        int n = TimeSeries.Length;
        int seasonalPeriod = 12;
        int nInner = 2;
        int nOuter = 1;
        int trendWindow = (int)Math.Ceiling(1.5 * seasonalPeriod / (1 - 1.5 / seasonalPeriod));
        trendWindow = trendWindow % 2 == 0 ? trendWindow + 1 : trendWindow;

        Vector<T> trend = new Vector<T>(n);
        Vector<T> seasonal = new Vector<T>(n);
        Vector<T> detrended = new Vector<T>(n);

        for (int i = 0; i < nOuter; i++)
        {
            for (int j = 0; j < nInner; j++)
            {
                // Step 1: Detrending
                detrended = SubtractVectors(TimeSeries, trend);

                // Step 2: Cycle-subseries Smoothing
                seasonal = CycleSubseriesSmoothing(detrended, seasonalPeriod);

                // Step 3: Low-pass Filtering of Smoothed Cycle-subseries
                Vector<T> lowPassSeasonal = LowPassFilter(seasonal, seasonalPeriod);

                // Step 4: Detrending of Smoothed Cycle-subseries
                seasonal = SubtractVectors(seasonal, lowPassSeasonal);

                // Step 5: Deseasonalizing
                Vector<T> deseasonalized = SubtractVectors(TimeSeries, seasonal);

                // Step 6: Trend Smoothing
                trend = LoessSmoothing(deseasonalized, trendWindow);
            }
        }

        return (trend, seasonal);
    }

    /// <summary>
    /// Subtracts one vector from another element by element
    /// </summary>
    /// <param name="a">The first vector</param>
    /// <param name="b">The vector to subtract</param>
    /// <returns>A new vector containing the result of a - b</returns>
    private Vector<T> SubtractVectors(Vector<T> a, Vector<T> b)
    {
        return Engine.Subtract(a, b);
    }

    /// <summary>
    /// Performs cycle-subseries smoothing on the data
    /// </summary>
    /// <param name="data">The data to smooth</param>
    /// <param name="period">The seasonal period</param>
    /// <returns>The smoothed data</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method breaks down the data into "subseries" based on the seasonal period.
    /// For example, with monthly data (period=12), it creates 12 subseries: all January values, all February values, etc.
    /// Each subseries is then smoothed separately to better capture the seasonal pattern.
    /// </remarks>
    private Vector<T> CycleSubseriesSmoothing(Vector<T> data, int period)
    {
        int n = data.Length;
        if (n == 0)
        {
            return new Vector<T>(0);
        }

        // Column p of the phase grid is the subseries data[p], data[p + period], ...; the first
        // fullPhases columns have `rows` members and the rest have rows - 1.
        int rows = (n + period - 1) / period;
        int fullPhases = n - (rows - 1) * period;
        var grid = ToPhaseGrid(data, period);
        var smoothed = SmoothSubseriesColumns(grid, period);

        if (fullPhases < period && rows > 1)
        {
            var top = Engine.TensorSlice(grid, new[] { 0, 0 }, new[] { rows - 1, period });
            var shortSmoothed = Engine.TensorConcatenate(
                new[] { SmoothSubseriesColumns(top, period), new Tensor<T>(new[] { 1, period }) }, 0);
            var phase = Engine.TensorBroadcastTo(
                Engine.TensorArange<T>(0, period, 1).Reshape(new[] { 1, period }), new[] { rows, period });
            smoothed = Engine.TensorWhere(
                Engine.TensorLessThan(phase, NumOps.FromDouble(fullPhases)), smoothed, shortSmoothed);
        }

        return Engine.TensorSlice(smoothed.Reshape(new[] { rows * period }), new[] { 0 }, new[] { n }).ToVector();
    }

    /// <summary>
    /// LOESS (span 0.75, tri-cube weights) of every column of <paramref name="grid"/>, where all
    /// columns are subseries of the same length whose members are <paramref name="period"/> apart.
    /// </summary>
    private Tensor<T> SmoothSubseriesColumns(Tensor<T> grid, int period)
    {
        int count = grid.Shape[0];
        int windowSize = (int)(0.75 * count);

        // Pairwise distances |x_a - x_b| = period * |a - b| scaled by the half window.
        var index = Engine.TensorArange<T>(0, count, 1);
        var rowsIndex = Engine.TensorBroadcastTo(index.Reshape(new[] { count, 1 }), new[] { count, count });
        var colsIndex = Engine.TensorBroadcastTo(index.Reshape(new[] { 1, count }), new[] { count, count });
        var distance = Engine.TensorMultiplyScalar(
            Engine.TensorAbs(Engine.TensorSubtract(colsIndex, rowsIndex)), NumOps.FromDouble(period));
        var weights = TriCube(Engine.TensorMultiplyScalar(distance, NumOps.FromDouble(1.0 / (windowSize / 2.0))));

        var weightedSums = Engine.TensorMatMul(weights, grid);
        var totals = Engine.TensorMatMul(weights, Engine.TensorOnesLike(grid));
        return DivideElementwise(weightedSums, totals);
    }

    /// <summary>
    /// Applies a low-pass filter to the data
    /// </summary>
    /// <param name="data">The data to filter</param>
    /// <param name="period">The period used to determine the window size</param>
    /// <returns>The filtered data</returns>
    /// <remarks>
    /// <b>For Beginners:</b> A low-pass filter removes high-frequency variations (rapid changes) and keeps
    /// low-frequency variations (slow changes). It's like blurring a photo to remove fine details
    /// but keep the overall shapes. This helps isolate the trend component.
    /// </remarks>
    private Vector<T> LowPassFilter(Vector<T> data, int period)
    {
        int windowSize = period + 1;
        return CenteredWindowAverage(data, BoxKernel(windowSize / 2));
    }

    /// <summary>
    /// Applies LOESS (Locally Estimated Scatterplot Smoothing) to a vector of data
    /// </summary>
    /// <param name="data">The data to smooth</param>
    /// <param name="windowSize">The size of the window to use for smoothing</param>
    /// <returns>A smoothed version of the input data</returns>
    /// <remarks>
    /// <b>For Beginners:</b> LOESS smoothing is like drawing a smooth curve through noisy data points.
    /// It works by looking at a small "window" of nearby points around each position and calculating
    /// a weighted average, giving more importance to closer points. This helps reveal underlying
    /// patterns by reducing noise in the data.
    /// </remarks>
    private Vector<T> LoessSmoothing(Vector<T> data, int windowSize)
    {
        return CenteredWindowAverage(data, TriCubeKernel(windowSize / 2));
    }

    /// <summary>
    /// Calculates the trend component using a moving average method
    /// </summary>
    /// <returns>A vector containing the trend component of the time series</returns>
    /// <remarks>
    /// <b>For Beginners:</b> A moving average calculates the average of a "window" of data points
    /// that moves through the time series. For each position, it averages the values of nearby
    /// points (both before and after). This smooths out short-term fluctuations and highlights
    /// longer-term trends.
    /// </remarks>
    private Vector<T> CalculateTrendMovingAverage()
    {
        int _windowSize = 7;
        return CenteredWindowAverage(TimeSeries, BoxKernel(_windowSize / 2));
    }

    /// <summary>
    /// Calculates the seasonal component using the moving average method
    /// </summary>
    /// <param name="trend">The trend component previously calculated</param>
    /// <returns>A vector containing the seasonal component of the time series</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method finds repeating patterns in your data by:
    /// 1. Removing the trend (subtracting it from the original data)
    /// 2. Grouping what's left by season (e.g., all Januaries together)
    /// 3. Calculating the average pattern for each season
    /// 4. Using these averages as the seasonal component
    /// 
    /// For example, with monthly data, it identifies how much each month typically
    /// deviates from the trend.
    /// </remarks>
    private Vector<T> CalculateSeasonalMovingAverage(Vector<T> trend)
    {
        int _seasonalPeriod = 12;
        Vector<T> detrended = Engine.Subtract(TimeSeries, trend);

        // Average seasonal component for each period, repeated over the series.
        return RepeatPeriodic(PhaseMeans(detrended, _seasonalPeriod), TimeSeries.Length);
    }

    /// <summary>
    /// Calculates the trend component using exponential smoothing
    /// </summary>
    /// <returns>A vector containing the trend component of the time series</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Exponential smoothing calculates a trend by giving more weight to recent
    /// observations and less weight to older ones. It's like having a better memory of recent events.
    /// 
    /// The alpha parameter (between 0 and 1) controls how quickly the influence of past observations
    /// decreases - higher values make the trend respond more quickly to recent changes.
    /// </remarks>
    private Vector<T> CalculateTrendExponentialSmoothing()
    {
        T _alpha = NumOps.FromDouble(0.2); // Smoothing factor
        Vector<T> trend = new Vector<T>(TimeSeries.Length);
        trend[0] = TimeSeries[0];

        for (int i = 1; i < TimeSeries.Length; i++)
        {
            T prevSmoothed = trend[i - 1];
            T observation = TimeSeries[i];
            trend[i] = NumOps.Add(
                NumOps.Multiply(_alpha, observation),
                NumOps.Multiply(NumOps.Subtract(NumOps.One, _alpha), prevSmoothed)
            );
        }

        return trend;
    }

    /// <summary>
    /// Calculates the seasonal component using exponential smoothing
    /// </summary>
    /// <param name="trend">The trend component previously calculated</param>
    /// <returns>A vector containing the seasonal component of the time series</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method identifies repeating patterns in your data (like higher sales in December)
    /// using exponential smoothing. It works by:
    /// 
    /// 1. First calculating initial seasonal values by subtracting the trend from the original data
    /// 2. Then updating these seasonal values over time, giving more weight to recent observations
    /// 
    /// The gamma parameter (between 0 and 1) controls how quickly the seasonal pattern can change:
    /// - Higher gamma (closer to 1): Seasonal patterns adapt quickly to changes
    /// - Lower gamma (closer to 0): Seasonal patterns change slowly over time
    /// 
    /// This is useful for data where seasonal patterns might gradually evolve rather than
    /// staying exactly the same each year.
    /// </remarks>
    private Vector<T> CalculateSeasonalExponentialSmoothing(Vector<T> trend)
    {
        T _gamma = NumOps.FromDouble(0.3); // Seasonal smoothing factor
        int _seasonalPeriod = 12;
        Vector<T> seasonal = new Vector<T>(TimeSeries.Length);

        // Initialize seasonal components
        for (int i = 0; i < _seasonalPeriod; i++)
        {
            seasonal[i] = NumOps.Subtract(TimeSeries[i], trend[i]);
        }

        for (int i = _seasonalPeriod; i < TimeSeries.Length; i++)
        {
            int seasonIndex = i % _seasonalPeriod;
            T observation = TimeSeries[i];
            T levelTrend = trend[i];
            T prevSeasonal = seasonal[i - _seasonalPeriod];

            seasonal[i] = NumOps.Add(
                NumOps.Multiply(_gamma, NumOps.Subtract(observation, levelTrend)),
                NumOps.Multiply(NumOps.Subtract(NumOps.One, _gamma), prevSeasonal)
            );
        }

        return seasonal;
    }
}
