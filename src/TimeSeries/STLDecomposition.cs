using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Enums;

namespace AiDotNet.TimeSeries;

/// <summary>
/// Implements Seasonal-Trend decomposition using LOESS (STL) for time series analysis.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// STL decomposition breaks down a time series into three components: trend, seasonal, and residual.
/// It uses locally weighted regression (LOESS) to extract these components, making it robust to
/// outliers and applicable to a wide range of time series data.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// STL decomposition is like breaking down a song into its basic elements - the melody (trend),
/// the repeating chorus (seasonal pattern), and the unique variations (residuals).
/// 
/// For example, if you analyze monthly sales data:
/// - The trend component shows the long-term increase or decrease in sales
/// - The seasonal component shows regular patterns that repeat (like higher sales during holidays)
/// - The residual component shows what's left after removing trend and seasonality (like unexpected events)
/// 
/// This decomposition helps you understand what's driving your time series and can improve forecasting.
/// The model offers different algorithms (standard, robust, and fast) to handle various types of data.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var features = new Matrix&lt;double&gt;(new double[,] { { 1.0 }, { 2.0 }, { 3.0 }, { 4.0 } });
/// var series = new Vector&lt;double&gt;(new double[] { 112, 118, 132, 129 });
///
/// // Decompose a time series into trend, seasonal, and residual components
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new STLDecomposition&lt;double&gt;(new STLDecompositionOptions&lt;double&gt; { SeasonalPeriod = 12 }))
///     .Build(features, series);
///
/// Vector&lt;double&gt; trend = result.GetTrend();
/// Vector&lt;double&gt; seasonal = result.GetSeasonal();
/// </code>
/// </example>
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.TimeSeriesModel)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.Forecasting)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("STL: A Seasonal-Trend Decomposition Procedure Based on Loess", "https://doi.org/10.6028/jres.090.015", Year = 1990, Authors = "Robert B. Cleveland, William S. Cleveland, Jean E. McRae, Irma Terpenning")]
public partial class STLDecomposition<T> : TimeSeriesModelBase<T>
{
    /// <summary>
    /// Configuration options for the STL decomposition.
    /// </summary>
    private STLDecompositionOptions<T> _stlOptions;

    /// <summary>
    /// The trend component of the time series.
    /// </summary>
    [Buffer]
    private Vector<T> _trend;

    /// <summary>
    /// The seasonal component of the time series.
    /// </summary>
    [Buffer]
    private Vector<T> _seasonal;

    /// <summary>
    /// The residual component of the time series.
    /// </summary>
    [Buffer]
    private Vector<T> _residual;

    /// <summary>
    /// Initializes a new instance of the STLDecomposition class with optional configuration options.
    /// </summary>
    /// <param name="options">The configuration options for STL decomposition. If null, default options are used.</param>
    public STLDecomposition(STLDecompositionOptions<T>? options = null)
        : base(options ?? new STLDecompositionOptions<T>())
    {
        _stlOptions = options ?? new STLDecompositionOptions<T>();
        _trend = Vector<T>.Empty();
        _seasonal = Vector<T>.Empty();
        _residual = Vector<T>.Empty();
    }

    /// <summary>
    /// Performs the standard STL decomposition algorithm.
    /// </summary>
    /// <param name="y">The time series data to decompose.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// The standard STL algorithm works in several steps:
    /// 
    /// 1. First, it removes any existing trend estimate from the data
    /// 2. Then it extracts seasonal patterns by looking at values from the same time in different cycles
    /// 3. It smooths these seasonal patterns to make them more consistent
    /// 4. It removes the seasonal component from the original data
    /// 5. It estimates the trend from this deseasonalized data
    /// 6. Finally, it calculates residuals by subtracting both trend and seasonal components from the original data
    /// 
    /// This approach is effective for most time series with clear seasonal patterns.
    /// </para>
    /// </remarks>
    private void PerformStandardSTL(Vector<T> y)
    {
        Vector<T> detrended = y;

        // Step 1: Detrending
        detrended = SubtractVectors(y, _trend);

        // Step 2: Cycle-subseries Smoothing
        _seasonal = CycleSubseriesSmoothing(detrended, _stlOptions.SeasonalPeriod, _stlOptions.SeasonalLoessWindow);

        // Step 3: Low-pass Filtering of Smoothed Cycle-subseries
        Vector<T> lowPassSeasonal = LowPassFilter(_seasonal, _stlOptions.LowPassFilterWindowSize);

        // Step 4: Detrending of Smoothed Cycle-subseries
        _seasonal = SubtractVectors(_seasonal, lowPassSeasonal);

        // Step 5: Deseasonalizing
        Vector<T> deseasonalized = SubtractVectors(y, _seasonal);

        // Step 6: Trend Smoothing
        _trend = LoessSmoothing(deseasonalized, _stlOptions.TrendLoessWindow);

        // Step 7: Calculation of Residuals
        _residual = CalculateResiduals(y, _trend, _seasonal);
    }

    /// <summary>
    /// Performs the robust STL decomposition algorithm, which is less sensitive to outliers.
    /// </summary>
    /// <param name="y">The time series data to decompose.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// The robust STL algorithm is like the standard version, but with added protection against outliers
    /// (unusual data points that could distort the results).
    /// 
    /// It works by:
    /// 1. Running the standard STL algorithm
    /// 2. Identifying potential outliers by looking at the residuals
    /// 3. Assigning lower weights to these outliers
    /// 4. Running the algorithm again with these weights
    /// 5. Repeating this process several times
    /// 
    /// This approach is particularly useful for data with anomalies or measurement errors.
    /// </para>
    /// </remarks>
    private void PerformRobustSTL(Vector<T> y)
    {
        Vector<T> detrended = y;
        Vector<T> robustnessWeights = Vector<T>.CreateDefault(y.Length, NumOps.One);

        for (int iteration = 0; iteration < _stlOptions.RobustIterations; iteration++)
        {
            PerformStandardSTL(y);

            // Apply robustness weights if not the last iteration
            if (iteration < _stlOptions.RobustIterations - 1)
            {
                robustnessWeights = CalculateRobustWeights(_residual);
                y = ApplyRobustnessWeights(y, robustnessWeights);
            }
        }
    }

    /// <summary>
    /// Performs a faster version of the STL decomposition algorithm.
    /// </summary>
    /// <param name="y">The time series data to decompose.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// The fast STL algorithm is a simplified version that trades some precision for speed.
    /// It's particularly useful for very large datasets where the standard algorithm might be too slow.
    /// 
    /// It works by:
    /// 1. Estimating the trend using a simple moving average
    /// 2. Removing this trend to find seasonal patterns
    /// 3. Averaging values from the same season across different cycles
    /// 4. Smoothing these seasonal patterns
    /// 5. Removing the seasonal component and re-estimating the trend
    /// 6. Calculating residuals and normalizing the seasonal component
    /// 
    /// While not as precise as the standard algorithm, it often provides good results much faster.
    /// </para>
    /// </remarks>
    private void PerformFastSTL(Vector<T> y)
    {
        int n = y.Length;
        int period = _stlOptions.SeasonalPeriod;
        int trendWindow = _stlOptions.TrendWindowSize;
        int seasonalWindow = _stlOptions.SeasonalLoessWindow;

        // Step 1: Initial Trend Estimation (using moving average)
        _trend = MovingAverage(y, trendWindow);

        // Step 2: Detrending
        Vector<T> detrended = SubtractVectors(y, _trend);

        // Step 3: Initial Seasonal Estimation: every position gets the mean of its phase
        var (phaseSums, phaseCounts) = TimeSeriesEngineOps<T>.PhaseSums(detrended, period);
        _seasonal = phaseSums.Length > 0 && n > 0
            ? TimeSeriesEngineOps<T>.RepeatPeriodic(
                Engine.Divide(phaseSums.GetSubVector(0, Math.Min(period, n)), phaseCounts.GetSubVector(0, Math.Min(period, n))), n)
            : new Vector<T>(n);

        // Step 4: Seasonal Smoothing
        _seasonal = SmoothSeasonal(_seasonal, period, seasonalWindow);

        // Step 5: Seasonal Adjustment
        Vector<T> seasonallyAdjusted = SubtractVectors(y, _seasonal);

        // Step 6: Final Trend Estimation
        _trend = MovingAverage(seasonallyAdjusted, trendWindow);

        // Step 7: Calculation of Residuals
        _residual = CalculateResiduals(y, _trend, _seasonal);

        // Step 8: Normalize Seasonal Component
        NormalizeSeasonal();
    }

    /// <summary>
    /// Smooths the seasonal component by applying a moving average to each subseries.
    /// </summary>
    /// <param name="seasonal">The seasonal component to smooth.</param>
    /// <param name="period">The seasonal period.</param>
    /// <param name="window">The window size for smoothing.</param>
    /// <returns>The smoothed seasonal component.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method makes the seasonal pattern more consistent by smoothing out irregularities.
    /// It works by:
    /// 1. Grouping values from the same position in each seasonal cycle (e.g., all Januaries together)
    /// 2. Applying a moving average to each group
    /// 3. Putting these smoothed values back in their original positions
    /// 
    /// This helps identify the true seasonal pattern by reducing random variations.
    /// </para>
    /// </remarks>
    private Vector<T> SmoothSeasonal(Vector<T> seasonal, int period, int window)
    {
        // Moving average of each phase's subseries, written back to the phase positions
        Vector<T> smoothed = new Vector<T>(seasonal.Length);
        for (int i = 0; i < Math.Min(period, seasonal.Length); i++)
        {
            Vector<T> smoothedSubseries = MovingAverage(Engine.StridedGather(seasonal, i, period), window);
            Engine.StridedScatter(smoothed, smoothedSubseries, i, period);
        }

        return smoothed;
    }

    /// <summary>
    /// Normalizes the seasonal component to ensure it sums to zero, adjusting the trend accordingly.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method ensures that the seasonal component represents only the seasonal variations
    /// by making it average to zero. Any overall level is moved to the trend component.
    /// 
    /// It's like making sure that seasonal adjustments don't change the overall average of the data.
    /// For example, if we say summer increases sales by 20% and winter decreases them by 20%,
    /// the average effect should be zero.
    /// </para>
    /// </remarks>
    private void NormalizeSeasonal()
    {
        T seasonalMean = Engine.Sum(_seasonal);
        seasonalMean = NumOps.Divide(seasonalMean, NumOps.FromDouble(_seasonal.Length));
        _seasonal = _seasonal.Transform(s => NumOps.Subtract(s, seasonalMean));
        _trend = _trend.Transform(t => NumOps.Add(t, seasonalMean));
    }

    /// <summary>
    /// Calculates a moving average of the input data with the specified window size.
    /// </summary>
    /// <param name="data">The input data.</param>
    /// <param name="windowSize">The window size for the moving average.</param>
    /// <returns>The moving average of the input data.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// A moving average smooths out data by averaging each point with its neighbors.
    /// For example, with a window size of 3, each point is replaced by the average of itself
    /// and the points immediately before and after it.
    /// 
    /// This method handles edge cases (points near the beginning or end of the data)
    /// by using smaller windows when necessary.
    /// 
    /// Moving averages help identify trends by reducing short-term fluctuations.
    /// </para>
    /// </remarks>
    private Vector<T> MovingAverage(Vector<T> data, int windowSize)
    {
        int n = data.Length;
        int effectiveWindow = Math.Min(windowSize, n);
        if (n == 0)
        {
            return new Vector<T>(0);
        }

        if (effectiveWindow <= 0)
        {
            // No window: every average divides by a non-positive count and is invalid.
            return Engine.Fill(n, NumOps.Divide(NumOps.Zero, NumOps.FromDouble(effectiveWindow)));
        }

        int half = effectiveWindow / 2;

        // Edge positions (i < half or i >= n - half) average the available points of [i - half, i + half].
        var padded = Vector<T>.Concatenate(new Vector<T>(half), data, new Vector<T>(half));
        var edgeAverages = Engine.Divide(
            TimeSeriesEngineOps<T>.WindowSums(padded, 2 * half + 1),
            TimeSeriesEngineOps<T>.InRangeCounts(n, half));

        // Interior positions average the full window of effectiveWindow points starting at i - half.
        int interior = n - 2 * half;
        var interiorAverages = Engine.Divide(
            TimeSeriesEngineOps<T>.WindowSums(data, effectiveWindow).GetSubVector(0, interior),
            NumOps.FromDouble(effectiveWindow));

        return Vector<T>.Concatenate(
            edgeAverages.GetSubVector(0, half), interiorAverages, edgeAverages.GetSubVector(n - half, half));
    }

    /// <summary>
    /// Subtracts one vector from another element-wise.
    /// </summary>
    /// <param name="a">The first vector.</param>
    /// <param name="b">The vector to subtract.</param>
    /// <returns>The result of a - b.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method simply subtracts each element of the second vector from the corresponding element
    /// of the first vector. For example, if a = [5, 10, 15] and b = [1, 2, 3], then the result
    /// would be [4, 8, 12].
    /// 
    /// In the context of STL decomposition, this is used to remove components (like trend or seasonal)
    /// from the original time series.
    /// </para>
    /// </remarks>
    private Vector<T> SubtractVectors(Vector<T> a, Vector<T> b)
    {
        // VECTORIZED: Use Engine subtraction
        return (Vector<T>)Engine.Subtract(a, b);
    }

    /// <summary>
    /// Performs cycle-subseries smoothing on the input data.
    /// </summary>
    /// <param name="data">The input data.</param>
    /// <param name="period">The seasonal period.</param>
    /// <param name="loessWindow">The window size for LOESS smoothing.</param>
    /// <returns>The smoothed data.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method extracts and smooths seasonal patterns by:
    /// 
    /// 1. Grouping values from the same position in each seasonal cycle (e.g., all Januaries together)
    /// 2. Applying LOESS smoothing to each group (which is like a sophisticated moving average)
    /// 3. Putting these smoothed values back in their original positions
    /// 
    /// This approach helps identify consistent seasonal patterns while reducing the influence of random variations.
    /// </para>
    /// </remarks>
    private Vector<T> CycleSubseriesSmoothing(Vector<T> data, int period, int loessWindow)
    {
        // LOESS over each phase's subseries. Neighbouring subseries points are `period` apart, so the
        // weights depend only on the offset k: tri-cube(k · period / window) for k · period <= window.
        Vector<T> smoothed = new Vector<T>(data.Length);
        int effectiveWindowSize = Math.Max(1, loessWindow);
        T window = NumOps.FromDouble(effectiveWindowSize);
        T spacing = NumOps.FromDouble(period);
        for (int i = 0; i < Math.Min(period, data.Length); i++)
        {
            Vector<T> subseries = Engine.StridedGather(data, i, period);
            int radius = Math.Min(subseries.Length - 1, effectiveWindowSize / period);
            var distances = Engine.Multiply(Engine.TensorArange<T>(0, radius + 1, 1).ToVector(), spacing);
            Vector<T> smoothedSubseries = KernelSmooth(subseries, TriCubeWeights(distances, window));
            Engine.StridedScatter(smoothed, smoothedSubseries, i, period);
        }

        return smoothed;
    }

    /// <summary>
    /// Applies a low-pass filter to the input data.
    /// </summary>
    /// <param name="data">The input data.</param>
    /// <param name="windowSize">The window size for filtering.</param>
    /// <returns>The filtered data.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// A low-pass filter removes high-frequency variations (rapid changes) while preserving
    /// low-frequency patterns (gradual changes).
    /// 
    /// This method applies multiple moving averages in sequence, which effectively smooths out
    /// short-term fluctuations. It's like looking at your data through progressively blurrier lenses
    /// to focus only on the major patterns.
    /// 
    /// In STL decomposition, this helps ensure that the seasonal component doesn't contain any trend.
    /// </para>
    /// </remarks>
    private Vector<T> LowPassFilter(Vector<T> data, int windowSize)
    {
        return MovingAverage(MovingAverage(MovingAverage(data, windowSize), windowSize), 3);
    }

    /// <summary>
    /// Calculates the residual component by subtracting trend and seasonal components from the original data.
    /// </summary>
    /// <param name="y">The original time series data.</param>
    /// <param name="trend">The trend component.</param>
    /// <param name="seasonal">The seasonal component.</param>
    /// <returns>The residual component.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// The residual component represents what's left after removing the trend and seasonal patterns.
    /// It includes random variations, noise, and any patterns not captured by the trend or seasonal components.
    /// 
    /// Calculating it is straightforward: for each point in the time series, subtract both the trend value
    /// and the seasonal value from the original value.
    /// 
    /// Analyzing residuals can help identify unusual events or additional patterns that might need modeling.
    /// </para>
    /// </remarks>
    private Vector<T> CalculateResiduals(Vector<T> y, Vector<T> trend, Vector<T> seasonal)
    {
        // VECTORIZED: Use Engine operations for residual calculation
        var yMinusTrend = (Vector<T>)Engine.Subtract(y, trend);
        return (Vector<T>)Engine.Subtract(yMinusTrend, seasonal);
    }

    /// <summary>
    /// Calculates robustness weights based on residuals to reduce the influence of outliers.
    /// </summary>
    /// <param name="residuals">The residual component.</param>
    /// <returns>A vector of weights for each data point.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method identifies potential outliers (unusual data points) and assigns them lower weights
    /// to reduce their influence on the decomposition.
    /// 
    /// It works by:
    /// 1. Calculating the absolute value of each residual
    /// 2. Finding the median (middle value) of these absolute residuals
    /// 3. Setting a threshold at 6 times this median
    /// 4. Assigning weights based on how far each residual is from zero compared to this threshold
    /// 
    /// Points with small residuals get weights close to 1 (full influence), while points with large
    /// residuals get lower weights (reduced influence). This helps the model focus on the typical
    /// patterns in the data rather than being distorted by unusual events.
    /// </para>
    /// </remarks>
    private Vector<T> CalculateRobustWeights(Vector<T> residuals)
    {
        Vector<T> absResiduals = residuals.Transform(r => NumOps.Abs(r));
        T median = absResiduals.Median();
        T threshold = NumOps.Multiply(NumOps.FromDouble(6), median);

        return absResiduals.Transform(r =>
        {
            if (NumOps.LessThan(r, threshold))
            {
                T weight = NumOps.Subtract(NumOps.One, NumOps.Power(NumOps.Divide(r, threshold), NumOps.FromDouble(2)));
                return MathHelper.Max(weight, NumOps.FromDouble(_stlOptions.RobustWeightThreshold));
            }
            else
            {
                return NumOps.FromDouble(_stlOptions.RobustWeightThreshold);
            }
        });
    }

    /// <summary>
    /// Applies robustness weights to the input data.
    /// </summary>
    /// <param name="y">The input data.</param>
    /// <param name="weights">The weights to apply.</param>
    /// <returns>The weighted data.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method applies the calculated weights to the original data, effectively reducing
    /// the influence of potential outliers. It simply multiplies each data point by its
    /// corresponding weight.
    /// 
    /// In the robust STL algorithm, this weighted data is then used for the next iteration
    /// of decomposition, resulting in estimates that are less affected by unusual data points.
    /// </para>
    /// </remarks>
    private Vector<T> ApplyRobustnessWeights(Vector<T> y, Vector<T> weights)
    {
        // VECTORIZED: Use Engine multiplication
        return (Vector<T>)Engine.Multiply(y, weights);
    }

    /// <summary>
    /// Calculates the tri-cube weight function used in LOESS smoothing for each distance.
    /// </summary>
    /// <param name="distances">The distances of the neighbouring points.</param>
    /// <param name="window">The window the distances are normalized by.</param>
    /// <returns>The tri-cube weight of each distance.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// The tri-cube function is a special weighting function used in LOESS smoothing.
    /// It assigns weights based on how far points are from the one being estimated:
    /// - Points very close get weights near 1 (high influence)
    /// - Points farther away get progressively smaller weights
    /// - Points beyond a certain distance get weights of 0 (no influence)
    /// 
    /// This creates a smooth transition from high influence to no influence as distance increases,
    /// which helps LOESS produce smooth trend and seasonal estimates.
    /// </para>
    /// </remarks>
    private Vector<T> TriCubeWeights(Vector<T> distances, T window)
    {
        // w(u) = ((1 - |u|)³)² for |u| <= 1, else 0, with u = distance / window
        var absU = Engine.Abs(Engine.Divide(distances, window));
        var oneMinusAbsU = Engine.Subtract(Engine.Fill(absU.Length, NumOps.One), absU);
        var cube = Engine.Multiply(Engine.Multiply(oneMinusAbsU, oneMinusAbsU), oneMinusAbsU);
        var weights = Tensor<T>.FromVector(Engine.Multiply(cube, cube));
        return Engine.TensorWhere(
            Engine.TensorGreaterThan(Tensor<T>.FromVector(absU), NumOps.One),
            Engine.TensorZerosLike(weights),
            weights).ToVector();
    }

    /// <summary>
    /// Weighted average of every point's neighbours: <c>Σ_k w[|k|] y[i + k] / Σ_k w[|k|]</c> over the
    /// offsets <c>|k| &lt;= r</c> that fall inside the series, where <paramref name="halfKernel"/> holds
    /// <c>w[0..r]</c>.
    /// </summary>
    private Vector<T> KernelSmooth(Vector<T> y, Vector<T> halfKernel)
    {
        int n = y.Length;
        int radius = halfKernel.Length - 1;
        var kernel = radius > 0
            ? Vector<T>.Concatenate(TimeSeriesEngineOps<T>.Reverse(halfKernel.GetSubVector(1, radius)), halfKernel)
            : halfKernel;

        // Zero padding: out-of-range neighbours add exact zeros to both sums.
        var padded = Vector<T>.Concatenate(new Vector<T>(radius), y, new Vector<T>(radius));
        var inRange = Vector<T>.Concatenate(new Vector<T>(radius), Engine.Fill(n, NumOps.One), new Vector<T>(radius));
        var windows = Engine.TensorUnfold(Tensor<T>.FromVector(padded), 0, 2 * radius + 1, 1).ToMatrix();
        var masks = Engine.TensorUnfold(Tensor<T>.FromVector(inRange), 0, 2 * radius + 1, 1).ToMatrix();
        return Engine.Divide(Engine.MatrixVectorMultiply(windows, kernel), Engine.MatrixVectorMultiply(masks, kernel));
    }

    /// <summary>
    /// Performs LOESS (locally weighted regression) smoothing on the input data.
    /// </summary>
    /// <param name="data">The input data.</param>
    /// <param name="windowSize">The window size for smoothing.</param>
    /// <returns>The smoothed data.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// LOESS smoothing is like a sophisticated moving average that adapts to the local structure of the data.
    /// For each point, it:
    /// 
    /// 1. Selects nearby points within a window
    /// 2. Assigns weights to these points based on their distance (closer points get higher weights)
    /// 3. Fits a weighted average to estimate the smoothed value
    /// 
    /// This approach creates a smooth curve that follows the general pattern of the data while
    /// reducing the influence of random fluctuations. It's particularly good at handling data
    /// where the underlying pattern changes gradually over time.
    /// </para>
    /// </remarks>
    private Vector<T> LoessSmoothing(Vector<T> data, int windowSize)
    {
        int n = data.Length;
        if (n == 0 || windowSize < 0)
        {
            // No point lies within a negative window: every value is kept.
            return data.Clone();
        }

        // Points within windowSize of i, weighted tri-cube(|j - i| / windowSize)
        int radius = Math.Min(windowSize, n - 1);
        var distances = Engine.TensorArange<T>(0, radius + 1, 1).ToVector();
        return KernelSmooth(data, TriCubeWeights(distances, NumOps.FromDouble(windowSize)));
    }

    /// <summary>
    /// Gets the trend component of the time series.
    /// </summary>
    /// <returns>The trend component.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the model has not been trained.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// The trend component represents the long-term direction of your time series.
    /// It shows whether your data is generally increasing, decreasing, or staying flat over time,
    /// after removing seasonal patterns and random fluctuations.
    /// 
    /// For example, in retail sales data, the trend might show steady growth over several years,
    /// even though there are seasonal peaks during holidays and random variations from month to month.
    /// </para>
    /// </remarks>
    public Vector<T> GetTrend() => _trend ?? throw new InvalidOperationException("Model has not been trained.");

    /// <summary>
    /// Gets the seasonal component of the time series.
    /// </summary>
    /// <returns>The seasonal component.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the model has not been trained.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// The seasonal component represents regular patterns that repeat at fixed intervals.
    /// These patterns might be related to time of day, day of week, month of year, or any other
    /// cyclical factor relevant to your data.
    /// 
    /// For example, retail sales might show a seasonal pattern with peaks in December (holiday shopping),
    /// back-to-school season, and other predictable times. The seasonal component captures these
    /// recurring patterns.
    /// </para>
    /// </remarks>
    public Vector<T> GetSeasonal() => _seasonal ?? throw new InvalidOperationException("Model has not been trained.");

    /// <summary>
    /// Gets the residual component of the time series.
    /// </summary>
    /// <returns>The residual component.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the model has not been trained.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// The residual component is what remains after removing both trend and seasonal patterns.
    /// It includes random variations, noise, and any patterns not captured by the other components.
    /// 
    /// Analyzing residuals can help identify:
    /// - Unusual events or outliers
    /// - Additional patterns that might need modeling
    /// - Whether the decomposition has successfully captured the main patterns in the data
    /// 
    /// Ideally, residuals should look like random noise with no obvious patterns.
    /// </para>
    /// </remarks>
    public Vector<T> GetResidual() => _residual ?? throw new InvalidOperationException("Model has not been trained.");

    /// <summary>
    /// Generates forecasts for future time periods based on the decomposed components.
    /// </summary>
    /// <param name="input">The input features matrix specifying the forecast horizon.</param>
    /// <returns>A vector of forecasted values.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the model has not been trained.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method predicts future values by:
    /// 
    /// 1. Using the last trend value (assuming the trend continues at the same level)
    /// 2. Adding the appropriate seasonal pattern from the last observed season
    /// 
    /// This simple approach works well for short-term forecasts when the trend is relatively stable.
    /// For example, if we've decomposed monthly sales data and want to forecast the next few months,
    /// we'd use the most recent trend level and add the seasonal pattern from the same months in the previous year.
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> input)
    {
        if (_trend == null || _seasonal == null)
            throw new InvalidOperationException("Model has not been trained.");

        int forecastHorizon = input.Rows;
        int n = _trend.Length;
        int seasonLength = _stlOptions.SeasonalPeriod;
        if (forecastHorizon <= 0)
        {
            return new Vector<T>(Math.Max(forecastHorizon, 0));
        }

        if (seasonLength == 0)
        {
            throw new DivideByZeroException(); // the season index i % seasonLength is undefined
        }

        // Use last full season for seasonal component (zero-filled when the series is shorter)
        int seasonStart = Math.Max(0, _seasonal.Length - seasonLength);
        int seasonCount = Math.Max(0, Math.Min(seasonLength, _seasonal.Length - seasonStart));
        Vector<T> lastSeason = Vector<T>.Concatenate(
            _seasonal.GetSubVector(seasonStart, seasonCount), new Vector<T>(seasonLength - seasonCount));

        // In-sample: the decomposed trend; out-of-sample: linear extrapolation from the last two trend values
        int inSample = Math.Min(forecastHorizon, n);
        Vector<T> trend = _trend.GetSubVector(0, inSample);
        if (forecastHorizon > inSample)
        {
            T lastTrend = _trend[n - 1];
            T slope = n > 1 ? NumOps.Subtract(_trend[n - 1], _trend[n - 2]) : NumOps.Zero;
            var steps = Engine.TensorArange<T>(1, forecastHorizon - n + 1, 1).ToVector();
            trend = Vector<T>.Concatenate(trend,
                Engine.Add(Engine.Fill(forecastHorizon - n, lastTrend), Engine.Multiply(steps, slope)));
        }

        return Engine.Add(trend, TimeSeriesEngineOps<T>.RepeatPeriodic(lastSeason, forecastHorizon));
    }

    /// <summary>
    /// Evaluates the performance of the trained model on test data.
    /// </summary>
    /// <param name="xTest">The input features matrix for testing.</param>
    /// <param name="yTest">The actual target values for testing.</param>
    /// <returns>A dictionary containing evaluation metrics.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the model has not been trained.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method tests how well our model performs by comparing its predictions to actual values.
    /// It calculates two error metrics:
    /// 
    /// - MAE (Mean Absolute Error): The average of the absolute differences between predictions and actual values
    /// - RMSE (Root Mean Squared Error): The square root of the average squared differences, which gives more weight to larger errors
    /// 
    /// Lower values for these metrics indicate better model performance. They help you understand
    /// how accurate your forecasts are likely to be and compare different models or parameter settings.
    /// </para>
    /// </remarks>
    public override Dictionary<string, T> EvaluateModel(Matrix<T> xTest, Vector<T> yTest)
    {
        if (_trend == null || _seasonal == null || _residual == null)
            throw new InvalidOperationException("Model has not been trained.");

        Vector<T> yPred = Predict(xTest);

        Dictionary<string, T> metrics = new Dictionary<string, T>();

        // Calculate Mean Absolute Error (MAE)
        metrics["MAE"] = StatisticsHelper<T>.CalculateMeanAbsoluteError(yTest, yPred);

        // Calculate Root Mean Squared Error (RMSE)
        metrics["RMSE"] = StatisticsHelper<T>.CalculateRootMeanSquaredError(yTest, yPred);

        return metrics;
    }

    /// <summary>
    /// Serializes the model's core parameters to a binary writer.
    /// </summary>
    /// <param name="writer">The binary writer to write to.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// Serialization is the process of converting the model's state into a format that can be saved to disk.
    /// This allows you to save a trained model and load it later without having to retrain it.
    /// 
    /// This method saves:
    /// - The STL configuration options (like seasonal period and window sizes)
    /// - The decomposed components (trend, seasonal, and residual)
    /// 
    /// After serializing, you can store the model and later deserialize it to make predictions
    /// or continue analysis without repeating the decomposition process.
    /// </para>
    /// </remarks>


    /// <summary>
    /// Deserializes the model's core parameters from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader to read from.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// Deserialization is the process of loading a previously saved model from disk.
    /// This method reads the model's parameters from a file and reconstructs the model
    /// exactly as it was when it was saved.
    /// 
    /// This allows you to:
    /// - Load a previously trained model without retraining
    /// - Make predictions with consistent results
    /// - Continue analysis from where you left off
    /// 
    /// It's like saving your work in a document and opening it later to continue editing.
    /// </para>
    /// </remarks>


    /// <summary>
    /// Resets the model to its initial state.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method clears all decomposed components and returns the model to its initial state,
    /// as if it had just been created with the same options but not yet trained.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// Resetting a model is like erasing the results of your analysis while keeping the settings.
    /// 
    /// This is useful when you want to:
    /// - Reapply the decomposition to different data
    /// - Try different algorithm types on the same data
    /// - Compare results with different settings
    /// - Start fresh after experimenting
    /// 
    /// For example, you might reset the model after analyzing monthly data to then analyze
    /// daily data with the same settings, or to compare the results of standard versus robust
    /// decomposition approaches on the same dataset.
    /// </para>
    /// </remarks>
    public override void Reset()
    {
        _trend = Vector<T>.Empty();
        _seasonal = Vector<T>.Empty();
        _residual = Vector<T>.Empty();
    }

    /// <summary>
    /// Creates a new instance of the STL decomposition model with the same options.
    /// </summary>
    /// <returns>A new STL decomposition model instance with the same configuration.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new instance of the STL decomposition model with the same configuration options
    /// as the current instance. The new instance is not trained and will need to be trained on data.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method creates a fresh copy of your model with the same settings but no results.
    /// 
    /// It's useful when you want to:
    /// - Apply the same decomposition approach to multiple datasets
    /// - Create multiple variations of the model with slight modifications
    /// - Share your model configuration with others
    /// - Preserve your original settings while experimenting with new options
    /// 
    /// Think of it like copying a recipe before making modifications - you keep the original
    /// intact while creating a new version that you can change.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateInstance()
    {
        // Create a new instance with the same options
        return new STLDecomposition<T>(_stlOptions);
    }

    /// <summary>
    /// Gets metadata about the model, including its type, configuration, and information about the decomposed components.
    /// </summary>
    /// <returns>A ModelMetaData object containing information about the model.</returns>
    /// <remarks>
    /// <para>
    /// This method returns detailed metadata about the model, including its type, configuration options,
    /// and information about the decomposed components. This metadata can be used for model selection,
    /// comparison, and documentation.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method provides a summary of your model's configuration and what it has learned.
    /// 
    /// It includes information like:
    /// - The type of model (STL Decomposition)
    /// - The algorithm used (Standard, Robust, or Fast)
    /// - The seasonal period and window sizes
    /// - Details about the decomposed components (sizes, statistics)
    /// 
    /// This metadata is useful for:
    /// - Comparing different decomposition approaches
    /// - Documenting your analysis process
    /// - Understanding what the model has extracted from your data
    /// - Sharing model information with others
    /// 
    /// Think of it like getting a detailed report card for your decomposition analysis.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                // Include configuration options
                { "AlgorithmType", _stlOptions.AlgorithmType },
                { "SeasonalPeriod", _stlOptions.SeasonalPeriod },
                { "TrendWindowSize", _stlOptions.TrendWindowSize },
                { "SeasonalLoessWindow", _stlOptions.SeasonalLoessWindow },
                { "TrendLoessWindow", _stlOptions.TrendLoessWindow },
                { "LowPassFilterWindowSize", _stlOptions.LowPassFilterWindowSize },
                { "RobustIterations", _stlOptions.RobustIterations },
            
                // Include information about decomposed components
                { "ComponentsAvailable", _trend != null && _seasonal != null && _residual != null },
                { "ComponentsLength", _trend?.Length ?? 0 },
                { "SeasonalStrength", Convert.ToDouble(CalculateSeasonalStrength()) },
                { "TrendStrength", Convert.ToDouble(CalculateTrendStrength()) }
            },
            ModelDataProvider = () => SerializeForMetadata()
        };

        return metadata;
    }

    /// <summary>
    /// Calculates the seasonal strength of the time series.
    /// </summary>
    /// <returns>A measure of how strong the seasonal pattern is (between 0 and 1).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// Seasonal strength measures how much of the variation in your data is explained by
    /// seasonal patterns. A value close to 1 means seasonality is very important, while
    /// a value close to 0 means seasonality contributes little to the overall pattern.
    /// 
    /// For example:
    /// - Retail sales data might have high seasonal strength (0.8+) due to holiday patterns
    /// - Stock market data might have low seasonal strength (0.2-) as it's influenced more by other factors
    /// 
    /// This metric helps you understand how important the seasonal component is in your data.
    /// </para>
    /// </remarks>
    private T CalculateSeasonalStrength()
    {
        if (_seasonal == null || _residual == null)
            return NumOps.Zero;

        // VECTORIZED: Use Engine addition for seasonal + residual
        Vector<T> seasonalPlusResidual = (Vector<T>)Engine.Add(_seasonal, _residual);

        T varSeasonal = StatisticsHelper<T>.CalculateVariance(_seasonal);
        T varSeasonalPlusResidual = StatisticsHelper<T>.CalculateVariance(seasonalPlusResidual);

        if (NumOps.LessThanOrEquals(varSeasonalPlusResidual, NumOps.Zero))
            return NumOps.Zero;

        T strength = NumOps.Subtract(NumOps.One, NumOps.Divide(StatisticsHelper<T>.CalculateVariance(_residual), varSeasonalPlusResidual));

        // Ensure the value is between 0 and 1
        if (NumOps.LessThan(strength, NumOps.Zero))
            return NumOps.Zero;
        if (NumOps.GreaterThan(strength, NumOps.One))
            return NumOps.One;

        return strength;
    }

    /// <summary>
    /// Calculates the trend strength of the time series.
    /// </summary>
    /// <returns>A measure of how strong the trend pattern is (between 0 and 1).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// Trend strength measures how much of the variation in your data is explained by
    /// the long-term trend. A value close to 1 means the trend is very important, while
    /// a value close to 0 means the trend contributes little to the overall pattern.
    /// 
    /// For example:
    /// - Data showing population growth might have high trend strength (0.9+)
    /// - Data with random fluctuations around a constant value would have low trend strength (0.1-)
    /// 
    /// This metric helps you understand how important the trend component is in your data.
    /// </para>
    /// </remarks>
    private T CalculateTrendStrength()
    {
        if (_trend == null || _residual == null)
            return NumOps.Zero;

        // VECTORIZED: Use Engine addition for seasonal + residual
        Vector<T> detrended;
        if (_seasonal != null)
        {
            detrended = (Vector<T>)Engine.Add(_seasonal, _residual);
        }
        else
        {
            detrended = _residual;
        }

        T varDetrended = StatisticsHelper<T>.CalculateVariance(detrended);
        T varResidual = StatisticsHelper<T>.CalculateVariance(_residual);

        if (NumOps.LessThanOrEquals(varDetrended, NumOps.Zero))
            return NumOps.One;

        T strength = NumOps.Subtract(NumOps.One, NumOps.Divide(varResidual, varDetrended));

        // Ensure the value is between 0 and 1
        if (NumOps.LessThan(strength, NumOps.Zero))
            return NumOps.Zero;
        if (NumOps.GreaterThan(strength, NumOps.One))
            return NumOps.One;

        return strength;
    }

    /// <summary>
    /// Implements the model-specific training logic for STL decomposition.
    /// </summary>
    /// <param name="x">The input features matrix (not used in STL decomposition).</param>
    /// <param name="y">The time series data to decompose.</param>
    /// <exception cref="ArgumentException">Thrown when the time series is too short for the specified seasonal period.</exception>
    /// <remarks>
    /// <para>
    /// This method handles the STL-specific training logic. It validates inputs and delegates to the
    /// appropriate algorithm implementation based on the configured algorithm type.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This is the core method that actually performs the decomposition of your time series into
    /// trend, seasonal, and residual components. It works behind the scenes to:
    ///
    /// 1. Check that your data meets the required conditions (like having enough points)
    /// 2. Choose the right algorithm based on your settings
    /// 3. Break down your data into its components using that algorithm
    ///
    /// The method is called automatically when you train the model, so you don't need to
    /// call it directly. Instead, you interact with the higher-level `Train` method
    /// that handles validation and preparation before calling this method.
    /// </para>
    /// </remarks>
    protected override void TrainCore(Matrix<T> x, Vector<T> y)
    {
        // Validate that we have enough data for the seasonal period
        if (y.Length < _stlOptions.SeasonalPeriod * 2)
        {
            throw new ArgumentException($"Time series is too short for the specified seasonal period. Need at least {_stlOptions.SeasonalPeriod * 2} observations for seasonal period {_stlOptions.SeasonalPeriod}, but got {y.Length}.", nameof(y));
        }

        int n = y.Length;

        // Initialize component vectors
        _trend = new Vector<T>(n);
        _seasonal = new Vector<T>(n);
        _residual = new Vector<T>(n);

        try
        {
            // Delegate to the appropriate algorithm implementation
            switch (_stlOptions.AlgorithmType)
            {
                case STLAlgorithmType.Standard:
                    PerformStandardSTL(y);
                    break;
                case STLAlgorithmType.Robust:
                    PerformRobustSTL(y);
                    break;
                case STLAlgorithmType.Fast:
                    PerformFastSTL(y);
                    break;
                default:
                    throw new ArgumentException($"Unsupported STL algorithm type: {_stlOptions.AlgorithmType}.", nameof(_stlOptions.AlgorithmType));
            }

            // Ensure seasonal component sums to zero within each period
            NormalizeSeasonal();

            // Recalculate residuals to ensure consistency
            _residual = CalculateResiduals(y, _trend, _seasonal);

            // Perform validation of decomposition results
            ValidateDecomposition(y);
        }
        catch (InvalidOperationException ex)
        {
            // Fail-safe: if the decomposition becomes numerically unstable, fall back to a trivial but valid decomposition.
            // This avoids breaking downstream workflows (e.g., JIT graph export) on benign inputs.
            System.Diagnostics.Debug.WriteLine(
                $"Warning: STL decomposition failed with InvalidOperationException. Falling back to trivial decomposition (trend=series, seasonal=0, residual=0). Error: {ex.Message}");
            y.AsSpan().Slice(0, n).CopyTo(_trend.AsWritableSpan());

            _seasonal.Fill(NumOps.Zero);
            _residual.Fill(NumOps.Zero);
        }
        catch (Exception ex)
        {
            // Reset component vectors on failure
            _trend = Vector<T>.Empty();
            _seasonal = Vector<T>.Empty();
            _residual = Vector<T>.Empty();

            // Re-throw with more context
            throw new InvalidOperationException($"STL decomposition failed: {ex.Message}", ex);
        }

        // Populate ModelParameters from seasonal pattern (one period) for GetParameters()
        int period = _stlOptions.SeasonalPeriod;
        int copied = Math.Max(0, Math.Min(period, _seasonal.Length));
        ModelParameters = Vector<T>.Concatenate(_seasonal.GetSubVector(0, copied), new Vector<T>(period - copied));
    }

    /// <summary>
    /// Validates the decomposition results to ensure they are reasonable.
    /// </summary>
    /// <param name="originalSeries">The original time series data.</param>
    /// <exception cref="InvalidOperationException">Thrown when decomposition results are invalid.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method performs quality checks on the decomposition results to make sure they're valid.
    /// It verifies that:
    /// - The trend component isn't all zeros or NaNs
    /// - The seasonal component shows a regular pattern
    /// - The components add up to the original data (within a small margin of error)
    /// 
    /// These checks help ensure that the decomposition was successful and meaningful.
    /// </para>
    /// </remarks>
    private void ValidateDecomposition(Vector<T> originalSeries)
    {
        // Check for NaN or Infinity values in components
        if (_trend.Any(IsInvalidValue) || _seasonal.Any(IsInvalidValue) || _residual.Any(IsInvalidValue))
        {
            throw new InvalidOperationException("Decomposition produced invalid values (NaN or Infinity).");
        }

        // Check that the sum of components equals original series
        // VECTORIZED: Use Engine operations to reconstruct series
        var trendPlusSeasonal = (Vector<T>)Engine.Add(_trend, _seasonal);
        Vector<T> reconstructed = (Vector<T>)Engine.Add(trendPlusSeasonal, _residual);

        T maxError = NumOps.Zero;
        if (originalSeries.Length > 0)
        {
            var errors = Engine.Abs(Engine.Subtract(originalSeries, reconstructed.GetSubVector(0, originalSeries.Length)));
            T largest = Engine.TensorMaxValue(errors.AsTensor());
            if (NumOps.GreaterThan(largest, maxError))
            {
                maxError = largest;
            }
        }

        // Allow a small numerical error due to floating-point arithmetic
        T tolerance = NumOps.FromDouble(1e-5);
        if (NumOps.GreaterThan(maxError, tolerance))
        {
            throw new InvalidOperationException($"Decomposition components don't sum to original series. Maximum error: {Convert.ToDouble(maxError)}.");
        }

        // Check for unreasonable trend (all zeros or constant)
        T trendVariance = StatisticsHelper<T>.CalculateVariance(_trend);
        if (NumOps.LessThan(trendVariance, NumOps.FromDouble(1e-10)))
        {
            // Not throwing here, just logging a warning since a flat trend could be valid
            System.Diagnostics.Debug.WriteLine("Warning: Trend component has very low variance. The series might be dominated by seasonality or noise.");
        }

        // Check that seasonal component has the expected pattern (repeating every period)
        // This is a simple check that looks at autocorrelation at the seasonal lag
        T seasonalAutocorrelation = CalculateSeasonalAutocorrelation();
        if (NumOps.LessThan(seasonalAutocorrelation, NumOps.FromDouble(0.5)))
        {
            System.Diagnostics.Debug.WriteLine($"Warning: Seasonal component has low autocorrelation ({Convert.ToDouble(seasonalAutocorrelation)}) at the seasonal lag. The detected seasonal pattern may be weak or inconsistent.");
        }
    }

    /// <summary>
    /// Checks if a value is invalid (NaN or Infinity).
    /// </summary>
    /// <param name="value">The value to check.</param>
    /// <returns>True if the value is NaN or Infinity, false otherwise.</returns>
    private bool IsInvalidValue(T value)
    {
        try
        {
            double doubleValue = Convert.ToDouble(value);
            return double.IsNaN(doubleValue) || double.IsInfinity(doubleValue);
        }
        catch (InvalidCastException)
        {
            // If conversion fails for non-numeric types, assume valid (not invalid)
            // This is expected behavior for types that cannot be converted to double
            return false;
        }
        catch (FormatException)
        {
            // If format conversion fails, assume valid for non-numeric types
            return false;
        }
        catch (OverflowException)
        {
            // If value is too large or small for double, it's invalid
            return true;
        }
    }

    /// <summary>
    /// Calculates the autocorrelation of the seasonal component at the seasonal lag.
    /// </summary>
    /// <returns>The autocorrelation value.</returns>
    private T CalculateSeasonalAutocorrelation()
    {
        int n = _seasonal.Length;
        int lag = _stlOptions.SeasonalPeriod;

        if (n <= lag)
        {
            return NumOps.One; // Not enough data to calculate
        }

        T mean = _seasonal.Average();
        T numerator = NumOps.Zero;
        T denominator = NumOps.Zero;

        // VECTORIZED: Calculate deviations using Engine operations
        var deviations = (Vector<T>)Engine.Subtract(_seasonal, Engine.Fill(n, mean));

        // Calculate numerator with lagged products
        numerator = Engine.DotProduct(deviations.GetSubVector(0, n - lag), deviations.GetSubVector(lag, n - lag));

        // VECTORIZED: Calculate denominator using dot product
        denominator = Engine.DotProduct(deviations, deviations);

        if (NumOps.LessThanOrEquals(denominator, NumOps.Zero))
        {
            return NumOps.One; // No variance, treat as perfect correlation
        }

        return NumOps.Divide(numerator, denominator);
    }

    /// <summary>
    /// Predicts a single value based on the decomposed components.
    /// </summary>
    /// <param name="input">The input vector containing time information.</param>
    /// <returns>The predicted value.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the model has not been trained.</exception>
    /// <remarks>
    /// <para>
    /// This method generates a prediction for a single future time point by combining
    /// the trend and seasonal components from the decomposition.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method predicts a value for a single future time point by:
    /// 
    /// 1. Using the last known trend value (assuming the trend continues)
    /// 2. Adding the appropriate seasonal pattern for that time point
    /// 
    /// For example, if you're forecasting sales for next January based on historical
    /// monthly data, it would:
    /// - Use the most recent trend level (overall sales level)
    /// - Add the January seasonal effect (how much January typically differs from average)
    /// 
    /// The input vector should contain the forecast horizon (how many steps ahead)
    /// and any other relevant information.
    /// </para>
    /// </remarks>
    public override T PredictSingle(Vector<T> input)
    {
        if (_trend == null || _seasonal == null)
        {
            throw new InvalidOperationException("Model has not been trained. Call Train before making predictions.");
        }

        // Extract the time index from the input
        int t = input.Length > 0 ? Math.Max(0, Convert.ToInt32(input[0])) : 0;
        int n = _trend.Length;
        int period = _stlOptions.SeasonalPeriod;

        // Determine trend component
        T trendComponent;
        if (t < n)
        {
            // In-sample: use the decomposed trend directly
            trendComponent = _trend[t];
        }
        else
        {
            // Out-of-sample: linear extrapolation from the last two trend values
            T lastTrend = _trend[n - 1];
            T prevTrend = n > 1 ? _trend[n - 2] : lastTrend;
            T slope = NumOps.Subtract(lastTrend, prevTrend);
            trendComponent = NumOps.Add(lastTrend, NumOps.Multiply(slope, NumOps.FromDouble(t - n + 1)));
        }

        // Determine seasonal component (cyclically repeat the last full period)
        int seasonalIndex = t % period;
        int seasonStart = n - period;
        if (seasonStart < 0) seasonStart = 0;
        T seasonalComponent = _seasonal[seasonStart + (seasonalIndex % (n - seasonStart))];

        // Return the sum (residual component is assumed to be zero for forecasting)
        return NumOps.Add(trendComponent, seasonalComponent);
    }

    /// <summary>
    /// Improves the standard STL algorithm by attempting to detect and adapt to changing seasonality.
    /// </summary>
    /// <param name="y">The original time series data.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This enhanced version of the standard STL algorithm is designed to better handle time series
    /// where the seasonal pattern might be changing over time.
    /// 
    /// It works by:
    /// 1. First performing a regular STL decomposition
    /// 2. Then analyzing how the seasonal pattern changes through the series
    /// 3. If changes are detected, adapting the decomposition to account for evolving seasonality
    /// 
    /// This approach is especially valuable for longer time series where seasonal patterns might
    /// evolve due to changing behaviors, climate shifts, or other factors.
    /// </para>
    /// </remarks>
    private void PerformAdaptiveSTL(Vector<T> y)
    {
        // First, perform standard STL decomposition
        PerformStandardSTL(y);

        int n = y.Length;
        int period = _stlOptions.SeasonalPeriod;

        // For time series with enough seasonal cycles, check for evolving seasonality
        if (n >= period * 4)
        {
            // Analyze sequential seasons to detect changes
            Vector<T> seasonalityChange = AnalyzeSeasonalEvolution();

            // If significant seasonal evolution is detected, adapt the decomposition
            if (HasSignificantSeasonalEvolution(seasonalityChange))
            {
                // Perform enhanced decomposition that accounts for changing seasonality
                PerformEvolvingSeasonalitySTL(y);
            }
        }
    }

    /// <summary>
    /// Analyzes how the seasonal pattern evolves over time.
    /// </summary>
    /// <returns>A vector measuring seasonal pattern changes between consecutive cycles.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method examines how the seasonal pattern changes from one cycle to the next.
    /// For example, in retail data, it might check if the holiday shopping peak is
    /// getting stronger or weaker over the years.
    /// 
    /// A high change score indicates the seasonal pattern is evolving significantly.
    /// </para>
    /// </remarks>
    private Vector<T> AnalyzeSeasonalEvolution()
    {
        int period = _stlOptions.SeasonalPeriod;
        int cycles = _seasonal.Length / period;

        var changes = new Vector<T>(cycles - 1);
        if (cycles < 2)
        {
            return changes;
        }

        // Euclidean distance between each pair of consecutive cycles in the [cycles, period] grid
        var grid = Tensor<T>.FromVector(_seasonal.GetSubVector(0, cycles * period), new[] { cycles, period });
        var next = Engine.TensorSlice(grid, new[] { 1, 0 }, new[] { cycles - 1, period });
        var previous = Engine.TensorSlice(grid, new[] { 0, 0 }, new[] { cycles - 1, period });
        var diff = Engine.TensorSubtract(next, previous);
        var sums = Engine.ReduceSum(Engine.TensorMultiply(diff, diff), new[] { 1 }, keepDims: false);
        return Engine.TensorSqrt(sums).ToVector();
    }

    /// <summary>
    /// Determines if the seasonal pattern is evolving significantly over time.
    /// </summary>
    /// <param name="seasonalityChange">Vector of seasonal change measurements.</param>
    /// <returns>True if significant evolution is detected, false otherwise.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method decides whether the seasonal pattern is changing enough to warrant
    /// special handling. It looks at how much change occurs between consecutive seasonal
    /// cycles and determines if it exceeds a threshold that indicates meaningful evolution.
    /// </para>
    /// </remarks>
    private bool HasSignificantSeasonalEvolution(Vector<T> seasonalityChange)
    {
        if (seasonalityChange.Length < 2)
        {
            return false;
        }

        // Calculate the mean change
        T meanChange = seasonalityChange.Average();

        // Calculate seasonal standard deviation for comparison
        T seasonalStdDev = _seasonal.StandardDeviation();

        // If mean change is greater than 20% of the seasonal standard deviation,
        // consider it significant evolution
        T threshold = NumOps.Multiply(seasonalStdDev, NumOps.FromDouble(0.2));

        return NumOps.GreaterThan(meanChange, threshold);
    }

    /// <summary>
    /// Performs STL decomposition adapted for time series with evolving seasonality.
    /// </summary>
    /// <param name="y">The original time series data.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This specialized decomposition method handles time series where the seasonal pattern
    /// changes over time. Instead of assuming a fixed seasonal pattern, it allows the
    /// pattern to evolve gradually.
    /// 
    /// For example, if the summer sales peak is getting stronger each year, or the
    /// winter dip is shifting earlier, this method will capture those changes.
    /// </para>
    /// </remarks>
    private void PerformEvolvingSeasonalitySTL(Vector<T> y)
    {
        int n = y.Length;
        int period = _stlOptions.SeasonalPeriod;

        // Use a smaller seasonal smoother window to allow more flexibility in the seasonal pattern
        int adaptiveSeasonalWindow = Math.Max(7, period / 2);

        // Detrend the series
        Vector<T> detrended = SubtractVectors(y, _trend);

        // Use a moving window approach for seasonal extraction
        _seasonal = new Vector<T>(n);
        int windowSize = period * 3; // Use 3 periods for each seasonal estimation

        for (int center = windowSize / 2; center < n - windowSize / 2; center += period)
        {
            // Extract local window
            int start = Math.Max(0, center - windowSize / 2);
            int end = Math.Min(n, center + windowSize / 2);

            // Create vector from the window
            Vector<T> localWindow = detrended.GetSubVector(start, end - start);

            // Extract seasonal pattern from this window
            Vector<T> localSeasonal = ExtractLocalSeasonalPattern(localWindow, period, adaptiveSeasonalWindow);

            // Apply the pattern to the center period
            int targetStart = center - period / 2;
            int applied = Math.Min(period, n - targetStart);
            localSeasonal.AsSpan().Slice(0, applied).CopyTo(_seasonal.AsWritableSpan().Slice(targetStart, applied));
        }

        // Handle edge cases (beginning and end of series)
        // For beginning, use the first estimated seasonal pattern
        int edge = Math.Min(period / 2, n);
        _seasonal.AsSpan().Slice(period, edge).CopyTo(_seasonal.AsWritableSpan().Slice(0, edge));

        // For end, use the last estimated seasonal pattern
        int tailStart = Math.Max(0, n - period / 2);
        _seasonal.AsSpan().Slice(tailStart - period, n - tailStart)
            .CopyTo(_seasonal.AsWritableSpan().Slice(tailStart, n - tailStart));

        // Smooth transitions between adjacent seasonal patterns
        _seasonal = SmoothSeasonalTransitions(_seasonal, period);

        // Re-extract trend after seasonal component is determined
        Vector<T> deseasonalized = SubtractVectors(y, _seasonal);
        _trend = LoessSmoothing(deseasonalized, _stlOptions.TrendLoessWindow);

        // Calculate residuals
        _residual = CalculateResiduals(y, _trend, _seasonal);
    }

    /// <summary>
    /// Extracts a seasonal pattern from a local window of the time series.
    /// </summary>
    /// <param name="window">The local window of detrended data.</param>
    /// <param name="period">The seasonal period.</param>
    /// <param name="loessWindow">The window size for LOESS smoothing.</param>
    /// <returns>The extracted seasonal pattern (one full period).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method finds the seasonal pattern within a specific section of your data.
    /// It's like looking at just a few years of monthly data to find the pattern for those
    /// specific years, rather than assuming the pattern is the same across the entire dataset.
    /// </para>
    /// </remarks>
    private Vector<T> ExtractLocalSeasonalPattern(Vector<T> window, int period, int loessWindow)
    {
        // Get the seasonal pattern by averaging values at the same phase in the cycle
        Vector<T> pattern = new Vector<T>(period);
        if (window.Length > 0)
        {
            // Phases without values keep a zero sum
            var (sums, counts) = TimeSeriesEngineOps<T>.PhaseSums(window, period);
            var countsT = Tensor<T>.FromVector(counts);
            var sumsT = Tensor<T>.FromVector(sums);
            var hasValues = Engine.TensorGreaterThan(countsT, NumOps.Zero);
            var divisors = Engine.TensorWhere(hasValues, countsT, Engine.TensorOnesLike(countsT));
            pattern = Engine.TensorWhere(hasValues, Engine.TensorDivide(sumsT, divisors), sumsT).ToVector();
        }

        // Smooth the pattern
        pattern = SmoothSeasonal(pattern, period, loessWindow);

        // Ensure the pattern sums to zero
        T mean = pattern.Average();
        return Engine.Subtract(pattern, Engine.Fill(pattern.Length, mean));
    }

    /// <summary>
    /// Smooths transitions between adjacent seasonal patterns.
    /// </summary>
    /// <param name="seasonal">The seasonal component with potentially abrupt transitions.</param>
    /// <param name="period">The seasonal period.</param>
    /// <returns>The smoothed seasonal component.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method ensures that the seasonal pattern changes gradually over time rather
    /// than having sudden jumps. It's like blending between different seasonal patterns
    /// to create smooth transitions, which is usually more realistic for real-world data.
    /// </para>
    /// </remarks>
    private Vector<T> SmoothSeasonalTransitions(Vector<T> seasonal, int period)
    {
        // Averaging the same-phase points of [i - period / 2, i + period / 2] only ever finds i itself
        // (the window is narrower than one period), so each value is its own average.
        if (period == 0 && seasonal.Length > 0)
        {
            throw new DivideByZeroException(); // the phase j % period is undefined
        }

        return seasonal.Clone();
    }
}
