using AiDotNet.Models.Options;

namespace AiDotNet.Preprocessing.TimeSeries;

/// <summary>
/// Computes rolling volatility and return-based features for financial time series.
/// </summary>
/// <remarks>
/// <para>
/// This transformer calculates volatility measures commonly used in quantitative finance,
/// including realized volatility, Parkinson, and Garman-Klass estimators.
/// </para>
/// <para><b>For Beginners:</b> Volatility measures how much a price moves up and down over time.
///
/// High volatility means:
/// - Prices change a lot day-to-day
/// - Higher risk but also higher potential returns
/// - Common during market uncertainty
///
/// Low volatility means:
/// - Prices are relatively stable
/// - Lower risk but also lower potential returns
/// - Common in stable market conditions
///
/// This transformer creates features like:
/// - Simple returns: (Today's price - Yesterday's price) / Yesterday's price
/// - Log returns: ln(Today's price / Yesterday's price) - preferred for statistical analysis
/// - Realized volatility: Standard deviation of returns over a window
/// - Momentum: How much price has changed over a period
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class RollingVolatilityTransformer<T> : TimeSeriesTransformerBase<T>
{
    #region Fields

    /// <summary>
    /// The enabled volatility measures.
    /// </summary>
    private readonly VolatilityMeasures _enabledMeasures;

    /// <summary>
    /// The annualization factor for volatility scaling.
    /// </summary>
    private readonly double _annualizationFactor;

    /// <summary>
    /// Whether to calculate returns.
    /// </summary>
    private readonly bool _calculateReturns;

    /// <summary>
    /// Whether to calculate momentum.
    /// </summary>
    private readonly bool _calculateMomentum;

    /// <summary>
    /// Cached operation names.
    /// </summary>
    private readonly string[] _operationNames;

    /// <summary>
    /// OHLC column configuration for proper volatility calculations.
    /// </summary>
    private readonly OhlcColumnConfig? _ohlcConfig;

    /// <summary>
    /// EWMA decay factor (lambda).
    /// </summary>
    private readonly double _ewmaLambda;

    /// <summary>
    /// GARCH omega (constant term).
    /// </summary>
    private readonly double _garchOmega;

    /// <summary>
    /// GARCH alpha (squared return coefficient).
    /// </summary>
    private readonly double _garchAlpha;

    /// <summary>
    /// GARCH beta (lagged variance coefficient).
    /// </summary>
    private readonly double _garchBeta;

    #endregion

    #region Constructor

    /// <summary>
    /// Creates a new rolling volatility transformer with the specified options.
    /// </summary>
    /// <param name="options">Configuration options, or null for defaults.</param>
    public RollingVolatilityTransformer(TimeSeriesFeatureOptions? options = null)
        : base(options)
    {
        _annualizationFactor = Options.AnnualizationFactor;
        _calculateReturns = Options.CalculateReturns;
        _calculateMomentum = Options.CalculateMomentum;
        _ohlcConfig = Options.OhlcColumns;

        // EWMA and GARCH parameters
        _ewmaLambda = Options.EwmaDecayFactor;
        _garchOmega = Options.GarchOmega;
        _garchAlpha = Options.GarchAlpha;
        _garchBeta = Options.GarchBeta;

        // Apply flags to mask enabled measures
        _enabledMeasures = Options.EnabledVolatilityMeasures;
        if (!_calculateReturns)
        {
            _enabledMeasures &= ~(VolatilityMeasures.SimpleReturns | VolatilityMeasures.LogReturns);
        }
        if (!_calculateMomentum)
        {
            _enabledMeasures &= ~VolatilityMeasures.Momentum;
        }

        _operationNames = BuildOperationNames();
    }

    #endregion

    #region Properties

    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    #endregion

    #region Core Implementation

    /// <inheritdoc />
    protected override void FitCore(Tensor<T> data)
    {
        // Volatility calculations don't need to learn parameters
    }

    /// <inheritdoc />
    protected override Tensor<T> TransformCore(Tensor<T> data)
    {
        int inputTimeSteps = GetTimeSteps(data);
        int inputFeatures = InputFeatureCount;
        int outputFeatures = OutputFeatureCount;

        // Determine output dimensions based on edge handling
        int outputTimeSteps = GetOutputTimeSteps(inputTimeSteps);
        int startIndex = GetOutputStartIndex();

        // Handle Truncate mode with empty output
        if (outputTimeSteps <= 0)
        {
            return new Tensor<T>(new[] { 0, outputFeatures });
        }

        var output = new Tensor<T>(new[] { outputTimeSteps, outputFeatures });

        // Track first valid index for forward fill
        int firstValidIndex = -1;
        int maxWindow = GetMaxWindowSize();

        for (int outT = 0; outT < outputTimeSteps; outT++)
        {
            int t = outT + startIndex; // Map to input time step
            int outputIdx = 0;

            foreach (int windowSize in WindowSizes)
            {
                bool isEdge = IsEdgeRegion(t, windowSize);

                // Handle edge cases based on EdgeHandling mode
                if (isEdge)
                {
                    switch (Options.EdgeHandling)
                    {
                        case EdgeHandling.NaN:
                            int measuresCount = CountEnabledMeasures() * inputFeatures;
                            for (int i = 0; i < measuresCount; i++)
                            {
                                output[outT, outputIdx++] = GetNaN();
                            }
                            continue;

                        case EdgeHandling.Partial:
                        case EdgeHandling.ForwardFill:
                            // Use partial window
                            break;

                        case EdgeHandling.Truncate:
                            break;
                    }
                }

                // Track first valid index for forward fill
                if (firstValidIndex < 0 && !IsEdgeRegion(t, maxWindow))
                {
                    firstValidIndex = outT;
                }

                for (int f = 0; f < inputFeatures; f++)
                {
                    int effectiveWindow = ShouldComputePartialWindows() && isEdge
                        ? GetEffectiveWindowSize(t, windowSize)
                        : windowSize;
                    ComputeVolatilityFeatures(data, t, f, windowSize, effectiveWindow, output, outT, ref outputIdx);
                }
            }
        }

        // Apply forward fill if needed
        if (Options.EdgeHandling == EdgeHandling.ForwardFill && firstValidIndex > 0)
        {
            ApplyForwardFill(output, firstValidIndex);
        }

        return output;
    }

    /// <inheritdoc />
    protected override Tensor<T> TransformParallel(Tensor<T> data)
    {
        int inputTimeSteps = GetTimeSteps(data);
        int inputFeatures = InputFeatureCount;
        int outputFeatures = OutputFeatureCount;

        // Determine output dimensions based on edge handling
        int outputTimeSteps = GetOutputTimeSteps(inputTimeSteps);
        int startIndex = GetOutputStartIndex();
        int maxWindow = GetMaxWindowSize();

        // Handle Truncate mode with empty output
        if (outputTimeSteps <= 0)
        {
            return new Tensor<T>(new[] { 0, outputFeatures });
        }

        var output = new Tensor<T>(new[] { outputTimeSteps, outputFeatures });
        int featuresPerWindowFeature = CountEnabledMeasures();

        // Track first valid index for forward fill (thread-safe)
        int firstValidIndex = -1;
        object lockObj = new object();

        Parallel.For(0, outputTimeSteps, outT =>
        {
            int t = outT + startIndex; // Map to input time step
            int outputIdx = 0;

            foreach (int windowSize in WindowSizes)
            {
                bool isEdge = IsEdgeRegion(t, windowSize);

                // Handle edge cases based on EdgeHandling mode
                if (isEdge)
                {
                    switch (Options.EdgeHandling)
                    {
                        case EdgeHandling.NaN:
                            int measuresCount = featuresPerWindowFeature * inputFeatures;
                            for (int i = 0; i < measuresCount; i++)
                            {
                                output[outT, outputIdx++] = GetNaN();
                            }
                            continue;

                        case EdgeHandling.Partial:
                        case EdgeHandling.ForwardFill:
                            // Use partial window
                            break;

                        case EdgeHandling.Truncate:
                            break;
                    }
                }

                // Track first valid index for forward fill (thread-safe)
                if (!IsEdgeRegion(t, maxWindow))
                {
                    lock (lockObj)
                    {
                        if (firstValidIndex < 0 || outT < firstValidIndex)
                            firstValidIndex = outT;
                    }
                }

                for (int f = 0; f < inputFeatures; f++)
                {
                    int effectiveWindow = ShouldComputePartialWindows() && isEdge
                        ? GetEffectiveWindowSize(t, windowSize)
                        : windowSize;
                    int localIdx = outputIdx;
                    ComputeVolatilityFeaturesThreadSafe(data, t, f, windowSize, effectiveWindow, output, outT, localIdx);
                    outputIdx += featuresPerWindowFeature;
                }
            }
        });

        // Apply forward fill if needed (sequential post-processing)
        if (Options.EdgeHandling == EdgeHandling.ForwardFill && firstValidIndex > 0)
        {
            ApplyForwardFill(output, firstValidIndex);
        }

        return output;
    }

    #endregion

    #region Feature Naming

    /// <inheritdoc />
    protected override string[] GenerateFeatureNames()
    {
        var names = new List<string>();
        var inputNames = GetInputFeatureNames();
        var sep = GetSeparator();
        var ops = GetOperationNames();

        foreach (int windowSize in WindowSizes)
        {
            foreach (string inputName in inputNames)
            {
                foreach (string op in ops)
                {
                    names.Add($"{inputName}{sep}{op}{sep}{windowSize}");
                }
            }
        }

        return [.. names];
    }

    /// <inheritdoc />
    protected override string[] GetOperationNames()
    {
        return _operationNames;
    }

    private string[] BuildOperationNames()
    {
        var ops = new List<string>();

        if (_enabledMeasures.HasFlag(VolatilityMeasures.SimpleReturns))
            ops.Add("simple_return");

        if (_enabledMeasures.HasFlag(VolatilityMeasures.LogReturns))
            ops.Add("log_return");

        if (_enabledMeasures.HasFlag(VolatilityMeasures.RealizedVolatility))
            ops.Add("realized_vol");

        if (_enabledMeasures.HasFlag(VolatilityMeasures.ParkinsonVolatility))
            ops.Add("parkinson_vol");

        if (_enabledMeasures.HasFlag(VolatilityMeasures.GarmanKlassVolatility))
            ops.Add("garman_klass_vol");

        if (_enabledMeasures.HasFlag(VolatilityMeasures.Momentum))
            ops.Add("momentum");

        if (_enabledMeasures.HasFlag(VolatilityMeasures.EwmaVolatility))
            ops.Add("ewma_vol");

        if (_enabledMeasures.HasFlag(VolatilityMeasures.GarchVolatility))
            ops.Add("garch_vol");

        if (_enabledMeasures.HasFlag(VolatilityMeasures.YangZhangVolatility))
            ops.Add("yang_zhang_vol");

        if (_enabledMeasures.HasFlag(VolatilityMeasures.RogersSatchellVolatility))
            ops.Add("rogers_satchell_vol");

        return [.. ops];
    }

    private int CountEnabledMeasures()
    {
        return _operationNames.Length;
    }

    #endregion

    #region Volatility Computation

    private void ComputeVolatilityFeatures(
        Tensor<T> data, int t, int f, int windowSize,
        Tensor<T> output, ref int outputIdx)
    {
        ComputeVolatilityFeatures(data, t, f, windowSize, windowSize, output, t, ref outputIdx);
    }

    private void ComputeVolatilityFeatures(
        Tensor<T> data, int inputT, int f, int windowSize, int effectiveWindow,
        Tensor<T> output, int outputT, ref int outputIdx)
    {
        // Extract price window (may be partial)
        var prices = ExtractWindowWithSize(data, inputT, f, windowSize, effectiveWindow);

        // Extract OHLC data if configured
        double[]? highPrices = null;
        double[]? lowPrices = null;
        double[]? openPrices = null;
        double[]? closePrices = null;

        if (_ohlcConfig != null)
        {
            if (_ohlcConfig.HighIndex.HasValue)
                highPrices = ExtractWindowWithSize(data, inputT, _ohlcConfig.HighIndex.Value, windowSize, effectiveWindow);
            if (_ohlcConfig.LowIndex.HasValue)
                lowPrices = ExtractWindowWithSize(data, inputT, _ohlcConfig.LowIndex.Value, windowSize, effectiveWindow);
            if (_ohlcConfig.OpenIndex.HasValue)
                openPrices = ExtractWindowWithSize(data, inputT, _ohlcConfig.OpenIndex.Value, windowSize, effectiveWindow);
            if (_ohlcConfig.CloseIndex.HasValue)
                closePrices = ExtractWindowWithSize(data, inputT, _ohlcConfig.CloseIndex.Value, windowSize, effectiveWindow);
        }

        // Compute returns for this window
        var simpleReturns = ComputeSimpleReturns(closePrices ?? prices);
        var logReturns = ComputeLogReturns(closePrices ?? prices);

        bool hasValidReturns = simpleReturns.Length > 0 &&
                               simpleReturns.Any(r => !double.IsNaN(r));

        // Simple returns
        if (_enabledMeasures.HasFlag(VolatilityMeasures.SimpleReturns))
        {
            double val = hasValidReturns ? simpleReturns.Where(r => !double.IsNaN(r)).LastOrDefault() : double.NaN;
            output[outputT, outputIdx++] = NumOps.FromDouble(val);
        }

        // Log returns
        if (_enabledMeasures.HasFlag(VolatilityMeasures.LogReturns))
        {
            double val = hasValidReturns ? logReturns.Where(r => !double.IsNaN(r)).LastOrDefault() : double.NaN;
            output[outputT, outputIdx++] = NumOps.FromDouble(val);
        }

        // Realized volatility (annualized std of log returns)
        if (_enabledMeasures.HasFlag(VolatilityMeasures.RealizedVolatility))
        {
            var validLogReturns = logReturns.Where(r => !double.IsNaN(r)).ToArray();
            double val = validLogReturns.Length >= 2
                ? ComputeStdDev(validLogReturns) * Math.Sqrt(_annualizationFactor)
                : double.NaN;
            output[outputT, outputIdx++] = NumOps.FromDouble(val);
        }

        // Parkinson volatility (uses high/low prices)
        if (_enabledMeasures.HasFlag(VolatilityMeasures.ParkinsonVolatility))
        {
            double val = ComputeParkinsonVolatility(highPrices, lowPrices, prices, effectiveWindow);
            output[outputT, outputIdx++] = NumOps.FromDouble(val);
        }

        // Garman-Klass volatility (uses OHLC prices)
        if (_enabledMeasures.HasFlag(VolatilityMeasures.GarmanKlassVolatility))
        {
            double val = ComputeGarmanKlassVolatility(openPrices, highPrices, lowPrices, closePrices, prices, effectiveWindow);
            output[outputT, outputIdx++] = NumOps.FromDouble(val);
        }

        // Momentum (price change ratio)
        if (_enabledMeasures.HasFlag(VolatilityMeasures.Momentum))
        {
            double val = ComputeMomentum(closePrices ?? prices, effectiveWindow);
            output[outputT, outputIdx++] = NumOps.FromDouble(val);
        }

        // EWMA volatility
        if (_enabledMeasures.HasFlag(VolatilityMeasures.EwmaVolatility))
        {
            var validLogReturns = logReturns.Where(r => !double.IsNaN(r)).ToArray();
            double val = validLogReturns.Length >= 2
                ? ComputeEwmaVolatility(validLogReturns, _ewmaLambda) * Math.Sqrt(_annualizationFactor)
                : double.NaN;
            output[outputT, outputIdx++] = NumOps.FromDouble(val);
        }

        // GARCH volatility
        if (_enabledMeasures.HasFlag(VolatilityMeasures.GarchVolatility))
        {
            var validLogReturns = logReturns.Where(r => !double.IsNaN(r)).ToArray();
            double val = validLogReturns.Length >= 2
                ? ComputeGarchVolatility(validLogReturns, _garchOmega, _garchAlpha, _garchBeta) * Math.Sqrt(_annualizationFactor)
                : double.NaN;
            output[outputT, outputIdx++] = NumOps.FromDouble(val);
        }

        // Yang-Zhang volatility (uses OHLC)
        if (_enabledMeasures.HasFlag(VolatilityMeasures.YangZhangVolatility))
        {
            double val = ComputeYangZhangVolatility(openPrices, highPrices, lowPrices, closePrices, prices, effectiveWindow);
            output[outputT, outputIdx++] = NumOps.FromDouble(val);
        }

        // Rogers-Satchell volatility (uses OHLC)
        if (_enabledMeasures.HasFlag(VolatilityMeasures.RogersSatchellVolatility))
        {
            double val = ComputeRogersSatchellVolatility(openPrices, highPrices, lowPrices, closePrices, prices, effectiveWindow);
            output[outputT, outputIdx++] = NumOps.FromDouble(val);
        }
    }

    private void ComputeVolatilityFeaturesThreadSafe(
        Tensor<T> data, int inputT, int f, int windowSize, int effectiveWindow,
        Tensor<T> output, int outputT, int startIdx)
    {
        int outputIdx = startIdx;

        var prices = ExtractWindowWithSize(data, inputT, f, windowSize, effectiveWindow);

        // Extract OHLC data if configured
        double[]? highPrices = null;
        double[]? lowPrices = null;
        double[]? openPrices = null;
        double[]? closePrices = null;

        if (_ohlcConfig != null)
        {
            if (_ohlcConfig.HighIndex.HasValue)
                highPrices = ExtractWindowWithSize(data, inputT, _ohlcConfig.HighIndex.Value, windowSize, effectiveWindow);
            if (_ohlcConfig.LowIndex.HasValue)
                lowPrices = ExtractWindowWithSize(data, inputT, _ohlcConfig.LowIndex.Value, windowSize, effectiveWindow);
            if (_ohlcConfig.OpenIndex.HasValue)
                openPrices = ExtractWindowWithSize(data, inputT, _ohlcConfig.OpenIndex.Value, windowSize, effectiveWindow);
            if (_ohlcConfig.CloseIndex.HasValue)
                closePrices = ExtractWindowWithSize(data, inputT, _ohlcConfig.CloseIndex.Value, windowSize, effectiveWindow);
        }

        var simpleReturns = ComputeSimpleReturns(closePrices ?? prices);
        var logReturns = ComputeLogReturns(closePrices ?? prices);

        bool hasValidReturns = simpleReturns.Length > 0 &&
                               simpleReturns.Any(r => !double.IsNaN(r));

        if (_enabledMeasures.HasFlag(VolatilityMeasures.SimpleReturns))
        {
            double val = hasValidReturns ? simpleReturns.Where(r => !double.IsNaN(r)).LastOrDefault() : double.NaN;
            output[outputT, outputIdx++] = NumOps.FromDouble(val);
        }

        if (_enabledMeasures.HasFlag(VolatilityMeasures.LogReturns))
        {
            double val = hasValidReturns ? logReturns.Where(r => !double.IsNaN(r)).LastOrDefault() : double.NaN;
            output[outputT, outputIdx++] = NumOps.FromDouble(val);
        }

        if (_enabledMeasures.HasFlag(VolatilityMeasures.RealizedVolatility))
        {
            var validLogReturns = logReturns.Where(r => !double.IsNaN(r)).ToArray();
            double val = validLogReturns.Length >= 2
                ? ComputeStdDev(validLogReturns) * Math.Sqrt(_annualizationFactor)
                : double.NaN;
            output[outputT, outputIdx++] = NumOps.FromDouble(val);
        }

        if (_enabledMeasures.HasFlag(VolatilityMeasures.ParkinsonVolatility))
        {
            double val = ComputeParkinsonVolatility(highPrices, lowPrices, prices, effectiveWindow);
            output[outputT, outputIdx++] = NumOps.FromDouble(val);
        }

        if (_enabledMeasures.HasFlag(VolatilityMeasures.GarmanKlassVolatility))
        {
            double val = ComputeGarmanKlassVolatility(openPrices, highPrices, lowPrices, closePrices, prices, effectiveWindow);
            output[outputT, outputIdx++] = NumOps.FromDouble(val);
        }

        if (_enabledMeasures.HasFlag(VolatilityMeasures.Momentum))
        {
            double val = ComputeMomentum(closePrices ?? prices, effectiveWindow);
            output[outputT, outputIdx++] = NumOps.FromDouble(val);
        }

        // EWMA volatility
        if (_enabledMeasures.HasFlag(VolatilityMeasures.EwmaVolatility))
        {
            var validLogReturns = logReturns.Where(r => !double.IsNaN(r)).ToArray();
            double val = validLogReturns.Length >= 2
                ? ComputeEwmaVolatility(validLogReturns, _ewmaLambda) * Math.Sqrt(_annualizationFactor)
                : double.NaN;
            output[outputT, outputIdx++] = NumOps.FromDouble(val);
        }

        // GARCH volatility
        if (_enabledMeasures.HasFlag(VolatilityMeasures.GarchVolatility))
        {
            var validLogReturns = logReturns.Where(r => !double.IsNaN(r)).ToArray();
            double val = validLogReturns.Length >= 2
                ? ComputeGarchVolatility(validLogReturns, _garchOmega, _garchAlpha, _garchBeta) * Math.Sqrt(_annualizationFactor)
                : double.NaN;
            output[outputT, outputIdx++] = NumOps.FromDouble(val);
        }

        // Yang-Zhang volatility
        if (_enabledMeasures.HasFlag(VolatilityMeasures.YangZhangVolatility))
        {
            double val = ComputeYangZhangVolatility(openPrices, highPrices, lowPrices, closePrices, prices, effectiveWindow);
            output[outputT, outputIdx++] = NumOps.FromDouble(val);
        }

        // Rogers-Satchell volatility
        if (_enabledMeasures.HasFlag(VolatilityMeasures.RogersSatchellVolatility))
        {
            double val = ComputeRogersSatchellVolatility(openPrices, highPrices, lowPrices, closePrices, prices, effectiveWindow);
            output[outputT, outputIdx++] = NumOps.FromDouble(val);
        }
    }

    #endregion

    #region Window Extraction

    private double[] ExtractWindow(Tensor<T> data, int endTime, int feature, int windowSize)
    {
        return ExtractWindowWithSize(data, endTime, feature, windowSize, windowSize);
    }

    private double[] ExtractWindowWithSize(Tensor<T> data, int endTime, int feature, int requestedSize, int effectiveSize)
    {
        // For partial windows, we only extract available data
        int actualSize = Math.Min(effectiveSize, endTime + 1);
        actualSize = Math.Max(1, actualSize); // At least 1 element

        int startTime = endTime - actualSize + 1;
        var window = new double[actualSize];

        for (int i = 0; i < actualSize; i++)
        {
            int t = startTime + i;
            window[i] = t < 0
                ? double.NaN
                : NumOps.ToDouble(GetValue(data, t, feature));
        }

        return window;
    }

    #endregion

    #region Return Calculations

    /// <summary>
    /// Computes simple returns: (P_t - P_{t-1}) / P_{t-1}.
    /// </summary>
    private static double[] ComputeSimpleReturns(double[] prices)
    {
        if (prices.Length < 2) return [];

        var returns = new double[prices.Length - 1];
        for (int i = 1; i < prices.Length; i++)
        {
            // Use a reasonable tolerance (1e-10) instead of double.Epsilon for near-zero price check
            returns[i - 1] = double.IsNaN(prices[i]) || double.IsNaN(prices[i - 1]) || Math.Abs(prices[i - 1]) < 1e-10
                ? double.NaN
                : (prices[i] - prices[i - 1]) / prices[i - 1];
        }

        return returns;
    }

    /// <summary>
    /// Computes log returns: ln(P_t / P_{t-1}).
    /// </summary>
    private static double[] ComputeLogReturns(double[] prices)
    {
        if (prices.Length < 2) return [];

        var returns = new double[prices.Length - 1];
        for (int i = 1; i < prices.Length; i++)
        {
            returns[i - 1] = double.IsNaN(prices[i]) || double.IsNaN(prices[i - 1]) ||
                prices[i] <= 0 || prices[i - 1] <= 0
                    ? double.NaN
                    : Math.Log(prices[i] / prices[i - 1]);
        }

        return returns;
    }

    #endregion

    #region Volatility Estimators

    /// <summary>
    /// Computes sample standard deviation.
    /// </summary>
    private static double ComputeStdDev(double[] values)
    {
        if (values.Length < 2) return double.NaN;

        double mean = values.Average();
        double sumSq = values.Select(x => (x - mean) * (x - mean)).Sum();
        return Math.Sqrt(sumSq / (values.Length - 1));
    }

    /// <summary>
    /// Computes Parkinson volatility estimator using OHLC data when available.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The Parkinson volatility uses the high-low range of prices,
    /// which captures intraday volatility. It's more efficient than close-to-close volatility.
    ///
    /// Formula: sqrt(1/(4n*ln(2)) * sum(ln(High/Low)^2))
    ///
    /// When OHLC data is provided, we use actual high/low values for accurate calculation.
    /// Otherwise, we approximate using a rolling max/min of the price column.
    /// </para>
    /// </remarks>
    private double ComputeParkinsonVolatility(double[]? highPrices, double[]? lowPrices, double[] fallbackPrices, int windowSize)
    {
        // Use OHLC data if available
        if (highPrices != null && lowPrices != null && highPrices.Length > 0 && lowPrices.Length > 0)
        {
            return ComputeParkinsonFromOhlc(highPrices, lowPrices);
        }

        // Fallback to approximation using min/max of price column
        return ComputeParkinsonApproximation(fallbackPrices);
    }

    /// <summary>
    /// Computes Parkinson volatility from actual OHLC high/low data.
    /// </summary>
    private double ComputeParkinsonFromOhlc(double[] highPrices, double[] lowPrices)
    {
        int n = Math.Min(highPrices.Length, lowPrices.Length);
        if (n < 1) return double.NaN;

        double sumLogRangeSq = 0;
        int validCount = 0;

        for (int i = 0; i < n; i++)
        {
            double high = highPrices[i];
            double low = lowPrices[i];

            if (double.IsNaN(high) || double.IsNaN(low) || high <= 0 || low <= 0 || low > high)
                continue;

            double logRange = Math.Log(high / low);
            sumLogRangeSq += logRange * logRange;
            validCount++;
        }

        if (validCount < 1) return double.NaN;

        // Parkinson formula: sqrt(sum(ln(H/L)^2) / (4*n*ln(2)))
        double variance = sumLogRangeSq / (4.0 * validCount * Math.Log(2));
        return Math.Sqrt(variance * _annualizationFactor);
    }

    /// <summary>
    /// Computes Parkinson volatility approximation when OHLC data is not available.
    /// </summary>
    private double ComputeParkinsonApproximation(double[] prices)
    {
        var validPrices = prices.Where(p => !double.IsNaN(p) && p > 0).ToArray();

        if (validPrices.Length < 2) return double.NaN;

        double high = validPrices.Max();
        double low = validPrices.Min();

        if (low <= 0 || high <= 0) return double.NaN;

        // Single-period approximation using rolling range
        double logRange = Math.Log(high / low);
        double variance = logRange * logRange / (4 * Math.Log(2));

        return Math.Sqrt(variance * _annualizationFactor);
    }

    /// <summary>
    /// Computes Garman-Klass volatility estimator using OHLC data when available.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Garman-Klass is an efficient volatility estimator that uses
    /// open, high, low, close (OHLC) prices. It's more accurate than Parkinson.
    ///
    /// Formula: 0.5 * ln(H/L)^2 - (2*ln(2) - 1) * ln(C/O)^2
    ///
    /// When OHLC data is provided, we use actual values for accurate calculation.
    /// Otherwise, we approximate using the price column.
    /// </para>
    /// </remarks>
    private double ComputeGarmanKlassVolatility(double[]? openPrices, double[]? highPrices,
        double[]? lowPrices, double[]? closePrices, double[] fallbackPrices, int windowSize)
    {
        // Use full OHLC data if available
        if (openPrices != null && highPrices != null && lowPrices != null && closePrices != null &&
            openPrices.Length > 0 && highPrices.Length > 0 && lowPrices.Length > 0 && closePrices.Length > 0)
        {
            return ComputeGarmanKlassFromOhlc(openPrices, highPrices, lowPrices, closePrices);
        }

        // Fallback to approximation
        return ComputeGarmanKlassApproximation(fallbackPrices);
    }

    /// <summary>
    /// Computes Garman-Klass volatility from actual OHLC data.
    /// </summary>
    private double ComputeGarmanKlassFromOhlc(double[] openPrices, double[] highPrices, double[] lowPrices, double[] closePrices)
    {
        int n = Math.Min(Math.Min(openPrices.Length, highPrices.Length),
                        Math.Min(lowPrices.Length, closePrices.Length));
        if (n < 1) return double.NaN;

        double sumVariance = 0;
        int validCount = 0;

        for (int i = 0; i < n; i++)
        {
            double open = openPrices[i];
            double high = highPrices[i];
            double low = lowPrices[i];
            double close = closePrices[i];

            if (double.IsNaN(open) || double.IsNaN(high) || double.IsNaN(low) || double.IsNaN(close) ||
                open <= 0 || high <= 0 || low <= 0 || close <= 0 || low > high)
                continue;

            double logHL = Math.Log(high / low);
            double logCO = Math.Log(close / open);

            // Garman-Klass formula per period
            double periodVariance = 0.5 * logHL * logHL - (2 * Math.Log(2) - 1) * logCO * logCO;
            sumVariance += Math.Max(0, periodVariance);  // Ensure non-negative
            validCount++;
        }

        if (validCount < 1) return double.NaN;

        double avgVariance = sumVariance / validCount;
        return Math.Sqrt(avgVariance * _annualizationFactor);
    }

    /// <summary>
    /// Computes Garman-Klass volatility approximation when OHLC data is not available.
    /// </summary>
    private double ComputeGarmanKlassApproximation(double[] prices)
    {
        var validPrices = prices.Where(p => !double.IsNaN(p) && p > 0).ToArray();

        if (validPrices.Length < 3) return double.NaN;

        // Approximate OHLC from close prices
        double open = validPrices[0];
        double close = validPrices[^1];
        double high = validPrices.Max();
        double low = validPrices.Min();

        if (low <= 0 || high <= 0 || open <= 0 || close <= 0) return double.NaN;

        // Single-period Garman-Klass formula
        double logHL = Math.Log(high / low);
        double logCO = Math.Log(close / open);

        double variance = 0.5 * logHL * logHL - (2 * Math.Log(2) - 1) * logCO * logCO;

        if (variance < 0) variance = 0; // Handle edge case

        return Math.Sqrt(variance * _annualizationFactor);
    }

    /// <summary>
    /// Computes price momentum (rate of change).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Momentum measures how much the price has changed
    /// over the window period. Positive momentum means prices are rising,
    /// negative means they're falling.
    ///
    /// Formula: (Current Price / Past Price) - 1
    /// </para>
    /// </remarks>
    private static double ComputeMomentum(double[] prices, int windowSize)
    {
        var validPrices = prices.Where(p => !double.IsNaN(p) && p > 0).ToArray();

        if (validPrices.Length < 2) return double.NaN;

        double pastPrice = validPrices[0];
        double currentPrice = validPrices[^1];

        if (pastPrice <= 0) return double.NaN;

        return (currentPrice / pastPrice) - 1;
    }

    #endregion

    #region Incremental Computation

    /// <summary>
    /// Computes volatility features incrementally from the circular buffer.
    /// </summary>
    protected override T[] ComputeIncrementalFeatures(IncrementalState<T> state, T[] newDataPoint)
    {
        var features = new T[OutputFeatureCount];
        int featureIdx = 0;

        foreach (int windowSize in WindowSizes)
        {
            for (int f = 0; f < InputFeatureCount; f++)
            {
                // Extract window data from the circular buffer
                var windowData = ExtractIncrementalWindow(state, f, windowSize);

                // Compute volatility features
                ComputeVolatilityFeaturesIncremental(windowData, features, ref featureIdx);
            }
        }

        return features;
    }

    /// <summary>
    /// Computes volatility features for a window of data incrementally.
    /// </summary>
    private void ComputeVolatilityFeaturesIncremental(double[] prices, T[] features, ref int featureIdx)
    {
        var simpleReturns = ComputeSimpleReturns(prices);
        var logReturns = ComputeLogReturns(prices);

        bool hasValidReturns = simpleReturns.Length > 0 && simpleReturns.Any(r => !double.IsNaN(r));

        if (_enabledMeasures.HasFlag(VolatilityMeasures.SimpleReturns))
        {
            double val = hasValidReturns ? simpleReturns.Where(r => !double.IsNaN(r)).LastOrDefault() : double.NaN;
            features[featureIdx++] = NumOps.FromDouble(val);
        }

        if (_enabledMeasures.HasFlag(VolatilityMeasures.LogReturns))
        {
            double val = hasValidReturns ? logReturns.Where(r => !double.IsNaN(r)).LastOrDefault() : double.NaN;
            features[featureIdx++] = NumOps.FromDouble(val);
        }

        if (_enabledMeasures.HasFlag(VolatilityMeasures.RealizedVolatility))
        {
            var validLogReturns = logReturns.Where(r => !double.IsNaN(r)).ToArray();
            double val = validLogReturns.Length >= 2
                ? ComputeStdDev(validLogReturns) * Math.Sqrt(_annualizationFactor)
                : double.NaN;
            features[featureIdx++] = NumOps.FromDouble(val);
        }

        if (_enabledMeasures.HasFlag(VolatilityMeasures.ParkinsonVolatility))
        {
            double val = ComputeParkinsonApproximation(prices);
            features[featureIdx++] = NumOps.FromDouble(val);
        }

        if (_enabledMeasures.HasFlag(VolatilityMeasures.GarmanKlassVolatility))
        {
            double val = ComputeGarmanKlassApproximation(prices);
            features[featureIdx++] = NumOps.FromDouble(val);
        }

        if (_enabledMeasures.HasFlag(VolatilityMeasures.Momentum))
        {
            double val = ComputeMomentum(prices, prices.Length);
            features[featureIdx++] = NumOps.FromDouble(val);
        }

        if (_enabledMeasures.HasFlag(VolatilityMeasures.EwmaVolatility))
        {
            var validLogReturns = logReturns.Where(r => !double.IsNaN(r)).ToArray();
            double val = validLogReturns.Length >= 2
                ? ComputeEwmaVolatility(validLogReturns, _ewmaLambda) * Math.Sqrt(_annualizationFactor)
                : double.NaN;
            features[featureIdx++] = NumOps.FromDouble(val);
        }

        if (_enabledMeasures.HasFlag(VolatilityMeasures.GarchVolatility))
        {
            var validLogReturns = logReturns.Where(r => !double.IsNaN(r)).ToArray();
            double val = validLogReturns.Length >= 2
                ? ComputeGarchVolatility(validLogReturns, _garchOmega, _garchAlpha, _garchBeta) * Math.Sqrt(_annualizationFactor)
                : double.NaN;
            features[featureIdx++] = NumOps.FromDouble(val);
        }

        if (_enabledMeasures.HasFlag(VolatilityMeasures.YangZhangVolatility))
        {
            // Fallback to realized volatility approximation for incremental
            var validLogReturns = logReturns.Where(r => !double.IsNaN(r)).ToArray();
            double val = validLogReturns.Length >= 2
                ? ComputeStdDev(validLogReturns) * Math.Sqrt(_annualizationFactor)
                : double.NaN;
            features[featureIdx++] = NumOps.FromDouble(val);
        }

        if (_enabledMeasures.HasFlag(VolatilityMeasures.RogersSatchellVolatility))
        {
            double val = ComputeParkinsonApproximation(prices);
            features[featureIdx++] = NumOps.FromDouble(val);
        }
    }

    #endregion

    #region Serialization

    /// <summary>
    /// Exports transformer-specific parameters for serialization.
    /// </summary>
    protected override Dictionary<string, object> ExportParameters()
    {
        return new Dictionary<string, object>
        {
            ["EnabledMeasures"] = (int)_enabledMeasures,
            ["AnnualizationFactor"] = _annualizationFactor,
            ["CalculateReturns"] = _calculateReturns,
            ["CalculateMomentum"] = _calculateMomentum,
            ["EwmaLambda"] = _ewmaLambda,
            ["GarchOmega"] = _garchOmega,
            ["GarchAlpha"] = _garchAlpha,
            ["GarchBeta"] = _garchBeta,
            ["OperationNames"] = _operationNames
        };
    }

    /// <summary>
    /// Imports transformer-specific parameters for validation.
    /// </summary>
    protected override void ImportParameters(Dictionary<string, object> parameters)
    {
        if (parameters.TryGetValue("EnabledMeasures", out var measuresObj))
        {
            int savedMeasures = Convert.ToInt32(measuresObj);
            if (savedMeasures != (int)_enabledMeasures)
            {
                throw new ArgumentException(
                    $"Saved EnabledMeasures ({savedMeasures}) does not match current configuration ({(int)_enabledMeasures}).");
            }
        }
    }

    #endregion

    #region Advanced Volatility Estimators

    /// <summary>
    /// Computes EWMA (Exponentially Weighted Moving Average) volatility.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> EWMA gives more weight to recent observations using a decay factor.
    /// This makes it more responsive to recent market changes than simple volatility.
    ///
    /// Formula: sigma²_t = lambda * sigma²_{t-1} + (1 - lambda) * r²_{t-1}
    /// where lambda is the decay factor (e.g., 0.94 for RiskMetrics).
    /// </para>
    /// </remarks>
    private static double ComputeEwmaVolatility(double[] returns, double lambda)
    {
        if (returns.Length < 2) return double.NaN;

        // Initialize with sample variance
        double variance = 0;
        for (int i = 0; i < returns.Length; i++)
        {
            variance += returns[i] * returns[i];
        }
        variance /= returns.Length;

        // Apply EWMA recursively
        double ewmaVariance = variance;
        for (int i = 0; i < returns.Length; i++)
        {
            ewmaVariance = lambda * ewmaVariance + (1 - lambda) * returns[i] * returns[i];
        }

        return Math.Sqrt(ewmaVariance);
    }

    /// <summary>
    /// Computes GARCH(1,1) volatility estimate.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> GARCH models capture volatility clustering - the tendency
    /// for high volatility periods to follow high volatility, and low to follow low.
    ///
    /// GARCH(1,1) formula: sigma²_t = omega + alpha * r²_{t-1} + beta * sigma²_{t-1}
    ///
    /// The unconditional (long-run) variance is: omega / (1 - alpha - beta)
    /// For stability: alpha + beta must be less than 1.
    /// </para>
    /// </remarks>
    private static double ComputeGarchVolatility(double[] returns, double omega, double alpha, double beta)
    {
        if (returns.Length < 2) return double.NaN;

        // Check stationarity condition
        if (alpha + beta >= 1)
        {
            // Fall back to simple variance if GARCH parameters are invalid
            return ComputeStdDev(returns);
        }

        // Initialize with unconditional variance
        double unconditionalVariance = omega / (1 - alpha - beta);
        double variance = unconditionalVariance;

        // Apply GARCH recursively
        for (int i = 0; i < returns.Length; i++)
        {
            variance = omega + alpha * returns[i] * returns[i] + beta * variance;
        }

        return Math.Sqrt(variance);
    }

    /// <summary>
    /// Computes Yang-Zhang volatility estimator using OHLC data.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Yang-Zhang is the most accurate OHLC-based volatility estimator.
    /// It combines three components:
    /// 1. Overnight volatility (close-to-open)
    /// 2. Open-to-close volatility
    /// 3. Rogers-Satchell volatility
    ///
    /// This captures both overnight jumps and intraday price movements.
    /// </para>
    /// </remarks>
    private double ComputeYangZhangVolatility(double[]? openPrices, double[]? highPrices,
        double[]? lowPrices, double[]? closePrices, double[] fallbackPrices, int windowSize)
    {
        // Need full OHLC data for Yang-Zhang
        if (openPrices == null || highPrices == null || lowPrices == null || closePrices == null ||
            openPrices.Length < 2 || highPrices.Length < 2 || lowPrices.Length < 2 || closePrices.Length < 2)
        {
            // Fall back to realized volatility approximation
            var logReturns = ComputeLogReturns(fallbackPrices);
            var validReturns = logReturns.Where(r => !double.IsNaN(r)).ToArray();
            if (validReturns.Length < 2) return double.NaN;
            return ComputeStdDev(validReturns) * Math.Sqrt(_annualizationFactor);
        }

        int n = Math.Min(Math.Min(openPrices.Length, highPrices.Length),
                        Math.Min(lowPrices.Length, closePrices.Length));

        // Component 1: Overnight variance (close to open)
        double overnightVariance = 0;
        int overnightCount = 0;
        for (int i = 1; i < n; i++)
        {
            double prevClose = closePrices[i - 1];
            double open = openPrices[i];
            if (double.IsNaN(prevClose) || double.IsNaN(open) || prevClose <= 0 || open <= 0)
                continue;
            double logReturn = Math.Log(open / prevClose);
            overnightVariance += logReturn * logReturn;
            overnightCount++;
        }

        // Component 2: Open-to-close variance
        double openCloseVariance = 0;
        int openCloseCount = 0;
        for (int i = 0; i < n; i++)
        {
            double open = openPrices[i];
            double close = closePrices[i];
            if (double.IsNaN(open) || double.IsNaN(close) || open <= 0 || close <= 0)
                continue;
            double logReturn = Math.Log(close / open);
            openCloseVariance += logReturn * logReturn;
            openCloseCount++;
        }

        // Component 3: Rogers-Satchell variance
        double rsVariance = ComputeRogersSatchellVariance(openPrices, highPrices, lowPrices, closePrices, n);

        if (overnightCount < 1 || openCloseCount < 1 || double.IsNaN(rsVariance))
            return double.NaN;

        overnightVariance /= overnightCount;
        openCloseVariance /= openCloseCount;

        // Yang-Zhang combination: use k = 0.34 / (1 + (n+1)/(n-1)) for optimal k
        double k = 0.34 / (1.0 + (n + 1.0) / (n - 1.0));
        double yangZhangVariance = overnightVariance + k * openCloseVariance + (1 - k) * rsVariance;

        return Math.Sqrt(yangZhangVariance * _annualizationFactor);
    }

    /// <summary>
    /// Computes Rogers-Satchell variance component.
    /// </summary>
    private static double ComputeRogersSatchellVariance(double[] openPrices, double[] highPrices,
        double[] lowPrices, double[] closePrices, int n)
    {
        double variance = 0;
        int validCount = 0;

        for (int i = 0; i < n; i++)
        {
            double open = openPrices[i];
            double high = highPrices[i];
            double low = lowPrices[i];
            double close = closePrices[i];

            if (double.IsNaN(open) || double.IsNaN(high) || double.IsNaN(low) || double.IsNaN(close) ||
                open <= 0 || high <= 0 || low <= 0 || close <= 0 || low > high)
                continue;

            // Rogers-Satchell formula per period
            double logHC = Math.Log(high / close);
            double logHO = Math.Log(high / open);
            double logLC = Math.Log(low / close);
            double logLO = Math.Log(low / open);

            double periodVariance = logHC * logHO + logLC * logLO;
            variance += periodVariance;
            validCount++;
        }

        if (validCount < 1) return double.NaN;
        return variance / validCount;
    }

    /// <summary>
    /// Computes Rogers-Satchell volatility estimator.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Rogers-Satchell is a drift-independent volatility estimator.
    /// It's unbiased even when the asset has a trend, making it more robust than close-to-close methods.
    ///
    /// Formula per period: ln(H/C) * ln(H/O) + ln(L/C) * ln(L/O)
    /// where H=High, L=Low, O=Open, C=Close.
    /// </para>
    /// </remarks>
    private double ComputeRogersSatchellVolatility(double[]? openPrices, double[]? highPrices,
        double[]? lowPrices, double[]? closePrices, double[] fallbackPrices, int windowSize)
    {
        // Need full OHLC data for Rogers-Satchell
        if (openPrices == null || highPrices == null || lowPrices == null || closePrices == null ||
            openPrices.Length < 1 || highPrices.Length < 1 || lowPrices.Length < 1 || closePrices.Length < 1)
        {
            // Fall back to Parkinson approximation
            return ComputeParkinsonApproximation(fallbackPrices);
        }

        int n = Math.Min(Math.Min(openPrices.Length, highPrices.Length),
                        Math.Min(lowPrices.Length, closePrices.Length));

        double variance = ComputeRogersSatchellVariance(openPrices, highPrices, lowPrices, closePrices, n);

        if (double.IsNaN(variance) || variance < 0) return double.NaN;

        return Math.Sqrt(variance * _annualizationFactor);
    }

    #endregion
}
