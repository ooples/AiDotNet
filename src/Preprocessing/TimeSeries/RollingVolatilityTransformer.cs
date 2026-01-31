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
    private string[] _operationNames;

    #endregion

    #region Constructor

    /// <summary>
    /// Creates a new rolling volatility transformer with the specified options.
    /// </summary>
    /// <param name="options">Configuration options, or null for defaults.</param>
    public RollingVolatilityTransformer(TimeSeriesFeatureOptions? options = null)
        : base(options)
    {
        _enabledMeasures = Options.EnabledVolatilityMeasures;
        _annualizationFactor = Options.AnnualizationFactor;
        _calculateReturns = Options.CalculateReturns;
        _calculateMomentum = Options.CalculateMomentum;
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
        int timeSteps = GetTimeSteps(data);
        int inputFeatures = InputFeatureCount;
        int outputFeatures = OutputFeatureCount;

        var output = new Tensor<T>(new[] { timeSteps, outputFeatures });

        for (int t = 0; t < timeSteps; t++)
        {
            int outputIdx = 0;

            foreach (int windowSize in WindowSizes)
            {
                for (int f = 0; f < inputFeatures; f++)
                {
                    ComputeVolatilityFeatures(data, t, f, windowSize, output, ref outputIdx);
                }
            }
        }

        return output;
    }

    /// <inheritdoc />
    protected override Tensor<T> TransformParallel(Tensor<T> data)
    {
        int timeSteps = GetTimeSteps(data);
        int inputFeatures = InputFeatureCount;
        int outputFeatures = OutputFeatureCount;

        var output = new Tensor<T>(new[] { timeSteps, outputFeatures });
        int featuresPerWindowFeature = CountEnabledMeasures();

        Parallel.For(0, timeSteps, t =>
        {
            int outputIdx = 0;

            foreach (int windowSize in WindowSizes)
            {
                for (int f = 0; f < inputFeatures; f++)
                {
                    int localIdx = outputIdx;
                    ComputeVolatilityFeaturesThreadSafe(data, t, f, windowSize, output, localIdx);
                    outputIdx += featuresPerWindowFeature;
                }
            }
        });

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
        // Extract price window
        var prices = ExtractWindow(data, t, f, windowSize);

        // Compute returns for this window
        var simpleReturns = ComputeSimpleReturns(prices);
        var logReturns = ComputeLogReturns(prices);

        bool hasValidReturns = simpleReturns.Length > 0 &&
                               simpleReturns.Any(r => !double.IsNaN(r));

        // Simple returns
        if (_enabledMeasures.HasFlag(VolatilityMeasures.SimpleReturns))
        {
            double val = hasValidReturns ? simpleReturns.Where(r => !double.IsNaN(r)).LastOrDefault() : double.NaN;
            output[t, outputIdx++] = NumOps.FromDouble(val);
        }

        // Log returns
        if (_enabledMeasures.HasFlag(VolatilityMeasures.LogReturns))
        {
            double val = hasValidReturns ? logReturns.Where(r => !double.IsNaN(r)).LastOrDefault() : double.NaN;
            output[t, outputIdx++] = NumOps.FromDouble(val);
        }

        // Realized volatility (annualized std of log returns)
        if (_enabledMeasures.HasFlag(VolatilityMeasures.RealizedVolatility))
        {
            var validLogReturns = logReturns.Where(r => !double.IsNaN(r)).ToArray();
            double val = validLogReturns.Length >= 2
                ? ComputeStdDev(validLogReturns) * Math.Sqrt(_annualizationFactor)
                : double.NaN;
            output[t, outputIdx++] = NumOps.FromDouble(val);
        }

        // Parkinson volatility (requires high/low prices - approximate using price range)
        if (_enabledMeasures.HasFlag(VolatilityMeasures.ParkinsonVolatility))
        {
            double val = ComputeParkinsonVolatility(prices, windowSize);
            output[t, outputIdx++] = NumOps.FromDouble(val);
        }

        // Garman-Klass volatility (requires OHLC - approximate using available data)
        if (_enabledMeasures.HasFlag(VolatilityMeasures.GarmanKlassVolatility))
        {
            double val = ComputeGarmanKlassVolatility(prices, windowSize);
            output[t, outputIdx++] = NumOps.FromDouble(val);
        }

        // Momentum (price change ratio)
        if (_enabledMeasures.HasFlag(VolatilityMeasures.Momentum))
        {
            double val = ComputeMomentum(prices, windowSize);
            output[t, outputIdx++] = NumOps.FromDouble(val);
        }
    }

    private void ComputeVolatilityFeaturesThreadSafe(
        Tensor<T> data, int t, int f, int windowSize,
        Tensor<T> output, int startIdx)
    {
        int outputIdx = startIdx;

        var prices = ExtractWindow(data, t, f, windowSize);
        var simpleReturns = ComputeSimpleReturns(prices);
        var logReturns = ComputeLogReturns(prices);

        bool hasValidReturns = simpleReturns.Length > 0 &&
                               simpleReturns.Any(r => !double.IsNaN(r));

        if (_enabledMeasures.HasFlag(VolatilityMeasures.SimpleReturns))
        {
            double val = hasValidReturns ? simpleReturns.Where(r => !double.IsNaN(r)).LastOrDefault() : double.NaN;
            output[t, outputIdx++] = NumOps.FromDouble(val);
        }

        if (_enabledMeasures.HasFlag(VolatilityMeasures.LogReturns))
        {
            double val = hasValidReturns ? logReturns.Where(r => !double.IsNaN(r)).LastOrDefault() : double.NaN;
            output[t, outputIdx++] = NumOps.FromDouble(val);
        }

        if (_enabledMeasures.HasFlag(VolatilityMeasures.RealizedVolatility))
        {
            var validLogReturns = logReturns.Where(r => !double.IsNaN(r)).ToArray();
            double val = validLogReturns.Length >= 2
                ? ComputeStdDev(validLogReturns) * Math.Sqrt(_annualizationFactor)
                : double.NaN;
            output[t, outputIdx++] = NumOps.FromDouble(val);
        }

        if (_enabledMeasures.HasFlag(VolatilityMeasures.ParkinsonVolatility))
        {
            double val = ComputeParkinsonVolatility(prices, windowSize);
            output[t, outputIdx++] = NumOps.FromDouble(val);
        }

        if (_enabledMeasures.HasFlag(VolatilityMeasures.GarmanKlassVolatility))
        {
            double val = ComputeGarmanKlassVolatility(prices, windowSize);
            output[t, outputIdx++] = NumOps.FromDouble(val);
        }

        if (_enabledMeasures.HasFlag(VolatilityMeasures.Momentum))
        {
            double val = ComputeMomentum(prices, windowSize);
            output[t, outputIdx++] = NumOps.FromDouble(val);
        }
    }

    #endregion

    #region Window Extraction

    private double[] ExtractWindow(Tensor<T> data, int endTime, int feature, int windowSize)
    {
        int startTime = endTime - windowSize + 1;
        var window = new double[windowSize];

        for (int i = 0; i < windowSize; i++)
        {
            int t = startTime + i;
            if (t < 0)
            {
                window[i] = double.NaN;
            }
            else
            {
                window[i] = NumOps.ToDouble(GetValue(data, t, feature));
            }
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
            if (double.IsNaN(prices[i]) || double.IsNaN(prices[i - 1]) || prices[i - 1] == 0)
            {
                returns[i - 1] = double.NaN;
            }
            else
            {
                returns[i - 1] = (prices[i] - prices[i - 1]) / prices[i - 1];
            }
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
            if (double.IsNaN(prices[i]) || double.IsNaN(prices[i - 1]) ||
                prices[i] <= 0 || prices[i - 1] <= 0)
            {
                returns[i - 1] = double.NaN;
            }
            else
            {
                returns[i - 1] = Math.Log(prices[i] / prices[i - 1]);
            }
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
    /// Computes Parkinson volatility estimator.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The Parkinson volatility uses the high-low range of prices,
    /// which captures intraday volatility. It's more efficient than close-to-close volatility.
    ///
    /// Formula: sqrt(1/(4n*ln(2)) * sum(ln(High/Low)^2))
    ///
    /// Since we may not have high/low data, we approximate using a rolling max/min.
    /// </para>
    /// </remarks>
    private double ComputeParkinsonVolatility(double[] prices, int windowSize)
    {
        var validPrices = prices.Where(p => !double.IsNaN(p) && p > 0).ToArray();

        if (validPrices.Length < 2) return double.NaN;

        double high = validPrices.Max();
        double low = validPrices.Min();

        if (low <= 0 || high <= 0) return double.NaN;

        // Parkinson formula approximation
        double logRange = Math.Log(high / low);
        double variance = logRange * logRange / (4 * Math.Log(2));

        return Math.Sqrt(variance * _annualizationFactor);
    }

    /// <summary>
    /// Computes Garman-Klass volatility estimator.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Garman-Klass is an efficient volatility estimator that uses
    /// open, high, low, close (OHLC) prices. It's more accurate than Parkinson.
    ///
    /// Since we may only have close prices, we approximate using close prices and inferred ranges.
    /// </para>
    /// </remarks>
    private double ComputeGarmanKlassVolatility(double[] prices, int windowSize)
    {
        var validPrices = prices.Where(p => !double.IsNaN(p) && p > 0).ToArray();

        if (validPrices.Length < 3) return double.NaN;

        // Approximate OHLC from close prices
        double open = validPrices[0];
        double close = validPrices[^1];
        double high = validPrices.Max();
        double low = validPrices.Min();

        if (low <= 0 || high <= 0 || open <= 0 || close <= 0) return double.NaN;

        // Garman-Klass formula
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
}
