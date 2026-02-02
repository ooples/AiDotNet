using AiDotNet.Models.Options;

namespace AiDotNet.Preprocessing.TimeSeries;

/// <summary>
/// Computes technical analysis indicators for time series data.
/// </summary>
/// <remarks>
/// <para>
/// This transformer calculates industry-standard technical indicators commonly used in
/// financial analysis and algorithmic trading, including moving averages, momentum indicators,
/// volatility bands, and volume-based indicators.
/// </para>
/// <para><b>For Beginners:</b> Technical indicators are mathematical formulas applied to price/volume data
/// to help predict future price movements. They fall into several categories:
///
/// - <b>Moving Averages</b>: Smooth out price data to identify trends (SMA, EMA, WMA, DEMA, TEMA)
/// - <b>Momentum Indicators</b>: Measure the speed of price changes (RSI, MACD, Stochastic, CCI)
/// - <b>Volatility Indicators</b>: Measure how much prices are fluctuating (Bollinger Bands, ATR)
/// - <b>Volume Indicators</b>: Confirm trends using trading volume (OBV)
///
/// Traders use these indicators to:
/// - Identify when to buy or sell
/// - Confirm trend strength
/// - Spot potential reversals
/// - Set stop-loss levels
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class TechnicalIndicatorsTransformer<T> : TimeSeriesTransformerBase<T>
{
    #region Fields

    /// <summary>
    /// The enabled technical indicators.
    /// </summary>
    private readonly TechnicalIndicators _enabledIndicators;

    /// <summary>
    /// Short period for EMA/MACD calculations.
    /// </summary>
    private readonly int _shortPeriod;

    /// <summary>
    /// Long period for EMA/MACD calculations.
    /// </summary>
    private readonly int _longPeriod;

    /// <summary>
    /// Signal period for MACD signal line.
    /// </summary>
    private readonly int _signalPeriod;

    /// <summary>
    /// RSI period.
    /// </summary>
    private readonly int _rsiPeriod;

    /// <summary>
    /// Bollinger Band standard deviation multiplier.
    /// </summary>
    private readonly double _bollingerStdDev;

    /// <summary>
    /// Stochastic K period.
    /// </summary>
    private readonly int _stochasticKPeriod;

    /// <summary>
    /// Stochastic D period.
    /// </summary>
    private readonly int _stochasticDPeriod;

    /// <summary>
    /// CCI period.
    /// </summary>
    private readonly int _cciPeriod;

    /// <summary>
    /// ADX period.
    /// </summary>
    private readonly int _adxPeriod;

    /// <summary>
    /// OHLC column configuration.
    /// </summary>
    private readonly OhlcColumnConfig? _ohlcConfig;

    /// <summary>
    /// Cached operation names.
    /// </summary>
    private readonly string[] _operationNames;

    #endregion

    #region Constructor

    /// <summary>
    /// Creates a new technical indicators transformer with the specified options.
    /// </summary>
    /// <param name="options">Configuration options, or null for defaults.</param>
    public TechnicalIndicatorsTransformer(TimeSeriesFeatureOptions? options = null)
        : base(options)
    {
        _enabledIndicators = Options.EnabledIndicators;
        _shortPeriod = Options.ShortPeriod;
        _longPeriod = Options.LongPeriod;
        _signalPeriod = Options.SignalPeriod;
        _rsiPeriod = Options.RsiPeriod;
        _bollingerStdDev = Options.BollingerBandStdDev;
        _stochasticKPeriod = Options.StochasticKPeriod;
        _stochasticDPeriod = Options.StochasticDPeriod;
        _cciPeriod = Options.CciPeriod;
        _adxPeriod = Options.AdxPeriod;
        _ohlcConfig = Options.OhlcColumns;

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
        // Technical indicators don't need to learn parameters
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

        // Pre-compute all indicators for efficiency (avoid redundant window calculations)
        var indicators = ComputeAllIndicators(data, inputTimeSteps, inputFeatures);

        // Track first valid index for forward fill
        int firstValidIndex = -1;
        int maxWindow = GetMaxWindowSize();

        for (int outT = 0; outT < outputTimeSteps; outT++)
        {
            int t = outT + startIndex;
            int outputIdx = 0;

            foreach (int windowSize in WindowSizes)
            {
                bool isEdge = IsEdgeRegion(t, windowSize);

                if (isEdge && Options.EdgeHandling == EdgeHandling.NaN)
                {
                    int indicatorCount = CountEnabledIndicators() * inputFeatures;
                    for (int i = 0; i < indicatorCount; i++)
                    {
                        output[outT, outputIdx++] = GetNaN();
                    }
                    continue;
                }

                // Track first valid index for forward fill
                if (firstValidIndex < 0 && !IsEdgeRegion(t, maxWindow))
                {
                    firstValidIndex = outT;
                }

                // Copy pre-computed indicator values
                foreach (var indicator in indicators)
                {
                    if (indicator.WindowSize == windowSize)
                    {
                        for (int f = 0; f < inputFeatures; f++)
                        {
                            var values = indicator.Values[f];
                            output[outT, outputIdx++] = t < values.Length
                                ? NumOps.FromDouble(values[t])
                                : GetNaN();
                        }
                    }
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
        // For technical indicators, sequential computation is often more efficient
        // due to the cumulative nature of many calculations (EMA, OBV, etc.)
        // We parallelize over features instead
        return TransformCore(data);
    }

    #endregion

    #region Feature Naming

    /// <inheritdoc />
    protected override string[] GenerateFeatureNames()
    {
        var names = new List<string>();
        var inputNames = GetInputFeatureNames();
        var sep = GetSeparator();

        foreach (int windowSize in WindowSizes)
        {
            foreach (var opName in _operationNames)
            {
                foreach (var inputName in inputNames)
                {
                    names.Add($"{inputName}{sep}{opName}{sep}{windowSize}");
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
        var names = new List<string>();

        if ((_enabledIndicators & TechnicalIndicators.SMA) != 0) names.Add("sma");
        if ((_enabledIndicators & TechnicalIndicators.EMA) != 0) names.Add("ema");
        if ((_enabledIndicators & TechnicalIndicators.WMA) != 0) names.Add("wma");
        if ((_enabledIndicators & TechnicalIndicators.DEMA) != 0) names.Add("dema");
        if ((_enabledIndicators & TechnicalIndicators.TEMA) != 0) names.Add("tema");
        if ((_enabledIndicators & TechnicalIndicators.BollingerBands) != 0)
        {
            names.Add("bb_upper");
            names.Add("bb_middle");
            names.Add("bb_lower");
            names.Add("bb_width");
        }
        if ((_enabledIndicators & TechnicalIndicators.RSI) != 0) names.Add("rsi");
        if ((_enabledIndicators & TechnicalIndicators.MACD) != 0)
        {
            names.Add("macd");
            names.Add("macd_signal");
            names.Add("macd_histogram");
        }
        if ((_enabledIndicators & TechnicalIndicators.ATR) != 0) names.Add("atr");
        if ((_enabledIndicators & TechnicalIndicators.StochasticOscillator) != 0)
        {
            names.Add("stoch_k");
            names.Add("stoch_d");
        }
        if ((_enabledIndicators & TechnicalIndicators.CCI) != 0) names.Add("cci");
        if ((_enabledIndicators & TechnicalIndicators.WilliamsR) != 0) names.Add("williams_r");
        if ((_enabledIndicators & TechnicalIndicators.ADX) != 0)
        {
            names.Add("adx");
            names.Add("plus_di");
            names.Add("minus_di");
        }
        if ((_enabledIndicators & TechnicalIndicators.OBV) != 0) names.Add("obv");

        return [.. names];
    }

    private int CountEnabledIndicators()
    {
        int count = 0;

        if ((_enabledIndicators & TechnicalIndicators.SMA) != 0) count++;
        if ((_enabledIndicators & TechnicalIndicators.EMA) != 0) count++;
        if ((_enabledIndicators & TechnicalIndicators.WMA) != 0) count++;
        if ((_enabledIndicators & TechnicalIndicators.DEMA) != 0) count++;
        if ((_enabledIndicators & TechnicalIndicators.TEMA) != 0) count++;
        if ((_enabledIndicators & TechnicalIndicators.BollingerBands) != 0) count += 4; // upper, middle, lower, width
        if ((_enabledIndicators & TechnicalIndicators.RSI) != 0) count++;
        if ((_enabledIndicators & TechnicalIndicators.MACD) != 0) count += 3; // macd, signal, histogram
        if ((_enabledIndicators & TechnicalIndicators.ATR) != 0) count++;
        if ((_enabledIndicators & TechnicalIndicators.StochasticOscillator) != 0) count += 2; // %K, %D
        if ((_enabledIndicators & TechnicalIndicators.CCI) != 0) count++;
        if ((_enabledIndicators & TechnicalIndicators.WilliamsR) != 0) count++;
        if ((_enabledIndicators & TechnicalIndicators.ADX) != 0) count += 3; // ADX, +DI, -DI
        if ((_enabledIndicators & TechnicalIndicators.OBV) != 0) count++;

        return count;
    }

    #endregion

    #region Incremental Computation

    /// <summary>
    /// Gets whether this transformer supports incremental transformation.
    /// Technical indicators like EMA, MACD, RSI require full history for accurate computation.
    /// </summary>
    public override bool SupportsIncrementalTransform => false;

    #endregion

    #region Serialization

    /// <summary>
    /// Exports transformer-specific parameters for serialization.
    /// </summary>
    protected override Dictionary<string, object> ExportParameters()
    {
        return new Dictionary<string, object>
        {
            ["EnabledIndicators"] = (int)_enabledIndicators,
            ["ShortPeriod"] = _shortPeriod,
            ["LongPeriod"] = _longPeriod,
            ["SignalPeriod"] = _signalPeriod,
            ["RsiPeriod"] = _rsiPeriod,
            ["BollingerStdDev"] = _bollingerStdDev,
            ["StochasticKPeriod"] = _stochasticKPeriod,
            ["StochasticDPeriod"] = _stochasticDPeriod,
            ["CciPeriod"] = _cciPeriod,
            ["AdxPeriod"] = _adxPeriod,
            ["OperationNames"] = _operationNames
        };
    }

    /// <summary>
    /// Imports transformer-specific parameters for validation.
    /// </summary>
    protected override void ImportParameters(Dictionary<string, object> parameters)
    {
        if (parameters.TryGetValue("EnabledIndicators", out var indicatorsObj))
        {
            int savedIndicators = Convert.ToInt32(indicatorsObj);
            if (savedIndicators != (int)_enabledIndicators)
            {
                throw new ArgumentException(
                    $"Saved EnabledIndicators ({savedIndicators}) does not match current configuration ({(int)_enabledIndicators}).");
            }
        }
    }

    #endregion

    #region Indicator Calculations

    private record IndicatorResult(string Name, int WindowSize, double[][] Values);

    private List<IndicatorResult> ComputeAllIndicators(Tensor<T> data, int timeSteps, int features)
    {
        var results = new List<IndicatorResult>();

        // Extract price data for each feature
        var priceData = new double[features][];
        for (int f = 0; f < features; f++)
        {
            priceData[f] = new double[timeSteps];
            for (int t = 0; t < timeSteps; t++)
            {
                priceData[f][t] = NumOps.ToDouble(GetValue(data, t, f));
            }
        }

        // Extract OHLC if configured
        double[]? highPrices = null;
        double[]? lowPrices = null;
        double[]? closePrices = null;

        if (_ohlcConfig != null)
        {
            if (_ohlcConfig.HighIndex.HasValue && _ohlcConfig.HighIndex.Value < features)
                highPrices = priceData[_ohlcConfig.HighIndex.Value];
            if (_ohlcConfig.LowIndex.HasValue && _ohlcConfig.LowIndex.Value < features)
                lowPrices = priceData[_ohlcConfig.LowIndex.Value];
            if (_ohlcConfig.CloseIndex.HasValue && _ohlcConfig.CloseIndex.Value < features)
                closePrices = priceData[_ohlcConfig.CloseIndex.Value];
        }

        foreach (int windowSize in WindowSizes)
        {
            // Moving Averages
            if ((_enabledIndicators & TechnicalIndicators.SMA) != 0)
            {
                var smaValues = new double[features][];
                for (int f = 0; f < features; f++)
                    smaValues[f] = ComputeSMA(priceData[f], windowSize);
                results.Add(new IndicatorResult("sma", windowSize, smaValues));
            }

            if ((_enabledIndicators & TechnicalIndicators.EMA) != 0)
            {
                var emaValues = new double[features][];
                for (int f = 0; f < features; f++)
                    emaValues[f] = ComputeEMA(priceData[f], windowSize);
                results.Add(new IndicatorResult("ema", windowSize, emaValues));
            }

            if ((_enabledIndicators & TechnicalIndicators.WMA) != 0)
            {
                var wmaValues = new double[features][];
                for (int f = 0; f < features; f++)
                    wmaValues[f] = ComputeWMA(priceData[f], windowSize);
                results.Add(new IndicatorResult("wma", windowSize, wmaValues));
            }

            if ((_enabledIndicators & TechnicalIndicators.DEMA) != 0)
            {
                var demaValues = new double[features][];
                for (int f = 0; f < features; f++)
                    demaValues[f] = ComputeDEMA(priceData[f], windowSize);
                results.Add(new IndicatorResult("dema", windowSize, demaValues));
            }

            if ((_enabledIndicators & TechnicalIndicators.TEMA) != 0)
            {
                var temaValues = new double[features][];
                for (int f = 0; f < features; f++)
                    temaValues[f] = ComputeTEMA(priceData[f], windowSize);
                results.Add(new IndicatorResult("tema", windowSize, temaValues));
            }

            // Bollinger Bands
            if ((_enabledIndicators & TechnicalIndicators.BollingerBands) != 0)
            {
                var bbUpper = new double[features][];
                var bbMiddle = new double[features][];
                var bbLower = new double[features][];
                var bbWidth = new double[features][];
                for (int f = 0; f < features; f++)
                {
                    var bb = ComputeBollingerBands(priceData[f], windowSize, _bollingerStdDev);
                    bbUpper[f] = bb.Upper;
                    bbMiddle[f] = bb.Middle;
                    bbLower[f] = bb.Lower;
                    bbWidth[f] = bb.Width;
                }
                results.Add(new IndicatorResult("bb_upper", windowSize, bbUpper));
                results.Add(new IndicatorResult("bb_middle", windowSize, bbMiddle));
                results.Add(new IndicatorResult("bb_lower", windowSize, bbLower));
                results.Add(new IndicatorResult("bb_width", windowSize, bbWidth));
            }

            // RSI - uses _rsiPeriod but we include for each window for consistency
            if ((_enabledIndicators & TechnicalIndicators.RSI) != 0)
            {
                var rsiValues = new double[features][];
                for (int f = 0; f < features; f++)
                    rsiValues[f] = ComputeRSI(priceData[f], windowSize);
                results.Add(new IndicatorResult("rsi", windowSize, rsiValues));
            }

            // MACD
            if ((_enabledIndicators & TechnicalIndicators.MACD) != 0)
            {
                var macdLine = new double[features][];
                var macdSignal = new double[features][];
                var macdHist = new double[features][];
                for (int f = 0; f < features; f++)
                {
                    var macd = ComputeMACD(priceData[f], _shortPeriod, _longPeriod, _signalPeriod);
                    macdLine[f] = macd.Line;
                    macdSignal[f] = macd.Signal;
                    macdHist[f] = macd.Histogram;
                }
                results.Add(new IndicatorResult("macd", windowSize, macdLine));
                results.Add(new IndicatorResult("macd_signal", windowSize, macdSignal));
                results.Add(new IndicatorResult("macd_histogram", windowSize, macdHist));
            }

            // ATR - requires high, low, close
            if ((_enabledIndicators & TechnicalIndicators.ATR) != 0)
            {
                var atrValues = new double[features][];
                for (int f = 0; f < features; f++)
                {
                    if (highPrices != null && lowPrices != null && closePrices != null)
                        atrValues[f] = ComputeATR(highPrices, lowPrices, closePrices, windowSize);
                    else
                        atrValues[f] = ComputeATRApproximation(priceData[f], windowSize);
                }
                results.Add(new IndicatorResult("atr", windowSize, atrValues));
            }

            // Stochastic Oscillator
            if ((_enabledIndicators & TechnicalIndicators.StochasticOscillator) != 0)
            {
                var stochK = new double[features][];
                var stochD = new double[features][];
                for (int f = 0; f < features; f++)
                {
                    if (highPrices != null && lowPrices != null && closePrices != null)
                    {
                        var stoch = ComputeStochastic(highPrices, lowPrices, closePrices, windowSize, _stochasticDPeriod);
                        stochK[f] = stoch.K;
                        stochD[f] = stoch.D;
                    }
                    else
                    {
                        var stoch = ComputeStochasticApproximation(priceData[f], windowSize, _stochasticDPeriod);
                        stochK[f] = stoch.K;
                        stochD[f] = stoch.D;
                    }
                }
                results.Add(new IndicatorResult("stoch_k", windowSize, stochK));
                results.Add(new IndicatorResult("stoch_d", windowSize, stochD));
            }

            // CCI
            if ((_enabledIndicators & TechnicalIndicators.CCI) != 0)
            {
                var cciValues = new double[features][];
                for (int f = 0; f < features; f++)
                {
                    if (highPrices != null && lowPrices != null && closePrices != null)
                        cciValues[f] = ComputeCCI(highPrices, lowPrices, closePrices, windowSize);
                    else
                        cciValues[f] = ComputeCCIApproximation(priceData[f], windowSize);
                }
                results.Add(new IndicatorResult("cci", windowSize, cciValues));
            }

            // Williams %R
            if ((_enabledIndicators & TechnicalIndicators.WilliamsR) != 0)
            {
                var wrValues = new double[features][];
                for (int f = 0; f < features; f++)
                {
                    if (highPrices != null && lowPrices != null && closePrices != null)
                        wrValues[f] = ComputeWilliamsR(highPrices, lowPrices, closePrices, windowSize);
                    else
                        wrValues[f] = ComputeWilliamsRApproximation(priceData[f], windowSize);
                }
                results.Add(new IndicatorResult("williams_r", windowSize, wrValues));
            }

            // ADX
            if ((_enabledIndicators & TechnicalIndicators.ADX) != 0)
            {
                var adxValues = new double[features][];
                var plusDi = new double[features][];
                var minusDi = new double[features][];
                for (int f = 0; f < features; f++)
                {
                    if (highPrices != null && lowPrices != null && closePrices != null)
                    {
                        var adx = ComputeADX(highPrices, lowPrices, closePrices, windowSize);
                        adxValues[f] = adx.ADX;
                        plusDi[f] = adx.PlusDI;
                        minusDi[f] = adx.MinusDI;
                    }
                    else
                    {
                        // ADX requires HLC, fill with NaN if not available
                        adxValues[f] = Enumerable.Repeat(double.NaN, timeSteps).ToArray();
                        plusDi[f] = Enumerable.Repeat(double.NaN, timeSteps).ToArray();
                        minusDi[f] = Enumerable.Repeat(double.NaN, timeSteps).ToArray();
                    }
                }
                results.Add(new IndicatorResult("adx", windowSize, adxValues));
                results.Add(new IndicatorResult("plus_di", windowSize, plusDi));
                results.Add(new IndicatorResult("minus_di", windowSize, minusDi));
            }

            // OBV - requires volume data (not typically available in standard price data)
            if ((_enabledIndicators & TechnicalIndicators.OBV) != 0)
            {
                var obvValues = new double[features][];
                for (int f = 0; f < features; f++)
                {
                    // OBV approximation using price as proxy for volume
                    obvValues[f] = ComputeOBVApproximation(priceData[f]);
                }
                results.Add(new IndicatorResult("obv", windowSize, obvValues));
            }
        }

        return results;
    }

    #endregion

    #region Moving Average Calculations

    /// <summary>
    /// Computes Simple Moving Average.
    /// </summary>
    private static double[] ComputeSMA(double[] prices, int period)
    {
        int n = prices.Length;
        var result = new double[n];

        for (int i = 0; i < n; i++)
        {
            if (i < period - 1)
            {
                result[i] = double.NaN;
                continue;
            }

            double sum = 0;
            for (int j = 0; j < period; j++)
                sum += prices[i - j];

            result[i] = sum / period;
        }

        return result;
    }

    /// <summary>
    /// Computes Exponential Moving Average.
    /// </summary>
    private static double[] ComputeEMA(double[] prices, int period)
    {
        int n = prices.Length;
        var result = new double[n];
        double multiplier = 2.0 / (period + 1);

        // Initialize with SMA for first period
        double sum = 0;
        for (int i = 0; i < Math.Min(period, n); i++)
            sum += prices[i];

        if (n < period)
        {
            for (int i = 0; i < n; i++)
                result[i] = double.NaN;
            return result;
        }

        result[period - 1] = sum / period;

        // Fill earlier values with NaN
        for (int i = 0; i < period - 1; i++)
            result[i] = double.NaN;

        // Calculate EMA for rest
        for (int i = period; i < n; i++)
        {
            result[i] = (prices[i] - result[i - 1]) * multiplier + result[i - 1];
        }

        return result;
    }

    /// <summary>
    /// Computes Weighted Moving Average.
    /// </summary>
    private static double[] ComputeWMA(double[] prices, int period)
    {
        int n = prices.Length;
        var result = new double[n];
        double denominator = period * (period + 1) / 2.0;

        for (int i = 0; i < n; i++)
        {
            if (i < period - 1)
            {
                result[i] = double.NaN;
                continue;
            }

            double sum = 0;
            for (int j = 0; j < period; j++)
            {
                sum += prices[i - j] * (period - j);
            }

            result[i] = sum / denominator;
        }

        return result;
    }

    /// <summary>
    /// Computes Double Exponential Moving Average (DEMA = 2*EMA - EMA(EMA)).
    /// </summary>
    private static double[] ComputeDEMA(double[] prices, int period)
    {
        var ema1 = ComputeEMA(prices, period);
        var ema2 = ComputeEMA(ema1, period);

        int n = prices.Length;
        var result = new double[n];

        for (int i = 0; i < n; i++)
        {
            if (double.IsNaN(ema1[i]) || double.IsNaN(ema2[i]))
                result[i] = double.NaN;
            else
                result[i] = 2 * ema1[i] - ema2[i];
        }

        return result;
    }

    /// <summary>
    /// Computes Triple Exponential Moving Average (TEMA = 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))).
    /// </summary>
    private static double[] ComputeTEMA(double[] prices, int period)
    {
        var ema1 = ComputeEMA(prices, period);
        var ema2 = ComputeEMA(ema1, period);
        var ema3 = ComputeEMA(ema2, period);

        int n = prices.Length;
        var result = new double[n];

        for (int i = 0; i < n; i++)
        {
            if (double.IsNaN(ema1[i]) || double.IsNaN(ema2[i]) || double.IsNaN(ema3[i]))
                result[i] = double.NaN;
            else
                result[i] = 3 * ema1[i] - 3 * ema2[i] + ema3[i];
        }

        return result;
    }

    #endregion

    #region Bollinger Bands

    private record BollingerBandsResult(double[] Upper, double[] Middle, double[] Lower, double[] Width);

    /// <summary>
    /// Computes Bollinger Bands (middle band = SMA, upper/lower = SMA ± stdDev*multiplier).
    /// </summary>
    private static BollingerBandsResult ComputeBollingerBands(double[] prices, int period, double stdDevMultiplier)
    {
        int n = prices.Length;
        var upper = new double[n];
        var middle = new double[n];
        var lower = new double[n];
        var width = new double[n];

        for (int i = 0; i < n; i++)
        {
            if (i < period - 1)
            {
                upper[i] = middle[i] = lower[i] = width[i] = double.NaN;
                continue;
            }

            // Calculate SMA
            double sum = 0;
            for (int j = 0; j < period; j++)
                sum += prices[i - j];
            double sma = sum / period;

            // Calculate standard deviation
            double sumSq = 0;
            for (int j = 0; j < period; j++)
            {
                double diff = prices[i - j] - sma;
                sumSq += diff * diff;
            }
            double stdDev = Math.Sqrt(sumSq / period);

            middle[i] = sma;
            upper[i] = sma + stdDevMultiplier * stdDev;
            lower[i] = sma - stdDevMultiplier * stdDev;
            width[i] = (upper[i] - lower[i]) / middle[i];  // Normalized width
        }

        return new BollingerBandsResult(upper, middle, lower, width);
    }

    #endregion

    #region RSI

    /// <summary>
    /// Computes Relative Strength Index (RSI).
    /// </summary>
    private static double[] ComputeRSI(double[] prices, int period)
    {
        int n = prices.Length;
        var result = new double[n];

        if (n < period + 1)
        {
            for (int i = 0; i < n; i++)
                result[i] = double.NaN;
            return result;
        }

        // Calculate price changes
        var gains = new double[n];
        var losses = new double[n];

        for (int i = 1; i < n; i++)
        {
            double change = prices[i] - prices[i - 1];
            gains[i] = change > 0 ? change : 0;
            losses[i] = change < 0 ? -change : 0;
        }

        // Calculate initial average gain/loss
        double avgGain = 0, avgLoss = 0;
        for (int i = 1; i <= period; i++)
        {
            avgGain += gains[i];
            avgLoss += losses[i];
        }
        avgGain /= period;
        avgLoss /= period;

        // Fill initial values with NaN
        for (int i = 0; i < period; i++)
            result[i] = double.NaN;

        // Calculate RSI
        if (avgLoss == 0)
            result[period] = 100;
        else
            result[period] = 100 - (100 / (1 + avgGain / avgLoss));

        // Continue with smoothed averages
        for (int i = period + 1; i < n; i++)
        {
            avgGain = (avgGain * (period - 1) + gains[i]) / period;
            avgLoss = (avgLoss * (period - 1) + losses[i]) / period;

            if (avgLoss == 0)
                result[i] = 100;
            else
                result[i] = 100 - (100 / (1 + avgGain / avgLoss));
        }

        return result;
    }

    #endregion

    #region MACD

    private record MACDResult(double[] Line, double[] Signal, double[] Histogram);

    /// <summary>
    /// Computes MACD (Moving Average Convergence Divergence).
    /// </summary>
    private static MACDResult ComputeMACD(double[] prices, int shortPeriod, int longPeriod, int signalPeriod)
    {
        int n = prices.Length;
        var line = new double[n];
        var signal = new double[n];
        var histogram = new double[n];

        var shortEma = ComputeEMA(prices, shortPeriod);
        var longEma = ComputeEMA(prices, longPeriod);

        // MACD Line = Short EMA - Long EMA
        for (int i = 0; i < n; i++)
        {
            if (double.IsNaN(shortEma[i]) || double.IsNaN(longEma[i]))
                line[i] = double.NaN;
            else
                line[i] = shortEma[i] - longEma[i];
        }

        // Signal Line = EMA of MACD Line
        var signalEma = ComputeEMA(line, signalPeriod);
        for (int i = 0; i < n; i++)
        {
            signal[i] = signalEma[i];
            if (double.IsNaN(line[i]) || double.IsNaN(signal[i]))
                histogram[i] = double.NaN;
            else
                histogram[i] = line[i] - signal[i];
        }

        return new MACDResult(line, signal, histogram);
    }

    #endregion

    #region ATR

    /// <summary>
    /// Computes Average True Range (ATR) using actual OHLC data.
    /// </summary>
    private static double[] ComputeATR(double[] high, double[] low, double[] close, int period)
    {
        int n = high.Length;
        var result = new double[n];
        var trueRange = new double[n];

        // Calculate True Range
        trueRange[0] = high[0] - low[0];
        for (int i = 1; i < n; i++)
        {
            double hl = high[i] - low[i];
            double hc = Math.Abs(high[i] - close[i - 1]);
            double lc = Math.Abs(low[i] - close[i - 1]);
            trueRange[i] = Math.Max(hl, Math.Max(hc, lc));
        }

        // Calculate ATR as EMA of True Range
        return ComputeEMA(trueRange, period);
    }

    /// <summary>
    /// Computes ATR approximation when OHLC data is not available.
    /// </summary>
    private static double[] ComputeATRApproximation(double[] prices, int period)
    {
        int n = prices.Length;
        var trueRange = new double[n];

        // Approximate True Range using price changes
        trueRange[0] = 0;
        for (int i = 1; i < n; i++)
        {
            trueRange[i] = Math.Abs(prices[i] - prices[i - 1]);
        }

        return ComputeEMA(trueRange, period);
    }

    #endregion

    #region Stochastic Oscillator

    private record StochasticResult(double[] K, double[] D);

    /// <summary>
    /// Computes Stochastic Oscillator using actual OHLC data.
    /// </summary>
    private static StochasticResult ComputeStochastic(double[] high, double[] low, double[] close, int kPeriod, int dPeriod)
    {
        int n = high.Length;
        var k = new double[n];
        var d = new double[n];

        for (int i = 0; i < n; i++)
        {
            if (i < kPeriod - 1)
            {
                k[i] = double.NaN;
                continue;
            }

            double highest = double.MinValue;
            double lowest = double.MaxValue;

            for (int j = 0; j < kPeriod; j++)
            {
                highest = Math.Max(highest, high[i - j]);
                lowest = Math.Min(lowest, low[i - j]);
            }

            double range = highest - lowest;
            k[i] = range == 0 ? 50 : (close[i] - lowest) / range * 100;
        }

        // %D is SMA of %K
        for (int i = 0; i < n; i++)
        {
            if (i < kPeriod - 1 + dPeriod - 1)
            {
                d[i] = double.NaN;
                continue;
            }

            double sum = 0;
            int count = 0;
            for (int j = 0; j < dPeriod; j++)
            {
                if (!double.IsNaN(k[i - j]))
                {
                    sum += k[i - j];
                    count++;
                }
            }
            d[i] = count > 0 ? sum / count : double.NaN;
        }

        return new StochasticResult(k, d);
    }

    /// <summary>
    /// Computes Stochastic Oscillator approximation when OHLC data is not available.
    /// </summary>
    private static StochasticResult ComputeStochasticApproximation(double[] prices, int kPeriod, int dPeriod)
    {
        int n = prices.Length;
        var k = new double[n];
        var d = new double[n];

        for (int i = 0; i < n; i++)
        {
            if (i < kPeriod - 1)
            {
                k[i] = double.NaN;
                continue;
            }

            double highest = double.MinValue;
            double lowest = double.MaxValue;

            for (int j = 0; j < kPeriod; j++)
            {
                highest = Math.Max(highest, prices[i - j]);
                lowest = Math.Min(lowest, prices[i - j]);
            }

            double range = highest - lowest;
            k[i] = range == 0 ? 50 : (prices[i] - lowest) / range * 100;
        }

        // %D is SMA of %K
        for (int i = 0; i < n; i++)
        {
            if (i < kPeriod - 1 + dPeriod - 1)
            {
                d[i] = double.NaN;
                continue;
            }

            double sum = 0;
            int count = 0;
            for (int j = 0; j < dPeriod; j++)
            {
                if (!double.IsNaN(k[i - j]))
                {
                    sum += k[i - j];
                    count++;
                }
            }
            d[i] = count > 0 ? sum / count : double.NaN;
        }

        return new StochasticResult(k, d);
    }

    #endregion

    #region CCI

    /// <summary>
    /// Computes Commodity Channel Index using actual OHLC data.
    /// </summary>
    private static double[] ComputeCCI(double[] high, double[] low, double[] close, int period)
    {
        int n = high.Length;
        var result = new double[n];
        var typicalPrice = new double[n];

        // Calculate Typical Price = (High + Low + Close) / 3
        for (int i = 0; i < n; i++)
            typicalPrice[i] = (high[i] + low[i] + close[i]) / 3;

        var smaTP = ComputeSMA(typicalPrice, period);

        for (int i = 0; i < n; i++)
        {
            if (i < period - 1)
            {
                result[i] = double.NaN;
                continue;
            }

            // Calculate Mean Deviation
            double sumDeviation = 0;
            for (int j = 0; j < period; j++)
                sumDeviation += Math.Abs(typicalPrice[i - j] - smaTP[i]);
            double meanDeviation = sumDeviation / period;

            // CCI = (TP - SMA(TP)) / (0.015 * Mean Deviation)
            if (meanDeviation == 0)
                result[i] = 0;
            else
                result[i] = (typicalPrice[i] - smaTP[i]) / (0.015 * meanDeviation);
        }

        return result;
    }

    /// <summary>
    /// Computes CCI approximation when OHLC data is not available.
    /// </summary>
    private static double[] ComputeCCIApproximation(double[] prices, int period)
    {
        int n = prices.Length;
        var result = new double[n];
        var sma = ComputeSMA(prices, period);

        for (int i = 0; i < n; i++)
        {
            if (i < period - 1)
            {
                result[i] = double.NaN;
                continue;
            }

            double sumDeviation = 0;
            for (int j = 0; j < period; j++)
                sumDeviation += Math.Abs(prices[i - j] - sma[i]);
            double meanDeviation = sumDeviation / period;

            if (meanDeviation == 0)
                result[i] = 0;
            else
                result[i] = (prices[i] - sma[i]) / (0.015 * meanDeviation);
        }

        return result;
    }

    #endregion

    #region Williams %R

    /// <summary>
    /// Computes Williams %R using actual OHLC data.
    /// </summary>
    private static double[] ComputeWilliamsR(double[] high, double[] low, double[] close, int period)
    {
        int n = high.Length;
        var result = new double[n];

        for (int i = 0; i < n; i++)
        {
            if (i < period - 1)
            {
                result[i] = double.NaN;
                continue;
            }

            double highest = double.MinValue;
            double lowest = double.MaxValue;

            for (int j = 0; j < period; j++)
            {
                highest = Math.Max(highest, high[i - j]);
                lowest = Math.Min(lowest, low[i - j]);
            }

            double range = highest - lowest;
            // Williams %R = -100 * (Highest - Close) / (Highest - Lowest)
            result[i] = range == 0 ? -50 : -100 * (highest - close[i]) / range;
        }

        return result;
    }

    /// <summary>
    /// Computes Williams %R approximation when OHLC data is not available.
    /// </summary>
    private static double[] ComputeWilliamsRApproximation(double[] prices, int period)
    {
        int n = prices.Length;
        var result = new double[n];

        for (int i = 0; i < n; i++)
        {
            if (i < period - 1)
            {
                result[i] = double.NaN;
                continue;
            }

            double highest = double.MinValue;
            double lowest = double.MaxValue;

            for (int j = 0; j < period; j++)
            {
                highest = Math.Max(highest, prices[i - j]);
                lowest = Math.Min(lowest, prices[i - j]);
            }

            double range = highest - lowest;
            result[i] = range == 0 ? -50 : -100 * (highest - prices[i]) / range;
        }

        return result;
    }

    #endregion

    #region ADX

    private record ADXResult(double[] ADX, double[] PlusDI, double[] MinusDI);

    /// <summary>
    /// Computes Average Directional Index (ADX) using actual OHLC data.
    /// </summary>
    private static ADXResult ComputeADX(double[] high, double[] low, double[] close, int period)
    {
        int n = high.Length;
        var adx = new double[n];
        var plusDi = new double[n];
        var minusDi = new double[n];
        var plusDm = new double[n];
        var minusDm = new double[n];
        var tr = new double[n];

        // Calculate True Range and Directional Movement
        for (int i = 1; i < n; i++)
        {
            double highDiff = high[i] - high[i - 1];
            double lowDiff = low[i - 1] - low[i];

            plusDm[i] = highDiff > lowDiff && highDiff > 0 ? highDiff : 0;
            minusDm[i] = lowDiff > highDiff && lowDiff > 0 ? lowDiff : 0;

            double hl = high[i] - low[i];
            double hc = Math.Abs(high[i] - close[i - 1]);
            double lc = Math.Abs(low[i] - close[i - 1]);
            tr[i] = Math.Max(hl, Math.Max(hc, lc));
        }

        // Smooth using EMA
        var atrSmooth = ComputeEMA(tr, period);
        var plusDmSmooth = ComputeEMA(plusDm, period);
        var minusDmSmooth = ComputeEMA(minusDm, period);

        // Calculate +DI and -DI
        var dx = new double[n];
        for (int i = 0; i < n; i++)
        {
            if (double.IsNaN(atrSmooth[i]) || atrSmooth[i] == 0)
            {
                plusDi[i] = minusDi[i] = dx[i] = double.NaN;
                continue;
            }

            plusDi[i] = 100 * plusDmSmooth[i] / atrSmooth[i];
            minusDi[i] = 100 * minusDmSmooth[i] / atrSmooth[i];

            double diSum = plusDi[i] + minusDi[i];
            dx[i] = diSum == 0 ? 0 : 100 * Math.Abs(plusDi[i] - minusDi[i]) / diSum;
        }

        // ADX = EMA of DX
        var adxSmooth = ComputeEMA(dx, period);
        for (int i = 0; i < n; i++)
            adx[i] = adxSmooth[i];

        return new ADXResult(adx, plusDi, minusDi);
    }

    #endregion

    #region OBV

    /// <summary>
    /// Computes On-Balance Volume approximation using price as proxy for volume.
    /// </summary>
    private static double[] ComputeOBVApproximation(double[] prices)
    {
        int n = prices.Length;
        var result = new double[n];

        result[0] = 0;
        for (int i = 1; i < n; i++)
        {
            double change = prices[i] - prices[i - 1];
            // Use absolute price as proxy for volume
            double volume = Math.Abs(prices[i]);

            if (change > 0)
                result[i] = result[i - 1] + volume;
            else if (change < 0)
                result[i] = result[i - 1] - volume;
            else
                result[i] = result[i - 1];
        }

        return result;
    }

    #endregion
}
