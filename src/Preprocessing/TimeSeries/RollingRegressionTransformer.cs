using AiDotNet.Models.Options;

namespace AiDotNet.Preprocessing.TimeSeries;

/// <summary>
/// Computes rolling regression features for financial time series analysis.
/// </summary>
/// <remarks>
/// <para>
/// This transformer calculates rolling statistics that measure the relationship between
/// an asset and a benchmark, including alpha, beta, R², Sharpe ratio, and Sortino ratio.
/// </para>
/// <para><b>For Beginners:</b> This transformer helps you understand how an asset (like a stock)
/// performs relative to a benchmark (like the S&P 500 market index).
///
/// Key features:
/// - Beta: How much the asset moves when the market moves (sensitivity)
/// - Alpha: Extra return the asset provides beyond what market exposure explains
/// - R²: How much of the asset's movement is explained by the market
/// - Sharpe Ratio: Return earned per unit of total risk taken
/// - Sortino Ratio: Like Sharpe but only penalizes downside risk
///
/// These rolling calculations show how these relationships change over time.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class RollingRegressionTransformer<T> : TimeSeriesTransformerBase<T>
{
    #region Fields

    /// <summary>
    /// The enabled regression features.
    /// </summary>
    private readonly RollingRegressionFeatures _enabledFeatures;

    /// <summary>
    /// The benchmark column index.
    /// </summary>
    private readonly int _benchmarkIndex;

    /// <summary>
    /// The period-adjusted risk-free rate.
    /// </summary>
    private readonly double _riskFreeRate;

    /// <summary>
    /// The minimum acceptable return for Sortino calculation.
    /// </summary>
    private readonly double _mar;

    /// <summary>
    /// The annualization factor for ratio calculations.
    /// </summary>
    private readonly double _annualizationFactor;

    /// <summary>
    /// Cached operation names.
    /// </summary>
    private readonly string[] _operationNames;

    #endregion

    #region Constructor

    /// <summary>
    /// Creates a new rolling regression transformer with the specified options.
    /// </summary>
    /// <param name="options">Configuration options, or null for defaults.</param>
    public RollingRegressionTransformer(TimeSeriesFeatureOptions? options = null)
        : base(options)
    {
        _enabledFeatures = Options.EnabledRegressionFeatures;
        _benchmarkIndex = Options.BenchmarkColumnIndex ?? 0;
        _annualizationFactor = Options.AnnualizationFactor;

        // Adjust risk-free rate for period if needed
        _riskFreeRate = Options.RiskFreeRateIsPeriodAdjusted
            ? Options.RiskFreeRate
            : Options.RiskFreeRate / Options.AnnualizationFactor;

        _mar = Options.MinimumAcceptableReturn / Options.AnnualizationFactor;

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
        // Guard against 1D inputs
        if (data.Rank < 2)
        {
            throw new ArgumentException(
                $"RollingRegressionTransformer requires at least 2D input data, but got {data.Rank}D tensor.");
        }

        // Validate benchmark index
        if (_benchmarkIndex < 0 || _benchmarkIndex >= data.Shape[1])
        {
            throw new ArgumentException(
                $"Benchmark column index {_benchmarkIndex} is out of range for data with {data.Shape[1]} columns");
        }
    }

    /// <inheritdoc />
    protected override Tensor<T> TransformCore(Tensor<T> data)
    {
        int inputTimeSteps = GetTimeSteps(data);
        int inputFeatures = InputFeatureCount;
        int outputFeatures = OutputFeatureCount;

        int outputTimeSteps = GetOutputTimeSteps(inputTimeSteps);
        int startIndex = GetOutputStartIndex();

        if (outputTimeSteps <= 0)
        {
            return new Tensor<T>(new[] { 0, outputFeatures });
        }

        var output = new Tensor<T>(new[] { outputTimeSteps, outputFeatures });

        // Extract benchmark returns
        var benchmarkPrices = ExtractSeries(data, _benchmarkIndex, inputTimeSteps);
        var benchmarkReturns = ComputeLogReturns(benchmarkPrices);

        int maxWindow = GetMaxWindowSize();
        int firstValidIndex = -1;

        for (int outT = 0; outT < outputTimeSteps; outT++)
        {
            int t = outT + startIndex;
            int outputIdx = 0;

            foreach (int windowSize in WindowSizes)
            {
                bool isEdge = IsEdgeRegion(t, windowSize);

                if (isEdge && Options.EdgeHandling == EdgeHandling.NaN)
                {
                    int featuresPerWindow = CountFeaturesPerWindow();
                    for (int i = 0; i < featuresPerWindow; i++)
                    {
                        output[outT, outputIdx++] = GetNaN();
                    }
                    continue;
                }

                if (firstValidIndex < 0 && !IsEdgeRegion(t, maxWindow))
                {
                    firstValidIndex = outT;
                }

                int effectiveWindow = ShouldComputePartialWindows() && isEdge
                    ? GetEffectiveWindowSize(t, windowSize)
                    : windowSize;

                // Compute features for each non-benchmark column
                for (int f = 0; f < inputFeatures; f++)
                {
                    if (f == _benchmarkIndex)
                        continue;

                    var assetPrices = ExtractWindow(data, t, f, effectiveWindow);
                    var assetReturns = ComputeLogReturns(assetPrices);
                    var benchmarkWindowReturns = ExtractReturnsWindow(benchmarkReturns, t, effectiveWindow);

                    ComputeRegressionFeatures(assetReturns, benchmarkWindowReturns, output, outT, ref outputIdx);
                }
            }
        }

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

        int outputTimeSteps = GetOutputTimeSteps(inputTimeSteps);
        int startIndex = GetOutputStartIndex();
        int maxWindow = GetMaxWindowSize();

        if (outputTimeSteps <= 0)
        {
            return new Tensor<T>(new[] { 0, outputFeatures });
        }

        var output = new Tensor<T>(new[] { outputTimeSteps, outputFeatures });

        // Extract benchmark returns
        var benchmarkPrices = ExtractSeries(data, _benchmarkIndex, inputTimeSteps);
        var benchmarkReturns = ComputeLogReturns(benchmarkPrices);

        int firstValidIndex = -1;
        object lockObj = new object();

        int featuresPerWindow = CountFeaturesPerWindow();
        int nonBenchmarkFeatures = inputFeatures - 1;

        Parallel.For(0, outputTimeSteps, outT =>
        {
            int t = outT + startIndex;
            int windowOffset = 0;

            foreach (int windowSize in WindowSizes)
            {
                bool isEdge = IsEdgeRegion(t, windowSize);

                if (isEdge && Options.EdgeHandling == EdgeHandling.NaN)
                {
                    int baseIdx = windowOffset * nonBenchmarkFeatures * CountEnabledFeatures();
                    for (int i = 0; i < nonBenchmarkFeatures * CountEnabledFeatures(); i++)
                    {
                        output[outT, baseIdx + i] = GetNaN();
                    }
                    windowOffset++;
                    continue;
                }

                if (!IsEdgeRegion(t, maxWindow))
                {
                    lock (lockObj)
                    {
                        if (firstValidIndex < 0 || outT < firstValidIndex)
                            firstValidIndex = outT;
                    }
                }

                int effectiveWindow = ShouldComputePartialWindows() && isEdge
                    ? GetEffectiveWindowSize(t, windowSize)
                    : windowSize;

                int featureOffset = 0;
                for (int f = 0; f < inputFeatures; f++)
                {
                    if (f == _benchmarkIndex)
                        continue;

                    var assetPrices = ExtractWindow(data, t, f, effectiveWindow);
                    var assetReturns = ComputeLogReturns(assetPrices);
                    var benchmarkWindowReturns = ExtractReturnsWindow(benchmarkReturns, t, effectiveWindow);

                    int outputIdx = windowOffset * nonBenchmarkFeatures * CountEnabledFeatures()
                                  + featureOffset * CountEnabledFeatures();

                    ComputeRegressionFeaturesThreadSafe(assetReturns, benchmarkWindowReturns, output, outT, outputIdx);
                    featureOffset++;
                }

                windowOffset++;
            }
        });

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
            for (int f = 0; f < inputNames.Length; f++)
            {
                if (f == _benchmarkIndex)
                    continue;

                foreach (string op in ops)
                {
                    names.Add($"{inputNames[f]}{sep}{op}{sep}{windowSize}");
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

        if (_enabledFeatures.HasFlag(RollingRegressionFeatures.Beta))
            ops.Add("beta");

        if (_enabledFeatures.HasFlag(RollingRegressionFeatures.Alpha))
            ops.Add("alpha");

        if (_enabledFeatures.HasFlag(RollingRegressionFeatures.RSquared))
            ops.Add("r_squared");

        if (_enabledFeatures.HasFlag(RollingRegressionFeatures.SharpeRatio))
            ops.Add("sharpe");

        if (_enabledFeatures.HasFlag(RollingRegressionFeatures.SortinoRatio))
            ops.Add("sortino");

        if (_enabledFeatures.HasFlag(RollingRegressionFeatures.Correlation))
            ops.Add("correlation");

        if (_enabledFeatures.HasFlag(RollingRegressionFeatures.TrackingError))
            ops.Add("tracking_error");

        if (_enabledFeatures.HasFlag(RollingRegressionFeatures.InformationRatio))
            ops.Add("info_ratio");

        return [.. ops];
    }

    private int CountEnabledFeatures()
    {
        return _operationNames.Length;
    }

    private int CountFeaturesPerWindow()
    {
        return CountEnabledFeatures() * (InputFeatureCount - 1);
    }

    #endregion

    #region Data Extraction

    private double[] ExtractSeries(Tensor<T> data, int feature, int timeSteps)
    {
        var series = new double[timeSteps];
        for (int t = 0; t < timeSteps; t++)
        {
            series[t] = NumOps.ToDouble(GetValue(data, t, feature));
        }
        return series;
    }

    private double[] ExtractWindow(Tensor<T> data, int endTime, int feature, int windowSize)
    {
        int actualSize = Math.Min(windowSize, endTime + 1);
        actualSize = Math.Max(1, actualSize);

        int startTime = endTime - actualSize + 1;
        var window = new double[actualSize];

        for (int i = 0; i < actualSize; i++)
        {
            int t = startTime + i;
            window[i] = t < 0 ? double.NaN : NumOps.ToDouble(GetValue(data, t, feature));
        }

        return window;
    }

    private static double[] ExtractReturnsWindow(double[] returns, int endTime, int windowSize)
    {
        // Returns array is 1 shorter than prices, so adjust
        int returnsEndIdx = Math.Min(endTime, returns.Length - 1);
        int actualSize = Math.Min(windowSize - 1, returnsEndIdx + 1);
        actualSize = Math.Max(1, actualSize);

        int startIdx = returnsEndIdx - actualSize + 1;
        var window = new double[actualSize];

        for (int i = 0; i < actualSize; i++)
        {
            int idx = startIdx + i;
            window[i] = idx >= 0 && idx < returns.Length ? returns[idx] : double.NaN;
        }

        return window;
    }

    #endregion

    #region Return Calculations

    /// <summary>
    /// Computes log returns from prices.
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

    #region Regression Calculations

    private void ComputeRegressionFeatures(double[] assetReturns, double[] benchmarkReturns,
        Tensor<T> output, int outT, ref int outputIdx)
    {
        // Align arrays and filter valid pairs
        var (validAsset, validBenchmark) = GetValidPairs(assetReturns, benchmarkReturns);

        if (validAsset.Length < 2)
        {
            // Not enough data
            int numFeatures = CountEnabledFeatures();
            for (int i = 0; i < numFeatures; i++)
            {
                output[outT, outputIdx++] = GetNaN();
            }
            return;
        }

        // Compute common statistics
        double assetMean = validAsset.Average();
        double benchmarkMean = validBenchmark.Average();
        double assetStd = ComputeStdDev(validAsset);
        double benchmarkStd = ComputeStdDev(validBenchmark);

        // Compute covariance and regression coefficients
        double covariance = ComputeCovariance(validAsset, validBenchmark, assetMean, benchmarkMean);
        double benchmarkVar = benchmarkStd * benchmarkStd;

        double beta = benchmarkVar > 1e-10 ? covariance / benchmarkVar : 0;
        double alpha = assetMean - beta * benchmarkMean;
        double correlation = assetStd > 1e-10 && benchmarkStd > 1e-10
            ? covariance / (assetStd * benchmarkStd)
            : 0;
        double rSquared = correlation * correlation;

        // Beta
        if (_enabledFeatures.HasFlag(RollingRegressionFeatures.Beta))
        {
            output[outT, outputIdx++] = NumOps.FromDouble(beta);
        }

        // Alpha (annualized)
        if (_enabledFeatures.HasFlag(RollingRegressionFeatures.Alpha))
        {
            double annualizedAlpha = alpha * _annualizationFactor;
            output[outT, outputIdx++] = NumOps.FromDouble(annualizedAlpha);
        }

        // R-squared
        if (_enabledFeatures.HasFlag(RollingRegressionFeatures.RSquared))
        {
            output[outT, outputIdx++] = NumOps.FromDouble(rSquared);
        }

        // Sharpe Ratio
        if (_enabledFeatures.HasFlag(RollingRegressionFeatures.SharpeRatio))
        {
            double sharpe = assetStd > 1e-10
                ? (assetMean - _riskFreeRate) / assetStd * Math.Sqrt(_annualizationFactor)
                : 0;
            output[outT, outputIdx++] = NumOps.FromDouble(sharpe);
        }

        // Sortino Ratio
        if (_enabledFeatures.HasFlag(RollingRegressionFeatures.SortinoRatio))
        {
            double downsideStd = ComputeDownsideDeviation(validAsset, _mar);
            double sortino = downsideStd > 1e-10
                ? (assetMean - _mar) / downsideStd * Math.Sqrt(_annualizationFactor)
                : 0;
            output[outT, outputIdx++] = NumOps.FromDouble(sortino);
        }

        // Correlation
        if (_enabledFeatures.HasFlag(RollingRegressionFeatures.Correlation))
        {
            output[outT, outputIdx++] = NumOps.FromDouble(correlation);
        }

        // Tracking Error
        if (_enabledFeatures.HasFlag(RollingRegressionFeatures.TrackingError))
        {
            var differences = new double[validAsset.Length];
            for (int i = 0; i < validAsset.Length; i++)
            {
                differences[i] = validAsset[i] - validBenchmark[i];
            }
            double trackingError = ComputeStdDev(differences) * Math.Sqrt(_annualizationFactor);
            output[outT, outputIdx++] = NumOps.FromDouble(trackingError);
        }

        // Information Ratio
        if (_enabledFeatures.HasFlag(RollingRegressionFeatures.InformationRatio))
        {
            var differences = new double[validAsset.Length];
            for (int i = 0; i < validAsset.Length; i++)
            {
                differences[i] = validAsset[i] - validBenchmark[i];
            }
            double trackingError = ComputeStdDev(differences);
            double meanDiff = differences.Average();
            double infoRatio = trackingError > 1e-10
                ? meanDiff / trackingError * Math.Sqrt(_annualizationFactor)
                : 0;
            output[outT, outputIdx++] = NumOps.FromDouble(infoRatio);
        }
    }

    private void ComputeRegressionFeaturesThreadSafe(double[] assetReturns, double[] benchmarkReturns,
        Tensor<T> output, int outT, int startIdx)
    {
        int outputIdx = startIdx;

        var (validAsset, validBenchmark) = GetValidPairs(assetReturns, benchmarkReturns);

        if (validAsset.Length < 2)
        {
            int numFeatures = CountEnabledFeatures();
            for (int i = 0; i < numFeatures; i++)
            {
                output[outT, outputIdx++] = GetNaN();
            }
            return;
        }

        double assetMean = validAsset.Average();
        double benchmarkMean = validBenchmark.Average();
        double assetStd = ComputeStdDev(validAsset);
        double benchmarkStd = ComputeStdDev(validBenchmark);

        double covariance = ComputeCovariance(validAsset, validBenchmark, assetMean, benchmarkMean);
        double benchmarkVar = benchmarkStd * benchmarkStd;

        double beta = benchmarkVar > 1e-10 ? covariance / benchmarkVar : 0;
        double alpha = assetMean - beta * benchmarkMean;
        double correlation = assetStd > 1e-10 && benchmarkStd > 1e-10
            ? covariance / (assetStd * benchmarkStd)
            : 0;
        double rSquared = correlation * correlation;

        if (_enabledFeatures.HasFlag(RollingRegressionFeatures.Beta))
        {
            output[outT, outputIdx++] = NumOps.FromDouble(beta);
        }

        if (_enabledFeatures.HasFlag(RollingRegressionFeatures.Alpha))
        {
            double annualizedAlpha = alpha * _annualizationFactor;
            output[outT, outputIdx++] = NumOps.FromDouble(annualizedAlpha);
        }

        if (_enabledFeatures.HasFlag(RollingRegressionFeatures.RSquared))
        {
            output[outT, outputIdx++] = NumOps.FromDouble(rSquared);
        }

        if (_enabledFeatures.HasFlag(RollingRegressionFeatures.SharpeRatio))
        {
            double sharpe = assetStd > 1e-10
                ? (assetMean - _riskFreeRate) / assetStd * Math.Sqrt(_annualizationFactor)
                : 0;
            output[outT, outputIdx++] = NumOps.FromDouble(sharpe);
        }

        if (_enabledFeatures.HasFlag(RollingRegressionFeatures.SortinoRatio))
        {
            double downsideStd = ComputeDownsideDeviation(validAsset, _mar);
            double sortino = downsideStd > 1e-10
                ? (assetMean - _mar) / downsideStd * Math.Sqrt(_annualizationFactor)
                : 0;
            output[outT, outputIdx++] = NumOps.FromDouble(sortino);
        }

        if (_enabledFeatures.HasFlag(RollingRegressionFeatures.Correlation))
        {
            output[outT, outputIdx++] = NumOps.FromDouble(correlation);
        }

        if (_enabledFeatures.HasFlag(RollingRegressionFeatures.TrackingError))
        {
            var differences = new double[validAsset.Length];
            for (int i = 0; i < validAsset.Length; i++)
            {
                differences[i] = validAsset[i] - validBenchmark[i];
            }
            double trackingError = ComputeStdDev(differences) * Math.Sqrt(_annualizationFactor);
            output[outT, outputIdx++] = NumOps.FromDouble(trackingError);
        }

        if (_enabledFeatures.HasFlag(RollingRegressionFeatures.InformationRatio))
        {
            var differences = new double[validAsset.Length];
            for (int i = 0; i < validAsset.Length; i++)
            {
                differences[i] = validAsset[i] - validBenchmark[i];
            }
            double trackingError = ComputeStdDev(differences);
            double meanDiff = differences.Average();
            double infoRatio = trackingError > 1e-10
                ? meanDiff / trackingError * Math.Sqrt(_annualizationFactor)
                : 0;
            output[outT, outputIdx++] = NumOps.FromDouble(infoRatio);
        }
    }

    #endregion

    #region Incremental Computation

    /// <summary>
    /// Gets whether this transformer supports incremental transformation.
    /// Rolling regression requires benchmark data synchronization which is complex for incremental.
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
            ["EnabledFeatures"] = (int)_enabledFeatures,
            ["BenchmarkIndex"] = _benchmarkIndex,
            ["RiskFreeRate"] = _riskFreeRate,
            ["MAR"] = _mar,
            ["AnnualizationFactor"] = _annualizationFactor,
            ["OperationNames"] = _operationNames
        };
    }

    /// <summary>
    /// Imports transformer-specific parameters for validation.
    /// </summary>
    protected override void ImportParameters(Dictionary<string, object> parameters)
    {
        if (parameters.TryGetValue("EnabledFeatures", out var featuresObj))
        {
            int savedFeatures = Convert.ToInt32(featuresObj);
            if (savedFeatures != (int)_enabledFeatures)
            {
                throw new ArgumentException(
                    $"Saved EnabledFeatures ({savedFeatures}) does not match current configuration ({(int)_enabledFeatures}).");
            }
        }

        if (parameters.TryGetValue("BenchmarkIndex", out var benchmarkObj))
        {
            int savedBenchmark = Convert.ToInt32(benchmarkObj);
            if (savedBenchmark != _benchmarkIndex)
            {
                throw new ArgumentException(
                    $"Saved BenchmarkIndex ({savedBenchmark}) does not match current configuration ({_benchmarkIndex}).");
            }
        }
    }

    #endregion

    #region Statistical Helpers

    /// <summary>
    /// Filters and aligns two arrays to contain only valid (non-NaN) paired values.
    /// </summary>
    private static (double[] Asset, double[] Benchmark) GetValidPairs(double[] assetReturns, double[] benchmarkReturns)
    {
        int minLen = Math.Min(assetReturns.Length, benchmarkReturns.Length);
        var validAsset = new List<double>();
        var validBenchmark = new List<double>();

        for (int i = 0; i < minLen; i++)
        {
            if (!double.IsNaN(assetReturns[i]) && !double.IsNaN(benchmarkReturns[i]))
            {
                validAsset.Add(assetReturns[i]);
                validBenchmark.Add(benchmarkReturns[i]);
            }
        }

        return ([.. validAsset], [.. validBenchmark]);
    }

    /// <summary>
    /// Computes sample standard deviation.
    /// </summary>
    private static double ComputeStdDev(double[] values)
    {
        if (values.Length < 2) return 0;

        double mean = values.Average();
        double sumSq = values.Select(x => (x - mean) * (x - mean)).Sum();
        return Math.Sqrt(sumSq / (values.Length - 1));
    }

    /// <summary>
    /// Computes sample covariance.
    /// </summary>
    private static double ComputeCovariance(double[] x, double[] y, double xMean, double yMean)
    {
        if (x.Length != y.Length || x.Length < 2) return 0;

        double sum = 0;
        for (int i = 0; i < x.Length; i++)
        {
            sum += (x[i] - xMean) * (y[i] - yMean);
        }

        return sum / (x.Length - 1);
    }

    /// <summary>
    /// Computes downside deviation (semi-deviation) for Sortino ratio.
    /// </summary>
    private static double ComputeDownsideDeviation(double[] returns, double mar)
    {
        var downsideReturns = returns.Where(r => r < mar).ToArray();

        if (downsideReturns.Length < 2) return 0;

        double sumSq = downsideReturns.Select(r => (r - mar) * (r - mar)).Sum();
        return Math.Sqrt(sumSq / returns.Length); // Use full count for semi-deviation
    }

    #endregion
}
