using AiDotNet.Models.Options;

namespace AiDotNet.Preprocessing.TimeSeries;

/// <summary>
/// Applies differencing and stationarity transformations to time series data.
/// </summary>
/// <remarks>
/// <para>
/// This transformer provides various methods to make time series stationary, which is
/// a requirement for many forecasting models (ARIMA, VAR, etc.). Stationary data has
/// constant statistical properties over time.
/// </para>
/// <para><b>For Beginners:</b> Many forecasting methods assume your data doesn't have trends
/// or seasonal patterns. This transformer removes those patterns through differencing.
///
/// Common transforms include:
/// - <b>First Difference</b>: Change from previous value (removes linear trend)
/// - <b>Seasonal Difference</b>: Change from same time in previous season (removes seasonality)
/// - <b>Detrending</b>: Subtract a fitted trend line
/// - <b>Decomposition</b>: Separate trend, seasonal, and residual components
///
/// After transformation, you can check stationarity using statistical tests like
/// the Augmented Dickey-Fuller test.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class DifferencingTransformer<T> : TimeSeriesTransformerBase<T>
{
    #region Fields

    /// <summary>
    /// The enabled differencing features.
    /// </summary>
    private readonly DifferencingFeatures _enabledFeatures;

    /// <summary>
    /// Differencing order.
    /// </summary>
    private readonly int _differencingOrder;

    /// <summary>
    /// Seasonal differencing period.
    /// </summary>
    private readonly int _seasonalPeriod;

    /// <summary>
    /// Polynomial degree for detrending.
    /// </summary>
    private readonly int _polynomialDegree;

    /// <summary>
    /// Hodrick-Prescott filter lambda.
    /// </summary>
    private readonly double _hpLambda;

    /// <summary>
    /// STL seasonal period.
    /// </summary>
    private readonly int _stlPeriod;

    /// <summary>
    /// STL robust iterations.
    /// </summary>
    private readonly int _stlIterations;

    /// <summary>
    /// Cached feature names.
    /// </summary>
    private readonly string[] _featureNames;

    /// <summary>
    /// Fitted trend coefficients for detrending (per feature).
    /// </summary>
    private double[][]? _trendCoefficients;

    #endregion

    #region Constructor

    /// <summary>
    /// Creates a new differencing transformer with the specified options.
    /// </summary>
    /// <param name="options">Configuration options, or null for defaults.</param>
    public DifferencingTransformer(TimeSeriesFeatureOptions? options = null)
        : base(options)
    {
        _enabledFeatures = Options.EnabledDifferencingFeatures;
        _differencingOrder = Options.DifferencingOrder;
        _seasonalPeriod = Options.SeasonalDifferencingPeriod;
        _polynomialDegree = Options.DetrendingPolynomialDegree;
        _hpLambda = Options.HodrickPrescottLambda;
        _stlPeriod = Options.StlSeasonalPeriod;
        _stlIterations = Options.StlRobustIterations;

        _featureNames = GenerateFeatureNames();
    }

    #endregion

    #region Properties

    /// <inheritdoc />
    public override bool SupportsInverseTransform => true;

    #endregion

    #region Core Implementation

    /// <inheritdoc />
    protected override void FitCore(Tensor<T> data)
    {
        int timeSteps = GetTimeSteps(data);
        int features = InputFeatureCount;

        // Fit trend coefficients for detrending
        if ((_enabledFeatures & (DifferencingFeatures.LinearDetrend | DifferencingFeatures.PolynomialDetrend)) != 0)
        {
            _trendCoefficients = new double[features][];
            for (int f = 0; f < features; f++)
            {
                var series = ExtractSeries(data, f, timeSteps);
                int degree = (_enabledFeatures & DifferencingFeatures.PolynomialDetrend) != 0
                    ? _polynomialDegree
                    : 1;
                _trendCoefficients[f] = FitPolynomial(series, degree);
            }
        }
    }

    /// <inheritdoc />
    protected override Tensor<T> TransformCore(Tensor<T> data)
    {
        int timeSteps = GetTimeSteps(data);
        int inputFeatures = InputFeatureCount;
        int outputFeatures = CountOutputFeatures(inputFeatures);

        var output = new Tensor<T>(new[] { timeSteps, outputFeatures });

        for (int f = 0; f < inputFeatures; f++)
        {
            var series = ExtractSeries(data, f, timeSteps);
            int outputIdx = f * CountFeaturesPerInput();

            // First difference
            if ((_enabledFeatures & DifferencingFeatures.FirstDifference) != 0)
            {
                var diff = ComputeDifference(series, 1);
                CopyToOutput(output, diff, outputIdx++);
            }

            // Second difference
            if ((_enabledFeatures & DifferencingFeatures.SecondDifference) != 0)
            {
                var diff = ComputeDifference(series, 2);
                CopyToOutput(output, diff, outputIdx++);
            }

            // Seasonal difference
            if ((_enabledFeatures & DifferencingFeatures.SeasonalDifference) != 0)
            {
                var diff = ComputeSeasonalDifference(series, _seasonalPeriod);
                CopyToOutput(output, diff, outputIdx++);
            }

            // Percent change
            if ((_enabledFeatures & DifferencingFeatures.PercentChange) != 0)
            {
                var pct = ComputePercentChange(series);
                CopyToOutput(output, pct, outputIdx++);
            }

            // Log difference
            if ((_enabledFeatures & DifferencingFeatures.LogDifference) != 0)
            {
                var logDiff = ComputeLogDifference(series);
                CopyToOutput(output, logDiff, outputIdx++);
            }

            // Linear detrend
            if ((_enabledFeatures & DifferencingFeatures.LinearDetrend) != 0)
            {
                var detrended = ComputeDetrended(series, _trendCoefficients?[f], 1);
                CopyToOutput(output, detrended, outputIdx++);
            }

            // Polynomial detrend
            if ((_enabledFeatures & DifferencingFeatures.PolynomialDetrend) != 0)
            {
                var detrended = ComputeDetrended(series, _trendCoefficients?[f], _polynomialDegree);
                CopyToOutput(output, detrended, outputIdx++);
            }

            // Hodrick-Prescott filter
            if ((_enabledFeatures & DifferencingFeatures.HodrickPrescottFilter) != 0)
            {
                var (trend, cycle) = ComputeHodrickPrescott(series, _hpLambda);
                CopyToOutput(output, trend, outputIdx++);
                CopyToOutput(output, cycle, outputIdx++);
            }

            // STL decomposition
            if ((_enabledFeatures & DifferencingFeatures.StlDecomposition) != 0)
            {
                var (seasonal, trend, residual) = ComputeStlDecomposition(series, _stlPeriod, _stlIterations);
                CopyToOutput(output, seasonal, outputIdx++);
                CopyToOutput(output, trend, outputIdx++);
                CopyToOutput(output, residual, outputIdx++);
            }
        }

        return output;
    }

    /// <inheritdoc />
    protected override Tensor<T> TransformParallel(Tensor<T> data)
    {
        int timeSteps = GetTimeSteps(data);
        int inputFeatures = InputFeatureCount;
        int outputFeatures = CountOutputFeatures(inputFeatures);
        int featuresPerInput = CountFeaturesPerInput();

        var output = new Tensor<T>(new[] { timeSteps, outputFeatures });

        Parallel.For(0, inputFeatures, f =>
        {
            var series = ExtractSeries(data, f, timeSteps);
            int outputIdx = f * featuresPerInput;

            if ((_enabledFeatures & DifferencingFeatures.FirstDifference) != 0)
            {
                var diff = ComputeDifference(series, 1);
                CopyToOutput(output, diff, outputIdx++);
            }

            if ((_enabledFeatures & DifferencingFeatures.SecondDifference) != 0)
            {
                var diff = ComputeDifference(series, 2);
                CopyToOutput(output, diff, outputIdx++);
            }

            if ((_enabledFeatures & DifferencingFeatures.SeasonalDifference) != 0)
            {
                var diff = ComputeSeasonalDifference(series, _seasonalPeriod);
                CopyToOutput(output, diff, outputIdx++);
            }

            if ((_enabledFeatures & DifferencingFeatures.PercentChange) != 0)
            {
                var pct = ComputePercentChange(series);
                CopyToOutput(output, pct, outputIdx++);
            }

            if ((_enabledFeatures & DifferencingFeatures.LogDifference) != 0)
            {
                var logDiff = ComputeLogDifference(series);
                CopyToOutput(output, logDiff, outputIdx++);
            }

            if ((_enabledFeatures & DifferencingFeatures.LinearDetrend) != 0)
            {
                var detrended = ComputeDetrended(series, _trendCoefficients?[f], 1);
                CopyToOutput(output, detrended, outputIdx++);
            }

            if ((_enabledFeatures & DifferencingFeatures.PolynomialDetrend) != 0)
            {
                var detrended = ComputeDetrended(series, _trendCoefficients?[f], _polynomialDegree);
                CopyToOutput(output, detrended, outputIdx++);
            }

            if ((_enabledFeatures & DifferencingFeatures.HodrickPrescottFilter) != 0)
            {
                var (trend, cycle) = ComputeHodrickPrescott(series, _hpLambda);
                CopyToOutput(output, trend, outputIdx++);
                CopyToOutput(output, cycle, outputIdx++);
            }

            if ((_enabledFeatures & DifferencingFeatures.StlDecomposition) != 0)
            {
                var (seasonal, trend, residual) = ComputeStlDecomposition(series, _stlPeriod, _stlIterations);
                CopyToOutput(output, seasonal, outputIdx++);
                CopyToOutput(output, trend, outputIdx++);
                CopyToOutput(output, residual, outputIdx++);
            }
        });

        return output;
    }

    #endregion

    #region Differencing Methods

    /// <summary>
    /// Computes n-th order differencing.
    /// </summary>
    private static double[] ComputeDifference(double[] series, int order)
    {
        int n = series.Length;
        var result = new double[n];

        // Copy series for iterative differencing
        var current = (double[])series.Clone();

        for (int d = 0; d < order; d++)
        {
            var next = new double[n];
            next[0] = double.NaN;

            for (int t = 1; t < n; t++)
            {
                if (double.IsNaN(current[t]) || double.IsNaN(current[t - 1]))
                    next[t] = double.NaN;
                else
                    next[t] = current[t] - current[t - 1];
            }

            current = next;
        }

        return current;
    }

    /// <summary>
    /// Computes seasonal differencing.
    /// </summary>
    private static double[] ComputeSeasonalDifference(double[] series, int period)
    {
        int n = series.Length;
        var result = new double[n];

        for (int t = 0; t < n; t++)
        {
            if (t < period)
                result[t] = double.NaN;
            else if (double.IsNaN(series[t]) || double.IsNaN(series[t - period]))
                result[t] = double.NaN;
            else
                result[t] = series[t] - series[t - period];
        }

        return result;
    }

    /// <summary>
    /// Computes percent change.
    /// </summary>
    private static double[] ComputePercentChange(double[] series)
    {
        int n = series.Length;
        var result = new double[n];
        result[0] = double.NaN;

        for (int t = 1; t < n; t++)
        {
            if (double.IsNaN(series[t]) || double.IsNaN(series[t - 1]) || series[t - 1] == 0)
                result[t] = double.NaN;
            else
                result[t] = (series[t] - series[t - 1]) / series[t - 1];
        }

        return result;
    }

    /// <summary>
    /// Computes log difference (log returns).
    /// </summary>
    private static double[] ComputeLogDifference(double[] series)
    {
        int n = series.Length;
        var result = new double[n];
        result[0] = double.NaN;

        for (int t = 1; t < n; t++)
        {
            if (double.IsNaN(series[t]) || double.IsNaN(series[t - 1]) ||
                series[t] <= 0 || series[t - 1] <= 0)
                result[t] = double.NaN;
            else
                result[t] = Math.Log(series[t]) - Math.Log(series[t - 1]);
        }

        return result;
    }

    #endregion

    #region Detrending Methods

    /// <summary>
    /// Fits a polynomial to the series using least squares.
    /// </summary>
    private static double[] FitPolynomial(double[] series, int degree)
    {
        int n = series.Length;
        int terms = degree + 1;

        // Build design matrix X and target vector y
        var validIndices = new List<int>();
        for (int i = 0; i < n; i++)
        {
            if (!double.IsNaN(series[i]))
                validIndices.Add(i);
        }

        if (validIndices.Count < terms)
            return new double[terms]; // Return zeros if not enough data

        int m = validIndices.Count;
        var X = new double[m, terms];
        var y = new double[m];

        for (int i = 0; i < m; i++)
        {
            int t = validIndices[i];
            y[i] = series[t];
            double x = (double)t / n; // Normalize to [0, 1]
            double power = 1.0;
            for (int j = 0; j < terms; j++)
            {
                X[i, j] = power;
                power *= x;
            }
        }

        // Solve normal equations: (X'X) * coeffs = X'y
        return SolveNormalEquations(X, y, m, terms);
    }

    /// <summary>
    /// Solves normal equations using Cholesky decomposition.
    /// </summary>
    private static double[] SolveNormalEquations(double[,] X, double[] y, int m, int terms)
    {
        // Compute X'X
        var XtX = new double[terms, terms];
        for (int i = 0; i < terms; i++)
        {
            for (int j = 0; j < terms; j++)
            {
                double sum = 0;
                for (int k = 0; k < m; k++)
                    sum += X[k, i] * X[k, j];
                XtX[i, j] = sum;
            }
        }

        // Compute X'y
        var Xty = new double[terms];
        for (int i = 0; i < terms; i++)
        {
            double sum = 0;
            for (int k = 0; k < m; k++)
                sum += X[k, i] * y[k];
            Xty[i] = sum;
        }

        // Simple Gaussian elimination for small systems
        var coeffs = new double[terms];
        var A = (double[,])XtX.Clone();
        var b = (double[])Xty.Clone();

        for (int i = 0; i < terms; i++)
        {
            // Find pivot
            int maxRow = i;
            for (int k = i + 1; k < terms; k++)
            {
                if (Math.Abs(A[k, i]) > Math.Abs(A[maxRow, i]))
                    maxRow = k;
            }

            // Swap rows
            for (int k = i; k < terms; k++)
                (A[i, k], A[maxRow, k]) = (A[maxRow, k], A[i, k]);
            (b[i], b[maxRow]) = (b[maxRow], b[i]);

            // Eliminate
            if (Math.Abs(A[i, i]) < 1e-10) continue;

            for (int k = i + 1; k < terms; k++)
            {
                double factor = A[k, i] / A[i, i];
                for (int j = i; j < terms; j++)
                    A[k, j] -= factor * A[i, j];
                b[k] -= factor * b[i];
            }
        }

        // Back substitution
        for (int i = terms - 1; i >= 0; i--)
        {
            double sum = b[i];
            for (int j = i + 1; j < terms; j++)
                sum -= A[i, j] * coeffs[j];
            coeffs[i] = Math.Abs(A[i, i]) > 1e-10 ? sum / A[i, i] : 0;
        }

        return coeffs;
    }

    /// <summary>
    /// Computes detrended series by subtracting fitted polynomial.
    /// </summary>
    private static double[] ComputeDetrended(double[] series, double[]? coeffs, int degree)
    {
        int n = series.Length;
        var result = new double[n];

        if (coeffs == null || coeffs.Length == 0)
        {
            Array.Copy(series, result, n);
            return result;
        }

        for (int t = 0; t < n; t++)
        {
            if (double.IsNaN(series[t]))
            {
                result[t] = double.NaN;
                continue;
            }

            double x = (double)t / n;
            double trend = 0;
            double power = 1.0;
            for (int j = 0; j < coeffs.Length; j++)
            {
                trend += coeffs[j] * power;
                power *= x;
            }

            result[t] = series[t] - trend;
        }

        return result;
    }

    #endregion

    #region Hodrick-Prescott Filter

    /// <summary>
    /// Computes Hodrick-Prescott filter decomposition.
    /// Solves: min sum(y_t - tau_t)^2 + lambda * sum(tau_{t+1} - 2*tau_t + tau_{t-1})^2
    /// Uses pentadiagonal matrix solver for (I + lambda*K'K)*tau = y.
    /// </summary>
    private static (double[] Trend, double[] Cycle) ComputeHodrickPrescott(double[] series, double lambda)
    {
        int n = series.Length;
        var trend = new double[n];
        var cycle = new double[n];

        if (n < 3)
        {
            // Too short for HP filter, return original as trend
            Array.Copy(series, trend, n);
            for (int t = 0; t < n; t++)
                cycle[t] = 0;
            return (trend, cycle);
        }

        // Handle NaN values by interpolating
        var y = InterpolateNaN(series);

        // Build pentadiagonal matrix (I + lambda*K'K) where K is second difference operator
        // K'K has structure:
        //   Row 0:    [1, -2, 1, 0, ...]
        //   Row 1:    [-2, 5, -4, 1, 0, ...]
        //   Row i:    [1, -4, 6, -4, 1] (interior rows)
        //   Row n-2:  [..., 1, -4, 5, -2]
        //   Row n-1:  [..., 1, -2, 1]

        // Diagonals: a (i-2), b (i-1), c (i), d (i+1), e (i+2)
        var a = new double[n];  // sub-sub diagonal (offset -2)
        var b = new double[n];  // sub diagonal (offset -1)
        var c = new double[n];  // main diagonal
        var d = new double[n];  // super diagonal (offset +1)
        var e = new double[n];  // super-super diagonal (offset +2)

        // Fill diagonals based on K'K structure
        for (int i = 0; i < n; i++)
        {
            if (i == 0)
            {
                c[i] = 1 + lambda;
                d[i] = -2 * lambda;
                e[i] = lambda;
            }
            else if (i == 1)
            {
                b[i] = -2 * lambda;
                c[i] = 1 + 5 * lambda;
                d[i] = -4 * lambda;
                e[i] = lambda;
            }
            else if (i == n - 2)
            {
                a[i] = lambda;
                b[i] = -4 * lambda;
                c[i] = 1 + 5 * lambda;
                d[i] = -2 * lambda;
            }
            else if (i == n - 1)
            {
                a[i] = lambda;
                b[i] = -2 * lambda;
                c[i] = 1 + lambda;
            }
            else
            {
                // Interior points
                a[i] = lambda;
                b[i] = -4 * lambda;
                c[i] = 1 + 6 * lambda;
                d[i] = -4 * lambda;
                e[i] = lambda;
            }
        }

        // Solve pentadiagonal system using forward elimination
        trend = SolvePentadiagonal(a, b, c, d, e, y);

        // Cycle = original - trend
        for (int t = 0; t < n; t++)
        {
            if (double.IsNaN(series[t]))
            {
                trend[t] = double.NaN;
                cycle[t] = double.NaN;
            }
            else
            {
                cycle[t] = series[t] - trend[t];
            }
        }

        return (trend, cycle);
    }

    /// <summary>
    /// Solves a pentadiagonal system Ax = r using Gaussian elimination.
    /// Diagonals: a (offset -2), b (offset -1), c (main), d (offset +1), e (offset +2).
    /// </summary>
    private static double[] SolvePentadiagonal(double[] a, double[] b, double[] c, double[] d, double[] e, double[] r)
    {
        int n = r.Length;

        // Make copies to avoid modifying inputs
        var aa = (double[])a.Clone();
        var bb = (double[])b.Clone();
        var cc = (double[])c.Clone();
        var dd = (double[])d.Clone();
        var ee = (double[])e.Clone();
        var rr = (double[])r.Clone();

        // Forward elimination
        for (int i = 0; i < n - 1; i++)
        {
            // Eliminate element at (i+1, i) using row i
            if (Math.Abs(cc[i]) > 1e-15)
            {
                double factor = bb[i + 1] / cc[i];
                bb[i + 1] = 0;
                cc[i + 1] -= factor * dd[i];
                if (i + 2 < n)
                    dd[i + 1] -= factor * ee[i];
                rr[i + 1] -= factor * rr[i];
            }

            // Eliminate element at (i+2, i) using row i (if exists)
            if (i + 2 < n && Math.Abs(cc[i]) > 1e-15)
            {
                double factor = aa[i + 2] / cc[i];
                aa[i + 2] = 0;
                bb[i + 2] -= factor * dd[i];
                cc[i + 2] -= factor * ee[i];
                rr[i + 2] -= factor * rr[i];
            }
        }

        // Back substitution
        var x = new double[n];

        for (int i = n - 1; i >= 0; i--)
        {
            double sum = rr[i];
            if (i + 1 < n)
                sum -= dd[i] * x[i + 1];
            if (i + 2 < n)
                sum -= ee[i] * x[i + 2];

            if (Math.Abs(cc[i]) > 1e-15)
                x[i] = sum / cc[i];
            else
                x[i] = 0;
        }

        return x;
    }

    #endregion

    #region STL Decomposition

    /// <summary>
    /// Computes STL (Seasonal-Trend decomposition using LOESS) decomposition.
    /// </summary>
    private static (double[] Seasonal, double[] Trend, double[] Residual) ComputeStlDecomposition(
        double[] series, int period, int robustIterations)
    {
        int n = series.Length;
        var seasonal = new double[n];
        var trend = new double[n];
        var residual = new double[n];

        // Handle NaN values
        var cleanSeries = InterpolateNaN(series);

        // Initialize
        for (int i = 0; i < n; i++)
            seasonal[i] = 0.0;
        Array.Copy(cleanSeries, trend, n);

        // STL iterations
        for (int iter = 0; iter < robustIterations + 1; iter++)
        {
            // Step 1: Detrend
            var detrended = new double[n];
            for (int t = 0; t < n; t++)
                detrended[t] = cleanSeries[t] - trend[t];

            // Step 2: Cycle-subseries smoothing (compute seasonal)
            var newSeasonal = new double[n];
            for (int s = 0; s < period; s++)
            {
                // Extract subseries for this seasonal position
                var subseries = new List<(int Index, double Value)>();
                for (int t = s; t < n; t += period)
                    subseries.Add((t, detrended[t]));

                // Smooth the subseries (simple moving average)
                var smoothed = SmoothSubseries(subseries);
                foreach (var (idx, val) in smoothed)
                    newSeasonal[idx] = val;
            }

            // Mean-center seasonal component
            double seasonalMean = 0;
            for (int t = 0; t < n; t++)
                seasonalMean += newSeasonal[t];
            seasonalMean /= n;
            for (int t = 0; t < n; t++)
                newSeasonal[t] -= seasonalMean;

            seasonal = newSeasonal;

            // Step 3: Deseasonalize
            var deseasonalized = new double[n];
            for (int t = 0; t < n; t++)
                deseasonalized[t] = cleanSeries[t] - seasonal[t];

            // Step 4: Trend smoothing (LOESS-like moving average)
            trend = SmoothTrend(deseasonalized, Math.Max(3, n / 10));
        }

        // Compute residual
        for (int t = 0; t < n; t++)
        {
            if (double.IsNaN(series[t]))
            {
                seasonal[t] = double.NaN;
                trend[t] = double.NaN;
                residual[t] = double.NaN;
            }
            else
            {
                residual[t] = series[t] - seasonal[t] - trend[t];
            }
        }

        return (seasonal, trend, residual);
    }

    /// <summary>
    /// Smooths a subseries using weighted average.
    /// </summary>
    private static List<(int Index, double Value)> SmoothSubseries(List<(int Index, double Value)> subseries)
    {
        int m = subseries.Count;
        var result = new List<(int Index, double Value)>();

        for (int i = 0; i < m; i++)
        {
            // Simple 3-point weighted average
            double sum = 0, weight = 0;
            for (int j = Math.Max(0, i - 1); j <= Math.Min(m - 1, i + 1); j++)
            {
                double w = 1.0 - Math.Abs(j - i) * 0.5;
                sum += subseries[j].Value * w;
                weight += w;
            }
            result.Add((subseries[i].Index, sum / weight));
        }

        return result;
    }

    /// <summary>
    /// Smooths trend using moving average.
    /// </summary>
    private static double[] SmoothTrend(double[] series, int windowSize)
    {
        int n = series.Length;
        var result = new double[n];
        int halfWindow = windowSize / 2;

        for (int t = 0; t < n; t++)
        {
            double sum = 0;
            int count = 0;
            for (int i = Math.Max(0, t - halfWindow); i <= Math.Min(n - 1, t + halfWindow); i++)
            {
                sum += series[i];
                count++;
            }
            result[t] = sum / count;
        }

        return result;
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Extracts a single feature series from tensor.
    /// </summary>
    private double[] ExtractSeries(Tensor<T> data, int feature, int timeSteps)
    {
        var series = new double[timeSteps];
        for (int t = 0; t < timeSteps; t++)
            series[t] = NumOps.ToDouble(GetValue(data, t, feature));
        return series;
    }

    /// <summary>
    /// Copies a double array to the output tensor at the specified column.
    /// </summary>
    private void CopyToOutput(Tensor<T> output, double[] series, int column)
    {
        for (int t = 0; t < series.Length; t++)
            output[t, column] = NumOps.FromDouble(series[t]);
    }

    /// <summary>
    /// Interpolates NaN values using linear interpolation.
    /// </summary>
    private static double[] InterpolateNaN(double[] series)
    {
        int n = series.Length;
        var result = (double[])series.Clone();

        // Find first valid value
        int firstValid = -1;
        for (int i = 0; i < n; i++)
        {
            if (!double.IsNaN(result[i]))
            {
                firstValid = i;
                break;
            }
        }

        if (firstValid < 0) return result; // All NaN

        // Forward fill start
        for (int i = 0; i < firstValid; i++)
            result[i] = result[firstValid];

        // Interpolate middle NaNs
        int lastValid = firstValid;
        for (int i = firstValid + 1; i < n; i++)
        {
            if (!double.IsNaN(result[i]))
            {
                // Interpolate between lastValid and i
                int gap = i - lastValid;
                for (int j = lastValid + 1; j < i; j++)
                {
                    double t = (double)(j - lastValid) / gap;
                    result[j] = result[lastValid] * (1 - t) + result[i] * t;
                }
                lastValid = i;
            }
        }

        // Backward fill end
        for (int i = lastValid + 1; i < n; i++)
            result[i] = result[lastValid];

        return result;
    }

    private int CountFeaturesPerInput()
    {
        int count = 0;
        if ((_enabledFeatures & DifferencingFeatures.FirstDifference) != 0) count++;
        if ((_enabledFeatures & DifferencingFeatures.SecondDifference) != 0) count++;
        if ((_enabledFeatures & DifferencingFeatures.SeasonalDifference) != 0) count++;
        if ((_enabledFeatures & DifferencingFeatures.PercentChange) != 0) count++;
        if ((_enabledFeatures & DifferencingFeatures.LogDifference) != 0) count++;
        if ((_enabledFeatures & DifferencingFeatures.LinearDetrend) != 0) count++;
        if ((_enabledFeatures & DifferencingFeatures.PolynomialDetrend) != 0) count++;
        if ((_enabledFeatures & DifferencingFeatures.HodrickPrescottFilter) != 0) count += 2; // trend + cycle
        if ((_enabledFeatures & DifferencingFeatures.StlDecomposition) != 0) count += 3; // seasonal + trend + residual
        return count;
    }

    private int CountOutputFeatures(int inputFeatures)
    {
        return inputFeatures * CountFeaturesPerInput();
    }

    #endregion

    #region Incremental Computation

    /// <summary>
    /// Computes differencing features incrementally from the circular buffer.
    /// Note: Only simple differencing operations are supported incrementally.
    /// Complex operations like detrending, HP filter, and STL require full series.
    /// </summary>
    protected override T[] ComputeIncrementalFeatures(IncrementalState<T> state, T[] newDataPoint)
    {
        var features = new T[OutputFeatureCount];
        int featureIdx = 0;

        for (int f = 0; f < InputFeatureCount; f++)
        {
            double currentValue = NumOps.ToDouble(newDataPoint[f]);
            int bufferLen = state.RollingBuffer[f].Length;
            // state.BufferPosition is where current value was just written
            // Previous value (t-1) is at BufferPosition - 1
            int currentPos = state.BufferPosition;

            // First difference: y[t] - y[t-1]
            if ((_enabledFeatures & DifferencingFeatures.FirstDifference) != 0)
            {
                int prev1Pos = (currentPos - 1 + bufferLen) % bufferLen;
                double prev1Value = NumOps.ToDouble(state.RollingBuffer[f][prev1Pos]);
                features[featureIdx++] = NumOps.FromDouble(currentValue - prev1Value);
            }

            // Second difference: (y[t] - y[t-1]) - (y[t-1] - y[t-2])
            if ((_enabledFeatures & DifferencingFeatures.SecondDifference) != 0)
            {
                int prev1Pos = (currentPos - 1 + bufferLen) % bufferLen;
                int prev2Pos = (currentPos - 2 + bufferLen) % bufferLen;
                double prev1Value = NumOps.ToDouble(state.RollingBuffer[f][prev1Pos]);
                double prev2Value = NumOps.ToDouble(state.RollingBuffer[f][prev2Pos]);
                double diff1 = currentValue - prev1Value;
                double diff0 = prev1Value - prev2Value;
                features[featureIdx++] = NumOps.FromDouble(diff1 - diff0);
            }

            // Seasonal difference: y[t] - y[t-period]
            if ((_enabledFeatures & DifferencingFeatures.SeasonalDifference) != 0)
            {
                if (_seasonalPeriod < bufferLen)
                {
                    int seasonPos = (currentPos - _seasonalPeriod + bufferLen) % bufferLen;
                    double seasonValue = NumOps.ToDouble(state.RollingBuffer[f][seasonPos]);
                    features[featureIdx++] = NumOps.FromDouble(currentValue - seasonValue);
                }
                else
                {
                    features[featureIdx++] = GetNaN();
                }
            }

            // Percent change: (y[t] - y[t-1]) / y[t-1]
            if ((_enabledFeatures & DifferencingFeatures.PercentChange) != 0)
            {
                int prev1Pos = (currentPos - 1 + bufferLen) % bufferLen;
                double prev1Value = NumOps.ToDouble(state.RollingBuffer[f][prev1Pos]);
                double pct = prev1Value != 0 ? (currentValue - prev1Value) / prev1Value : double.NaN;
                features[featureIdx++] = NumOps.FromDouble(pct);
            }

            // Log difference: log(y[t] / y[t-1])
            if ((_enabledFeatures & DifferencingFeatures.LogDifference) != 0)
            {
                int prev1Pos = (currentPos - 1 + bufferLen) % bufferLen;
                double prev1Value = NumOps.ToDouble(state.RollingBuffer[f][prev1Pos]);
                double logDiff = (currentValue > 0 && prev1Value > 0) ? Math.Log(currentValue / prev1Value) : double.NaN;
                features[featureIdx++] = NumOps.FromDouble(logDiff);
            }

            // Complex features require full series - return NaN for incremental
            if ((_enabledFeatures & DifferencingFeatures.LinearDetrend) != 0)
                features[featureIdx++] = GetNaN();
            if ((_enabledFeatures & DifferencingFeatures.PolynomialDetrend) != 0)
                features[featureIdx++] = GetNaN();
            if ((_enabledFeatures & DifferencingFeatures.HodrickPrescottFilter) != 0)
            {
                features[featureIdx++] = GetNaN(); // trend
                features[featureIdx++] = GetNaN(); // cycle
            }
            if ((_enabledFeatures & DifferencingFeatures.StlDecomposition) != 0)
            {
                features[featureIdx++] = GetNaN(); // seasonal
                features[featureIdx++] = GetNaN(); // trend
                features[featureIdx++] = GetNaN(); // residual
            }
        }

        return features;
    }

    #endregion

    #region Serialization

    /// <summary>
    /// Exports transformer-specific parameters for serialization.
    /// </summary>
    protected override Dictionary<string, object> ExportParameters()
    {
        var parameters = new Dictionary<string, object>
        {
            ["EnabledFeatures"] = (int)_enabledFeatures,
            ["DifferencingOrder"] = _differencingOrder,
            ["SeasonalPeriod"] = _seasonalPeriod,
            ["PolynomialDegree"] = _polynomialDegree,
            ["HpLambda"] = _hpLambda,
            ["StlPeriod"] = _stlPeriod,
            ["StlIterations"] = _stlIterations
        };

        // Include fitted trend coefficients if present
        if (_trendCoefficients != null)
        {
            parameters["TrendCoefficients"] = _trendCoefficients;
        }

        return parameters;
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

        // Restore trend coefficients if present
        if (parameters.TryGetValue("TrendCoefficients", out var coeffsObj) && coeffsObj is double[][] savedCoeffs)
        {
            _trendCoefficients = savedCoeffs;
        }
    }

    #endregion

    #region Feature Naming

    /// <inheritdoc />
    protected override string[] GenerateFeatureNames()
    {
        var names = new List<string>();
        var inputNames = GetInputFeatureNames();
        var sep = GetSeparator();

        foreach (var inputName in inputNames)
        {
            if ((_enabledFeatures & DifferencingFeatures.FirstDifference) != 0)
                names.Add($"{inputName}{sep}diff1");
            if ((_enabledFeatures & DifferencingFeatures.SecondDifference) != 0)
                names.Add($"{inputName}{sep}diff2");
            if ((_enabledFeatures & DifferencingFeatures.SeasonalDifference) != 0)
                names.Add($"{inputName}{sep}seasonal{sep}diff{sep}{_seasonalPeriod}");
            if ((_enabledFeatures & DifferencingFeatures.PercentChange) != 0)
                names.Add($"{inputName}{sep}pct{sep}change");
            if ((_enabledFeatures & DifferencingFeatures.LogDifference) != 0)
                names.Add($"{inputName}{sep}log{sep}diff");
            if ((_enabledFeatures & DifferencingFeatures.LinearDetrend) != 0)
                names.Add($"{inputName}{sep}detrend{sep}linear");
            if ((_enabledFeatures & DifferencingFeatures.PolynomialDetrend) != 0)
                names.Add($"{inputName}{sep}detrend{sep}poly{_polynomialDegree}");
            if ((_enabledFeatures & DifferencingFeatures.HodrickPrescottFilter) != 0)
            {
                names.Add($"{inputName}{sep}hp{sep}trend");
                names.Add($"{inputName}{sep}hp{sep}cycle");
            }
            if ((_enabledFeatures & DifferencingFeatures.StlDecomposition) != 0)
            {
                names.Add($"{inputName}{sep}stl{sep}seasonal");
                names.Add($"{inputName}{sep}stl{sep}trend");
                names.Add($"{inputName}{sep}stl{sep}residual");
            }
        }

        return [.. names];
    }

    /// <inheritdoc />
    protected override string[] GetOperationNames()
    {
        return ["differencing"];
    }

    #endregion
}
