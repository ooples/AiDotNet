using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.TimeSeries;

/// <summary>
/// Detects anomalies in time series using ARIMA model residuals.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> ARIMA (AutoRegressive Integrated Moving Average) is a classic
/// time series model. It predicts future values based on past values and errors. Points
/// where the prediction error is large are flagged as anomalies.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Fit an ARIMA(p,d,q) model to the time series
/// 2. Compute prediction residuals
/// 3. Points with large residuals (standardized) are anomalies
/// </para>
/// <para>
/// <b>When to use:</b>
/// - Stationary time series (or made stationary via differencing)
/// - Detecting point anomalies that don't fit the temporal pattern
/// - Well-understood temporal dependencies
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - p (AR order): 2
/// - d (differencing): 1
/// - q (MA order): 2
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Box, G.E.P., Jenkins, G.M. (1970). "Time Series Analysis: Forecasting and Control."
/// </para>
/// </remarks>
public class ARIMADetector<T> : AnomalyDetectorBase<T>
{
    private readonly int _p;
    private readonly int _d;
    private readonly int _q;
    private double[]? _arCoeffs;
    private double[]? _maCoeffs;
    private double _mean;
    private double _residualStd;
    private double[]? _lastValues;

    /// <summary>
    /// Gets the AR order (p).
    /// </summary>
    public int P => _p;

    /// <summary>
    /// Gets the differencing order (d).
    /// </summary>
    public int D => _d;

    /// <summary>
    /// Gets the MA order (q).
    /// </summary>
    public int Q => _q;

    /// <summary>
    /// Creates a new ARIMA anomaly detector.
    /// </summary>
    /// <param name="p">AR (autoregressive) order. Default is 2.</param>
    /// <param name="d">Differencing order. Default is 1.</param>
    /// <param name="q">MA (moving average) order. Default is 2.</param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public ARIMADetector(int p = 2, int d = 1, int q = 2,
        double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (p < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(p),
                "P must be non-negative. Recommended is 2.");
        }

        if (d < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(d),
                "D must be non-negative. Recommended is 1.");
        }

        if (q < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(q),
                "Q must be non-negative. Recommended is 2.");
        }

        _p = p;
        _d = d;
        _q = q;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        if (X.Columns != 1)
        {
            throw new ArgumentException(
                "ARIMA expects univariate time series (1 column).",
                nameof(X));
        }

        int n = X.Rows;

        // Extract values
        var values = new double[n];
        for (int i = 0; i < n; i++)
        {
            values[i] = NumOps.ToDouble(X[i, 0]);
        }

        // Apply differencing
        var diffValues = ApplyDifferencing(values, _d);

        // Store last values for prediction (undifferencing)
        _lastValues = new double[Math.Max(_p, _d) + 1];
        for (int i = 0; i < _lastValues.Length && i < n; i++)
        {
            _lastValues[i] = values[n - 1 - i];
        }

        // Fit AR coefficients using Yule-Walker equations (simplified)
        _mean = diffValues.Average();
        var centered = diffValues.Select(v => v - _mean).ToArray();

        _arCoeffs = FitARCoefficients(centered, _p);
        _maCoeffs = FitMACoefficients(centered, _arCoeffs, _q);

        // Compute residuals
        var residuals = ComputeResiduals(centered);
        _residualStd = Math.Sqrt(residuals.Average(r => r * r));
        if (_residualStd < 1e-10) _residualStd = 1;

        // Calculate scores for training data to set threshold
        var trainingScores = ScoreAnomaliesInternal(X);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
    }

    private double[] ApplyDifferencing(double[] values, int d)
    {
        var result = values;
        for (int i = 0; i < d; i++)
        {
            var diffed = new double[result.Length - 1];
            for (int j = 0; j < diffed.Length; j++)
            {
                diffed[j] = result[j + 1] - result[j];
            }
            result = diffed;
        }
        return result;
    }

    private double[] FitARCoefficients(double[] values, int p)
    {
        if (p == 0) return Array.Empty<double>();

        int n = values.Length;
        var coeffs = new double[p];

        // Compute autocorrelation
        var r = new double[p + 1];
        for (int k = 0; k <= p; k++)
        {
            double sum = 0;
            for (int t = k; t < n; t++)
            {
                sum += values[t] * values[t - k];
            }
            r[k] = sum / (n - k);
        }

        // Solve Yule-Walker using Levinson-Durbin algorithm
        var a = new double[p];
        var aNew = new double[p];

        a[0] = r[1] / r[0];
        double e = r[0] * (1 - a[0] * a[0]);

        for (int k = 1; k < p; k++)
        {
            double lambda = r[k + 1];
            for (int j = 0; j < k; j++)
            {
                lambda -= a[j] * r[k - j];
            }
            lambda /= e;

            aNew[k] = lambda;
            for (int j = 0; j < k; j++)
            {
                aNew[j] = a[j] - lambda * a[k - 1 - j];
            }

            Array.Copy(aNew, a, k + 1);
            e *= (1 - lambda * lambda);
            if (e <= 0) break;
        }

        Array.Copy(a, coeffs, p);
        return coeffs;
    }

    private double[] FitMACoefficients(double[] values, double[] arCoeffs, int q)
    {
        if (q == 0) return Array.Empty<double>();

        // Compute AR residuals
        int p = arCoeffs.Length;
        int n = values.Length;
        var residuals = new double[n];

        for (int t = p; t < n; t++)
        {
            double pred = 0;
            for (int j = 0; j < p; j++)
            {
                pred += arCoeffs[j] * values[t - 1 - j];
            }
            residuals[t] = values[t] - pred;
        }

        // Fit MA coefficients from residual autocorrelation (simplified)
        var maCoeffs = new double[q];
        var rr = new double[q + 1];

        for (int k = 0; k <= q; k++)
        {
            double sum = 0;
            int count = 0;
            for (int t = p + k; t < n; t++)
            {
                sum += residuals[t] * residuals[t - k];
                count++;
            }
            rr[k] = count > 0 ? sum / count : 0;
        }

        if (rr[0] > 1e-10)
        {
            for (int k = 0; k < q; k++)
            {
                maCoeffs[k] = rr[k + 1] / rr[0];
            }
        }

        return maCoeffs;
    }

    private double[] ComputeResiduals(double[] values)
    {
        int n = values.Length;
        int start = Math.Max(_p, _q);
        var residuals = new double[n];
        var errors = new double[n];

        var arCoeffs = _arCoeffs;
        var maCoeffs = _maCoeffs;
        if (arCoeffs == null || maCoeffs == null)
        {
            throw new InvalidOperationException("Model not properly fitted.");
        }

        for (int t = start; t < n; t++)
        {
            // AR prediction
            double pred = 0;
            for (int j = 0; j < arCoeffs.Length; j++)
            {
                pred += arCoeffs[j] * values[t - 1 - j];
            }

            // MA correction
            for (int j = 0; j < maCoeffs.Length && t - 1 - j >= 0; j++)
            {
                pred += maCoeffs[j] * errors[t - 1 - j];
            }

            errors[t] = values[t] - pred;
            residuals[t] = errors[t];
        }

        return residuals;
    }

    /// <inheritdoc/>
    public override Vector<T> ScoreAnomalies(Matrix<T> X)
    {
        EnsureFitted();
        return ScoreAnomaliesInternal(X);
    }

    private Vector<T> ScoreAnomaliesInternal(Matrix<T> X)
    {
        ValidateInput(X);

        if (X.Columns != 1)
        {
            throw new ArgumentException(
                "ARIMA expects univariate time series (1 column).",
                nameof(X));
        }

        int n = X.Rows;
        var values = new double[n];
        for (int i = 0; i < n; i++)
        {
            values[i] = NumOps.ToDouble(X[i, 0]);
        }

        // Apply differencing
        var diffValues = ApplyDifferencing(values, _d);
        var centered = diffValues.Select(v => v - _mean).ToArray();

        // Compute residuals
        var residuals = ComputeResiduals(centered);

        // Score based on standardized residuals
        var scores = new Vector<T>(n);
        int start = Math.Max(_p, _q) + _d;

        for (int i = 0; i < n; i++)
        {
            double score;
            if (i < start)
            {
                // Not enough history - use moderate score
                score = 0.5;
            }
            else
            {
                int residualIdx = i - _d;
                if (residualIdx >= 0 && residualIdx < residuals.Length)
                {
                    // Standardized absolute residual
                    score = Math.Abs(residuals[residualIdx]) / _residualStd;
                }
                else
                {
                    score = 0.5;
                }
            }
            scores[i] = NumOps.FromDouble(score);
        }

        return scores;
    }
}
