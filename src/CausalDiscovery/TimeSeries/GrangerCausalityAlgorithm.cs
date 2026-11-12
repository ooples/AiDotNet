using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.TimeSeries;

/// <summary>
/// Granger Causality — time series causal discovery via predictive improvement.
/// </summary>
/// <remarks>
/// <para>
/// Granger causality tests whether the past values of one variable X improve the prediction
/// of another variable Y beyond what Y's own past values provide. If so, X "Granger-causes" Y.
/// </para>
/// <para>
/// <b>Test procedure for each pair (i → j):</b>
/// <list type="number">
/// <item>Fit a restricted model: Y_t = f(Y_{t-1}, ..., Y_{t-L}) — autoregressive on Y only</item>
/// <item>Fit an unrestricted model: Y_t = f(Y_{t-1}, ..., Y_{t-L}, X_{t-1}, ..., X_{t-L})</item>
/// <item>Compare using F-test: F = ((RSS_r - RSS_u) / L) / (RSS_u / (n - 2L - 1))</item>
/// <item>If F is significant (p &lt; alpha), X Granger-causes Y</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> Imagine predicting tomorrow's temperature. If knowing yesterday's
/// humidity helps predict temperature better than just knowing past temperatures alone,
/// then humidity "Granger-causes" temperature. This doesn't prove true causation but
/// indicates a useful predictive relationship.
/// </para>
/// <para>
/// Reference: Granger (1969), "Investigating Causal Relations by Econometric Models
/// and Cross-spectral Methods", Econometrica.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.Causal)]
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.CausalModel)]
[ModelCategory(ModelCategory.Statistical)]
[ModelCategory(ModelCategory.Linear)]
[ModelTask(ModelTask.CausalInference)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
[ResearchPaper("Investigating Causal Relations by Econometric Models and Cross-spectral Methods", "https://doi.org/10.2307/1912791", Year = 1969, Authors = "Clive W. J. Granger")]
public class GrangerCausalityAlgorithm<T> : TimeSeriesCausalBase<T>
{
    private readonly double _significanceLevel = 0.05;

    /// <inheritdoc/>
    public override string Name => "Granger Causality";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    /// <summary>
    /// Initializes Granger Causality with optional configuration.
    /// </summary>
    public GrangerCausalityAlgorithm(CausalDiscoveryOptions? options = null)
    {
        ApplyTimeSeriesOptions(options);
        if (options?.SignificanceLevel.HasValue == true)
        {
            if (options.SignificanceLevel.Value <= 0 || options.SignificanceLevel.Value >= 1)
                throw new ArgumentOutOfRangeException(nameof(options), "SignificanceLevel must be in (0, 1).");
            _significanceLevel = options.SignificanceLevel.Value;
        }
        if (MaxLag <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "MaxLag must be > 0.");
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;

        int effectiveN = n - MaxLag;
        if (effectiveN <= 2 * MaxLag + 1)
        {
            // Not enough data for Granger test — return empty graph
            return new Matrix<T>(d, d);
        }

        var weights = new Dictionary<int, double>();

        for (int target = 0; target < d; target++)
        {
            // Restricted model: AR on target only (p parameters = MaxLag)
            double rssRestricted = ComputeARModelRSS(data, target, MaxLag, effectiveN);

            for (int cause = 0; cause < d; cause++)
            {
                if (cause == target) continue;

                // Unrestricted model: AR on target + lags of cause (p parameters = 2 * MaxLag)
                double rssUnrestricted = ComputeGrangerRSS(data, target, cause, MaxLag, effectiveN);

                // F-test: F = ((RSS_r - RSS_u) / q) / (RSS_u / (n - k))
                // where q = number of added regressors (MaxLag)
                // k = total parameters in unrestricted model (2 * MaxLag + 1 for intercept)
                int q = MaxLag; // number of restrictions
                int k = 2 * MaxLag; // parameters in unrestricted model (excluding intercept)
                int dfResidual = effectiveN - k - 1; // degrees of freedom for residuals

                if (dfResidual > 0 && rssRestricted >= rssUnrestricted)
                {
                    if (rssUnrestricted > 1e-10)
                    {
                        // Standard F-test path
                        double fStat = ((rssRestricted - rssUnrestricted) / q) /
                                       (rssUnrestricted / dfResidual);

                        if (fStat > 0)
                        {
                            double pValue = FDistributionSurvivalFunction(fStat, q, dfResidual);

                            if (pValue <= _significanceLevel)
                            {
                                double rSquaredImprovement = (rssRestricted - rssUnrestricted) / rssRestricted;
                                weights[cause * d + target] = rSquaredImprovement;
                            }
                        }
                    }
                    else if (rssRestricted > 1e-10)
                    {
                        // Near-perfect unrestricted fit: the cause's lags drive RSS to near zero
                        // while the restricted model still has residuals. This indicates the cause
                        // variable is strongly predictive.
                        double rSquaredImprovement = (rssRestricted - rssUnrestricted) / rssRestricted;
                        if (rSquaredImprovement > 0.01)
                        {
                            weights[cause * d + target] = Math.Min(rSquaredImprovement, 1.0);
                        }
                    }
                    else
                    {
                        // Both models achieve near-perfect fit (deterministic data).
                        // Fall back to lagged cross-correlation as a causality proxy:
                        // if the cause's past values correlate strongly with the target's current value,
                        // record an edge with the correlation magnitude as weight.
                        double crossCorr = ComputeLaggedCrossCorrelation(data, cause, target, MaxLag);
                        if (Math.Abs(crossCorr) > 0.5)
                        {
                            weights[cause * d + target] = Math.Abs(crossCorr);
                        }
                    }
                }
            }
        }

        return CausalTensorHelpers.FromStatistics<T>(d, d, weights);
    }

    private double ComputeARModelRSS(Matrix<T> data, int target, int lag, int effectiveN)
    {
        // Design: [effectiveN x (lag + 1)] — lag columns (column l = target[t + lag - l - 1]) + intercept
        var targetColumn = Engine.GetColumn(data, target).AsTensor().Reshape(new[] { data.Rows, 1 });
        var y = Engine.TensorSlice(targetColumn, new[] { lag, 0 }, new[] { effectiveN, 1 });
        var columns = new Tensor<T>[lag + 1];
        for (int l = 0; l < lag; l++)
            columns[l] = Engine.TensorSlice(targetColumn, new[] { lag - l - 1, 0 }, new[] { effectiveN, 1 });
        columns[lag] = Engine.TensorOnesLike(y); // intercept
        var design = Engine.TensorConcatenate(columns, 1);

        return ComputeRSS(design.ToMatrix(), y.Reshape(new[] { effectiveN }).ToVector(), effectiveN, lag + 1);
    }

    private double ComputeGrangerRSS(Matrix<T> data, int target, int cause, int lag, int effectiveN)
    {
        // Design: [effectiveN x (2*lag + 1)] — target lags + cause lags + intercept
        var targetColumn = Engine.GetColumn(data, target).AsTensor().Reshape(new[] { data.Rows, 1 });
        var causeColumn = Engine.GetColumn(data, cause).AsTensor().Reshape(new[] { data.Rows, 1 });
        var y = Engine.TensorSlice(targetColumn, new[] { lag, 0 }, new[] { effectiveN, 1 });
        var columns = new Tensor<T>[2 * lag + 1];
        for (int l = 0; l < lag; l++)
        {
            columns[l] = Engine.TensorSlice(targetColumn, new[] { lag - l - 1, 0 }, new[] { effectiveN, 1 });
            columns[lag + l] = Engine.TensorSlice(causeColumn, new[] { lag - l - 1, 0 }, new[] { effectiveN, 1 });
        }
        columns[2 * lag] = Engine.TensorOnesLike(y); // intercept
        var design = Engine.TensorConcatenate(columns, 1);

        return ComputeRSS(design.ToMatrix(), y.Reshape(new[] { effectiveN }).ToVector(), effectiveN, 2 * lag + 1);
    }

    /// <summary>
    /// Computes 1 - F_CDF(x; d1, d2) using the regularized incomplete beta function.
    /// P(F > x) = I_{d2/(d2 + d1*x)}(d2/2, d1/2)
    /// </summary>
    private static double FDistributionSurvivalFunction(double x, int d1, int d2)
    {
        if (x <= 0) return 1.0;
        if (d1 <= 0 || d2 <= 0) return 1.0;

        double a = d1 / 2.0;
        double b = d2 / 2.0;
        double t = d2 / (d2 + d1 * x);

        // P(F > x) = I_t(b, a) = regularized incomplete beta with swapped parameters
        return RegularizedIncompleteBeta(t, b, a);
    }

    /// <summary>
    /// Computes the regularized incomplete beta function I_x(a, b) using a continued fraction expansion.
    /// Uses the Lentz algorithm for numerical stability.
    /// Reference: Press et al., "Numerical Recipes", Chapter 6.4.
    /// </summary>
    private static double RegularizedIncompleteBeta(double x, double a, double b)
    {
        if (x <= 0) return 0;
        if (x >= 1) return 1;

        // Use the identity I_x(a,b) = 1 - I_{1-x}(b,a) when x > (a+1)/(a+b+2) for convergence
        if (x > (a + 1) / (a + b + 2))
            return 1.0 - RegularizedIncompleteBeta(1.0 - x, b, a);

        // Compute log(Beta(a,b)) = logGamma(a) + logGamma(b) - logGamma(a+b)
        double logBeta = LogGamma(a) + LogGamma(b) - LogGamma(a + b);

        // Front factor: x^a * (1-x)^b / (a * Beta(a,b))
        double front = Math.Exp(a * Math.Log(x) + b * Math.Log(1.0 - x) - logBeta) / a;

        // Continued fraction expansion (Lentz's method)
        double cf = ContinuedFractionBeta(a, b, x);

        return front * cf;
    }

    /// <summary>
    /// Evaluates the continued fraction for the incomplete beta function.
    /// </summary>
    private static double ContinuedFractionBeta(double a, double b, double x)
    {
        const int maxIterations = 200;
        const double epsilon = 1e-14;
        const double tiny = 1e-30;

        double qab = a + b;
        double qap = a + 1.0;
        double qam = a - 1.0;

        double c = 1.0;
        double d = 1.0 - qab * x / qap;
        if (Math.Abs(d) < tiny) d = tiny;
        d = 1.0 / d;
        double h = d;

        for (int m = 1; m <= maxIterations; m++)
        {
            int m2 = 2 * m;

            // Even step
            double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
            d = 1.0 + aa * d;
            if (Math.Abs(d) < tiny) d = tiny;
            c = 1.0 + aa / c;
            if (Math.Abs(c) < tiny) c = tiny;
            d = 1.0 / d;
            h *= d * c;

            // Odd step
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
            d = 1.0 + aa * d;
            if (Math.Abs(d) < tiny) d = tiny;
            c = 1.0 + aa / c;
            if (Math.Abs(c) < tiny) c = tiny;
            d = 1.0 / d;
            double del = d * c;
            h *= del;

            if (Math.Abs(del - 1.0) < epsilon)
                return h;
        }

        return h; // did not fully converge, return best estimate
    }

    /// <summary>
    /// Computes lagged cross-correlation between cause (at lag) and target (at present).
    /// Returns the maximum absolute cross-correlation across all lags from 1 to maxLag.
    /// Used as a fallback when the F-test cannot be computed (e.g., near-zero RSS).
    /// </summary>
    private double ComputeLaggedCrossCorrelation(Matrix<T> data, int cause, int target, int maxLag)
    {
        int n = data.Rows;
        var causeColumn = Engine.GetColumn(data, cause).AsTensor();
        var targetColumn = Engine.GetColumn(data, target).AsTensor();
        T tiny = NumOps.FromDouble(1e-15);

        // |corr| per usable lag; a lag whose denominator is not above 1e-15 (or whose correlation is
        // NaN) contributes 0, and the result is the maximum, starting from 0.
        var correlations = new List<Tensor<T>> { new Tensor<T>(new[] { 1 }) };
        for (int lag = 1; lag <= maxLag && lag < n; lag++)
        {
            int effectiveN = n - lag;
            if (effectiveN < 3) continue;

            // Compute correlation between cause[0..effectiveN-1] and target[lag..n-1]
            var c = Engine.TensorSlice(causeColumn, new[] { 0 }, new[] { effectiveN });
            var t = Engine.TensorSlice(targetColumn, new[] { lag }, new[] { effectiveN });
            var dx = Engine.TensorSubtractScalar(c, Engine.TensorMean(c));
            var dy = Engine.TensorSubtractScalar(t, Engine.TensorMean(t));
            var sxy = Engine.ReduceSum(Engine.TensorMultiply(dx, dy), new[] { 0 }, true);
            var denom = Engine.TensorSqrt(Engine.TensorMultiply(
                Engine.ReduceSum(Engine.TensorSquare(dx), new[] { 0 }, true),
                Engine.ReduceSum(Engine.TensorSquare(dy), new[] { 0 }, true)));
            var usable = Engine.TensorGreaterThan(denom, tiny);
            var corr = Engine.TensorAbs(Engine.TensorDivide(sxy,
                Engine.TensorWhere(usable, denom, Engine.TensorOnesLike(denom))));
            correlations.Add(Engine.TensorWhere(Engine.TensorMultiply(usable, Engine.TensorGreaterThan(corr, NumOps.Zero)),
                corr, Engine.TensorZerosLike(corr)));
        }

        return NumOps.ToDouble(Engine.TensorMaxValue(Engine.TensorConcatenate(correlations.ToArray(), 0)));
    }

    /// <summary>
    /// Computes log(Gamma(x)) using the Lanczos approximation.
    /// Accurate to about 15 decimal places.
    /// </summary>
    private static double LogGamma(double x)
    {
        if (x <= 0) return double.PositiveInfinity;

        // Lanczos coefficients (g=7)
        double[] coef =
        [
            0.99999999999980993,
            676.5203681218851,
            -1259.1392167224028,
            771.32342877765313,
            -176.61502916214059,
            12.507343278686905,
            -0.13857109526572012,
            9.9843695780195716e-6,
            1.5056327351493116e-7
        ];

        if (x < 0.5)
        {
            // Reflection formula: Gamma(x) * Gamma(1-x) = pi / sin(pi*x)
            return Math.Log(Math.PI / Math.Sin(Math.PI * x)) - LogGamma(1.0 - x);
        }

        x -= 1.0;
        double ag = coef[0];
        double t = x + 7.5;

        for (int i = 1; i < coef.Length; i++)
            ag += coef[i] / (x + i);

        return 0.5 * Math.Log(2.0 * Math.PI) + (x + 0.5) * Math.Log(t) - t + Math.Log(ag);
    }
}
