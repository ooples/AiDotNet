using AiDotNet.Distributions;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;

namespace AiDotNet.Regression;

/// <summary>
/// Zero-Inflated regression for count data with excess zeros.
/// </summary>
/// <remarks>
/// <para>
/// Zero-Inflated models handle count data where there are more zeros than a standard
/// count distribution would predict. They model the data as a mixture: with probability π,
/// the observation is a "structural zero," and with probability (1-π), it follows a count
/// distribution (Poisson or Negative Binomial).
/// </para>
/// <para>
/// <b>For Beginners:</b> Imagine counting how many times customers visit a store each month:
/// - Some people NEVER visit (structural zeros) - they live far away or shop elsewhere
/// - Some people visit sometimes but happened to visit 0 times this month (sampling zeros)
///
/// Standard Poisson regression treats all zeros the same, but Zero-Inflated models
/// recognize these two types of zeros:
///
/// 1. The "zero model" predicts WHO are structural zeros (π)
/// 2. The "count model" predicts HOW MANY for non-structural-zero people (λ)
///
/// Example interpretation:
/// - "30% of potential customers are 'never visitors' (π = 0.3)"
/// - "Among potential visitors, the average visit rate is 2.5 times/month (λ = 2.5)"
///
/// This gives better predictions and allows you to understand both processes.
/// </para>
/// <para>
/// Reference: Lambert, D. (1992). "Zero-Inflated Poisson Regression, with an Application
/// to Defects in Manufacturing". Technometrics, 34(1), 1-14.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class ZeroInflatedRegression<T> : AsyncDecisionTreeRegressionBase<T>
{
    /// <summary>
    /// Coefficients for the count model (λ).
    /// </summary>
    private Vector<T>? _countCoefficients;

    /// <summary>
    /// Intercept for the count model.
    /// </summary>
    private T _countIntercept;

    /// <summary>
    /// Coefficients for the zero-inflation model (π).
    /// </summary>
    private Vector<T>? _zeroCoefficients;

    /// <summary>
    /// Intercept for the zero-inflation model.
    /// </summary>
    private T _zeroIntercept;

    /// <summary>
    /// Dispersion parameter for Negative Binomial (if applicable).
    /// </summary>
    private T _dispersion;

    /// <summary>
    /// Number of features.
    /// </summary>
    private int _numFeatures;

    /// <summary>
    /// Configuration options.
    /// </summary>
    private readonly ZeroInflatedRegressionOptions _options;

    /// <inheritdoc/>
    public override int NumberOfTrees => 1;

    /// <summary>
    /// Gets the count model coefficients.
    /// </summary>
    public Vector<T>? CountCoefficients => _countCoefficients;

    /// <summary>
    /// Gets the zero-inflation model coefficients.
    /// </summary>
    public Vector<T>? ZeroCoefficients => _zeroCoefficients;

    /// <summary>
    /// Initializes a new instance of ZeroInflatedRegression.
    /// </summary>
    /// <param name="options">Configuration options.</param>
    /// <param name="regularization">Optional regularization.</param>
    public ZeroInflatedRegression(ZeroInflatedRegressionOptions? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(null, regularization)
    {
        _options = options ?? new ZeroInflatedRegressionOptions();
        _countIntercept = NumOps.Zero;
        _zeroIntercept = NumOps.Zero;
        _dispersion = NumOps.One;
        _numFeatures = 0;
    }

    /// <inheritdoc/>
    public override async Task TrainAsync(Matrix<T> x, Vector<T> y)
    {
        _numFeatures = x.Columns;
        int n = x.Rows;

        // Validate response values are non-negative integers
        T zero = NumOps.Zero;
        for (int i = 0; i < n; i++)
        {
            double yi = NumOps.ToDouble(y[i]);
            if (NumOps.LessThan(y[i], zero) || yi != Math.Floor(yi))
            {
                throw new ArgumentException($"Response must be non-negative integers. Found: {yi} at index {i}");
            }
        }

        // Initialize parameters
        InitializeParameters(y);

        T prevLogLik = NumOps.MinValue;
        T tolerance = NumOps.FromDouble(_options.Tolerance);

        // EM algorithm for optimization
        for (int iter = 0; iter < _options.MaxIterations; iter++)
        {
            // E-step: compute posterior probability of being a structural zero
            var (lambdas, pis) = ComputePredictions(x);
            var posteriorZero = ComputePosteriorZero(y, lambdas, pis);

            // M-step: update parameters
            // Update count model using weighted data (weight = 1 - posterior zero)
            UpdateCountModel(x, y, posteriorZero);

            // Update zero-inflation model
            if (_options.ModelZeroInflation)
            {
                UpdateZeroModel(x, posteriorZero);
            }

            // Update dispersion for Negative Binomial
            if (_options.DistributionFamily == ZeroInflatedDistributionFamily.NegativeBinomial)
            {
                UpdateDispersion(y, lambdas, posteriorZero);
            }

            // Check convergence
            (lambdas, pis) = ComputePredictions(x);
            T logLik = ComputeLogLikelihood(y, lambdas, pis);

            if (NumOps.LessThan(NumOps.Abs(NumOps.Subtract(logLik, prevLogLik)), tolerance))
            {
                break;
            }
            prevLogLik = logLik;
        }

        await CalculateFeatureImportancesAsync(x.Columns);
    }

    /// <inheritdoc/>
    public override async Task<Vector<T>> PredictAsync(Matrix<T> input)
    {
        var (lambdas, pis) = await Task.Run(() => ComputePredictions(input));
        var predictions = new Vector<T>(input.Rows);

        // Expected value = (1 - π) * λ
        for (int i = 0; i < input.Rows; i++)
        {
            predictions[i] = NumOps.Multiply(NumOps.Subtract(NumOps.One, pis[i]), lambdas[i]);
        }

        return predictions;
    }

    /// <summary>
    /// Predicts the probability of being a structural zero for each sample.
    /// </summary>
    /// <param name="input">Input feature matrix.</param>
    /// <returns>Vector of zero-inflation probabilities.</returns>
    public async Task<Vector<T>> PredictZeroProbabilityAsync(Matrix<T> input)
    {
        var (_, pis) = await Task.Run(() => ComputePredictions(input));
        return pis;
    }

    /// <summary>
    /// Predicts the expected count conditional on not being a structural zero.
    /// </summary>
    /// <param name="input">Input feature matrix.</param>
    /// <returns>Vector of conditional count expectations.</returns>
    public async Task<Vector<T>> PredictConditionalCountAsync(Matrix<T> input)
    {
        var (lambdas, _) = await Task.Run(() => ComputePredictions(input));
        return lambdas;
    }

    /// <summary>
    /// Predicts the probability mass function for each sample.
    /// </summary>
    /// <param name="input">Input feature matrix.</param>
    /// <param name="maxCount">Maximum count value to compute probability for.</param>
    /// <returns>Matrix where [i,k] is P(Y_i = k).</returns>
    public async Task<Matrix<T>> PredictPMFAsync(Matrix<T> input, int maxCount = 20)
    {
        var (lambdas, pis) = await Task.Run(() => ComputePredictions(input));
        var pmf = new Matrix<T>(input.Rows, maxCount + 1);

        for (int i = 0; i < input.Rows; i++)
        {
            T pi = pis[i];
            T oneMinusPi = NumOps.Subtract(NumOps.One, pi);
            double lambdaD = NumOps.ToDouble(lambdas[i]);

            for (int k = 0; k <= maxCount; k++)
            {
                // ComputeCountProbability is a numerical recipe — boundary conversion
                double pk = ComputeCountProbability(k, lambdaD);
                T pkT = NumOps.FromDouble(pk);

                if (k == 0)
                {
                    // P(Y=0) = π + (1-π) * P_count(0)
                    pmf[i, k] = NumOps.Add(pi, NumOps.Multiply(oneMinusPi, pkT));
                }
                else
                {
                    // P(Y=k) = (1-π) * P_count(k)
                    pmf[i, k] = NumOps.Multiply(oneMinusPi, pkT);
                }
            }
        }

        return pmf;
    }

    /// <summary>
    /// Initializes parameters from target values.
    /// </summary>
    private void InitializeParameters(Vector<T> y)
    {
        // Proportion of zeros
        int numZeros = 0;
        T sumNonZero = NumOps.Zero;
        int countNonZero = 0;

        for (int i = 0; i < y.Length; i++)
        {
            if (NumOps.Compare(y[i], NumOps.Zero) == 0)
            {
                numZeros++;
            }
            else
            {
                sumNonZero = NumOps.Add(sumNonZero, y[i]);
                countNonZero++;
            }
        }

        T pZero = NumOps.Divide(NumOps.FromDouble(numZeros), NumOps.FromDouble(y.Length));
        T meanNonZero = countNonZero > 0
            ? NumOps.Divide(sumNonZero, NumOps.FromDouble(countNonZero))
            : NumOps.One;

        // Initialize zero model intercept (logit scale)
        // Clamp pZero to [0.01, 0.99]
        T minP = NumOps.FromDouble(0.01);
        T maxP = NumOps.FromDouble(0.99);
        if (NumOps.LessThan(pZero, minP)) pZero = minP;
        if (NumOps.GreaterThan(pZero, maxP)) pZero = maxP;
        _zeroIntercept = NumOps.Log(NumOps.Divide(pZero, NumOps.Subtract(NumOps.One, pZero)));

        // Initialize count model intercept (log scale)
        T minMean = NumOps.FromDouble(0.1);
        if (NumOps.LessThan(meanNonZero, minMean)) meanNonZero = minMean;
        _countIntercept = NumOps.Log(meanNonZero);

        // Initialize coefficients
        _countCoefficients = new Vector<T>(_numFeatures);
        if (_options.ModelZeroInflation)
        {
            _zeroCoefficients = new Vector<T>(_numFeatures);
        }

        // Initialize dispersion
        _dispersion = NumOps.One;
    }

    /// <summary>
    /// Computes predictions for all samples.
    /// </summary>
    private (Vector<T> lambdas, Vector<T> pis) ComputePredictions(Matrix<T> x)
    {
        int n = x.Rows;
        var lambdas = new Vector<T>(n);
        var pis = new Vector<T>(n);
        T minLambda = NumOps.FromDouble(0.001);

        for (int i = 0; i < n; i++)
        {
            // Count model linear predictor
            T etaCount = _countIntercept;
            if (_countCoefficients != null)
            {
                for (int j = 0; j < _numFeatures; j++)
                {
                    etaCount = NumOps.Add(etaCount, NumOps.Multiply(_countCoefficients[j], x[i, j]));
                }
            }

            // Apply count link inverse (boundary: special math functions)
            T lambda;
            if (_options.CountLink == ZeroInflatedCountLink.Log)
            {
                lambda = NumOps.Exp(etaCount);
            }
            else if (_options.CountLink == ZeroInflatedCountLink.SquareRoot)
            {
                lambda = NumOps.Multiply(etaCount, etaCount);
            }
            else if (_options.CountLink == ZeroInflatedCountLink.Identity)
            {
                lambda = NumOps.LessThan(etaCount, minLambda) ? minLambda : etaCount;
            }
            else
            {
                lambda = NumOps.Exp(etaCount);
            }
            lambdas[i] = lambda;

            // Zero-inflation model linear predictor
            T etaZero = _zeroIntercept;
            if (_options.ModelZeroInflation && _zeroCoefficients != null)
            {
                for (int j = 0; j < _numFeatures; j++)
                {
                    etaZero = NumOps.Add(etaZero, NumOps.Multiply(_zeroCoefficients[j], x[i, j]));
                }
            }

            // Apply zero link inverse (boundary: special math functions for Probit/CLogLog)
            double etaZeroD = NumOps.ToDouble(etaZero);
            double pi = _options.ZeroLink switch
            {
                ZeroInflatedZeroLink.Logit => 1 / (1 + Math.Exp(-etaZeroD)),
                ZeroInflatedZeroLink.Probit => StandardNormalCdf(etaZeroD),
                ZeroInflatedZeroLink.CLogLog => 1 - Math.Exp(-Math.Exp(etaZeroD)),
                _ => 1 / (1 + Math.Exp(-etaZeroD))
            };
            pis[i] = NumOps.FromDouble(pi);
        }

        return (lambdas, pis);
    }

    /// <summary>
    /// Computes posterior probability of being a structural zero.
    /// </summary>
    private Vector<T> ComputePosteriorZero(Vector<T> y, Vector<T> lambdas, Vector<T> pis)
    {
        var posterior = new Vector<T>(y.Length);
        T half = NumOps.FromDouble(0.5);

        for (int i = 0; i < y.Length; i++)
        {
            if (NumOps.Compare(y[i], NumOps.Zero) == 0)
            {
                // P(Z=1 | Y=0) = π / (π + (1-π) * P(Y=0|Z=0))
                // ComputeCountProbability is a numerical recipe — boundary conversion
                double p0 = ComputeCountProbability(0, NumOps.ToDouble(lambdas[i]));
                T p0T = NumOps.FromDouble(p0);
                T pi = pis[i];
                T denom = NumOps.Add(pi, NumOps.Multiply(NumOps.Subtract(NumOps.One, pi), p0T));
                posterior[i] = NumOps.GreaterThan(denom, NumOps.Zero)
                    ? NumOps.Divide(pi, denom)
                    : half;
            }
            else
            {
                // If Y > 0, definitely not a structural zero
                posterior[i] = NumOps.Zero;
            }
        }

        return posterior;
    }

    /// <summary>
    /// Computes probability from the count distribution.
    /// </summary>
    private double ComputeCountProbability(int k, double lambda)
    {
        if (_options.DistributionFamily == ZeroInflatedDistributionFamily.Poisson)
        {
            // Poisson PMF
            if (lambda <= 0) return k == 0 ? 1.0 : 0.0;
            return Math.Exp(-lambda + k * Math.Log(lambda) - LogFactorial(k));
        }
        else
        {
            // Negative Binomial PMF
            double r = NumOps.ToDouble(_dispersion);
            double p = r / (r + lambda);
            return Math.Exp(LogGamma(k + r) - LogFactorial(k) - LogGamma(r) +
                           r * Math.Log(p) + k * Math.Log(1 - p));
        }
    }

    /// <summary>
    /// Updates the count model parameters.
    /// </summary>
    private void UpdateCountModel(Matrix<T> x, Vector<T> y, Vector<T> posteriorZero)
    {
        int n = x.Rows;
        int p = _numFeatures;
        T minWeight = NumOps.FromDouble(1e-10);

        // Weighted IRLS for Poisson/NB regression
        var weights = new Vector<T>(n);
        var z = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            T wi = NumOps.Subtract(NumOps.One, posteriorZero[i]);
            if (NumOps.LessThan(wi, minWeight)) wi = minWeight;

            T eta = _countIntercept;
            if (_countCoefficients != null)
            {
                for (int j = 0; j < p; j++)
                {
                    eta = NumOps.Add(eta, NumOps.Multiply(_countCoefficients[j], x[i, j]));
                }
            }

            T lambda = NumOps.Exp(eta);
            if (NumOps.LessThan(lambda, minWeight)) lambda = minWeight;

            // Working weight and response for log link
            weights[i] = NumOps.Multiply(wi, lambda);
            z[i] = NumOps.Add(eta, NumOps.Divide(NumOps.Subtract(y[i], lambda), lambda));
        }

        if (_countCoefficients is null)
        {
            throw new InvalidOperationException("Count coefficients not initialized.");
        }
        UpdateCoefficientsWLS(x, z, weights, ref _countCoefficients, ref _countIntercept);
    }

    /// <summary>
    /// Updates the zero-inflation model parameters.
    /// </summary>
    private void UpdateZeroModel(Matrix<T> x, Vector<T> posteriorZero)
    {
        int n = x.Rows;
        int p = _numFeatures;
        T minPi = NumOps.FromDouble(1e-10);
        T maxPi = NumOps.FromDouble(1 - 1e-10);

        // Weighted logistic regression for zero model
        var weights = new Vector<T>(n);
        var z = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            T pZero = posteriorZero[i];

            T eta = _zeroIntercept;
            if (_zeroCoefficients != null)
            {
                for (int j = 0; j < p; j++)
                {
                    eta = NumOps.Add(eta, NumOps.Multiply(_zeroCoefficients[j], x[i, j]));
                }
            }

            // Logistic function: 1 / (1 + exp(-eta))
            T pi = NumOps.Divide(NumOps.One, NumOps.Add(NumOps.One, NumOps.Exp(NumOps.Negate(eta))));
            if (NumOps.LessThan(pi, minPi)) pi = minPi;
            if (NumOps.GreaterThan(pi, maxPi)) pi = maxPi;

            // Working weight and response for logit link
            T piOneMinusPi = NumOps.Multiply(pi, NumOps.Subtract(NumOps.One, pi));
            weights[i] = piOneMinusPi;
            z[i] = NumOps.Add(eta, NumOps.Divide(NumOps.Subtract(pZero, pi), piOneMinusPi));
        }

        if (_zeroCoefficients != null)
        {
            UpdateCoefficientsWLS(x, z, weights, ref _zeroCoefficients, ref _zeroIntercept);
        }
    }

    /// <summary>
    /// Updates dispersion parameter for Negative Binomial.
    /// </summary>
    private void UpdateDispersion(Vector<T> y, Vector<T> lambdas, Vector<T> posteriorZero)
    {
        // Method of moments estimate
        T sumSqDev = NumOps.Zero;
        T sumLambda = NumOps.Zero;
        T totalWeight = NumOps.Zero;
        T minWeight = NumOps.FromDouble(1e-10);

        for (int i = 0; i < y.Length; i++)
        {
            T wi = NumOps.Subtract(NumOps.One, posteriorZero[i]);
            if (NumOps.LessThan(wi, minWeight)) continue;

            T diff = NumOps.Subtract(y[i], lambdas[i]);
            sumSqDev = NumOps.Add(sumSqDev, NumOps.Multiply(wi, NumOps.Multiply(diff, diff)));
            sumLambda = NumOps.Add(sumLambda, NumOps.Multiply(wi, lambdas[i]));
            totalWeight = NumOps.Add(totalWeight, wi);
        }

        if (NumOps.GreaterThan(totalWeight, NumOps.Zero) && NumOps.GreaterThan(sumLambda, NumOps.Zero))
        {
            T varianceEstimate = NumOps.Divide(sumSqDev, totalWeight);
            T meanEstimate = NumOps.Divide(sumLambda, totalWeight);

            // For NB: Var = μ + μ²/r, so r = μ² / (Var - μ)
            T excess = NumOps.Subtract(varianceEstimate, meanEstimate);
            T minExcess = NumOps.FromDouble(0.1);
            if (NumOps.GreaterThan(excess, minExcess))
            {
                T r = NumOps.Divide(NumOps.Multiply(meanEstimate, meanEstimate), excess);
                T minR = NumOps.FromDouble(0.1);
                T maxR = NumOps.FromDouble(1000);
                if (NumOps.LessThan(r, minR)) r = minR;
                if (NumOps.GreaterThan(r, maxR)) r = maxR;
                _dispersion = r;
            }
        }
    }

    /// <summary>
    /// Updates coefficients using weighted least squares.
    /// </summary>
    private void UpdateCoefficientsWLS(Matrix<T> x, Vector<T> z, Vector<T> weights,
        ref Vector<T> coefficients, ref T intercept)
    {
        int n = x.Rows;
        int p = _numFeatures;
        T minWeight = NumOps.FromDouble(1e-10);

        var xtwx = new Matrix<T>(p + 1, p + 1);
        var xtwz = new Vector<T>(p + 1);

        for (int i = 0; i < n; i++)
        {
            T w = weights[i];
            if (NumOps.LessThan(w, minWeight)) continue;

            xtwx[0, 0] = NumOps.Add(xtwx[0, 0], w);
            xtwz[0] = NumOps.Add(xtwz[0], NumOps.Multiply(w, z[i]));

            for (int j = 0; j < p; j++)
            {
                T wxij = NumOps.Multiply(w, x[i, j]);
                xtwx[0, j + 1] = NumOps.Add(xtwx[0, j + 1], wxij);
                xtwx[j + 1, 0] = NumOps.Add(xtwx[j + 1, 0], wxij);
                xtwz[j + 1] = NumOps.Add(xtwz[j + 1], NumOps.Multiply(wxij, z[i]));

                for (int k = 0; k <= j; k++)
                {
                    T val = NumOps.Add(xtwx[j + 1, k + 1], NumOps.Multiply(wxij, x[i, k]));
                    xtwx[j + 1, k + 1] = val;
                    if (k < j) xtwx[k + 1, j + 1] = val;
                }
            }
        }

        // Regularization
        if (_options.UseRegularization)
        {
            T lambda = NumOps.FromDouble(_options.RegularizationStrength);
            for (int j = 1; j <= p; j++)
            {
                xtwx[j, j] = NumOps.Add(xtwx[j, j], lambda);
            }
        }

        var solution = SolveSystem(xtwx, xtwz, p + 1);

        intercept = solution[0];
        for (int j = 0; j < p; j++)
        {
            coefficients[j] = solution[j + 1];
        }
    }

    /// <summary>
    /// Computes the log-likelihood.
    /// </summary>
    private T ComputeLogLikelihood(Vector<T> y, Vector<T> lambdas, Vector<T> pis)
    {
        T ll = NumOps.Zero;
        T tiny = NumOps.FromDouble(1e-300);

        for (int i = 0; i < y.Length; i++)
        {
            T pi = pis[i];
            T oneMinusPi = NumOps.Subtract(NumOps.One, pi);
            // ComputeCountProbability is a numerical recipe — boundary conversion
            double lambdaD = NumOps.ToDouble(lambdas[i]);

            if (NumOps.Compare(y[i], NumOps.Zero) == 0)
            {
                double p0 = ComputeCountProbability(0, lambdaD);
                T likelihood = NumOps.Add(pi, NumOps.Add(NumOps.Multiply(oneMinusPi, NumOps.FromDouble(p0)), tiny));
                ll = NumOps.Add(ll, NumOps.Log(likelihood));
            }
            else
            {
                int yi = (int)NumOps.ToDouble(y[i]);
                double pk = ComputeCountProbability(yi, lambdaD);
                T likelihood = NumOps.Add(NumOps.Multiply(oneMinusPi, NumOps.FromDouble(pk)), tiny);
                ll = NumOps.Add(ll, NumOps.Log(likelihood));
            }
        }
        return ll;
    }

    private static double StandardNormalCdf(double z)
    {
        return 0.5 * (1 + Erf(z / Math.Sqrt(2)));
    }

    private static double Erf(double x)
    {
        double sign = x < 0 ? -1.0 : 1.0;
        x = Math.Abs(x);
        double t = 1.0 / (1.0 + 0.3275911 * x);
        double y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.Exp(-x * x);
        return sign * y;
    }

    private static double LogFactorial(int n)
    {
        if (n <= 1) return 0;
        return LogGamma(n + 1);
    }

    private static double LogGamma(double x)
    {
        if (x <= 0) return double.PositiveInfinity;
        double[] c = { 76.18009172947146, -86.50532032941677, 24.01409824083091,
                       -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5 };
        double y = x, tmp = x + 5.5;
        tmp -= (x + 0.5) * Math.Log(tmp);
        double ser = 1.000000000190015;
        for (int j = 0; j < 6; j++) ser += c[j] / ++y;
        return -tmp + Math.Log(2.5066282746310005 * ser / x);
    }

    private Vector<T> SolveSystem(Matrix<T> a, Vector<T> b, int n)
    {
        var aug = new Matrix<T>(n, n + 1);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++) aug[i, j] = a[i, j];
            aug[i, n] = b[i];
        }

        T pivotThreshold = NumOps.FromDouble(1e-10);

        for (int col = 0; col < n; col++)
        {
            int maxRow = col;
            for (int row = col + 1; row < n; row++)
            {
                if (NumOps.GreaterThan(NumOps.Abs(aug[row, col]), NumOps.Abs(aug[maxRow, col])))
                    maxRow = row;
            }

            for (int j = 0; j <= n; j++)
                (aug[col, j], aug[maxRow, j]) = (aug[maxRow, j], aug[col, j]);

            T pivot = aug[col, col];
            if (NumOps.LessThan(NumOps.Abs(pivot), pivotThreshold))
                pivot = pivotThreshold;
            for (int j = 0; j <= n; j++) aug[col, j] = NumOps.Divide(aug[col, j], pivot);

            for (int row = 0; row < n; row++)
            {
                if (row != col)
                {
                    T factor = aug[row, col];
                    for (int j = 0; j <= n; j++)
                        aug[row, j] = NumOps.Subtract(aug[row, j], NumOps.Multiply(factor, aug[col, j]));
                }
            }
        }

        var sol = new Vector<T>(n);
        for (int i = 0; i < n; i++) sol[i] = aug[i, n];
        return sol;
    }

    /// <inheritdoc/>
    protected override Task CalculateFeatureImportancesAsync(int featureCount)
    {
        var importances = new Vector<T>(_numFeatures);

        for (int f = 0; f < _numFeatures; f++)
        {
            T imp = NumOps.Zero;
            if (_countCoefficients != null)
                imp = NumOps.Add(imp, NumOps.Abs(_countCoefficients[f]));
            if (_zeroCoefficients != null)
                imp = NumOps.Add(imp, NumOps.Abs(_zeroCoefficients[f]));
            importances[f] = imp;
        }

        T sum = NumOps.Zero;
        for (int f = 0; f < _numFeatures; f++)
        {
            sum = NumOps.Add(sum, importances[f]);
        }
        if (NumOps.GreaterThan(sum, NumOps.Zero))
        {
            for (int f = 0; f < _numFeatures; f++)
                importances[f] = NumOps.Divide(importances[f], sum);
        }

        FeatureImportances = importances;
        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.ZeroInflatedRegression,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "DistributionFamily", _options.DistributionFamily.ToString() },
                { "ModelZeroInflation", _options.ModelZeroInflation },
                { "NumberOfFeatures", _numFeatures }
            }
        };
    }

    /// <inheritdoc/>
    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        byte[] baseData = base.Serialize();
        writer.Write(baseData.Length);
        writer.Write(baseData);

        writer.Write((int)_options.DistributionFamily);
        writer.Write(_options.ModelZeroInflation);
        writer.Write(_numFeatures);
        writer.Write(NumOps.ToDouble(_countIntercept));
        writer.Write(NumOps.ToDouble(_zeroIntercept));
        writer.Write(NumOps.ToDouble(_dispersion));

        WriteVec(writer, _countCoefficients);
        WriteVec(writer, _zeroCoefficients);

        return ms.ToArray();
    }

    private void WriteVec(BinaryWriter w, Vector<T>? v)
    {
        w.Write(v != null);
        if (v != null)
        {
            w.Write(v.Length);
            for (int i = 0; i < v.Length; i++) w.Write(NumOps.ToDouble(v[i]));
        }
    }

    /// <inheritdoc/>
    public override void Deserialize(byte[] modelData)
    {
        using var ms = new MemoryStream(modelData);
        using var reader = new BinaryReader(ms);

        int baseLen = reader.ReadInt32();
        base.Deserialize(reader.ReadBytes(baseLen));

        _options.DistributionFamily = (ZeroInflatedDistributionFamily)reader.ReadInt32();
        _options.ModelZeroInflation = reader.ReadBoolean();
        _numFeatures = reader.ReadInt32();
        _countIntercept = NumOps.FromDouble(reader.ReadDouble());
        _zeroIntercept = NumOps.FromDouble(reader.ReadDouble());
        _dispersion = NumOps.FromDouble(reader.ReadDouble());

        _countCoefficients = ReadVec(reader);
        _zeroCoefficients = ReadVec(reader);
    }

    private Vector<T>? ReadVec(BinaryReader r)
    {
        if (!r.ReadBoolean()) return null;
        int len = r.ReadInt32();
        var v = new Vector<T>(len);
        for (int i = 0; i < len; i++) v[i] = NumOps.FromDouble(r.ReadDouble());
        return v;
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new ZeroInflatedRegression<T>(_options, Regularization);
    }
}
