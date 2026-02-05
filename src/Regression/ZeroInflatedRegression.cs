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
        for (int i = 0; i < n; i++)
        {
            double yi = NumOps.ToDouble(y[i]);
            if (yi < 0 || yi != Math.Floor(yi))
            {
                throw new ArgumentException($"Response must be non-negative integers. Found: {yi} at index {i}");
            }
        }

        // Initialize parameters
        InitializeParameters(y);

        double prevLogLik = double.MinValue;

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
            double logLik = ComputeLogLikelihood(y, lambdas, pis);

            if (Math.Abs(logLik - prevLogLik) < _options.Tolerance)
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
            double pi = NumOps.ToDouble(pis[i]);
            double lambda = NumOps.ToDouble(lambdas[i]);
            predictions[i] = NumOps.FromDouble((1 - pi) * lambda);
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
        return new Vector<T>(pis);
    }

    /// <summary>
    /// Predicts the expected count conditional on not being a structural zero.
    /// </summary>
    /// <param name="input">Input feature matrix.</param>
    /// <returns>Vector of conditional count expectations.</returns>
    public async Task<Vector<T>> PredictConditionalCountAsync(Matrix<T> input)
    {
        var (lambdas, _) = await Task.Run(() => ComputePredictions(input));
        return new Vector<T>(lambdas);
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
            double pi = NumOps.ToDouble(pis[i]);
            double lambda = NumOps.ToDouble(lambdas[i]);

            for (int k = 0; k <= maxCount; k++)
            {
                double prob;
                if (k == 0)
                {
                    // P(Y=0) = π + (1-π) * P_count(0)
                    double p0 = ComputeCountProbability(0, lambda);
                    prob = pi + (1 - pi) * p0;
                }
                else
                {
                    // P(Y=k) = (1-π) * P_count(k)
                    double pk = ComputeCountProbability(k, lambda);
                    prob = (1 - pi) * pk;
                }
                pmf[i, k] = NumOps.FromDouble(prob);
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
        double sumNonZero = 0;
        int countNonZero = 0;

        for (int i = 0; i < y.Length; i++)
        {
            double yi = NumOps.ToDouble(y[i]);
            if (yi == 0)
            {
                numZeros++;
            }
            else
            {
                sumNonZero += yi;
                countNonZero++;
            }
        }

        double pZero = (double)numZeros / y.Length;
        double meanNonZero = countNonZero > 0 ? sumNonZero / countNonZero : 1.0;

        // Initialize zero model intercept (logit scale)
        pZero = Math.Max(0.01, Math.Min(0.99, pZero));
        _zeroIntercept = NumOps.FromDouble(Math.Log(pZero / (1 - pZero)));

        // Initialize count model intercept (log scale)
        _countIntercept = NumOps.FromDouble(Math.Log(Math.Max(meanNonZero, 0.1)));

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
    private (T[] lambdas, T[] pis) ComputePredictions(Matrix<T> x)
    {
        int n = x.Rows;
        var lambdas = new T[n];
        var pis = new T[n];

        for (int i = 0; i < n; i++)
        {
            // Count model linear predictor
            double etaCount = NumOps.ToDouble(_countIntercept);
            if (_countCoefficients != null)
            {
                for (int j = 0; j < _numFeatures; j++)
                {
                    etaCount += NumOps.ToDouble(_countCoefficients[j]) * NumOps.ToDouble(x[i, j]);
                }
            }

            // Apply count link inverse (log link -> exp)
            double lambda = _options.CountLink switch
            {
                ZeroInflatedCountLink.Log => Math.Exp(etaCount),
                ZeroInflatedCountLink.SquareRoot => etaCount * etaCount,
                ZeroInflatedCountLink.Identity => Math.Max(0.001, etaCount),
                _ => Math.Exp(etaCount)
            };
            lambdas[i] = NumOps.FromDouble(lambda);

            // Zero-inflation model linear predictor
            double etaZero = NumOps.ToDouble(_zeroIntercept);
            if (_options.ModelZeroInflation && _zeroCoefficients != null)
            {
                for (int j = 0; j < _numFeatures; j++)
                {
                    etaZero += NumOps.ToDouble(_zeroCoefficients[j]) * NumOps.ToDouble(x[i, j]);
                }
            }

            // Apply zero link inverse
            double pi = _options.ZeroLink switch
            {
                ZeroInflatedZeroLink.Logit => 1 / (1 + Math.Exp(-etaZero)),
                ZeroInflatedZeroLink.Probit => StandardNormalCdf(etaZero),
                ZeroInflatedZeroLink.CLogLog => 1 - Math.Exp(-Math.Exp(etaZero)),
                _ => 1 / (1 + Math.Exp(-etaZero))
            };
            pis[i] = NumOps.FromDouble(pi);
        }

        return (lambdas, pis);
    }

    /// <summary>
    /// Computes posterior probability of being a structural zero.
    /// </summary>
    private double[] ComputePosteriorZero(Vector<T> y, T[] lambdas, T[] pis)
    {
        var posterior = new double[y.Length];

        for (int i = 0; i < y.Length; i++)
        {
            double yi = NumOps.ToDouble(y[i]);
            double pi = NumOps.ToDouble(pis[i]);
            double lambda = NumOps.ToDouble(lambdas[i]);

            if (yi == 0)
            {
                // P(Z=1 | Y=0) = π / (π + (1-π) * P(Y=0|Z=0))
                double p0 = ComputeCountProbability(0, lambda);
                double numer = pi;
                double denom = pi + (1 - pi) * p0;
                posterior[i] = denom > 0 ? numer / denom : 0.5;
            }
            else
            {
                // If Y > 0, definitely not a structural zero
                posterior[i] = 0;
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
    private void UpdateCountModel(Matrix<T> x, Vector<T> y, double[] posteriorZero)
    {
        int n = x.Rows;
        int p = _numFeatures;

        // Weighted IRLS for Poisson/NB regression
        var weights = new double[n];
        var z = new double[n];

        for (int i = 0; i < n; i++)
        {
            double yi = NumOps.ToDouble(y[i]);
            double wi = 1 - posteriorZero[i];  // Weight by probability of not being structural zero

            if (wi < 1e-10) wi = 1e-10;

            double eta = NumOps.ToDouble(_countIntercept);
            if (_countCoefficients != null)
            {
                for (int j = 0; j < p; j++)
                {
                    eta += NumOps.ToDouble(_countCoefficients[j]) * NumOps.ToDouble(x[i, j]);
                }
            }

            double lambda = Math.Exp(eta);
            lambda = Math.Max(lambda, 1e-10);

            // Working weight and response for log link
            weights[i] = wi * lambda;
            z[i] = eta + (yi - lambda) / lambda;
        }

        UpdateCoefficientsWLS(x, z, weights, ref _countCoefficients!, ref _countIntercept);
    }

    /// <summary>
    /// Updates the zero-inflation model parameters.
    /// </summary>
    private void UpdateZeroModel(Matrix<T> x, double[] posteriorZero)
    {
        int n = x.Rows;
        int p = _numFeatures;

        // Weighted logistic regression for zero model
        var weights = new double[n];
        var z = new double[n];

        for (int i = 0; i < n; i++)
        {
            double pZero = posteriorZero[i];

            double eta = NumOps.ToDouble(_zeroIntercept);
            if (_zeroCoefficients != null)
            {
                for (int j = 0; j < p; j++)
                {
                    eta += NumOps.ToDouble(_zeroCoefficients[j]) * NumOps.ToDouble(x[i, j]);
                }
            }

            double pi = 1 / (1 + Math.Exp(-eta));
            pi = Math.Max(1e-10, Math.Min(1 - 1e-10, pi));

            // Working weight and response for logit link
            weights[i] = pi * (1 - pi);
            z[i] = eta + (pZero - pi) / (pi * (1 - pi));
        }

        if (_zeroCoefficients != null)
        {
            UpdateCoefficientsWLS(x, z, weights, ref _zeroCoefficients, ref _zeroIntercept);
        }
    }

    /// <summary>
    /// Updates dispersion parameter for Negative Binomial.
    /// </summary>
    private void UpdateDispersion(Vector<T> y, T[] lambdas, double[] posteriorZero)
    {
        // Method of moments estimate
        double sumSqDev = 0;
        double sumLambda = 0;
        double totalWeight = 0;

        for (int i = 0; i < y.Length; i++)
        {
            double wi = 1 - posteriorZero[i];
            if (wi < 1e-10) continue;

            double yi = NumOps.ToDouble(y[i]);
            double lambda = NumOps.ToDouble(lambdas[i]);

            sumSqDev += wi * (yi - lambda) * (yi - lambda);
            sumLambda += wi * lambda;
            totalWeight += wi;
        }

        if (totalWeight > 0 && sumLambda > 0)
        {
            double varianceEstimate = sumSqDev / totalWeight;
            double meanEstimate = sumLambda / totalWeight;

            // For NB: Var = μ + μ²/r, so r = μ² / (Var - μ)
            double excess = varianceEstimate - meanEstimate;
            if (excess > 0.1)
            {
                double r = meanEstimate * meanEstimate / excess;
                r = Math.Max(0.1, Math.Min(1000, r));
                _dispersion = NumOps.FromDouble(r);
            }
        }
    }

    /// <summary>
    /// Updates coefficients using weighted least squares.
    /// </summary>
    private void UpdateCoefficientsWLS(Matrix<T> x, double[] z, double[] weights,
        ref Vector<T> coefficients, ref T intercept)
    {
        int n = x.Rows;
        int p = _numFeatures;

        var xtwx = new double[p + 1, p + 1];
        var xtwz = new double[p + 1];

        for (int i = 0; i < n; i++)
        {
            double w = weights[i];
            if (w < 1e-10) continue;

            xtwx[0, 0] += w;
            xtwz[0] += w * z[i];

            for (int j = 0; j < p; j++)
            {
                double xij = NumOps.ToDouble(x[i, j]);
                xtwx[0, j + 1] += w * xij;
                xtwx[j + 1, 0] += w * xij;
                xtwz[j + 1] += w * xij * z[i];

                for (int k = 0; k <= j; k++)
                {
                    double xik = NumOps.ToDouble(x[i, k]);
                    xtwx[j + 1, k + 1] += w * xij * xik;
                    if (k < j) xtwx[k + 1, j + 1] = xtwx[j + 1, k + 1];
                }
            }
        }

        // Regularization
        double lambda = _options.UseRegularization ? _options.RegularizationStrength : 0;
        for (int j = 1; j <= p; j++)
        {
            xtwx[j, j] += lambda;
        }

        var solution = SolveSystem(xtwx, xtwz, p + 1);

        intercept = NumOps.FromDouble(solution[0]);
        for (int j = 0; j < p; j++)
        {
            coefficients[j] = NumOps.FromDouble(solution[j + 1]);
        }
    }

    /// <summary>
    /// Computes the log-likelihood.
    /// </summary>
    private double ComputeLogLikelihood(Vector<T> y, T[] lambdas, T[] pis)
    {
        double ll = 0;
        for (int i = 0; i < y.Length; i++)
        {
            double yi = NumOps.ToDouble(y[i]);
            double pi = NumOps.ToDouble(pis[i]);
            double lambda = NumOps.ToDouble(lambdas[i]);

            if (yi == 0)
            {
                double p0 = ComputeCountProbability(0, lambda);
                ll += Math.Log(pi + (1 - pi) * p0 + 1e-300);
            }
            else
            {
                double pk = ComputeCountProbability((int)yi, lambda);
                ll += Math.Log((1 - pi) * pk + 1e-300);
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

    private double[] SolveSystem(double[,] a, double[] b, int n)
    {
        var aug = new double[n, n + 1];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++) aug[i, j] = a[i, j];
            aug[i, n] = b[i];
        }

        for (int col = 0; col < n; col++)
        {
            int maxRow = col;
            for (int row = col + 1; row < n; row++)
                if (Math.Abs(aug[row, col]) > Math.Abs(aug[maxRow, col])) maxRow = row;

            for (int j = 0; j <= n; j++)
                (aug[col, j], aug[maxRow, j]) = (aug[maxRow, j], aug[col, j]);

            double pivot = aug[col, col];
            if (Math.Abs(pivot) < 1e-10) pivot = 1e-10;
            for (int j = 0; j <= n; j++) aug[col, j] /= pivot;

            for (int row = 0; row < n; row++)
            {
                if (row != col)
                {
                    double factor = aug[row, col];
                    for (int j = 0; j <= n; j++) aug[row, j] -= factor * aug[col, j];
                }
            }
        }

        var sol = new double[n];
        for (int i = 0; i < n; i++) sol[i] = aug[i, n];
        return sol;
    }

    /// <inheritdoc/>
    protected override Task CalculateFeatureImportancesAsync(int featureCount)
    {
        var importances = new T[_numFeatures];

        for (int f = 0; f < _numFeatures; f++)
        {
            double imp = 0;
            if (_countCoefficients != null)
                imp += Math.Abs(NumOps.ToDouble(_countCoefficients[f]));
            if (_zeroCoefficients != null)
                imp += Math.Abs(NumOps.ToDouble(_zeroCoefficients[f]));
            importances[f] = NumOps.FromDouble(imp);
        }

        double sum = importances.Sum(x => NumOps.ToDouble(x));
        if (sum > 0)
        {
            for (int f = 0; f < _numFeatures; f++)
                importances[f] = NumOps.Divide(importances[f], NumOps.FromDouble(sum));
        }

        FeatureImportances = new Vector<T>(importances);
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
