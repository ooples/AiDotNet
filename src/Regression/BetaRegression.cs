using AiDotNet.Distributions;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;

namespace AiDotNet.Regression;

/// <summary>
/// Beta Regression for modeling proportions and rates bounded in (0, 1).
/// </summary>
/// <remarks>
/// <para>
/// Beta Regression is the appropriate choice when your response variable is a continuous
/// proportion or rate that must fall strictly between 0 and 1. It uses the Beta distribution
/// and can model both the mean and precision as functions of covariates.
/// </para>
/// <para>
/// <b>For Beginners:</b> When you need to predict proportions (like percentages),
/// regular regression can give impossible results (negative values or values > 1).
/// Beta Regression fixes this by:
///
/// 1. Always producing valid predictions between 0 and 1
/// 2. Naturally handling skewed proportions
/// 3. Allowing varying uncertainty (some predictions more reliable than others)
///
/// Example applications:
/// - Predicting market share (e.g., "37% market share")
/// - Modeling test pass rates
/// - Estimating probability scores
/// - Analyzing biological concentrations
///
/// The model uses a "link function" to transform proportions to a scale where linear
/// modeling works, then transforms predictions back to valid proportions.
/// </para>
/// <para>
/// Reference: Ferrari, S.L.P., Cribari-Neto, F. (2004). "Beta Regression for
/// Modelling Rates and Proportions". Journal of Applied Statistics, 31(7), 799-815.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class BetaRegression<T> : AsyncDecisionTreeRegressionBase<T>
{
    /// <summary>
    /// Coefficients for the mean (μ) model.
    /// </summary>
    private Vector<T>? _meanCoefficients;

    /// <summary>
    /// Intercept for the mean model.
    /// </summary>
    private T _meanIntercept;

    /// <summary>
    /// Coefficients for the precision (φ) model (if variable precision).
    /// </summary>
    private Vector<T>? _precisionCoefficients;

    /// <summary>
    /// Intercept for the precision model.
    /// </summary>
    private T _precisionIntercept;

    /// <summary>
    /// Number of features.
    /// </summary>
    private int _numFeatures;

    /// <summary>
    /// Configuration options.
    /// </summary>
    private readonly BetaRegressionOptions _options;

    /// <inheritdoc/>
    public override int NumberOfTrees => 1;

    /// <summary>
    /// Gets the mean model coefficients.
    /// </summary>
    public Vector<T>? MeanCoefficients => _meanCoefficients;

    /// <summary>
    /// Gets the mean model intercept.
    /// </summary>
    public T MeanIntercept => _meanIntercept;

    /// <summary>
    /// Gets the precision model coefficients (if variable precision is enabled).
    /// </summary>
    public Vector<T>? PrecisionCoefficients => _precisionCoefficients;

    /// <summary>
    /// Gets the precision (or its intercept if constant).
    /// </summary>
    public T Precision => _precisionIntercept;

    /// <summary>
    /// Initializes a new instance of BetaRegression.
    /// </summary>
    /// <param name="options">Configuration options.</param>
    /// <param name="regularization">Optional regularization.</param>
    public BetaRegression(BetaRegressionOptions? options = null, IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(null, regularization)
    {
        _options = options ?? new BetaRegressionOptions();
        _meanIntercept = NumOps.Zero;
        _precisionIntercept = NumOps.FromDouble(Math.Log(10));  // Initial precision = 10
        _numFeatures = 0;
    }

    /// <inheritdoc/>
    public override async Task TrainAsync(Matrix<T> x, Vector<T> y)
    {
        _numFeatures = x.Columns;
        int n = x.Rows;

        // Validate response values are in (0, 1)
        for (int i = 0; i < n; i++)
        {
            double yi = NumOps.ToDouble(y[i]);
            if (yi <= 0 || yi >= 1)
            {
                throw new ArgumentException($"Response values must be in (0, 1). Found: {yi} at index {i}");
            }
        }

        // Initialize parameters
        InitializeParameters(y);

        double prevLogLik = double.MinValue;

        // Fisher scoring / IRLS
        for (int iter = 0; iter < _options.MaxIterations; iter++)
        {
            // Compute current predictions
            var (mus, phis) = ComputePredictions(x);

            // Update mean model coefficients
            UpdateMeanModel(x, y, mus, phis);

            // Update precision model if variable
            if (_options.ModelVariablePrecision)
            {
                UpdatePrecisionModel(x, y, mus, phis);
            }

            // Check convergence
            (mus, phis) = ComputePredictions(x);
            double logLik = ComputeLogLikelihood(y, mus, phis);

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
        var (mus, _) = await Task.Run(() => ComputePredictions(input));
        return mus;
    }

    /// <summary>
    /// Predicts full Beta distributions for each input sample.
    /// </summary>
    /// <param name="input">Input feature matrix.</param>
    /// <returns>Array of predicted Beta distributions.</returns>
    public async Task<IParametricDistribution<T>[]> PredictDistributionsAsync(Matrix<T> input)
    {
        var (mus, phis) = await Task.Run(() => ComputePredictions(input));
        var distributions = new IParametricDistribution<T>[input.Rows];

        for (int i = 0; i < input.Rows; i++)
        {
            // Convert (μ, φ) to (α, β) parameterization
            double mu = NumOps.ToDouble(mus[i]);
            double phi = NumOps.ToDouble(phis[i]);
            double alpha = mu * phi;
            double beta = (1 - mu) * phi;

            distributions[i] = new BetaDistribution<T>(
                NumOps.FromDouble(alpha),
                NumOps.FromDouble(beta));
        }

        return distributions;
    }

    /// <summary>
    /// Gets prediction intervals for each input sample.
    /// </summary>
    /// <param name="input">Input feature matrix.</param>
    /// <param name="confidenceLevel">Confidence level (default 0.95).</param>
    /// <returns>Tuple of (lower bounds, upper bounds).</returns>
    public async Task<(Vector<T> Lower, Vector<T> Upper)> PredictIntervalAsync(Matrix<T> input, double confidenceLevel = 0.95)
    {
        var distributions = await PredictDistributionsAsync(input);
        var lower = new Vector<T>(input.Rows);
        var upper = new Vector<T>(input.Rows);

        double alpha = 1 - confidenceLevel;
        T alphaLower = NumOps.FromDouble(alpha / 2);
        T alphaUpper = NumOps.FromDouble(1 - alpha / 2);

        for (int i = 0; i < input.Rows; i++)
        {
            lower[i] = distributions[i].InverseCdf(alphaLower);
            upper[i] = distributions[i].InverseCdf(alphaUpper);
        }

        return (lower, upper);
    }

    /// <summary>
    /// Initializes parameters from target values.
    /// </summary>
    private void InitializeParameters(Vector<T> y)
    {
        // Initialize mean intercept using empirical logit
        double sumLogit = 0;
        for (int i = 0; i < y.Length; i++)
        {
            double yi = NumOps.ToDouble(y[i]);
            sumLogit += Math.Log(yi / (1 - yi));
        }
        _meanIntercept = NumOps.FromDouble(sumLogit / y.Length);

        // Initialize mean coefficients to zero
        _meanCoefficients = new Vector<T>(_numFeatures);

        // Initialize precision
        if (_options.ModelVariablePrecision)
        {
            _precisionCoefficients = new Vector<T>(_numFeatures);
        }
    }

    /// <summary>
    /// Computes mean (μ) and precision (φ) predictions for all samples.
    /// </summary>
    private (Vector<T> mus, Vector<T> phis) ComputePredictions(Matrix<T> x)
    {
        int n = x.Rows;
        var mus = new Vector<T>(n);
        var phis = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            // Linear predictor for mean
            double etaMu = NumOps.ToDouble(_meanIntercept);
            if (_meanCoefficients != null)
            {
                for (int j = 0; j < _numFeatures; j++)
                {
                    etaMu += NumOps.ToDouble(_meanCoefficients[j]) * NumOps.ToDouble(x[i, j]);
                }
            }

            // Apply link function inverse
            double mu = InverseLinkFunction(etaMu);
            mus[i] = NumOps.FromDouble(mu);

            // Linear predictor for precision
            double etaPhi = NumOps.ToDouble(_precisionIntercept);
            if (_options.ModelVariablePrecision && _precisionCoefficients != null)
            {
                for (int j = 0; j < _numFeatures; j++)
                {
                    etaPhi += NumOps.ToDouble(_precisionCoefficients[j]) * NumOps.ToDouble(x[i, j]);
                }
            }

            // Precision uses log link
            double phi = Math.Exp(etaPhi);
            phi = Math.Max(phi, 0.1);  // Ensure positive
            phis[i] = NumOps.FromDouble(phi);
        }

        return (mus, phis);
    }

    /// <summary>
    /// Updates the mean model using Fisher scoring.
    /// </summary>
    private void UpdateMeanModel(Matrix<T> x, Vector<T> y, Vector<T> mus, Vector<T> phis)
    {
        int n = x.Rows;
        int p = _numFeatures;

        // Working weights and adjusted dependent variable
        var weights = new Vector<double>(n);
        var z = new Vector<double>(n);

        for (int i = 0; i < n; i++)
        {
            double yi = NumOps.ToDouble(y[i]);
            double mu = NumOps.ToDouble(mus[i]);
            double phi = NumOps.ToDouble(phis[i]);

            // Derivative of link function at mu
            double dmu = LinkFunctionDerivative(mu);

            // Fisher weight: w = φ * mu * (1-mu) / g'(mu)²
            double v = mu * (1 - mu);
            double w = phi * v / (dmu * dmu);
            weights[i] = w;

            // Working response
            double eta = LinkFunction(mu);
            double residual = yi - mu;
            z[i] = eta + residual * dmu / v;
        }

        // Weighted least squares
        UpdateCoefficientsWLS(x, z, weights, ref _meanCoefficients!, ref _meanIntercept);
    }

    /// <summary>
    /// Updates the precision model using Fisher scoring.
    /// </summary>
    private void UpdatePrecisionModel(Matrix<T> x, Vector<T> y, Vector<T> mus, Vector<T> phis)
    {
        int n = x.Rows;

        var weights = new Vector<double>(n);
        var z = new Vector<double>(n);

        for (int i = 0; i < n; i++)
        {
            double yi = NumOps.ToDouble(y[i]);
            double mu = NumOps.ToDouble(mus[i]);
            double phi = NumOps.ToDouble(phis[i]);

            // Score for phi: d log L / d log(phi)
            double ystar = Math.Log(yi / (1 - yi));
            double mustar = Digamma(mu * phi) - Digamma((1 - mu) * phi);
            double score = mu * (ystar - mustar);

            // Fisher information for log(phi)
            double trigammaTerm = mu * mu * Trigamma(mu * phi) + (1 - mu) * (1 - mu) * Trigamma((1 - mu) * phi);
            double fisherInfo = phi * phi * trigammaTerm;
            fisherInfo = Math.Max(fisherInfo, 0.1);

            weights[i] = fisherInfo;
            double logPhi = Math.Log(phi);
            z[i] = logPhi + score / fisherInfo;
        }

        if (_precisionCoefficients != null)
        {
            UpdateCoefficientsWLS(x, z, weights, ref _precisionCoefficients, ref _precisionIntercept);
        }
    }

    /// <summary>
    /// Updates coefficients using weighted least squares.
    /// </summary>
    private void UpdateCoefficientsWLS(Matrix<T> x, Vector<double> z, Vector<double> weights, ref Vector<T> coefficients, ref T intercept)
    {
        int n = x.Rows;
        int p = _numFeatures;

        // X'WX and X'Wz
        var xtwx = new Matrix<double>(p + 1, p + 1);
        var xtwz = new Vector<double>(p + 1);

        for (int i = 0; i < n; i++)
        {
            double w = weights[i];

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

        // Solve
        var solution = SolveSystem(xtwx, xtwz, p + 1);

        // Update with learning rate
        double lr = _options.LearningRate;
        intercept = NumOps.Add(
            NumOps.Multiply(NumOps.FromDouble(1 - lr), intercept),
            NumOps.FromDouble(lr * solution[0]));

        for (int j = 0; j < p; j++)
        {
            coefficients[j] = NumOps.Add(
                NumOps.Multiply(NumOps.FromDouble(1 - lr), coefficients[j]),
                NumOps.FromDouble(lr * solution[j + 1]));
        }
    }

    /// <summary>
    /// Computes the log-likelihood.
    /// </summary>
    private double ComputeLogLikelihood(Vector<T> y, Vector<T> mus, Vector<T> phis)
    {
        double ll = 0;
        for (int i = 0; i < y.Length; i++)
        {
            double yi = NumOps.ToDouble(y[i]);
            double mu = NumOps.ToDouble(mus[i]);
            double phi = NumOps.ToDouble(phis[i]);

            double alpha = mu * phi;
            double beta = (1 - mu) * phi;

            // Log Beta PDF
            ll += LogGamma(phi) - LogGamma(alpha) - LogGamma(beta);
            ll += (alpha - 1) * Math.Log(yi) + (beta - 1) * Math.Log(1 - yi);
        }
        return ll;
    }

    /// <summary>
    /// Applies the link function.
    /// </summary>
    private double LinkFunction(double mu)
    {
        mu = Math.Max(1e-10, Math.Min(1 - 1e-10, mu));

        return _options.LinkFunction switch
        {
            BetaLinkFunction.Logit => Math.Log(mu / (1 - mu)),
            BetaLinkFunction.Probit => InverseStandardNormalCdf(mu),
            BetaLinkFunction.CLogLog => Math.Log(-Math.Log(1 - mu)),
            BetaLinkFunction.Log => Math.Log(mu),
            _ => Math.Log(mu / (1 - mu))
        };
    }

    /// <summary>
    /// Applies the inverse link function.
    /// </summary>
    private double InverseLinkFunction(double eta)
    {
        return _options.LinkFunction switch
        {
            BetaLinkFunction.Logit => 1 / (1 + Math.Exp(-eta)),
            BetaLinkFunction.Probit => StandardNormalCdf(eta),
            BetaLinkFunction.CLogLog => 1 - Math.Exp(-Math.Exp(eta)),
            BetaLinkFunction.Log => Math.Exp(eta),
            _ => 1 / (1 + Math.Exp(-eta))
        };
    }

    /// <summary>
    /// Computes the derivative of the link function.
    /// </summary>
    private double LinkFunctionDerivative(double mu)
    {
        mu = Math.Max(1e-10, Math.Min(1 - 1e-10, mu));

        return _options.LinkFunction switch
        {
            BetaLinkFunction.Logit => 1 / (mu * (1 - mu)),
            BetaLinkFunction.Probit => 1 / StandardNormalPdf(InverseStandardNormalCdf(mu)),
            BetaLinkFunction.CLogLog => 1 / ((1 - mu) * Math.Log(1 - mu)),
            BetaLinkFunction.Log => 1 / mu,
            _ => 1 / (mu * (1 - mu))
        };
    }

    private static double StandardNormalCdf(double z)
    {
        return 0.5 * (1 + Erf(z / Math.Sqrt(2)));
    }

    private static double StandardNormalPdf(double z)
    {
        return Math.Exp(-0.5 * z * z) / Math.Sqrt(2 * Math.PI);
    }

    private static double InverseStandardNormalCdf(double p)
    {
        // Rational approximation
        if (p <= 0) return double.NegativeInfinity;
        if (p >= 1) return double.PositiveInfinity;

        double t = Math.Sqrt(-2 * Math.Log(p < 0.5 ? p : 1 - p));
        double c0 = 2.515517, c1 = 0.802853, c2 = 0.010328;
        double d1 = 1.432788, d2 = 0.189269, d3 = 0.001308;
        double result = t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t);
        return p < 0.5 ? -result : result;
    }

    private static double Erf(double x)
    {
        double sign = x < 0 ? -1.0 : 1.0;
        x = Math.Abs(x);
        double t = 1.0 / (1.0 + 0.3275911 * x);
        double y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.Exp(-x * x);
        return sign * y;
    }

    private static double LogGamma(double x)
    {
        if (x <= 0) return double.PositiveInfinity;
        double[] c = { 76.18009172947146, -86.50532032941677, 24.01409824083091,
                       -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5 };
        double y = x;
        double tmp = x + 5.5;
        tmp -= (x + 0.5) * Math.Log(tmp);
        double ser = 1.000000000190015;
        for (int j = 0; j < 6; j++) ser += c[j] / ++y;
        return -tmp + Math.Log(2.5066282746310005 * ser / x);
    }

    private static double Digamma(double x)
    {
        if (x <= 0) return double.NaN;
        double result = 0;
        while (x < 6)
        {
            result -= 1 / x;
            x += 1;
        }
        result += Math.Log(x) - 1 / (2 * x) - 1 / (12 * x * x) + 1 / (120 * x * x * x * x);
        return result;
    }

    private static double Trigamma(double x)
    {
        if (x <= 0) return double.NaN;
        double result = 0;
        while (x < 6)
        {
            result += 1 / (x * x);
            x += 1;
        }
        result += 1 / x + 1 / (2 * x * x) + 1 / (6 * x * x * x);
        return result;
    }

    private Vector<double> SolveSystem(Matrix<double> a, Vector<double> b, int n)
    {
        var aug = new Matrix<double>(n, n + 1);
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

        var sol = new Vector<double>(n);
        for (int i = 0; i < n; i++) sol[i] = aug[i, n];
        return sol;
    }

    /// <inheritdoc/>
    protected override Task CalculateFeatureImportancesAsync(int featureCount)
    {
        var importances = new Vector<T>(_numFeatures);

        for (int f = 0; f < _numFeatures; f++)
        {
            double imp = 0;
            if (_meanCoefficients != null)
                imp += Math.Abs(NumOps.ToDouble(_meanCoefficients[f]));
            if (_precisionCoefficients != null)
                imp += Math.Abs(NumOps.ToDouble(_precisionCoefficients[f]));
            importances[f] = NumOps.FromDouble(imp);
        }

        double sum = 0;
        for (int f = 0; f < _numFeatures; f++)
            sum += NumOps.ToDouble(importances[f]);
        if (sum > 0)
        {
            for (int f = 0; f < _numFeatures; f++)
                importances[f] = NumOps.Divide(importances[f], NumOps.FromDouble(sum));
        }

        FeatureImportances = importances;
        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.BetaRegression,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "LinkFunction", _options.LinkFunction.ToString() },
                { "ModelVariablePrecision", _options.ModelVariablePrecision },
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

        writer.Write((int)_options.LinkFunction);
        writer.Write(_options.ModelVariablePrecision);
        writer.Write(_numFeatures);
        writer.Write(NumOps.ToDouble(_meanIntercept));
        writer.Write(NumOps.ToDouble(_precisionIntercept));

        WriteVector(writer, _meanCoefficients);
        WriteVector(writer, _precisionCoefficients);

        return ms.ToArray();
    }

    private void WriteVector(BinaryWriter w, Vector<T>? v)
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

        _options.LinkFunction = (BetaLinkFunction)reader.ReadInt32();
        _options.ModelVariablePrecision = reader.ReadBoolean();
        _numFeatures = reader.ReadInt32();
        _meanIntercept = NumOps.FromDouble(reader.ReadDouble());
        _precisionIntercept = NumOps.FromDouble(reader.ReadDouble());

        _meanCoefficients = ReadVector(reader);
        _precisionCoefficients = ReadVector(reader);
    }

    private Vector<T>? ReadVector(BinaryReader r)
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
        return new BetaRegression<T>(_options, Regularization);
    }
}
