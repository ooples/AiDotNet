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
    private const double MuFloor = 1e-10;
    private const double MuCeiling = 1.0 - 1e-10;
    private const double MinFisherValue = 0.1;
    private const double PivotThreshold = 1e-10;

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
        T zero = NumOps.Zero;
        T one = NumOps.One;
        for (int i = 0; i < n; i++)
        {
            if (!NumOps.GreaterThan(y[i], zero) || !NumOps.LessThan(y[i], one))
            {
                throw new ArgumentException($"Response values must be in (0, 1). Found: {NumOps.ToDouble(y[i])} at index {i}");
            }
        }

        // Initialize parameters
        InitializeParameters(y);

        T prevLogLik = NumOps.MinValue;
        T tolerance = NumOps.FromDouble(_options.Tolerance);

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
            T logLik = ComputeLogLikelihood(y, mus, phis);

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
            T alpha = NumOps.Multiply(mus[i], phis[i]);
            T beta = NumOps.Multiply(NumOps.Subtract(NumOps.One, mus[i]), phis[i]);

            distributions[i] = new BetaDistribution<T>(alpha, beta);
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
        T sumLogit = NumOps.Zero;
        for (int i = 0; i < y.Length; i++)
        {
            T yi = y[i];
            sumLogit = NumOps.Add(sumLogit, NumOps.Log(NumOps.Divide(yi, NumOps.Subtract(NumOps.One, yi))));
        }
        _meanIntercept = NumOps.Divide(sumLogit, NumOps.FromDouble(y.Length));

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
        T minPhi = NumOps.FromDouble(0.1);

        for (int i = 0; i < n; i++)
        {
            // Linear predictor for mean
            T etaMu = _meanIntercept;
            if (_meanCoefficients != null)
            {
                for (int j = 0; j < _numFeatures; j++)
                {
                    etaMu = NumOps.Add(etaMu, NumOps.Multiply(_meanCoefficients[j], x[i, j]));
                }
            }

            // Apply link function inverse and clamp away from beta distribution endpoints
            double muRaw = InverseLinkFunction(NumOps.ToDouble(etaMu));
            double muClamped = Math.Max(MuFloor, Math.Min(MuCeiling, muRaw));
            mus[i] = NumOps.FromDouble(muClamped);

            // Linear predictor for precision
            T etaPhi = _precisionIntercept;
            if (_options.ModelVariablePrecision && _precisionCoefficients != null)
            {
                for (int j = 0; j < _numFeatures; j++)
                {
                    etaPhi = NumOps.Add(etaPhi, NumOps.Multiply(_precisionCoefficients[j], x[i, j]));
                }
            }

            // Precision uses log link
            T phi = NumOps.Exp(etaPhi);
            if (NumOps.LessThan(phi, minPhi))
            {
                phi = minPhi;
            }
            phis[i] = phi;
        }

        return (mus, phis);
    }

    /// <summary>
    /// Updates the mean model using Fisher scoring.
    /// </summary>
    private void UpdateMeanModel(Matrix<T> x, Vector<T> y, Vector<T> mus, Vector<T> phis)
    {
        int n = x.Rows;

        // Working weights and adjusted dependent variable
        var weights = new Vector<T>(n);
        var z = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            T mu = mus[i];
            T phi = phis[i];
            double muD = NumOps.ToDouble(mu);

            // Derivative of link function at mu (boundary: special math)
            double dmu = LinkFunctionDerivative(muD);
            T dmuT = NumOps.FromDouble(dmu);

            // Fisher weight: w = φ * mu * (1-mu) / g'(mu)²
            T v = NumOps.Multiply(mu, NumOps.Subtract(NumOps.One, mu));
            T w = NumOps.Divide(NumOps.Multiply(phi, v), NumOps.Multiply(dmuT, dmuT));
            weights[i] = w;

            // Working response
            double eta = LinkFunction(muD);
            T residual = NumOps.Subtract(y[i], mu);
            z[i] = NumOps.Add(NumOps.FromDouble(eta), NumOps.Divide(NumOps.Multiply(residual, dmuT), v));
        }

        // Weighted least squares
        if (_meanCoefficients is null)
        {
            throw new InvalidOperationException("Mean coefficients not initialized.");
        }
        UpdateCoefficientsWLS(x, z, weights, ref _meanCoefficients, ref _meanIntercept);
    }

    /// <summary>
    /// Updates the precision model using Fisher scoring.
    /// </summary>
    private void UpdatePrecisionModel(Matrix<T> x, Vector<T> y, Vector<T> mus, Vector<T> phis)
    {
        int n = x.Rows;
        var weights = new Vector<T>(n);
        var z = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            // Boundary conversion for special functions (Digamma, Trigamma)
            double muD = NumOps.ToDouble(mus[i]);
            double phiD = NumOps.ToDouble(phis[i]);
            double yiD = NumOps.ToDouble(y[i]);

            // Score for phi: d log L / d log(phi)
            double ystar = Math.Log(yiD / (1 - yiD));
            double mustar = Digamma(muD * phiD) - Digamma((1 - muD) * phiD);
            double score = muD * (ystar - mustar);

            // Fisher information for log(phi)
            double trigammaTerm = muD * muD * Trigamma(muD * phiD) + (1 - muD) * (1 - muD) * Trigamma((1 - muD) * phiD);
            double fisherInfo = phiD * phiD * trigammaTerm;
            fisherInfo = Math.Max(fisherInfo, MinFisherValue);

            T fisherInfoT = NumOps.FromDouble(fisherInfo);
            weights[i] = fisherInfoT;
            T logPhi = NumOps.Log(phis[i]);
            z[i] = NumOps.Add(logPhi, NumOps.Divide(NumOps.FromDouble(score), fisherInfoT));
        }

        if (_precisionCoefficients != null)
        {
            UpdateCoefficientsWLS(x, z, weights, ref _precisionCoefficients, ref _precisionIntercept);
        }
    }

    /// <summary>
    /// Updates coefficients using weighted least squares.
    /// </summary>
    private void UpdateCoefficientsWLS(Matrix<T> x, Vector<T> z, Vector<T> weights, ref Vector<T> coefficients, ref T intercept)
    {
        int n = x.Rows;
        int p = _numFeatures;

        // X'WX and X'Wz
        var xtwx = new Matrix<T>(p + 1, p + 1);
        var xtwz = new Vector<T>(p + 1);

        for (int i = 0; i < n; i++)
        {
            T w = weights[i];

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

        // Solve
        var solution = SolveSystem(xtwx, xtwz, p + 1);

        // Update with learning rate
        T lr = NumOps.FromDouble(_options.LearningRate);
        T oneMinusLr = NumOps.Subtract(NumOps.One, lr);
        intercept = NumOps.Add(
            NumOps.Multiply(oneMinusLr, intercept),
            NumOps.Multiply(lr, solution[0]));

        for (int j = 0; j < p; j++)
        {
            coefficients[j] = NumOps.Add(
                NumOps.Multiply(oneMinusLr, coefficients[j]),
                NumOps.Multiply(lr, solution[j + 1]));
        }
    }

    /// <summary>
    /// Computes the log-likelihood.
    /// </summary>
    private T ComputeLogLikelihood(Vector<T> y, Vector<T> mus, Vector<T> phis)
    {
        T ll = NumOps.Zero;
        for (int i = 0; i < y.Length; i++)
        {
            T mu = mus[i];
            T phi = phis[i];
            T alpha = NumOps.Multiply(mu, phi);
            T beta = NumOps.Multiply(NumOps.Subtract(NumOps.One, mu), phi);

            // Log Beta PDF — LogGamma stays double (numerical recipe)
            double lgPhi = LogGamma(NumOps.ToDouble(phi));
            double lgAlpha = LogGamma(NumOps.ToDouble(alpha));
            double lgBeta = LogGamma(NumOps.ToDouble(beta));

            T logPdf = NumOps.FromDouble(lgPhi - lgAlpha - lgBeta);
            T logYi = NumOps.Log(y[i]);
            T logOneMinusYi = NumOps.Log(NumOps.Subtract(NumOps.One, y[i]));

            logPdf = NumOps.Add(logPdf, NumOps.Multiply(NumOps.Subtract(alpha, NumOps.One), logYi));
            logPdf = NumOps.Add(logPdf, NumOps.Multiply(NumOps.Subtract(beta, NumOps.One), logOneMinusYi));

            ll = NumOps.Add(ll, logPdf);
        }
        return ll;
    }

    /// <summary>
    /// Applies the link function.
    /// </summary>
    private double LinkFunction(double mu)
    {
        mu = Math.Max(MuFloor, Math.Min(MuCeiling, mu));

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
        mu = Math.Max(MuFloor, Math.Min(MuCeiling, mu));

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

    private Vector<T> SolveSystem(Matrix<T> a, Vector<T> b, int n)
    {
        var aug = new Matrix<T>(n, n + 1);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++) aug[i, j] = a[i, j];
            aug[i, n] = b[i];
        }

        T pivotThreshold = NumOps.FromDouble(PivotThreshold);

        for (int col = 0; col < n; col++)
        {
            // Partial pivoting
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
            if (_meanCoefficients != null)
                imp = NumOps.Add(imp, NumOps.Abs(_meanCoefficients[f]));
            if (_precisionCoefficients != null)
                imp = NumOps.Add(imp, NumOps.Abs(_precisionCoefficients[f]));
            importances[f] = imp;
        }

        T sum = NumOps.Zero;
        for (int f = 0; f < _numFeatures; f++)
            sum = NumOps.Add(sum, importances[f]);
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
