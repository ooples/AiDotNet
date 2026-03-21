using AiDotNet.Tensors.Engines;
using AiDotNet.Attributes;
using AiDotNet.Distributions;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;

namespace AiDotNet.Regression;

/// <summary>
/// GAMLSS (Generalized Additive Models for Location, Scale, and Shape) regression.
/// </summary>
/// <remarks>
/// <para>
/// GAMLSS extends generalized linear models by allowing any or all distribution parameters
/// to be modeled as functions of the explanatory variables. This enables heteroskedastic
/// modeling and full distributional regression.
/// </para>
/// <para>
/// <b>For Beginners:</b> Traditional regression models predict a single value (the mean).
/// GAMLSS predicts an entire probability distribution by modeling multiple parameters:
///
/// - Location (μ): Controls where the distribution is centered (like the mean)
/// - Scale (σ): Controls how spread out the distribution is (like standard deviation)
/// - Shape (ν, τ): Controls the shape (skewness, tail behavior)
///
/// This is powerful because:
/// 1. You get uncertainty estimates that vary with your inputs
/// 2. You can model phenomena where variance depends on the predictors
/// 3. You get proper prediction intervals instead of assuming constant variance
///
/// Example use cases:
/// - Financial forecasting where volatility depends on market conditions
/// - Medical studies where patient variability depends on treatment
/// - Any scenario where "it depends" applies to uncertainty, not just the average
/// </para>
/// <para>
/// Reference: Rigby, R.A., Stasinopoulos, D.M. (2005). "Generalized additive models
/// for location, scale and shape". Applied Statistics, 54, 507-554.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a GAMLSS model for distributional regression
/// var options = new GAMLSSRegressionOptions&lt;double&gt;();
/// var model = new GAMLSSRegression&lt;double&gt;(options);
///
/// // Prepare training data: 6 samples with 2 features each
/// var features = Matrix&lt;double&gt;.Build.Dense(6, 2, new double[] {
///     1, 2,  3, 4,  5, 6,  7, 8,  9, 10,  11, 12 });
/// var targets = new Vector&lt;double&gt;(new double[] { 3.0, 7.1, 11.0, 15.2, 19.0, 23.1 });
///
/// // Train to model location, scale, and shape parameters
/// model.Train(features, targets);
///
/// // Predict for a new sample (full distributional prediction)
/// var newSample = Matrix&lt;double&gt;.Build.Dense(1, 2, new double[] { 13, 14 });
/// var prediction = model.Predict(newSample);
/// </code>
/// </example>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ModelPaper("Generalized additive models for location, scale and shape", "https://doi.org/10.1111/j.1467-9876.2005.00510.x", Year = 2005, Authors = "Robert A. Rigby, D. Mikis Stasinopoulos")]
public class GAMLSSRegression<T> : AsyncDecisionTreeRegressionBase<T>
{
    /// <summary>
    /// Coefficients for the location parameter model.
    /// </summary>
    private Vector<T>? _locationCoefficients;

    /// <summary>
    /// Coefficients for the scale parameter model.
    /// </summary>
    private Vector<T>? _scaleCoefficients;

    /// <summary>
    /// Coefficients for the shape parameter model (if applicable).
    /// </summary>
    private Vector<T>? _shapeCoefficients;

    /// <summary>
    /// Intercept for the location parameter.
    /// </summary>
    private T _locationIntercept;

    /// <summary>
    /// Intercept for the scale parameter.
    /// </summary>
    private T _scaleIntercept;

    /// <summary>
    /// Intercept for the shape parameter.
    /// </summary>
    private T _shapeIntercept;

    /// <summary>
    /// Number of features.
    /// </summary>
    private int _numFeatures;

    /// <summary>
    /// Configuration options.
    /// </summary>
    private readonly GAMLSSOptions _options;

    /// <inheritdoc/>
    public override int NumberOfTrees => 1;

    /// <summary>
    /// Gets the location (mean) model coefficients.
    /// </summary>
    public Vector<T>? LocationCoefficients => _locationCoefficients;

    /// <summary>
    /// Gets the scale model coefficients.
    /// </summary>
    public Vector<T>? ScaleCoefficients => _scaleCoefficients;

    /// <summary>
    /// Initializes a new instance of GAMLSSRegression.
    /// </summary>
    /// <param name="options">Configuration options.</param>
    /// <param name="regularization">Optional regularization.</param>
    public GAMLSSRegression(GAMLSSOptions? options = null, IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(null, regularization)
    {
        _options = options ?? new GAMLSSOptions();
        _locationIntercept = NumOps.Zero;
        _scaleIntercept = NumOps.One;
        _shapeIntercept = NumOps.FromDouble(4.0);  // Default df for Student-t
        _numFeatures = 0;
    }

    /// <inheritdoc/>
    /// <summary>Y standardization for scale-invariant training.</summary>
    private double _yMean;
    private double _yStd = 1.0;

    public override async Task TrainAsync(Matrix<T> x, Vector<T> y)
    {
        _numFeatures = x.Columns;
        int n = x.Rows;

        // Standardize y for scale-invariant training
        double yMean = 0;
        for (int i = 0; i < n; i++) yMean += NumOps.ToDouble(y[i]);
        yMean /= n;
        double yVar = 0;
        for (int i = 0; i < n; i++) { double d = NumOps.ToDouble(y[i]) - yMean; yVar += d * d; }
        double yStd = Math.Sqrt(yVar / n);
        if (yStd < 1e-10) yStd = 1.0;
        _yMean = yMean;
        _yStd = yStd;
        var yStandardized = new Vector<T>(n);
        for (int i = 0; i < n; i++)
            yStandardized[i] = NumOps.FromDouble((NumOps.ToDouble(y[i]) - yMean) / yStd);
        y = yStandardized;

        // Initialize parameters
        InitializeParameters(y);

        // Initialize linear predictors
        var etaLocation = new Vector<T>(n);
        var etaScale = new Vector<T>(n);
        var etaShape = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            etaLocation[i] = _locationIntercept;
            etaScale[i] = _scaleIntercept;
            etaShape[i] = _shapeIntercept;
        }

        double prevDeviance = double.MaxValue;

        // Outer iteration: cycle through parameters
        for (int outer = 0; outer < _options.MaxOuterIterations; outer++)
        {
            // Fit location parameter
            if (_options.LocationModelType != GAMLSSModelType.Constant)
            {
                FitLocationParameter(x, y, etaLocation, etaScale, etaShape);
            }

            // Update linear predictor for location
            UpdateLinearPredictor(x, etaLocation, _locationCoefficients, _locationIntercept);

            // Fit scale parameter
            if (_options.ScaleModelType != GAMLSSModelType.Constant)
            {
                FitScaleParameter(x, y, etaLocation, etaScale, etaShape);
            }

            // Update linear predictor for scale
            UpdateLinearPredictor(x, etaScale, _scaleCoefficients, _scaleIntercept, useExpLink: true);

            // Fit shape parameter (if applicable)
            if (_options.ShapeModelType != GAMLSSModelType.Constant &&
                _options.DistributionFamily == GAMLSSDistributionFamily.StudentT)
            {
                FitShapeParameter(x, y, etaLocation, etaScale, etaShape);
            }

            // Check convergence
            double deviance = ComputeDeviance(y, etaLocation, etaScale, etaShape);
            if (Math.Abs(prevDeviance - deviance) < _options.Tolerance)
            {
                break;
            }
            prevDeviance = deviance;
        }

        await CalculateFeatureImportancesAsync(x.Columns);
    }

    /// <inheritdoc/>
    public override async Task<Vector<T>> PredictAsync(Matrix<T> input)
    {
        var distributions = await PredictDistributionsAsync(input);
        var predictions = new Vector<T>(input.Rows);

        for (int i = 0; i < input.Rows; i++)
        {
            // Denormalize: prediction = standardized_mean * yStd + yMean
            double mean = NumOps.ToDouble(distributions[i].Mean);
            double pred = mean * _yStd + _yMean;
            // Guard against NaN/Infinity from degenerate data (e.g., collinear features)
            if (double.IsNaN(pred) || double.IsInfinity(pred))
                pred = _yMean;
            predictions[i] = NumOps.FromDouble(pred);
        }

        return predictions;
    }

    /// <summary>
    /// Predicts full probability distributions for each input sample.
    /// </summary>
    /// <param name="input">Input feature matrix.</param>
    /// <returns>Array of predicted distributions.</returns>
    public async Task<IParametricDistribution<T>[]> PredictDistributionsAsync(Matrix<T> input)
    {
        int n = input.Rows;
        var distributions = new IParametricDistribution<T>[n];

        await Task.Run(() =>
        {
            for (int i = 0; i < n; i++)
            {
                T location = PredictParameter(input.GetRow(i), _locationCoefficients, _locationIntercept, useExpLink: false);
                T scale = PredictParameter(input.GetRow(i), _scaleCoefficients, _scaleIntercept, useExpLink: true);
                T shape = _shapeIntercept;

                if (_options.ShapeModelType != GAMLSSModelType.Constant && _shapeCoefficients != null)
                {
                    shape = PredictParameter(input.GetRow(i), _shapeCoefficients, _shapeIntercept, useExpLink: true);
                }

                distributions[i] = CreateDistribution(location, scale, shape);
            }
        });

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
    /// Initializes distribution parameters from the target values.
    /// </summary>
    private void InitializeParameters(Vector<T> y)
    {
        // Compute initial estimates
        T mean = NumOps.Zero;
        for (int i = 0; i < y.Length; i++)
        {
            mean = NumOps.Add(mean, y[i]);
        }
        mean = NumOps.Divide(mean, NumOps.FromDouble(y.Length));

        T variance = NumOps.Zero;
        for (int i = 0; i < y.Length; i++)
        {
            T diff = NumOps.Subtract(y[i], mean);
            variance = NumOps.Add(variance, NumOps.Multiply(diff, diff));
        }
        variance = NumOps.Divide(variance, NumOps.FromDouble(y.Length));
        T minVariance = NumOps.FromDouble(1e-6);
        if (NumOps.LessThan(variance, minVariance))
        {
            variance = minVariance;
        }

        _locationIntercept = mean;
        // Log link for scale: log(sqrt(variance))
        _scaleIntercept = NumOps.FromDouble(Math.Log(Math.Sqrt(NumOps.ToDouble(variance))));

        // Initialize coefficients to zero
        if (_options.LocationModelType != GAMLSSModelType.Constant)
        {
            _locationCoefficients = new Vector<T>(_numFeatures);
        }
        if (_options.ScaleModelType != GAMLSSModelType.Constant)
        {
            _scaleCoefficients = new Vector<T>(_numFeatures);
        }
        if (_options.ShapeModelType != GAMLSSModelType.Constant)
        {
            _shapeCoefficients = new Vector<T>(_numFeatures);
        }
    }

    /// <summary>
    /// Fits the location parameter using IRLS (Iteratively Reweighted Least Squares).
    /// </summary>
    private void FitLocationParameter(Matrix<T> x, Vector<T> y, Vector<T> etaLocation, Vector<T> etaScale, Vector<T> etaShape)
    {
        int n = x.Rows;

        for (int iter = 0; iter < _options.MaxInnerIterations; iter++)
        {
            // Compute working weights and adjusted dependent variable
            var weights = new Vector<T>(n);
            var z = new Vector<T>(n);

            for (int i = 0; i < n; i++)
            {
                double mu = NumOps.ToDouble(etaLocation[i]);
                double sigma = Math.Exp(NumOps.ToDouble(etaScale[i]));
                double yi = NumOps.ToDouble(y[i]);

                // For normal distribution: weight = 1/sigma^2, z = y
                double w = 1.0 / (sigma * sigma);
                double zi = yi;

                weights[i] = NumOps.FromDouble(w);
                z[i] = NumOps.FromDouble(zi);
            }

            // Update coefficients using weighted least squares
            if (_locationCoefficients is null)
            {
                throw new InvalidOperationException("Location coefficients have not been initialized.");
            }
            UpdateCoefficients(x, z, weights, ref _locationCoefficients, ref _locationIntercept);
            UpdateLinearPredictor(x, etaLocation, _locationCoefficients, _locationIntercept);
        }
    }

    /// <summary>
    /// Fits the scale parameter using IRLS.
    /// </summary>
    private void FitScaleParameter(Matrix<T> x, Vector<T> y, Vector<T> etaLocation, Vector<T> etaScale, Vector<T> etaShape)
    {
        int n = x.Rows;

        for (int iter = 0; iter < _options.MaxInnerIterations; iter++)
        {
            var weights = new Vector<T>(n);
            var z = new Vector<T>(n);

            for (int i = 0; i < n; i++)
            {
                double mu = NumOps.ToDouble(etaLocation[i]);
                double logSigma = NumOps.ToDouble(etaScale[i]);
                double sigma = Math.Exp(logSigma);
                double yi = NumOps.ToDouble(y[i]);

                // Working response and weight for log(sigma)
                double residual = (yi - mu) / sigma;
                double dldLogSigma = residual * residual - 1;  // d log L / d log(sigma)
                double w = 2.0;  // Fisher information for log(sigma) in normal

                double zi = logSigma + dldLogSigma / w;  // Newton step

                weights[i] = NumOps.FromDouble(w);
                z[i] = NumOps.FromDouble(zi);
            }

            if (_scaleCoefficients is null)
            {
                throw new InvalidOperationException("Scale coefficients have not been initialized.");
            }
            UpdateCoefficients(x, z, weights, ref _scaleCoefficients, ref _scaleIntercept);
            UpdateLinearPredictor(x, etaScale, _scaleCoefficients, _scaleIntercept);
        }
    }

    /// <summary>
    /// Fits the shape parameter using IRLS.
    /// </summary>
    private void FitShapeParameter(Matrix<T> x, Vector<T> y, Vector<T> etaLocation, Vector<T> etaScale, Vector<T> etaShape)
    {
        int n = x.Rows;

        for (int iter = 0; iter < _options.MaxInnerIterations; iter++)
        {
            var weights = new Vector<T>(n);
            var z = new Vector<T>(n);

            for (int i = 0; i < n; i++)
            {
                double mu = NumOps.ToDouble(etaLocation[i]);
                double sigma = Math.Exp(NumOps.ToDouble(etaScale[i]));
                double nu = Math.Exp(NumOps.ToDouble(etaShape[i]));  // degrees of freedom
                double yi = NumOps.ToDouble(y[i]);

                // Simplified: constant weight, linear update
                weights[i] = NumOps.FromDouble(1.0);
                z[i] = etaShape[i];
            }

            if (_shapeCoefficients != null)
            {
                UpdateCoefficients(x, z, weights, ref _shapeCoefficients, ref _shapeIntercept);
                UpdateLinearPredictor(x, etaShape, _shapeCoefficients, _shapeIntercept);
            }
        }
    }

    /// <summary>
    /// Updates coefficients using weighted least squares.
    /// </summary>
    private void UpdateCoefficients(Matrix<T> x, Vector<T> z, Vector<T> weights, ref Vector<T> coefficients, ref T intercept)
    {
        int n = x.Rows;
        int p = _numFeatures;

        // Compute X'WX and X'Wz with intercept
        var xtwx = new Matrix<T>(p + 1, p + 1);
        var xtwz = new Vector<T>(p + 1);

        for (int i = 0; i < n; i++)
        {
            T w = weights[i];
            T zi = z[i];

            // Intercept term
            xtwx[0, 0] = NumOps.Add(xtwx[0, 0], w);
            xtwz[0] = NumOps.Add(xtwz[0], NumOps.Multiply(w, zi));

            for (int j = 0; j < p; j++)
            {
                T xij = x[i, j];
                T wxij = NumOps.Multiply(w, xij);
                xtwx[0, j + 1] = NumOps.Add(xtwx[0, j + 1], wxij);
                xtwx[j + 1, 0] = NumOps.Add(xtwx[j + 1, 0], wxij);
                xtwz[j + 1] = NumOps.Add(xtwz[j + 1], NumOps.Multiply(wxij, zi));

                for (int k = 0; k <= j; k++)
                {
                    T xik = x[i, k];
                    T val = NumOps.Multiply(wxij, xik);
                    xtwx[j + 1, k + 1] = NumOps.Add(xtwx[j + 1, k + 1], val);
                    if (k < j)
                        xtwx[k + 1, j + 1] = xtwx[j + 1, k + 1];
                }
            }
        }

        // Add regularization
        T lambda = _options.UseRegularization ? NumOps.FromDouble(_options.RegularizationStrength) : NumOps.Zero;
        for (int j = 1; j <= p; j++)
        {
            xtwx[j, j] = NumOps.Add(xtwx[j, j], lambda);
        }

        // Solve the system
        var solution = SolveLinearSystem(xtwx, xtwz, p + 1);

        // Update parameters with learning rate
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
    /// Updates the linear predictor for all samples.
    /// </summary>
    private void UpdateLinearPredictor(Matrix<T> x, Vector<T> eta, Vector<T>? coefficients, T intercept, bool useExpLink = false)
    {
        for (int i = 0; i < x.Rows; i++)
        {
            T val = intercept;

            if (coefficients != null)
            {
                for (int j = 0; j < _numFeatures; j++)
                {
                    val = NumOps.Add(val, NumOps.Multiply(coefficients[j], x[i, j]));
                }
            }

            eta[i] = val;
        }
    }

    /// <summary>
    /// Predicts a single parameter value for one sample.
    /// </summary>
    private T PredictParameter(Vector<T> sample, Vector<T>? coefficients, T intercept, bool useExpLink)
    {
        T val = intercept;

        if (coefficients != null)
        {
            val = NumOps.Add(val, AiDotNetEngine.Current.DotProduct(coefficients, sample));
        }

        if (useExpLink)
        {
            val = NumOps.Exp(val);
        }

        return val;
    }

    /// <summary>
    /// Computes the deviance (negative log-likelihood).
    /// </summary>
    private double ComputeDeviance(Vector<T> y, Vector<T> etaLocation, Vector<T> etaScale, Vector<T> etaShape)
    {
        double deviance = 0;

        for (int i = 0; i < y.Length; i++)
        {
            T location = etaLocation[i];
            T scale = NumOps.FromDouble(Math.Exp(NumOps.ToDouble(etaScale[i])));
            T shape = NumOps.FromDouble(Math.Exp(NumOps.ToDouble(etaShape[i])));

            var dist = CreateDistribution(location, scale, shape);
            T logPdf = dist.LogPdf(y[i]);
            deviance -= NumOps.ToDouble(logPdf);
        }

        return 2 * deviance;
    }

    /// <summary>
    /// Creates a distribution with the specified parameters.
    /// </summary>
    private IParametricDistribution<T> CreateDistribution(T location, T scale, T shape)
    {
        // Clamp variance to a minimum positive value to prevent exceptions from collinear/degenerate data
        T minVariance = NumOps.FromDouble(1e-10);
        T variance = NumOps.Multiply(scale, scale);
        double varianceD = NumOps.ToDouble(variance);
        if (double.IsNaN(varianceD) || double.IsInfinity(varianceD) || varianceD <= 0)
            variance = minVariance;

        return _options.DistributionFamily switch
        {
            GAMLSSDistributionFamily.Normal => new NormalDistribution<T>(location, variance),
            GAMLSSDistributionFamily.Laplace => new LaplaceDistribution<T>(location, NumOps.LessThanOrEquals(scale, NumOps.Zero) ? NumOps.FromDouble(1e-5) : scale),
            GAMLSSDistributionFamily.StudentT => new StudentTDistribution<T>(location, NumOps.LessThanOrEquals(scale, NumOps.Zero) ? NumOps.FromDouble(1e-5) : scale, shape),
            GAMLSSDistributionFamily.Gamma => new GammaDistribution<T>(shape, NumOps.Divide(shape, location)),
            GAMLSSDistributionFamily.LogNormal => new LogNormalDistribution<T>(location, variance),
            GAMLSSDistributionFamily.Poisson => new PoissonDistribution<T>(location),
            GAMLSSDistributionFamily.NegativeBinomial => new NegativeBinomialDistribution<T>(location, shape),
            _ => new NormalDistribution<T>(location, variance)
        };
    }

    /// <summary>
    /// Solves a linear system using Gaussian elimination.
    /// </summary>
    private Vector<T> SolveLinearSystem(Matrix<T> a, Vector<T> b, int n)
    {
        var augmented = new Matrix<T>(n, n + 1);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                augmented[i, j] = a[i, j];
            }
            augmented[i, n] = b[i];
        }

        T pivotThreshold = NumOps.FromDouble(1e-10);

        // Forward elimination with partial pivoting
        for (int col = 0; col < n; col++)
        {
            int maxRow = col;
            for (int row = col + 1; row < n; row++)
            {
                if (NumOps.GreaterThan(NumOps.Abs(augmented[row, col]), NumOps.Abs(augmented[maxRow, col])))
                {
                    maxRow = row;
                }
            }

            for (int j = 0; j <= n; j++)
            {
                (augmented[col, j], augmented[maxRow, j]) = (augmented[maxRow, j], augmented[col, j]);
            }

            T pivot = augmented[col, col];
            if (NumOps.LessThan(NumOps.Abs(pivot), pivotThreshold))
            {
                pivot = pivotThreshold;
            }

            for (int j = 0; j <= n; j++)
            {
                augmented[col, j] = NumOps.Divide(augmented[col, j], pivot);
            }

            for (int row = 0; row < n; row++)
            {
                if (row != col)
                {
                    T factor = augmented[row, col];
                    for (int j = 0; j <= n; j++)
                    {
                        augmented[row, j] = NumOps.Subtract(augmented[row, j], NumOps.Multiply(factor, augmented[col, j]));
                    }
                }
            }
        }

        var solution = new Vector<T>(n);
        for (int i = 0; i < n; i++)
        {
            solution[i] = augmented[i, n];
        }

        return solution;
    }

    /// <inheritdoc/>
    protected override Task CalculateFeatureImportancesAsync(int featureCount)
    {
        var importances = new Vector<T>(_numFeatures);

        // Combine importances from all parameter models
        for (int f = 0; f < _numFeatures; f++)
        {
            T importance = NumOps.Zero;

            if (_locationCoefficients != null)
            {
                importance = NumOps.Add(importance, NumOps.Abs(_locationCoefficients[f]));
            }
            if (_scaleCoefficients != null)
            {
                importance = NumOps.Add(importance, NumOps.Abs(_scaleCoefficients[f]));
            }
            if (_shapeCoefficients != null)
            {
                importance = NumOps.Add(importance, NumOps.Abs(_shapeCoefficients[f]));
            }

            importances[f] = importance;
        }

        // Normalize
        T sum = NumOps.Zero;
        for (int f = 0; f < _numFeatures; f++)
        {
            sum = NumOps.Add(sum, importances[f]);
        }
        if (NumOps.GreaterThan(sum, NumOps.Zero))
        {
            for (int f = 0; f < _numFeatures; f++)
            {
                importances[f] = NumOps.Divide(importances[f], sum);
            }
        }

        FeatureImportances = importances;
        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "DistributionFamily", _options.DistributionFamily.ToString() },
                { "LocationModelType", _options.LocationModelType.ToString() },
                { "ScaleModelType", _options.ScaleModelType.ToString() },
                { "ShapeModelType", _options.ShapeModelType.ToString() },
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

        // Options
        writer.Write((int)_options.DistributionFamily);
        writer.Write((int)_options.LocationModelType);
        writer.Write((int)_options.ScaleModelType);
        writer.Write((int)_options.ShapeModelType);

        // Y standardization
        writer.Write(_yMean);
        writer.Write(_yStd);

        // State
        writer.Write(_numFeatures);
        writer.Write(NumOps.ToDouble(_locationIntercept));
        writer.Write(NumOps.ToDouble(_scaleIntercept));
        writer.Write(NumOps.ToDouble(_shapeIntercept));

        // Coefficients
        SerializeVector(writer, _locationCoefficients);
        SerializeVector(writer, _scaleCoefficients);
        SerializeVector(writer, _shapeCoefficients);

        return ms.ToArray();
    }

    private void SerializeVector(BinaryWriter writer, Vector<T>? vec)
    {
        writer.Write(vec != null);
        if (vec != null)
        {
            writer.Write(vec.Length);
            for (int i = 0; i < vec.Length; i++)
            {
                writer.Write(NumOps.ToDouble(vec[i]));
            }
        }
    }

    /// <inheritdoc/>
    public override void Deserialize(byte[] modelData)
    {
        using var ms = new MemoryStream(modelData);
        using var reader = new BinaryReader(ms);

        int baseLen = reader.ReadInt32();
        byte[] baseData = reader.ReadBytes(baseLen);
        base.Deserialize(baseData);

        // Options
        _options.DistributionFamily = (GAMLSSDistributionFamily)reader.ReadInt32();
        _options.LocationModelType = (GAMLSSModelType)reader.ReadInt32();
        _options.ScaleModelType = (GAMLSSModelType)reader.ReadInt32();
        _options.ShapeModelType = (GAMLSSModelType)reader.ReadInt32();

        // Y standardization
        _yMean = reader.ReadDouble();
        _yStd = reader.ReadDouble();

        // State
        _numFeatures = reader.ReadInt32();
        _locationIntercept = NumOps.FromDouble(reader.ReadDouble());
        _scaleIntercept = NumOps.FromDouble(reader.ReadDouble());
        _shapeIntercept = NumOps.FromDouble(reader.ReadDouble());

        // Coefficients
        _locationCoefficients = DeserializeVector(reader);
        _scaleCoefficients = DeserializeVector(reader);
        _shapeCoefficients = DeserializeVector(reader);
    }

    private Vector<T>? DeserializeVector(BinaryReader reader)
    {
        bool hasValue = reader.ReadBoolean();
        if (!hasValue) return null;

        int length = reader.ReadInt32();
        var vec = new Vector<T>(length);
        for (int i = 0; i < length; i++)
        {
            vec[i] = NumOps.FromDouble(reader.ReadDouble());
        }
        return vec;
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new GAMLSSRegression<T>(_options, Regularization);
    }

    /// <summary>
    /// Creates a deep copy via serialization to preserve private coefficient state.
    /// </summary>
    public override IFullModel<T, Matrix<T>, Vector<T>> Clone()
    {
        var clone = new GAMLSSRegression<T>(_options, Regularization);
        var data = Serialize();
        clone.Deserialize(data);
        return clone;
    }
}
