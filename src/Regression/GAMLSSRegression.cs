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
/// var options = new GAMLSSOptions();
///
/// // Prepare training data: 6 samples with 2 features each
/// var features = new Matrix&lt;double&gt;(new double[,] { { 1, 2 }, { 3, 4 }, { 5, 6 }, { 7, 8 }, { 9, 10 }, { 11, 12 } });
/// var targets = new Vector&lt;double&gt;(new double[] { 3.0, 7.1, 11.0, 15.2, 19.0, 23.1 });
///
/// // Train to model location, scale, and shape parameters
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new GAMLSSRegression&lt;double&gt;(options))
///     .Build(features, targets);
///
/// // Predict for a new sample (full distributional prediction)
/// var newSample = new Matrix&lt;double&gt;(new double[,] { { 13, 14 } });
/// var prediction = result.Predict(newSample);
/// </code>
/// </example>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Generalized additive models for location, scale and shape", "https://doi.org/10.1111/j.1467-9876.2005.00510.x", Year = 2005, Authors = "Robert A. Rigby, D. Mikis Stasinopoulos")]
public partial class GAMLSSRegression<T> : AsyncDecisionTreeRegressionBase<T>
{
    // Bounds on the scale/shape linear predictors (log-link parameters such as σ and ν).
    // The RS algorithm of Rigby &amp; Stasinopoulos (2005), like the reference gamlss R package,
    // constrains the fitted parameters to a numerically safe range so the Fisher-scoring
    // iterations stay stable. Because the target is standardised to unit variance during
    // training (see TrainAsync), σ lives on a unit scale, so σ ∈ [1e-6, 1e6] (i.e. the log
    // predictor in [-13.8, 13.8]) is far wider than any genuine residual spread yet prevents
    // exp(η) from overflowing to ±∞ — without the bound a near-perfect location fit drives the
    // scale predictor toward -∞ each outer cycle until σ underflows and the location working
    // weight 1/σ² becomes ∞, corrupting every coefficient to NaN.
    private const double MinLogScale = -13.815510557964274; // ln(1e-6)
    private const double MaxLogScale = 13.815510557964274;  // ln(1e6)

    /// <summary>
    /// Coefficients for the location parameter model.
    /// </summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T>? _locationCoefficients;

    /// <summary>
    /// Coefficients for the scale parameter model.
    /// </summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T>? _scaleCoefficients;

    /// <summary>
    /// Coefficients for the shape parameter model (if applicable).
    /// </summary>
    [AiDotNet.Attributes.TrainableParameter]
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
        _yMean = NumOps.Zero;
        _yStd = NumOps.Zero;
        _options = options ?? new GAMLSSOptions();
        _locationIntercept = NumOps.Zero;
        _scaleIntercept = NumOps.One;
        _shapeIntercept = NumOps.FromDouble(4.0);  // Default df for Student-t
        _numFeatures = 0;
    }

    /// <inheritdoc/>
    /// <summary>Y standardization for scale-invariant training.</summary>

    private T _yMean;


    private T _yStd;


    /// <inheritdoc />
    public override async Task TrainAsync(Matrix<T> x, Vector<T> y)
    {
        _numFeatures = x.Columns;
        int n = x.Rows;

        // Standardize y for scale-invariant training (SIMD-accelerated via Engine)
        T yMeanT = Engine.Mean(y);
        var centered = (Vector<T>)Engine.Subtract(y, Vector<T>.CreateDefault(n, yMeanT));
        T yVarT = Engine.DotProduct(centered, centered);
        T yStdT = NumOps.Sqrt(NumOps.Divide(yVarT, NumOps.FromDouble(n)));
        T epsT = NumOps.FromDouble(1e-10);
        if (NumOps.LessThan(yStdT, epsT)) yStdT = NumOps.One;
        _yMean = yMeanT;
        _yStd = yStdT;
        var yStandardized = (Vector<T>)Engine.Divide(centered, Vector<T>.CreateDefault(n, yStdT));
        y = yStandardized;

        // Initialize parameters
        InitializeParameters(y);

        // Initialize linear predictors
        var etaLocation = Engine.Fill(n, _locationIntercept);
        var etaScale = Engine.Fill(n, _scaleIntercept);
        var etaShape = Engine.Fill(n, _shapeIntercept);

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
        if (distributions.Length == 0)
        {
            return new Vector<T>(0);
        }

        // Denormalize: prediction = standardized_mean * yStd + yMean
        var means = new Vector<T>(distributions.Select(distribution => distribution.Mean));
        var fallback = Engine.Fill(means.Length, _yMean);
        var predictions = Tensor<T>.FromVector(Engine.Add(Engine.Multiply(means, _yStd), fallback));

        // Guard against NaN/Infinity from degenerate data (e.g., collinear features)
        return Engine.TensorWhere(Engine.TensorIsFinite(predictions), predictions, Tensor<T>.FromVector(fallback)).ToVector();
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

        double alpha = 1 - confidenceLevel;
        T alphaLower = NumOps.FromDouble(alpha / 2);
        T alphaUpper = NumOps.FromDouble(1 - alpha / 2);

        var lower = new Vector<T>(distributions.Select(distribution => distribution.InverseCdf(alphaLower)));
        var upper = new Vector<T>(distributions.Select(distribution => distribution.InverseCdf(alphaUpper)));

        return (lower, upper);
    }

    /// <summary>
    /// Initializes distribution parameters from the target values.
    /// </summary>
    private void InitializeParameters(Vector<T> y)
    {
        // Compute initial estimates
        T mean = NumOps.Divide(y.Length > 0 ? Engine.Sum(y) : NumOps.Zero, NumOps.FromDouble(y.Length));

        var centered = Engine.Subtract(y, Engine.Fill(y.Length, mean));
        T variance = NumOps.Divide(y.Length > 0 ? Engine.DotProduct(centered, centered) : NumOps.Zero, NumOps.FromDouble(y.Length));

        T minVariance = NumOps.FromDouble(1e-6);
        if (NumOps.LessThan(variance, minVariance))
        {
            variance = minVariance;
        }

        _locationIntercept = mean;
        // Log link for scale: log(sqrt(variance))
        _scaleIntercept = NumOps.Log(NumOps.Sqrt(variance));

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
        for (int iter = 0; iter < _options.MaxInnerIterations; iter++)
        {
            // Compute working weights and adjusted dependent variable.
            // For normal distribution: weight = 1/sigma^2 with sigma = exp(etaScale), z = y
            var sigma = Engine.Exp(etaScale);
            var weights = Engine.Reciprocal(Engine.Multiply(sigma, sigma));
            var z = new Vector<T>(y);

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
        T fisherWeight = NumOps.FromDouble(2.0);  // Fisher information for log(sigma) in normal

        for (int iter = 0; iter < _options.MaxInnerIterations; iter++)
        {
            // Working response and weight for log(sigma):
            // residual = (y - mu) / sigma, dl/dlog(sigma) = residual^2 - 1, z = log(sigma) + dl / w (Newton step)
            var sigma = Engine.Exp(etaScale);
            var residual = Engine.Divide(Engine.Subtract(y, etaLocation), sigma);
            var score = Engine.Subtract(Engine.Multiply(residual, residual), Engine.Fill(n, NumOps.One));
            var z = Engine.Add(etaScale, Engine.Divide(score, fisherWeight));
            var weights = Engine.Fill(n, fisherWeight);

            if (_scaleCoefficients is null)
            {
                throw new InvalidOperationException("Scale coefficients have not been initialized.");
            }

            UpdateCoefficients(x, z, weights, ref _scaleCoefficients, ref _scaleIntercept);
            UpdateLinearPredictor(x, etaScale, _scaleCoefficients, _scaleIntercept, useExpLink: true);
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
            // Simplified: constant weight, linear update on the current shape predictor
            var weights = Engine.Fill(n, NumOps.One);
            var z = new Vector<T>(etaShape);

            if (_shapeCoefficients != null)
            {
                UpdateCoefficients(x, z, weights, ref _shapeCoefficients, ref _shapeIntercept);
                UpdateLinearPredictor(x, etaShape, _shapeCoefficients, _shapeIntercept, useExpLink: true);
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

        // Compute X'WX and X'Wz with an intercept column first
        var design = (x.Columns == p ? x : x.GetSubMatrix(0, 0, n, p)).AddConstantColumn(NumOps.One);
        var designTensor = Tensor<T>.FromMatrix(design);
        var weightColumn = Engine.TensorBroadcastTo(new Tensor<T>(new[] { n, 1 }, weights), new[] { n, p + 1 });
        var weightedDesign = Engine.TensorMultiply(designTensor, weightColumn).ToMatrix();
        var designTranspose = Engine.MatrixTranspose(design);
        var xtwx = Engine.MatrixMultiply(designTranspose, weightedDesign);
        var xtwz = Engine.MatrixVectorMultiply(designTranspose, Engine.Multiply(weights, z));

        // Add regularization (not on the intercept)
        T lambda = _options.UseRegularization ? NumOps.FromDouble(_options.RegularizationStrength) : NumOps.Zero;
        var penalty = Vector<T>.Concatenate(new[] { Engine.FillZero<T>(1), Engine.Fill(p, lambda) });
        xtwx = Engine.MatrixAdd(xtwx, Matrix<T>.CreateDiagonal(penalty));

        // Solve the system
        var solution = SolveLinearSystem(xtwx, xtwz, p + 1);

        // Update parameters with learning rate
        T lr = NumOps.FromDouble(_options.LearningRate);
        T oneMinusLr = NumOps.Subtract(NumOps.One, lr);

        intercept = NumOps.Add(
            NumOps.Multiply(oneMinusLr, intercept),
            NumOps.Multiply(lr, solution[0]));

        coefficients = Engine.Add(
            Engine.Multiply(coefficients, oneMinusLr),
            Engine.Multiply(solution.GetSubVector(1, p), lr));
    }

    /// <summary>
    /// Updates the linear predictor for all samples.
    /// </summary>
    private void UpdateLinearPredictor(Matrix<T> x, Vector<T> eta, Vector<T>? coefficients, T intercept, bool useExpLink = false)
    {
        int n = x.Rows;
        var values = Engine.Fill(n, intercept);
        if (coefficients != null && n > 0 && _numFeatures > 0)
        {
            var design = x.Columns == _numFeatures ? x : x.GetSubMatrix(0, 0, n, _numFeatures);
            values = Engine.Add(values, Engine.MatrixVectorMultiply(design, coefficients.GetSubVector(0, _numFeatures)));
        }

        // For log-link parameters (scale sigma, shape nu) the predictor feeds exp(.); bound it so
        // exp() cannot overflow/underflow and destabilise the IRLS working weights (see the
        // MinLogScale/MaxLogScale rationale on the class fields).
        if (useExpLink)
        {
            values = Engine.Clamp(values, NumOps.FromDouble(MinLogScale), NumOps.FromDouble(MaxLogScale));
        }

        // The predictor is updated in place (bulk copy) because the caller keeps the same vector
        values.AsSpan().CopyTo(eta.AsWritableSpan());
    }

    /// <summary>
    /// Clamps a log-link (scale/shape) linear-predictor value to the numerically safe range
    /// [<see cref="MinLogScale"/>, <see cref="MaxLogScale"/>].
    /// </summary>
    private T ClampLogScale(T value)
    {
        double v = NumOps.ToDouble(value);
        if (v < MinLogScale) return NumOps.FromDouble(MinLogScale);
        if (v > MaxLogScale) return NumOps.FromDouble(MaxLogScale);
        return value;
    }

    /// <summary>
    /// Predicts a single parameter value for one sample.
    /// </summary>
    private T PredictParameter(Vector<T> sample, Vector<T>? coefficients, T intercept, bool useExpLink)
    {
        T val = intercept;

        if (coefficients != null)
        {
            val = NumOps.Add(val, Engine.DotProduct(coefficients, sample));
        }

        if (useExpLink)
        {
            val = NumOps.Exp(ClampLogScale(val));
        }

        return val;
    }

    /// <summary>
    /// Computes the deviance (negative log-likelihood).
    /// </summary>
    private double ComputeDeviance(Vector<T> y, Vector<T> etaLocation, Vector<T> etaScale, Vector<T> etaShape)
    {
        if (y.Length == 0)
        {
            return 0;
        }

        // Log-density of every observation under its own fitted distribution
        var scales = Engine.Exp(etaScale);
        var shapes = Engine.Exp(etaShape);
        var logPdfs = etaLocation
            .Zip(scales, (location, scale) => (location, scale))
            .Zip(shapes, (ls, shape) => (ls.location, ls.scale, shape))
            .Zip(y, (parameters, observation) =>
                CreateDistribution(parameters.location, parameters.scale, parameters.shape).LogPdf(observation))
            .ToList();

        // The convergence check works in double; the single total is converted once.
        double deviance = -NumOps.ToDouble(Engine.Sum(new Vector<T>(logPdfs)));
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
        // Augmented [A | b]
        var augmented = Engine.TensorConcatenate(
            new[]
            {
                Tensor<T>.FromMatrix(a.GetSubMatrix(0, 0, n, n)),
                new Tensor<T>(new[] { n, 1 }, b.GetSubVector(0, n))
            }, 1).ToMatrix();

        T pivotThreshold = NumOps.FromDouble(1e-10);

        // Gauss-Jordan elimination with partial pivoting
        for (int col = 0; col < n; col++)
        {
            augmented = RegressionEngineOps.GaussJordanStep(augmented, col, n, pivotThreshold);
        }

        return augmented.GetColumn(n);
    }

    /// <inheritdoc/>
    protected override Task CalculateFeatureImportancesAsync(int featureCount)
    {
        var importances = new Vector<T>(_numFeatures);

        // Combine importances from all parameter models: sum of |coefficient| per feature
        foreach (var coefficients in new[] { _locationCoefficients, _scaleCoefficients, _shapeCoefficients })
        {
            if (coefficients != null && _numFeatures > 0)
            {
                importances = Engine.Add(importances, Engine.Abs(coefficients.GetSubVector(0, _numFeatures)));
            }
        }

        // Normalize
        T sum = _numFeatures > 0 ? Engine.Sum(importances) : NumOps.Zero;
        if (NumOps.GreaterThan(sum, NumOps.Zero))
        {
            importances = Engine.Divide(importances, sum);
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

}
