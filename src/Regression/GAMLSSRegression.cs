using AiDotNet.Distributions;
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
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
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
    public override async Task TrainAsync(Matrix<T> x, Vector<T> y)
    {
        _numFeatures = x.Columns;
        int n = x.Rows;

        // Initialize parameters
        InitializeParameters(y);

        // Initialize linear predictors
        var etaLocation = new T[n];
        var etaScale = new T[n];
        var etaShape = new T[n];

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
            predictions[i] = distributions[i].Mean;
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
        double mean = 0, variance = 0;
        for (int i = 0; i < y.Length; i++)
        {
            mean += NumOps.ToDouble(y[i]);
        }
        mean /= y.Length;

        for (int i = 0; i < y.Length; i++)
        {
            double diff = NumOps.ToDouble(y[i]) - mean;
            variance += diff * diff;
        }
        variance /= y.Length;
        variance = Math.Max(variance, 1e-6);

        _locationIntercept = NumOps.FromDouble(mean);
        _scaleIntercept = NumOps.FromDouble(Math.Log(Math.Sqrt(variance)));  // Log link for scale

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
    private void FitLocationParameter(Matrix<T> x, Vector<T> y, T[] etaLocation, T[] etaScale, T[] etaShape)
    {
        int n = x.Rows;

        for (int iter = 0; iter < _options.MaxInnerIterations; iter++)
        {
            // Compute working weights and adjusted dependent variable
            var weights = new T[n];
            var z = new T[n];

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
            UpdateCoefficients(x, z, weights, ref _locationCoefficients!, ref _locationIntercept);
            UpdateLinearPredictor(x, etaLocation, _locationCoefficients, _locationIntercept);
        }
    }

    /// <summary>
    /// Fits the scale parameter using IRLS.
    /// </summary>
    private void FitScaleParameter(Matrix<T> x, Vector<T> y, T[] etaLocation, T[] etaScale, T[] etaShape)
    {
        int n = x.Rows;

        for (int iter = 0; iter < _options.MaxInnerIterations; iter++)
        {
            var weights = new T[n];
            var z = new T[n];

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

            UpdateCoefficients(x, z, weights, ref _scaleCoefficients!, ref _scaleIntercept);
            UpdateLinearPredictor(x, etaScale, _scaleCoefficients, _scaleIntercept);
        }
    }

    /// <summary>
    /// Fits the shape parameter using IRLS.
    /// </summary>
    private void FitShapeParameter(Matrix<T> x, Vector<T> y, T[] etaLocation, T[] etaScale, T[] etaShape)
    {
        int n = x.Rows;

        for (int iter = 0; iter < _options.MaxInnerIterations; iter++)
        {
            var weights = new T[n];
            var z = new T[n];

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
    private void UpdateCoefficients(Matrix<T> x, T[] z, T[] weights, ref Vector<T> coefficients, ref T intercept)
    {
        int n = x.Rows;
        int p = _numFeatures;

        // Compute X'WX and X'Wz with intercept
        var xtwx = new double[p + 1, p + 1];
        var xtwz = new double[p + 1];

        for (int i = 0; i < n; i++)
        {
            double w = NumOps.ToDouble(weights[i]);
            double zi = NumOps.ToDouble(z[i]);

            // Intercept term
            xtwx[0, 0] += w;
            xtwz[0] += w * zi;

            for (int j = 0; j < p; j++)
            {
                double xij = NumOps.ToDouble(x[i, j]);
                xtwx[0, j + 1] += w * xij;
                xtwx[j + 1, 0] += w * xij;
                xtwz[j + 1] += w * xij * zi;

                for (int k = 0; k <= j; k++)
                {
                    double xik = NumOps.ToDouble(x[i, k]);
                    xtwx[j + 1, k + 1] += w * xij * xik;
                    if (k < j)
                        xtwx[k + 1, j + 1] = xtwx[j + 1, k + 1];
                }
            }
        }

        // Add regularization
        double lambda = _options.UseRegularization ? _options.RegularizationStrength : 0;
        for (int j = 1; j <= p; j++)
        {
            xtwx[j, j] += lambda;
        }

        // Solve the system
        var solution = SolveLinearSystem(xtwx, xtwz, p + 1);

        // Update parameters with learning rate
        intercept = NumOps.Add(
            NumOps.Multiply(NumOps.FromDouble(1 - _options.LearningRate), intercept),
            NumOps.FromDouble(_options.LearningRate * solution[0]));

        for (int j = 0; j < p; j++)
        {
            coefficients[j] = NumOps.Add(
                NumOps.Multiply(NumOps.FromDouble(1 - _options.LearningRate), coefficients[j]),
                NumOps.FromDouble(_options.LearningRate * solution[j + 1]));
        }
    }

    /// <summary>
    /// Updates the linear predictor for all samples.
    /// </summary>
    private void UpdateLinearPredictor(Matrix<T> x, T[] eta, Vector<T>? coefficients, T intercept, bool useExpLink = false)
    {
        for (int i = 0; i < x.Rows; i++)
        {
            double val = NumOps.ToDouble(intercept);

            if (coefficients != null)
            {
                for (int j = 0; j < _numFeatures; j++)
                {
                    val += NumOps.ToDouble(coefficients[j]) * NumOps.ToDouble(x[i, j]);
                }
            }

            eta[i] = NumOps.FromDouble(val);
        }
    }

    /// <summary>
    /// Predicts a single parameter value for one sample.
    /// </summary>
    private T PredictParameter(Vector<T> sample, Vector<T>? coefficients, T intercept, bool useExpLink)
    {
        double val = NumOps.ToDouble(intercept);

        if (coefficients != null)
        {
            for (int j = 0; j < _numFeatures; j++)
            {
                val += NumOps.ToDouble(coefficients[j]) * NumOps.ToDouble(sample[j]);
            }
        }

        if (useExpLink)
        {
            val = Math.Exp(val);
        }

        return NumOps.FromDouble(val);
    }

    /// <summary>
    /// Computes the deviance (negative log-likelihood).
    /// </summary>
    private double ComputeDeviance(Vector<T> y, T[] etaLocation, T[] etaScale, T[] etaShape)
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
        return _options.DistributionFamily switch
        {
            GAMLSSDistributionFamily.Normal => new NormalDistribution<T>(location, NumOps.Multiply(scale, scale)),
            GAMLSSDistributionFamily.Laplace => new LaplaceDistribution<T>(location, scale),
            GAMLSSDistributionFamily.StudentT => new StudentTDistribution<T>(location, scale, shape),
            GAMLSSDistributionFamily.Gamma => new GammaDistribution<T>(shape, NumOps.Divide(shape, location)),
            GAMLSSDistributionFamily.LogNormal => new LogNormalDistribution<T>(location, NumOps.Multiply(scale, scale)),
            GAMLSSDistributionFamily.Poisson => new PoissonDistribution<T>(location),
            GAMLSSDistributionFamily.NegativeBinomial => new NegativeBinomialDistribution<T>(location, shape),
            _ => new NormalDistribution<T>(location, NumOps.Multiply(scale, scale))
        };
    }

    /// <summary>
    /// Solves a linear system using Gaussian elimination.
    /// </summary>
    private double[] SolveLinearSystem(double[,] a, double[] b, int n)
    {
        var augmented = new double[n, n + 1];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                augmented[i, j] = a[i, j];
            }
            augmented[i, n] = b[i];
        }

        // Forward elimination with partial pivoting
        for (int col = 0; col < n; col++)
        {
            int maxRow = col;
            for (int row = col + 1; row < n; row++)
            {
                if (Math.Abs(augmented[row, col]) > Math.Abs(augmented[maxRow, col]))
                {
                    maxRow = row;
                }
            }

            for (int j = 0; j <= n; j++)
            {
                (augmented[col, j], augmented[maxRow, j]) = (augmented[maxRow, j], augmented[col, j]);
            }

            double pivot = augmented[col, col];
            if (Math.Abs(pivot) < 1e-10) pivot = 1e-10;

            for (int j = 0; j <= n; j++)
            {
                augmented[col, j] /= pivot;
            }

            for (int row = 0; row < n; row++)
            {
                if (row != col)
                {
                    double factor = augmented[row, col];
                    for (int j = 0; j <= n; j++)
                    {
                        augmented[row, j] -= factor * augmented[col, j];
                    }
                }
            }
        }

        var solution = new double[n];
        for (int i = 0; i < n; i++)
        {
            solution[i] = augmented[i, n];
        }

        return solution;
    }

    /// <inheritdoc/>
    protected override Task CalculateFeatureImportancesAsync(int featureCount)
    {
        var importances = new T[_numFeatures];

        // Combine importances from all parameter models
        for (int f = 0; f < _numFeatures; f++)
        {
            double importance = 0;

            if (_locationCoefficients != null)
            {
                importance += Math.Abs(NumOps.ToDouble(_locationCoefficients[f]));
            }
            if (_scaleCoefficients != null)
            {
                importance += Math.Abs(NumOps.ToDouble(_scaleCoefficients[f]));
            }
            if (_shapeCoefficients != null)
            {
                importance += Math.Abs(NumOps.ToDouble(_shapeCoefficients[f]));
            }

            importances[f] = NumOps.FromDouble(importance);
        }

        // Normalize
        double sum = importances.Sum(x => NumOps.ToDouble(x));
        if (sum > 0)
        {
            for (int f = 0; f < _numFeatures; f++)
            {
                importances[f] = NumOps.Divide(importances[f], NumOps.FromDouble(sum));
            }
        }

        FeatureImportances = new Vector<T>(importances);
        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.GAMLSS,
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
}
