using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;

namespace AiDotNet.Regression;

/// <summary>
/// Implements Inverse Gaussian regression, a generalized linear model for positive continuous data
/// with variance proportional to the cube of the mean.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// The Inverse Gaussian distribution (also known as Wald distribution) is appropriate for modeling
/// positive continuous response variables with heavy right tails. It's commonly used for modeling
/// response times, waiting times, and first passage times.
/// </para>
/// <para>
/// The Inverse Gaussian distribution has two parameters: μ (mean) and λ (shape), with:
/// - Mean: μ
/// - Variance: μ³/λ = φ × μ³, where φ = 1/λ is the dispersion parameter
/// </para>
/// <para>
/// The model is fitted using iteratively reweighted least squares (IRLS), a form of maximum likelihood estimation.
/// </para>
/// <para>
/// For Beginners:
/// Inverse Gaussian regression is used when you're trying to predict positive continuous values that have
/// heavy right tails (meaning extreme large values are possible and have high variability). Common examples include:
/// - Response times in cognitive experiments
/// - Time until failure for mechanical systems
/// - First passage times in physics
/// - Waiting times in queuing systems
///
/// Compared to Gamma regression which has variance proportional to μ², Inverse Gaussian has variance
/// proportional to μ³, meaning it handles even heavier tails where large values are much more variable.
/// </para>
/// </remarks>
public class InverseGaussianRegression<T> : RegressionBase<T>
{
    /// <summary>
    /// Configuration options for the Inverse Gaussian regression model.
    /// </summary>
    private readonly InverseGaussianRegressionOptions<T> _options;

    /// <summary>
    /// The estimated dispersion parameter (φ = 1/λ).
    /// </summary>
    private T _dispersion;

    /// <summary>
    /// Gets the estimated dispersion parameter.
    /// </summary>
    /// <value>
    /// The dispersion parameter φ, which controls the variance relative to the mean cubed.
    /// For Inverse Gaussian distribution, Variance = φ × μ³.
    /// </value>
    /// <remarks>
    /// <para>
    /// The dispersion parameter is estimated after model fitting using the Pearson residuals.
    /// Smaller values indicate less spread in the data relative to the mean cubed.
    /// </para>
    /// <para>
    /// For Beginners:
    /// The dispersion parameter tells you how spread out your data is relative to its average cubed.
    /// A smaller dispersion means the predictions are more precise; a larger dispersion means
    /// there's more variability in the actual values around the predicted values.
    /// </para>
    /// </remarks>
    public T Dispersion => _dispersion;

    /// <summary>
    /// Initializes a new instance of the InverseGaussianRegression class with the specified options and regularization.
    /// </summary>
    /// <param name="options">Configuration options for the Inverse Gaussian regression model. If null, default options will be used.</param>
    /// <param name="regularization">Regularization method to prevent overfitting. If null, no regularization will be applied.</param>
    /// <remarks>
    /// <para>
    /// The constructor initializes the model with either the provided options or default settings.
    /// </para>
    /// <para>
    /// For Beginners:
    /// This constructor sets up the Inverse Gaussian regression model with your specified settings or uses
    /// default settings if none are provided. Regularization is an optional technique to prevent the model
    /// from becoming too complex and overfitting to the training data.
    /// </para>
    /// </remarks>
    public InverseGaussianRegression(InverseGaussianRegressionOptions<T>? options = null, IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, regularization)
    {
        _options = options ?? new InverseGaussianRegressionOptions<T>();
        _dispersion = NumOps.FromDouble(_options.InitialDispersion);
    }

    /// <summary>
    /// Trains the Inverse Gaussian regression model on the provided data.
    /// </summary>
    /// <param name="x">The input features matrix where each row is a training example and each column is a feature.</param>
    /// <param name="y">The target positive continuous values vector corresponding to each training example.</param>
    /// <remarks>
    /// <para>
    /// This method implements the iteratively reweighted least squares (IRLS) algorithm to fit the model.
    /// The steps are:
    /// 1. Initialize coefficients and intercept
    /// 2. For each iteration:
    ///    a. Compute the predicted mean (mu) using the current coefficients and link function
    ///    b. Compute the weights matrix (W) based on mu and link function
    ///    c. Compute the working response (z)
    ///    d. Solve the weighted least squares problem to get new coefficients
    ///    e. Check for convergence
    /// 3. Estimate the dispersion parameter from the residuals
    /// </para>
    /// <para>
    /// For Beginners:
    /// Training is the process where the model learns from your data. The algorithm starts with initial guesses
    /// for the coefficients and then iteratively improves them until they converge to the best values.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown when any target value is not positive.</exception>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        ValidationHelper<T>.ValidateInputData(x, y);
        ValidateInverseGaussianData(y);

        int numFeatures = x.Columns;
        int numSamples = x.Rows;
        Coefficients = new Vector<T>(numFeatures);
        Intercept = NumOps.Zero;

        // Initialize coefficients using mean for appropriate link
        double meanY = 0;
        for (int i = 0; i < numSamples; i++)
        {
            meanY += NumOps.ToDouble(y[i]);
        }
        meanY /= numSamples;

        // Set initial intercept based on link function
        Intercept = _options.LinkFunction switch
        {
            InverseGaussianLinkFunction.Log => NumOps.FromDouble(Math.Log(meanY)),
            InverseGaussianLinkFunction.InverseSquared => NumOps.FromDouble(-1.0 / (2.0 * meanY * meanY)),
            InverseGaussianLinkFunction.Inverse => NumOps.FromDouble(1.0 / meanY),
            InverseGaussianLinkFunction.Identity => NumOps.FromDouble(meanY),
            _ => NumOps.FromDouble(Math.Log(meanY))
        };

        Matrix<T> xWithIntercept = x.AddColumn(Vector<T>.CreateDefault(x.Rows, NumOps.One));

        for (int iteration = 0; iteration < _options.MaxIterations; iteration++)
        {
            Vector<T> currentCoefficients = new([.. Coefficients, Intercept]);
            Vector<T> eta = xWithIntercept.Multiply(currentCoefficients);
            Vector<T> mu = ApplyInverseLink(eta);

            // Ensure mu values are positive and not too small
            mu = ClampMu(mu);

            Matrix<T> w = ComputeWeights(mu);
            Vector<T> z = ComputeWorkingResponse(eta, y, mu);

            Matrix<T> xTw = xWithIntercept.Transpose().Multiply(w);
            Matrix<T> xTwx = xTw.Multiply(xWithIntercept);
            Vector<T> xTwz = xTw.Multiply(z);

            // Add ridge regularization to ensure numerical stability
            var minRegularization = 1e-10;
            var userStrength = Regularization?.GetOptions().Strength ?? 0.0;
            var effectiveStrength = NumOps.FromDouble(Math.Max(minRegularization, userStrength));
            for (int i = 0; i < xTwx.Rows; i++)
            {
                xTwx[i, i] = NumOps.Add(xTwx[i, i], effectiveStrength);
            }

            Vector<T> newCoefficients = MatrixSolutionHelper.SolveLinearSystem(xTwx, xTwz, _options.DecompositionType);

            // Apply regularization to the coefficients
            if (Regularization != null)
            {
                newCoefficients = Regularization.Regularize(newCoefficients);
            }

            if (HasConverged(currentCoefficients, newCoefficients))
            {
                break;
            }

            Coefficients = new Vector<T>([.. newCoefficients.Take(numFeatures)]);
            Intercept = newCoefficients[numFeatures];
        }

        // Estimate dispersion parameter using Pearson residuals
        EstimateDispersion(x, y);
    }

    /// <summary>
    /// Validates that all target values are positive, as required for Inverse Gaussian regression.
    /// </summary>
    /// <param name="y">The target values vector to validate.</param>
    /// <exception cref="ArgumentException">Thrown when any value is not positive.</exception>
    /// <remarks>
    /// <para>
    /// The Inverse Gaussian distribution is only defined for positive values.
    /// </para>
    /// <para>
    /// For Beginners:
    /// Inverse Gaussian regression can only work with positive numbers (greater than zero).
    /// </para>
    /// </remarks>
    private void ValidateInverseGaussianData(Vector<T> y)
    {
        for (int i = 0; i < y.Length; i++)
        {
            double value = NumOps.ToDouble(y[i]);
            if (value <= 0)
            {
                throw new ArgumentException($"Inverse Gaussian regression requires strictly positive target values. Found value {value} at index {i}.");
            }
        }
    }

    /// <summary>
    /// Applies the inverse link function to convert the linear predictor to the mean.
    /// </summary>
    /// <param name="eta">The linear predictor (X × β).</param>
    /// <returns>The predicted mean values μ.</returns>
    /// <remarks>
    /// <para>
    /// The inverse link function converts from the linear scale to the response scale:
    /// - Log link: μ = exp(η)
    /// - InverseSquared link: μ = 1/sqrt(-2η)
    /// - Inverse link: μ = 1/η
    /// - Identity link: μ = η
    /// </para>
    /// <para>
    /// For Beginners:
    /// The link function transforms predictions to ensure they're always positive.
    /// </para>
    /// </remarks>
    private Vector<T> ApplyInverseLink(Vector<T> eta)
    {
        return _options.LinkFunction switch
        {
            InverseGaussianLinkFunction.Log => eta.Transform(NumOps.Exp),
            InverseGaussianLinkFunction.InverseSquared => eta.Transform(v =>
            {
                double etaVal = NumOps.ToDouble(v);
                // μ = 1/sqrt(-2η), but need to handle sign carefully
                if (etaVal >= 0) return NumOps.FromDouble(1e10); // Large positive if invalid
                return NumOps.FromDouble(1.0 / Math.Sqrt(-2.0 * etaVal));
            }),
            InverseGaussianLinkFunction.Inverse => eta.Transform(v => NumOps.Divide(NumOps.One, v)),
            InverseGaussianLinkFunction.Identity => eta.Clone(),
            _ => eta.Transform(NumOps.Exp)
        };
    }

    /// <summary>
    /// Clamps the mean values to ensure they're positive and numerically stable.
    /// </summary>
    /// <param name="mu">The mean values to clamp.</param>
    /// <returns>The clamped mean values.</returns>
    /// <remarks>
    /// <para>
    /// This prevents numerical issues when mu gets too close to zero or becomes negative.
    /// </para>
    /// <para>
    /// For Beginners:
    /// During the iterative fitting process, the predicted values might temporarily become
    /// very small or even slightly negative due to numerical approximations. This method
    /// ensures they stay positive and reasonably sized.
    /// </para>
    /// </remarks>
    private Vector<T> ClampMu(Vector<T> mu)
    {
        var result = new Vector<T>(mu.Length);
        double minValue = 1e-10;
        double maxValue = 1e10;

        for (int i = 0; i < mu.Length; i++)
        {
            double value = NumOps.ToDouble(mu[i]);
            value = Math.Max(minValue, Math.Min(maxValue, value));
            result[i] = NumOps.FromDouble(value);
        }

        return result;
    }

    /// <summary>
    /// Computes the weights matrix for the iteratively reweighted least squares algorithm.
    /// </summary>
    /// <param name="mu">The vector of predicted mean values.</param>
    /// <returns>A diagonal matrix of weights.</returns>
    /// <remarks>
    /// <para>
    /// For Inverse Gaussian regression, the weights depend on the link function.
    /// The weight formula is: W = 1 / (V(μ) × (g'(μ))²)
    /// where V(μ) = μ³ is the variance function for Inverse Gaussian.
    ///
    /// - Log link: g'(μ) = 1/μ, so W = 1 / (μ³ × (1/μ)²) = 1/μ
    /// - InverseSquared link: g'(μ) = 1/μ³, so W = 1 / (μ³ × (1/μ³)²) = μ³
    /// - Inverse link: g'(μ) = -1/μ², so W = 1 / (μ³ × (1/μ²)²) = μ
    /// - Identity link: g'(μ) = 1, so W = 1/μ³
    /// </para>
    /// <para>
    /// For Beginners:
    /// Each observation is given a weight that depends on its current predicted value.
    /// </para>
    /// </remarks>
    private Matrix<T> ComputeWeights(Vector<T> mu)
    {
        var weights = new Vector<T>(mu.Length);

        for (int i = 0; i < mu.Length; i++)
        {
            double muVal = NumOps.ToDouble(mu[i]);
            double weight = _options.LinkFunction switch
            {
                // W = 1 / (V(μ) × (g'(μ))²) = 1 / (μ³ × (1/μ)²) = 1/μ
                InverseGaussianLinkFunction.Log => 1.0 / muVal,
                // W = 1 / (V(μ) × (g'(μ))²) = 1 / (μ³ × (1/μ³)²) = μ³
                InverseGaussianLinkFunction.InverseSquared => muVal * muVal * muVal,
                // W = 1 / (V(μ) × (g'(μ))²) = 1 / (μ³ × (1/μ²)²) = μ
                InverseGaussianLinkFunction.Inverse => muVal,
                // W = 1 / (V(μ) × (g'(μ))²) = 1 / (μ³ × 1) = 1/μ³
                InverseGaussianLinkFunction.Identity => 1.0 / (muVal * muVal * muVal),
                _ => 1.0 / muVal
            };
            weights[i] = NumOps.FromDouble(weight);
        }

        return Matrix<T>.CreateDiagonal(weights);
    }

    /// <summary>
    /// Computes the working response for the iteratively reweighted least squares algorithm.
    /// </summary>
    /// <param name="eta">The linear predictor.</param>
    /// <param name="y">The target values vector.</param>
    /// <param name="mu">The vector of predicted mean values.</param>
    /// <returns>The working response vector.</returns>
    /// <remarks>
    /// <para>
    /// The working response is computed as: z = η + (y - μ) × g'(μ)
    /// where g'(μ) is the derivative of the link function.
    ///
    /// - Log link: z = η + (y - μ)/μ
    /// - InverseSquared link: z = η + (y - μ)/(μ³)
    /// - Inverse link: z = η - (y - μ)/μ²
    /// - Identity link: z = y
    /// </para>
    /// <para>
    /// For Beginners:
    /// The working response is an adjusted version of the target variable that helps convergence.
    /// </para>
    /// </remarks>
    private Vector<T> ComputeWorkingResponse(Vector<T> eta, Vector<T> y, Vector<T> mu)
    {
        var z = new Vector<T>(eta.Length);

        for (int i = 0; i < eta.Length; i++)
        {
            double etaVal = NumOps.ToDouble(eta[i]);
            double yVal = NumOps.ToDouble(y[i]);
            double muVal = NumOps.ToDouble(mu[i]);
            double diff = yVal - muVal;

            double zVal = _options.LinkFunction switch
            {
                // g'(μ) = 1/μ, so z = η + (y - μ)/μ
                InverseGaussianLinkFunction.Log => etaVal + diff / muVal,
                // g'(μ) = 1/μ³, so z = η + (y - μ)/μ³
                InverseGaussianLinkFunction.InverseSquared => etaVal + diff / (muVal * muVal * muVal),
                // g'(μ) = -1/μ², so z = η - (y - μ)/μ²
                InverseGaussianLinkFunction.Inverse => etaVal - diff / (muVal * muVal),
                // g'(μ) = 1, so z = η + (y - μ) = y
                InverseGaussianLinkFunction.Identity => yVal,
                _ => etaVal + diff / muVal
            };

            z[i] = NumOps.FromDouble(zVal);
        }

        return z;
    }

    /// <summary>
    /// Checks if the algorithm has converged by comparing the change in coefficients.
    /// </summary>
    /// <param name="oldCoefficients">The coefficients from the previous iteration.</param>
    /// <param name="newCoefficients">The coefficients from the current iteration.</param>
    /// <returns>True if the change is less than the tolerance; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// Convergence is determined by calculating the L2 norm between the old and new coefficients.
    /// </para>
    /// <para>
    /// For Beginners:
    /// This method checks if the model has found the best solution by measuring how much
    /// the coefficients changed in the last iteration.
    /// </para>
    /// </remarks>
    private bool HasConverged(Vector<T> oldCoefficients, Vector<T> newCoefficients)
    {
        T diff = oldCoefficients.Subtract(newCoefficients).L2Norm();
        return NumOps.LessThan(diff, NumOps.FromDouble(_options.Tolerance));
    }

    /// <summary>
    /// Estimates the dispersion parameter using Pearson residuals after model fitting.
    /// </summary>
    /// <param name="x">The input features matrix.</param>
    /// <param name="y">The target values vector.</param>
    /// <remarks>
    /// <para>
    /// The dispersion parameter is estimated as:
    /// φ = (1/(n-p)) × Σ((y_i - μ_i)²/μ_i³)
    /// where n is the number of samples and p is the number of parameters.
    /// </para>
    /// <para>
    /// For Beginners:
    /// The dispersion parameter measures how much the actual values vary compared to predictions.
    /// </para>
    /// </remarks>
    private void EstimateDispersion(Matrix<T> x, Vector<T> y)
    {
        Vector<T> predictions = Predict(x);
        int n = y.Length;
        int p = Coefficients.Length + 1;

        double sumPearsonResidualsSq = 0;
        for (int i = 0; i < n; i++)
        {
            double yVal = NumOps.ToDouble(y[i]);
            double muVal = NumOps.ToDouble(predictions[i]);
            // For Inverse Gaussian, variance = μ³, so Pearson residual = (y - μ) / sqrt(μ³)
            double variance = muVal * muVal * muVal;
            double pearsonResidualSq = (yVal - muVal) * (yVal - muVal) / variance;
            sumPearsonResidualsSq += pearsonResidualSq;
        }

        double dispersion = sumPearsonResidualsSq / Math.Max(1, n - p);
        _dispersion = NumOps.FromDouble(dispersion);
    }

    /// <summary>
    /// Makes predictions for the given input data.
    /// </summary>
    /// <param name="x">The input features matrix where each row is an example and each column is a feature.</param>
    /// <returns>A vector of predicted mean values for each input example.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the linear predictor and applies the inverse link function.
    /// </para>
    /// <para>
    /// For Beginners:
    /// After training, this method is used to make predictions on new data.
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> x)
    {
        Matrix<T> xWithIntercept = x.AddColumn(Vector<T>.CreateDefault(x.Rows, NumOps.One));
        Vector<T> coefficientsWithIntercept = new(Coefficients.Length + 1);

        for (int i = 0; i < Coefficients.Length; i++)
        {
            coefficientsWithIntercept[i] = Coefficients[i];
        }
        coefficientsWithIntercept[Coefficients.Length] = Intercept;

        Vector<T> eta = xWithIntercept.Multiply(coefficientsWithIntercept);
        Vector<T> mu = ApplyInverseLink(eta);

        return ClampMu(mu);
    }

    /// <summary>
    /// Serializes the model to a byte array.
    /// </summary>
    /// <returns>A byte array containing the serialized model data.</returns>
    /// <remarks>
    /// <para>
    /// Serializes the model including options, coefficients, and dispersion parameter.
    /// </para>
    /// <para>
    /// For Beginners:
    /// Serialization saves the model so you can load it later without retraining.
    /// </para>
    /// </remarks>
    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Serialize base class data
        byte[] baseData = base.Serialize();
        writer.Write(baseData.Length);
        writer.Write(baseData);

        // Serialize InverseGaussianRegression specific options
        writer.Write(_options.MaxIterations);
        writer.Write(_options.Tolerance);
        writer.Write((int)_options.LinkFunction);
        writer.Write((int)_options.DecompositionType);
        writer.Write(_options.InitialDispersion);
        writer.Write(NumOps.ToDouble(_dispersion));

        return ms.ToArray();
    }

    /// <summary>
    /// Deserializes the model from a byte array.
    /// </summary>
    /// <param name="data">The byte array containing the serialized model data.</param>
    /// <remarks>
    /// <para>
    /// Reconstructs the model's state from the serialized data.
    /// </para>
    /// <para>
    /// For Beginners:
    /// Deserialization loads a previously saved model.
    /// </para>
    /// </remarks>
    public override void Deserialize(byte[] data)
    {
        using var ms = new MemoryStream(data);
        using var reader = new BinaryReader(ms);

        // Deserialize base class data
        int baseDataLength = reader.ReadInt32();
        byte[] baseData = reader.ReadBytes(baseDataLength);
        base.Deserialize(baseData);

        // Deserialize InverseGaussianRegression specific options
        _options.MaxIterations = reader.ReadInt32();
        _options.Tolerance = reader.ReadDouble();
        _options.LinkFunction = (InverseGaussianLinkFunction)reader.ReadInt32();
        _options.DecompositionType = (MatrixDecompositionType)reader.ReadInt32();
        _options.InitialDispersion = reader.ReadDouble();
        _dispersion = NumOps.FromDouble(reader.ReadDouble());
    }

    /// <summary>
    /// Gets the type of the model.
    /// </summary>
    /// <returns>The model type identifier for Inverse Gaussian regression.</returns>
    /// <remarks>
    /// <para>
    /// This method is used for model identification and serialization purposes.
    /// </para>
    /// <para>
    /// For Beginners:
    /// Returns an identifier indicating this is an Inverse Gaussian regression model.
    /// </para>
    /// </remarks>
    protected override ModelType GetModelType()
    {
        return ModelType.InverseGaussianRegression;
    }

    /// <summary>
    /// Creates a new instance of the Inverse Gaussian Regression model with the same configuration.
    /// </summary>
    /// <returns>A new instance of the Inverse Gaussian Regression model.</returns>
    /// <remarks>
    /// <para>
    /// Creates a deep copy of the current model, including all options and coefficients.
    /// </para>
    /// <para>
    /// For Beginners:
    /// This method creates an exact copy of your trained model.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        var newOptions = new InverseGaussianRegressionOptions<T>
        {
            MaxIterations = _options.MaxIterations,
            Tolerance = _options.Tolerance,
            LinkFunction = _options.LinkFunction,
            DecompositionType = _options.DecompositionType,
            InitialDispersion = _options.InitialDispersion
        };

        var newModel = new InverseGaussianRegression<T>(newOptions, Regularization);

        // Copy coefficients if they exist
        if (Coefficients != null)
        {
            newModel.Coefficients = Coefficients.Clone();
        }

        // Copy the intercept and dispersion
        newModel.Intercept = Intercept;
        newModel._dispersion = _dispersion;

        return newModel;
    }
}
