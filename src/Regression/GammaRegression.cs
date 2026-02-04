using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;

namespace AiDotNet.Regression;

/// <summary>
/// Implements Gamma regression, a generalized linear model for positive continuous data with right-skewed distributions.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Gamma regression is appropriate when the response variable is positive continuous and often right-skewed,
/// with variance that increases with the mean. It's commonly used for modeling durations, costs, and other
/// positive quantities where the coefficient of variation is approximately constant.
/// </para>
/// <para>
/// The Gamma distribution has two parameters: shape (k) and scale (θ), with:
/// - Mean: μ = k × θ
/// - Variance: μ²/k = φ × μ², where φ = 1/k is the dispersion parameter
/// </para>
/// <para>
/// The model is fitted using iteratively reweighted least squares (IRLS), a form of maximum likelihood estimation.
/// </para>
/// <para>
/// For Beginners:
/// Gamma regression is used when you're trying to predict positive continuous values that are often right-skewed
/// (meaning most values are small but some are very large). Common examples include:
/// - Insurance claim amounts
/// - Hospital length of stay
/// - Income levels
/// - Time until an event occurs
/// - Costs and prices
///
/// Unlike linear regression which can predict negative values, Gamma regression ensures predictions are always
/// positive. It also handles the common pattern where larger values tend to be more variable.
/// </para>
/// </remarks>
public class GammaRegression<T> : RegressionBase<T>
{
    /// <summary>
    /// Configuration options for the Gamma regression model.
    /// </summary>
    private readonly GammaRegressionOptions<T> _options;

    /// <summary>
    /// The estimated dispersion parameter (φ = 1/shape).
    /// </summary>
    private T _dispersion;

    /// <summary>
    /// Gets the estimated dispersion parameter.
    /// </summary>
    /// <value>
    /// The dispersion parameter φ, which controls the variance relative to the mean.
    /// For Gamma distribution, Variance = φ × μ².
    /// </value>
    /// <remarks>
    /// <para>
    /// The dispersion parameter is estimated after model fitting using the Pearson residuals.
    /// Smaller values indicate less spread in the data relative to the mean.
    /// </para>
    /// <para>
    /// For Beginners:
    /// The dispersion parameter tells you how spread out your data is relative to its average.
    /// A smaller dispersion means the predictions are more precise; a larger dispersion means
    /// there's more variability in the actual values around the predicted values.
    /// </para>
    /// </remarks>
    public T Dispersion => _dispersion;

    /// <summary>
    /// Initializes a new instance of the GammaRegression class with the specified options and regularization.
    /// </summary>
    /// <param name="options">Configuration options for the Gamma regression model. If null, default options will be used.</param>
    /// <param name="regularization">Regularization method to prevent overfitting. If null, no regularization will be applied.</param>
    /// <remarks>
    /// <para>
    /// The constructor initializes the model with either the provided options or default settings.
    /// </para>
    /// <para>
    /// For Beginners:
    /// This constructor sets up the Gamma regression model with your specified settings or uses
    /// default settings if none are provided. Regularization is an optional technique to prevent the model
    /// from becoming too complex and overfitting to the training data.
    /// </para>
    /// </remarks>
    public GammaRegression(GammaRegressionOptions<T>? options = null, IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, regularization)
    {
        _options = options ?? new GammaRegressionOptions<T>();
        _dispersion = NumOps.FromDouble(_options.InitialDispersion);
    }

    /// <summary>
    /// Trains the Gamma regression model on the provided data.
    /// </summary>
    /// <param name="x">The input features matrix where each row is a training example and each column is a feature.</param>
    /// <param name="y">The target positive continuous values vector corresponding to each training example.</param>
    /// <remarks>
    /// <para>
    /// This method implements the iteratively reweighted least squares (IRLS) algorithm to fit the Gamma regression model.
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
    /// for the coefficients and then iteratively improves them until they converge to the best values. At each step,
    /// it calculates predicted values, compares them to the actual values, and adjusts the coefficients to reduce
    /// the difference. This process continues until the changes become very small (convergence) or until a maximum
    /// number of iterations is reached.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown when any target value is not positive.</exception>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        ValidationHelper<T>.ValidateInputData(x, y);
        ValidateGammaData(y);

        int numFeatures = x.Columns;
        int numSamples = x.Rows;
        Coefficients = new Vector<T>(numFeatures);
        Intercept = NumOps.Zero;

        // Initialize coefficients using log of mean for log link
        double meanY = 0;
        for (int i = 0; i < numSamples; i++)
        {
            meanY += NumOps.ToDouble(y[i]);
        }
        meanY /= numSamples;

        // Set initial intercept based on link function
        Intercept = _options.LinkFunction switch
        {
            GammaLinkFunction.Log => NumOps.FromDouble(Math.Log(meanY)),
            GammaLinkFunction.Inverse => NumOps.FromDouble(1.0 / meanY),
            GammaLinkFunction.Identity => NumOps.FromDouble(meanY),
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
    /// Validates that all target values are positive, as required for Gamma regression.
    /// </summary>
    /// <param name="y">The target values vector to validate.</param>
    /// <exception cref="ArgumentException">Thrown when any value is not positive.</exception>
    /// <remarks>
    /// <para>
    /// The Gamma distribution is only defined for positive values. This validation ensures
    /// the data is appropriate for Gamma regression.
    /// </para>
    /// <para>
    /// For Beginners:
    /// Gamma regression can only work with positive numbers (greater than zero). If your data
    /// contains zero or negative values, you'll need to transform it or use a different model.
    /// </para>
    /// </remarks>
    private void ValidateGammaData(Vector<T> y)
    {
        for (int i = 0; i < y.Length; i++)
        {
            double value = NumOps.ToDouble(y[i]);
            if (value <= 0)
            {
                throw new ArgumentException($"Gamma regression requires strictly positive target values. Found value {value} at index {i}.");
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
    /// - Inverse link: μ = 1/η
    /// - Identity link: μ = η
    /// </para>
    /// <para>
    /// For Beginners:
    /// The link function transforms predictions to ensure they're always positive.
    /// The log link is most common because it naturally ensures positive predictions
    /// through the exponential function.
    /// </para>
    /// </remarks>
    private Vector<T> ApplyInverseLink(Vector<T> eta)
    {
        return _options.LinkFunction switch
        {
            GammaLinkFunction.Log => eta.Transform(NumOps.Exp),
            GammaLinkFunction.Inverse => eta.Transform(v => NumOps.Divide(NumOps.One, v)),
            GammaLinkFunction.Identity => eta.Clone(),
            _ => eta.Transform(NumOps.Exp)
        };
    }

    /// <summary>
    /// Applies the link function to convert the mean to the linear predictor scale.
    /// </summary>
    /// <param name="mu">The mean values.</param>
    /// <returns>The linear predictor η.</returns>
    /// <remarks>
    /// <para>
    /// The link function maps from the response scale to the linear predictor scale:
    /// - Log link: η = log(μ)
    /// - Inverse link: η = 1/μ
    /// - Identity link: η = μ
    /// </para>
    /// <para>
    /// For Beginners:
    /// The link function is the mathematical transformation that connects your predicted
    /// values to the linear combination of features. It's the "bridge" between the feature
    /// weights and the actual predictions.
    /// </para>
    /// </remarks>
    private Vector<T> ApplyLink(Vector<T> mu)
    {
        return _options.LinkFunction switch
        {
            GammaLinkFunction.Log => mu.Transform(v => NumOps.FromDouble(Math.Log(NumOps.ToDouble(v)))),
            GammaLinkFunction.Inverse => mu.Transform(v => NumOps.Divide(NumOps.One, v)),
            GammaLinkFunction.Identity => mu.Clone(),
            _ => mu.Transform(v => NumOps.FromDouble(Math.Log(NumOps.ToDouble(v))))
        };
    }

    /// <summary>
    /// Clamps the mean values to ensure they're positive and numerically stable.
    /// </summary>
    /// <param name="mu">The mean values to clamp.</param>
    /// <returns>The clamped mean values.</returns>
    /// <remarks>
    /// <para>
    /// This prevents numerical issues when mu gets too close to zero or becomes negative
    /// due to numerical errors in the optimization process.
    /// </para>
    /// <para>
    /// For Beginners:
    /// During the iterative fitting process, the predicted values might temporarily become
    /// very small or even slightly negative due to numerical approximations. This method
    /// ensures they stay positive and reasonably sized to keep the calculations stable.
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
    /// For Gamma regression, the weights depend on the link function:
    /// - Log link: W = 1 (constant weights, since variance × (dη/dμ)² = μ² × (1/μ)² = 1)
    /// - Inverse link: W = μ² (canonical link, since variance × (dη/dμ)² = μ² × (1/μ²)² = 1/μ²)
    /// - Identity link: W = 1/μ² (since variance × (dη/dμ)² = μ² × 1 = μ²)
    ///
    /// The weight formula is: W = 1 / (V(μ) × (g'(μ))²)
    /// where V(μ) = μ² is the variance function for Gamma.
    /// </para>
    /// <para>
    /// For Beginners:
    /// In the iterative fitting process, each observation is given a weight that depends on
    /// its current predicted value and the link function being used. These weights help the
    /// algorithm converge to the correct solution by balancing the influence of different
    /// observations.
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
                // W = 1 / (V(μ) × (g'(μ))²) = 1 / (μ² × (1/μ)²) = 1
                GammaLinkFunction.Log => 1.0,
                // W = 1 / (V(μ) × (g'(μ))²) = 1 / (μ² × (1/μ²)²) = μ²
                GammaLinkFunction.Inverse => muVal * muVal,
                // W = 1 / (V(μ) × (g'(μ))²) = 1 / (μ² × 1) = 1/μ²
                GammaLinkFunction.Identity => 1.0 / (muVal * muVal),
                _ => 1.0
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
    /// For different link functions:
    /// - Log link: z = η + (y - μ)/μ
    /// - Inverse link: z = η - (y - μ)/μ²
    /// - Identity link: z = η + (y - μ) = y
    /// </para>
    /// <para>
    /// For Beginners:
    /// The working response is an adjusted version of the target variable that helps the algorithm
    /// converge to the correct solution. It combines the current linear predictor with the error term
    /// (difference between actual and predicted values) scaled appropriately for the link function.
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
                GammaLinkFunction.Log => etaVal + diff / muVal,
                // g'(μ) = -1/μ², so z = η - (y - μ)/μ²
                GammaLinkFunction.Inverse => etaVal - diff / (muVal * muVal),
                // g'(μ) = 1, so z = η + (y - μ) = y
                GammaLinkFunction.Identity => yVal,
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
    /// <returns>True if the change is less than the tolerance, indicating convergence; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// Convergence is determined by calculating the L2 norm (Euclidean distance) between the old and new coefficients
    /// and checking if it's less than the specified tolerance.
    /// </para>
    /// <para>
    /// For Beginners:
    /// This method checks if the model has "settled down" and found the best solution. It does this by measuring
    /// how much the coefficients changed in the last iteration. If the change is very small (less than the tolerance),
    /// we consider the model to have converged and stop the training process.
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
    /// φ = (1/(n-p)) × Σ((y_i - μ_i)/μ_i)²
    /// where n is the number of samples and p is the number of parameters.
    /// </para>
    /// <para>
    /// For Beginners:
    /// The dispersion parameter measures how much the actual values vary compared to what the model predicts.
    /// It's calculated after the model is fitted by looking at the squared standardized differences between
    /// actual and predicted values.
    /// </para>
    /// </remarks>
    private void EstimateDispersion(Matrix<T> x, Vector<T> y)
    {
        Vector<T> predictions = Predict(x);
        int n = y.Length;
        int p = Coefficients.Length + 1; // coefficients + intercept

        double sumPearsonResidualsSq = 0;
        for (int i = 0; i < n; i++)
        {
            double yVal = NumOps.ToDouble(y[i]);
            double muVal = NumOps.ToDouble(predictions[i]);
            double pearsonResidual = (yVal - muVal) / muVal;
            sumPearsonResidualsSq += pearsonResidual * pearsonResidual;
        }

        // Estimate dispersion as sum of squared Pearson residuals divided by degrees of freedom
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
    /// This method adds an intercept column to the input matrix, computes the linear predictor,
    /// and applies the inverse link function to get the predicted means.
    /// </para>
    /// <para>
    /// For Beginners:
    /// After training, this method is used to make predictions on new data. It takes your input features,
    /// applies the learned coefficients, and returns the predicted positive values. The predictions are always
    /// positive, which is appropriate for data like costs, durations, or amounts.
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
    /// This method serializes both the base class data and the Gamma regression specific options,
    /// including link function, maximum iterations, convergence tolerance, and dispersion parameter.
    /// </para>
    /// <para>
    /// For Beginners:
    /// Serialization converts the model's internal state into a format that can be saved to disk or
    /// transmitted over a network. This allows you to save a trained model and load it later without
    /// having to retrain it. Think of it like saving your progress in a video game.
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

        // Serialize GammaRegression specific options
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
    /// This method deserializes both the base class data and the Gamma regression specific options,
    /// reconstructing the model's state from the serialized data.
    /// </para>
    /// <para>
    /// For Beginners:
    /// Deserialization is the opposite of serialization - it takes the saved model data and reconstructs
    /// the model's internal state. This allows you to load a previously trained model and use it to make
    /// predictions without having to retrain it. It's like loading a saved game to continue where you left off.
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

        // Deserialize GammaRegression specific options
        _options.MaxIterations = reader.ReadInt32();
        _options.Tolerance = reader.ReadDouble();
        _options.LinkFunction = (GammaLinkFunction)reader.ReadInt32();
        _options.DecompositionType = (MatrixDecompositionType)reader.ReadInt32();
        _options.InitialDispersion = reader.ReadDouble();
        _dispersion = NumOps.FromDouble(reader.ReadDouble());
    }

    /// <summary>
    /// Gets the type of the model.
    /// </summary>
    /// <returns>The model type identifier for Gamma regression.</returns>
    /// <remarks>
    /// <para>
    /// This method is used for model identification and serialization purposes.
    /// </para>
    /// <para>
    /// For Beginners:
    /// This method simply returns an identifier that indicates this is a Gamma regression model.
    /// It's used internally by the library to keep track of different types of models.
    /// </para>
    /// </remarks>
    protected override ModelType GetModelType()
    {
        return ModelType.GammaRegression;
    }

    /// <summary>
    /// Creates a new instance of the Gamma Regression model with the same configuration.
    /// </summary>
    /// <returns>A new instance of the Gamma Regression model.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a deep copy of the current Gamma Regression model, including its options,
    /// coefficients, intercept, dispersion, and regularization settings.
    /// </para>
    /// <para>
    /// For Beginners:
    /// This method creates an exact copy of your trained model.
    ///
    /// Think of it like making a perfect duplicate:
    /// - It copies all the configuration settings (like link function, maximum iterations, and tolerance)
    /// - It preserves the coefficients (the weights for each feature)
    /// - It maintains the intercept and dispersion parameter
    ///
    /// Creating a copy is useful when you want to:
    /// - Create a backup before further modifying the model
    /// - Create variations of the same model for different purposes
    /// - Share the model with others while keeping your original intact
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        var newOptions = new GammaRegressionOptions<T>
        {
            MaxIterations = _options.MaxIterations,
            Tolerance = _options.Tolerance,
            LinkFunction = _options.LinkFunction,
            DecompositionType = _options.DecompositionType,
            InitialDispersion = _options.InitialDispersion
        };

        var newModel = new GammaRegression<T>(newOptions, Regularization);

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
