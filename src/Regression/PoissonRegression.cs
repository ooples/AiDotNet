namespace AiDotNet.Regression;

/// <summary>
/// Implements Poisson regression, a generalized linear model used for modeling count data and contingency tables.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Poisson regression is appropriate when the dependent variable represents counts, such as the number of occurrences
/// of an event in a fixed period of time or space. It assumes that the response variable follows a Poisson distribution
/// and uses a log link function to relate the expected value of the response to the linear predictor.
/// </para>
/// <para>
/// The model is fitted using iteratively reweighted least squares (IRLS), a form of maximum likelihood estimation.
/// </para>
/// <para>
/// For Beginners:
/// Poisson regression is used when you're trying to predict counts (like number of customer visits, number of accidents,
/// etc.). Unlike linear regression, it ensures predictions are always non-negative and can handle cases where the
/// variance increases with the mean, which is common in count data.
/// </para>
/// </remarks>
public class PoissonRegression<T> : RegressionBase<T>
{
    /// <summary>
    /// Configuration options for the Poisson regression model.
    /// </summary>
    /// <value>
    /// Contains settings like maximum iterations and convergence tolerance.
    /// </value>
    private readonly PoissonRegressionOptions<T> _options;

    /// <summary>
    /// Initializes a new instance of the PoissonRegression class with the specified options and regularization.
    /// </summary>
    /// <param name="options">Configuration options for the Poisson regression model. If null, default options will be used.</param>
    /// <param name="regularization">Regularization method to prevent overfitting. If null, no regularization will be applied.</param>
    /// <remarks>
    /// <para>
    /// The constructor initializes the model with either the provided options or default settings.
    /// </para>
    /// <para>
    /// For Beginners:
    /// This constructor sets up the Poisson regression model with your specified settings or uses
    /// default settings if none are provided. Regularization is an optional technique to prevent the model
    /// from becoming too complex and overfitting to the training data.
    /// </para>
    /// </remarks>
    public PoissonRegression(PoissonRegressionOptions<T>? options = null, IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, regularization)
    {
        _options = options ?? new PoissonRegressionOptions<T>();
    }

    /// <summary>
    /// Trains the Poisson regression model on the provided data.
    /// </summary>
    /// <param name="x">The input features matrix where each row is a training example and each column is a feature.</param>
    /// <param name="y">The target count values vector corresponding to each training example.</param>
    /// <remarks>
    /// <para>
    /// This method implements the iteratively reweighted least squares (IRLS) algorithm to fit the Poisson regression model.
    /// The steps are:
    /// 1. Initialize coefficients and intercept
    /// 2. For each iteration:
    ///    a. Compute the predicted mean (mu) using the current coefficients
    ///    b. Compute the weights matrix (W) based on mu
    ///    c. Compute the working response (z)
    ///    d. Solve the weighted least squares problem to get new coefficients
    ///    e. Check for convergence
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
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        ValidationHelper<T>.ValidateInputData(x, y);
        ValidationHelper<T>.ValidatePoissonData(y);

        int numFeatures = x.Columns;
        Coefficients = new Vector<T>(numFeatures);
        Intercept = NumOps.Zero;

        Matrix<T> xWithIntercept = x.AddColumn(Vector<T>.CreateDefault(x.Rows, NumOps.One));

        for (int iteration = 0; iteration < _options.MaxIterations; iteration++)
        {
            Vector<T> currentCoefficients = new([.. Coefficients, Intercept]);
            Vector<T> mu = PredictMean(xWithIntercept, currentCoefficients);
            Matrix<T> w = ComputeWeights(mu);
            Vector<T> z = ComputeWorkingResponse(xWithIntercept, y, mu, currentCoefficients);

            Matrix<T> xTw = xWithIntercept.Transpose().Multiply(w);
            Matrix<T> xTwx = xTw.Multiply(xWithIntercept);
            Vector<T> xTwz = xTw.Multiply(z);

            // Add ridge regularization to ensure numerical stability
            // This adds a small value to the diagonal to prevent singularity
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
    }

    /// <summary>
    /// Predicts the mean (expected value) for the given input data using the current model parameters.
    /// </summary>
    /// <param name="x">The input features matrix, including the intercept column.</param>
    /// <param name="coefficients">The model coefficients, including the intercept.</param>
    /// <returns>A vector of predicted mean values.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the linear predictor (X * coefficients) and applies the exponential function
    /// to get the predicted mean, as required by the Poisson regression model.
    /// </para>
    /// <para>
    /// For Beginners:
    /// In Poisson regression, we predict the expected count by first calculating a linear combination of
    /// the input features and coefficients, then applying the exponential function to ensure the result
    /// is always positive (since counts can't be negative).
    /// </para>
    /// </remarks>
    private Vector<T> PredictMean(Matrix<T> x, Vector<T> coefficients)
    {
        return x.Multiply(coefficients).Transform(NumOps.Exp);
    }

    /// <summary>
    /// Computes the weights matrix for the iteratively reweighted least squares algorithm.
    /// </summary>
    /// <param name="mu">The vector of predicted mean values.</param>
    /// <returns>A diagonal matrix of weights.</returns>
    /// <remarks>
    /// <para>
    /// For Poisson regression, the weights are equal to the predicted mean values.
    /// </para>
    /// <para>
    /// For Beginners:
    /// In the iterative fitting process, each observation is given a weight based on its predicted value.
    /// For Poisson regression, observations with higher predicted counts get higher weights in the next iteration.
    /// </para>
    /// </remarks>
    private Matrix<T> ComputeWeights(Vector<T> mu)
    {
        return Matrix<T>.CreateDiagonal(mu);
    }

    /// <summary>
    /// Computes the working response for the iteratively reweighted least squares algorithm.
    /// </summary>
    /// <param name="x">The input features matrix, including the intercept column.</param>
    /// <param name="y">The target count values vector.</param>
    /// <param name="mu">The vector of predicted mean values.</param>
    /// <param name="coefficients">The current model coefficients, including the intercept.</param>
    /// <returns>The working response vector.</returns>
    /// <remarks>
    /// <para>
    /// The working response is computed as eta + (y - mu) / mu, where eta is the linear predictor.
    /// </para>
    /// <para>
    /// For Beginners:
    /// The working response is an adjusted version of the target variable that helps the algorithm
    /// converge to the correct solution. It combines the current predictions with the error term
    /// (difference between actual and predicted values) in a way that's appropriate for the Poisson model.
    /// </para>
    /// </remarks>
    private static Vector<T> ComputeWorkingResponse(Matrix<T> x, Vector<T> y, Vector<T> mu, Vector<T> coefficients)
    {
        Vector<T> eta = x.Multiply(coefficients);
        return eta.Add(y.Subtract(mu).PointwiseDivide(mu));
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
    /// Makes predictions for the given input data.
    /// </summary>
    /// <param name="x">The input features matrix where each row is an example and each column is a feature.</param>
    /// <returns>A vector of predicted count values for each input example.</returns>
    /// <remarks>
    /// <para>
    /// This method adds an intercept column to the input matrix, combines the coefficients and intercept,
    /// and calls PredictMean to compute the predicted counts.
    /// </para>
    /// <para>
    /// For Beginners:
    /// After training, this method is used to make predictions on new data. It takes your input features,
    /// applies the learned coefficients, and returns the predicted counts. The predictions are always
    /// non-negative, which is appropriate for count data.
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

        return PredictMean(xWithIntercept, coefficientsWithIntercept);
    }

    /// <summary>
    /// Serializes the model to a byte array.
    /// </summary>
    /// <returns>A byte array containing the serialized model data.</returns>
    /// <remarks>
    /// <para>
    /// This method serializes both the base class data and the Poisson regression specific options,
    /// including maximum iterations and convergence tolerance.
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

        // Serialize PoissonRegression specific options
        writer.Write(_options.MaxIterations);
        writer.Write(Convert.ToDouble(_options.Tolerance));

        return ms.ToArray();
    }

    /// <summary>
    /// Deserializes the model from a byte array.
    /// </summary>
    /// <param name="data">The byte array containing the serialized model data.</param>
    /// <remarks>
    /// <para>
    /// This method deserializes both the base class data and the Poisson regression specific options,
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

        // Deserialize PoissonRegression specific options
        _options.MaxIterations = reader.ReadInt32();
        _options.Tolerance = Convert.ToDouble(reader.ReadDouble());
    }

    /// <summary>
    /// Gets the type of the model.
    /// </summary>
    /// <returns>The model type identifier for Poisson regression.</returns>
    /// <remarks>
    /// <para>
    /// This method is used for model identification and serialization purposes.
    /// </para>
    /// <para>
    /// For Beginners:
    /// This method simply returns an identifier that indicates this is a Poisson regression model.
    /// It's used internally by the library to keep track of different types of models.
    /// </para>
    /// </remarks>
    protected override ModelType GetModelType()
    {
        return ModelType.PoissonRegression;
    }

    /// <summary>
    /// Creates a new instance of the Poisson Regression model with the same configuration.
    /// </summary>
    /// <returns>A new instance of the Poisson Regression model.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the creation fails or required components are null.</exception>
    /// <remarks>
    /// <para>
    /// This method creates a deep copy of the current Poisson Regression model, including its options,
    /// coefficients, intercept, and regularization settings. The new instance is completely independent of the original,
    /// allowing modifications without affecting the original model.
    /// </para>
    /// <para>
    /// For Beginners:
    /// This method creates an exact copy of your trained model.
    /// 
    /// Think of it like making a perfect duplicate:
    /// - It copies all the configuration settings (like maximum iterations and tolerance)
    /// - It preserves the coefficients (the weights for each feature)
    /// - It maintains the intercept (the starting point of your model)
    /// 
    /// Creating a copy is useful when you want to:
    /// - Create a backup before further modifying the model
    /// - Create variations of the same model for different purposes
    /// - Share the model with others while keeping your original intact
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        var newModel = new PoissonRegression<T>(_options, Regularization);

        // Copy coefficients if they exist
        if (Coefficients != null)
        {
            newModel.Coefficients = Coefficients.Clone();
        }

        // Copy the intercept
        newModel.Intercept = Intercept;

        return newModel;
    }
}
