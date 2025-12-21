namespace AiDotNet.Regression;

/// <summary>
/// Represents a negative binomial regression model for count data that may exhibit overdispersion.
/// </summary>
/// <remarks>
/// <para>
/// Negative binomial regression is a type of generalized linear model used for modeling count data when the variance
/// exceeds the mean (overdispersion), which violates the assumption of Poisson regression. It extends Poisson regression
/// by adding a dispersion parameter that accounts for the extra variance in the data. The model uses a log link function
/// to ensure that predictions are always positive, as required for count data.
/// </para>
/// <para><b>For Beginners:</b> Negative binomial regression is a special model for predicting counts when your data shows more variation than expected.
/// 
/// Think of it like predicting the number of customer service calls a business receives each day:
/// - A simple model might assume consistent variation around the average (Poisson model)
/// - But real data often shows much more variation - some days have way more calls than expected
/// - Negative binomial regression handles this "extra randomness" by including a special adjustment (dispersion parameter)
/// 
/// For example, the model might predict that a business receives 15 calls per day on average, but also accounts for the fact
/// that some days might have 5 calls while others have 40, which is more extreme variation than simpler models would expect.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class NegativeBinomialRegression<T> : RegressionBase<T>
{
    /// <summary>
    /// The dispersion parameter that accounts for overdispersion in the data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The dispersion parameter captures the degree to which the variance in the data exceeds the mean. A value of 1
    /// indicates no overdispersion (equivalent to Poisson regression), while larger values indicate increasing levels
    /// of overdispersion.
    /// </para>
    /// <para><b>For Beginners:</b> This is the "extra randomness adjuster" in the model.
    /// 
    /// The dispersion parameter:
    /// - Measures how much extra variation exists in your data beyond what's expected
    /// - A value of 1 means normal variation (like a Poisson distribution)
    /// - Larger values mean more extreme ups and downs in your data
    /// 
    /// It's like a dial that adjusts how much random fluctuation the model expects - higher values tell the model
    /// that large deviations from the average are normal for this data.
    /// </para>
    /// </remarks>
    private T _dispersion;

    /// <summary>
    /// The configuration options for the negative binomial regression model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// These options control the behavior of the negative binomial regression algorithm during training, including
    /// parameters such as the maximum number of iterations, convergence tolerance, and the matrix decomposition method
    /// used for solving the linear system.
    /// </para>
    /// <para><b>For Beginners:</b> These are the settings that control how the model learns.
    /// 
    /// Key settings include:
    /// - How many attempts (iterations) the model makes to improve itself
    /// - How precise the model needs to be before it stops training
    /// - What mathematical method to use for calculations
    /// 
    /// These settings affect how quickly the model trains and how accurate it becomes.
    /// </para>
    /// </remarks>
    private readonly NegativeBinomialRegressionOptions<T> _options;

    /// <summary>
    /// Initializes a new instance of the <see cref="NegativeBinomialRegression{T}"/> class with optional custom options and regularization.
    /// </summary>
    /// <param name="options">Custom options for the negative binomial regression algorithm. If null, default options are used.</param>
    /// <param name="regularization">Regularization method to prevent overfitting. If null, no regularization is applied.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new negative binomial regression model with the specified options and regularization.
    /// If no options are provided, default values are used. Regularization helps prevent overfitting by penalizing
    /// large coefficient values. The dispersion parameter is initially set to 1, which is equivalent to a Poisson model.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new negative binomial regression model with your chosen settings.
    /// 
    /// When creating the model:
    /// - You can provide custom settings (options) or use the defaults
    /// - You can add regularization, which helps prevent the model from memorizing the training data
    /// - The model starts with a dispersion value of 1 (standard amount of randomness)
    /// 
    /// The model will adjust the dispersion parameter during training based on the actual variation in your data.
    /// </para>
    /// </remarks>
    public NegativeBinomialRegression(NegativeBinomialRegressionOptions<T>? options = null, IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, regularization)
    {
        _options = options ?? new NegativeBinomialRegressionOptions<T>();
        _dispersion = NumOps.One;
    }

    /// <summary>
    /// Trains the negative binomial regression model using the provided features and target values.
    /// </summary>
    /// <param name="X">The feature matrix where each row is a sample and each column is a feature.</param>
    /// <param name="y">The target vector containing the count values to predict.</param>
    /// <exception cref="ArgumentException">Thrown when the number of rows in X does not match the length of y.</exception>
    /// <remarks>
    /// <para>
    /// This method trains the negative binomial regression model using iteratively reweighted least squares (IRLS),
    /// which is a form of Fisher scoring. It iteratively updates the coefficients by solving a weighted least squares
    /// problem until convergence. After finding the optimal coefficients, it updates the dispersion parameter to account
    /// for the observed variance in the data.
    /// </para>
    /// <para><b>For Beginners:</b> This is where the model learns from your data.
    /// 
    /// During training:
    /// 1. The model starts with initial guesses for the coefficients
    /// 2. It calculates predictions based on current coefficients
    /// 3. It computes how much to adjust each coefficient to improve predictions
    /// 4. It updates the coefficients and checks if they've stabilized
    /// 5. Steps 2-4 repeat until the changes become very small or a maximum number of iterations is reached
    /// 6. Finally, it calculates the dispersion parameter to account for extra variation in the data
    /// 
    /// This process finds the best coefficients for predicting count data while accounting for
    /// the extra randomness that's often present in real-world counts.
    /// </para>
    /// </remarks>
    public override void Train(Matrix<T> X, Vector<T> y)
    {
        if (X.Rows != y.Length)
            throw new ArgumentException("The number of rows in X must match the length of y.");

        InitializeCoefficients(X.Columns);

        for (int iteration = 0; iteration < _options.MaxIterations; iteration++)
        {
            var oldCoefficients = Coefficients.Clone();

            // Calculate linear predictors and means
            var linearPredictors = X.Multiply(Coefficients).Add(Intercept);
            var means = linearPredictors.Transform(NumOps.Exp);

            // Calculate weights and working response
            var weights = CalculateWeights(means);
            var workingResponse = CalculateWorkingResponse(y, means, linearPredictors);

            // Weighted least squares step
            var weightedX = X.PointwiseMultiply(weights.Transform(NumOps.Sqrt));
            var weightedY = workingResponse.PointwiseMultiply(weights.Transform(NumOps.Sqrt));

            // Apply regularization to the design matrix
            var regularizedX = Regularization.Regularize(weightedX);

            // Solve the regularized system
            var newCoefficients = MatrixSolutionHelper.SolveLinearSystem(regularizedX, weightedY, MatrixDecompositionFactory.GetDecompositionType(_options.DecompositionMethod));

            // Apply regularization to the coefficients
            newCoefficients = Regularization.Regularize(newCoefficients);

            Coefficients = newCoefficients.Slice(1, newCoefficients.Length - 1);
            Intercept = newCoefficients[0];

            // Check for convergence
            if (NumOps.LessThan(Coefficients.Subtract(oldCoefficients).Norm(), NumOps.FromDouble(_options.Tolerance)))
                break;
        }

        UpdateDispersion(X, y);
    }

    /// <summary>
    /// Initializes the model coefficients to zeros.
    /// </summary>
    /// <param name="featureCount">The number of features in the model.</param>
    /// <remarks>
    /// <para>
    /// This method initializes the model coefficients to zeros. This provides a starting point for the iterative
    /// optimization procedure used in training.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets up the starting values for the model.
    /// 
    /// Before training begins:
    /// - The model creates a coefficient for each feature
    /// - All coefficients start at zero (no effect)
    /// - The intercept (baseline value) also starts at zero
    /// 
    /// These values will be adjusted during training to find the optimal prediction formula.
    /// </para>
    /// </remarks>
    private void InitializeCoefficients(int featureCount)
    {
        Coefficients = new Vector<T>(featureCount);
        Intercept = NumOps.Zero;
    }

    /// <summary>
    /// Makes predictions for new data points using the trained negative binomial regression model.
    /// </summary>
    /// <param name="X">The feature matrix where each row is a sample to predict.</param>
    /// <returns>A vector containing the predicted count values.</returns>
    /// <remarks>
    /// <para>
    /// This method makes predictions by applying the learned coefficients to the input features and transforming
    /// the result with the exponential function. This ensures that the predictions are always positive, as required
    /// for count data. The predictions represent the expected counts based on the input features.
    /// </para>
    /// <para><b>For Beginners:</b> This is where the model makes predictions on new data.
    /// 
    /// The prediction process:
    /// 1. For each data point, multiply each feature by its corresponding coefficient
    /// 2. Add all these products together (plus the intercept)
    /// 3. Apply the exponential function to ensure the result is positive
    /// 4. The final value is the predicted count
    /// 
    /// For example, if predicting daily customer service calls based on day of week, staffing level, and
    /// weather conditions, the model combines these factors mathematically to estimate the expected number of calls.
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> X)
    {
        var linearPredictors = X.Multiply(Coefficients).Add(Intercept);
        return linearPredictors.Transform(NumOps.Exp);
    }

    /// <summary>
    /// Calculates the weights for the iteratively reweighted least squares algorithm.
    /// </summary>
    /// <param name="means">The vector of predicted means.</param>
    /// <returns>A vector of weights for each observation.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the weights used in the iteratively reweighted least squares (IRLS) algorithm based on the
    /// predicted means and the current dispersion parameter. The weights are inversely proportional to the variance of
    /// each observation, which depends on both the mean and the dispersion parameter.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates how much to trust each data point during training.
    /// 
    /// The weights:
    /// - Determine how much influence each data point has on the model
    /// - Adjust based on the expected variance for each point
    /// - Account for the extra randomness captured by the dispersion parameter
    /// 
    /// Data points with higher expected variance get lower weights, meaning they have less
    /// influence on the model's learning process. This helps the model focus on more reliable patterns.
    /// </para>
    /// </remarks>
    private Vector<T> CalculateWeights(Vector<T> means)
    {
        return means.Transform(mu => NumOps.Divide(NumOps.Square(mu), NumOps.Add(mu, NumOps.Divide(NumOps.Square(mu), _dispersion))));
    }

    /// <summary>
    /// Calculates the working response for the iteratively reweighted least squares algorithm.
    /// </summary>
    /// <param name="y">The observed count values.</param>
    /// <param name="means">The predicted mean values.</param>
    /// <param name="linearPredictors">The linear predictors (log of the means).</param>
    /// <returns>A vector of working responses for each observation.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the working response used in the iteratively reweighted least squares (IRLS) algorithm.
    /// The working response is a transformed version of the original response that makes the problem more like a
    /// weighted linear regression problem. It combines the linear predictors with a term based on the residuals.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates adjusted target values that help the model learn more effectively.
    /// 
    /// The working response:
    /// - Transforms the problem into a form that's easier to solve mathematically
    /// - Combines the current predictions with information about the prediction errors
    /// - Helps the iterative algorithm converge to the optimal solution
    /// 
    /// It's like creating a temporary "target" for each step of the learning process that points
    /// the model in the right direction for improvement.
    /// </para>
    /// </remarks>
    private Vector<T> CalculateWorkingResponse(Vector<T> y, Vector<T> means, Vector<T> linearPredictors)
    {
        return linearPredictors.Add(y.Subtract(means).PointwiseDivide(means));
    }

    /// <summary>
    /// Updates the dispersion parameter based on the Pearson residuals.
    /// </summary>
    /// <param name="X">The feature matrix used for training.</param>
    /// <param name="y">The observed count values.</param>
    /// <remarks>
    /// <para>
    /// This method updates the dispersion parameter by calculating the Pearson residuals (the standardized differences
    /// between observed and predicted values) and computing their mean square. The dispersion parameter is estimated as
    /// the sum of squared Pearson residuals divided by the degrees of freedom (sample size minus number of parameters).
    /// </para>
    /// <para><b>For Beginners:</b> This method adjusts the "extra randomness" parameter based on the actual data.
    /// 
    /// To update the dispersion:
    /// 1. Calculate how far off each prediction is from the actual value
    /// 2. Standardize these differences to account for the expected variation
    /// 3. Square these standardized differences and find their average
    /// 4. This average tells us how much extra variation exists in the data
    /// 
    /// A larger dispersion value means there's more unexplained randomness in the counts,
    /// which the model needs to account for when making predictions and estimating uncertainty.
    /// </para>
    /// </remarks>
    private void UpdateDispersion(Matrix<T> X, Vector<T> y)
    {
        var predictions = Predict(X);
        var pearsonResiduals = y.Subtract(predictions).Transform(
            (yi, predi) => NumOps.Divide(NumOps.Subtract(yi, NumOps.FromDouble(predi)), NumOps.Sqrt(NumOps.FromDouble(predi))));
        var sumSquaredResiduals = pearsonResiduals.Transform(NumOps.Square).Sum();
        var degreesOfFreedom = NumOps.FromDouble(X.Rows - X.Columns);
        _dispersion = NumOps.Divide(sumSquaredResiduals, degreesOfFreedom);
    }

    /// <summary>
    /// Serializes the negative binomial regression model to a byte array for storage or transmission.
    /// </summary>
    /// <returns>A byte array containing the serialized model data.</returns>
    /// <remarks>
    /// <para>
    /// This method converts the entire negative binomial regression model, including its parameters and configuration,
    /// into a byte array that can be stored in a file or database, or transmitted over a network. The model can later be
    /// restored using the Deserialize method.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves the model to a format that can be stored or shared.
    /// 
    /// Serialization:
    /// - Converts all the model's data into a sequence of bytes
    /// - Preserves the model's coefficients, intercept, dispersion parameter, and options
    /// - Allows you to save the trained model to a file
    /// - Lets you load the model later without having to retrain it
    /// 
    /// It's like taking a snapshot of the model that you can use later or share with others.
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

        // Serialize NegativeBinomialRegression specific data
        writer.Write(Convert.ToDouble(_dispersion));
        writer.Write(_options.MaxIterations);
        writer.Write(Convert.ToDouble(_options.Tolerance));

        return ms.ToArray();
    }

    /// <summary>
    /// Deserializes the negative binomial regression model from a byte array.
    /// </summary>
    /// <param name="modelData">A byte array containing the serialized model data.</param>
    /// <remarks>
    /// <para>
    /// This method restores a negative binomial regression model from a serialized byte array, reconstructing its parameters
    /// and configuration. This allows a previously trained model to be loaded from storage or after being received over a network.
    /// </para>
    /// <para><b>For Beginners:</b> This method rebuilds the model from a saved format.
    /// 
    /// Deserialization:
    /// - Takes a sequence of bytes that represents a model
    /// - Reconstructs the original model with all its learned parameters
    /// - Restores the coefficients, intercept, dispersion parameter, and options
    /// - Allows you to use a previously trained model without retraining
    /// 
    /// It's like unpacking a model that was packed up for storage or sharing,
    /// so you can use it again exactly as it was before.
    /// </para>
    /// </remarks>
    public override void Deserialize(byte[] modelData)
    {
        using var ms = new MemoryStream(modelData);
        using var reader = new BinaryReader(ms);

        // Deserialize base class data
        int baseDataLength = reader.ReadInt32();
        byte[] baseData = reader.ReadBytes(baseDataLength);
        base.Deserialize(baseData);

        // Deserialize NegativeBinomialRegression specific data
        _dispersion = NumOps.FromDouble(reader.ReadDouble());
        _options.MaxIterations = reader.ReadInt32();
        _options.Tolerance = reader.ReadDouble();
    }

    /// <summary>
    /// Gets the type of regression model.
    /// </summary>
    /// <returns>The model type, in this case, NegativeBinomialRegression.</returns>
    /// <remarks>
    /// <para>
    /// This method returns an enumeration value indicating that this is a negative binomial regression model. This is used
    /// for type identification when working with different regression models in a unified manner.
    /// </para>
    /// <para><b>For Beginners:</b> This method simply identifies what kind of model this is.
    /// 
    /// It returns a label (NegativeBinomialRegression) that:
    /// - Identifies this specific type of model
    /// - Helps other code handle the model appropriately
    /// - Is used when saving or loading models
    /// 
    /// It's like a name tag that lets other parts of the program know what kind of model they're working with.
    /// </para>
    /// </remarks>
    protected override ModelType GetModelType()
    {
        return ModelType.NegativeBinomialRegression;
    }

    /// <summary>
    /// Creates a new instance of the Negative Binomial Regression model with the same configuration.
    /// </summary>
    /// <returns>A new instance of the Negative Binomial Regression model.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the creation fails or required components are null.</exception>
    /// <remarks>
    /// <para>
    /// This method creates a deep copy of the current Negative Binomial Regression model, including its options,
    /// coefficients, intercept, dispersion parameter, and regularization settings. The new instance is completely 
    /// independent of the original, allowing modifications without affecting the original model.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates an exact copy of your trained model.
    /// 
    /// Think of it like making a perfect duplicate:
    /// - It copies all the configuration settings (like maximum iterations and tolerance)
    /// - It preserves the coefficients (the importance values for each feature)
    /// - It maintains the intercept (the starting point or base value)
    /// - It keeps the dispersion parameter (the "extra randomness adjuster")
    /// 
    /// Creating a copy is useful when you want to:
    /// - Create a backup before further modifying the model
    /// - Create variations of the same model for different purposes
    /// - Share the model with others while keeping your original intact
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        var newModel = new NegativeBinomialRegression<T>(_options, Regularization);

        // Copy coefficients if they exist
        if (Coefficients != null)
        {
            newModel.Coefficients = Coefficients.Clone();
        }

        // Copy the intercept
        newModel.Intercept = Intercept;

        // Copy the dispersion parameter
        newModel._dispersion = _dispersion;

        return newModel;
    }
}
