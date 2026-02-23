namespace AiDotNet.Regression;

/// <summary>
/// Implements Quantile Regression, a technique that estimates the conditional quantiles of a response variable
/// distribution in the linear model, providing a more complete view of the relationship between variables.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Unlike ordinary least squares regression which estimates the conditional mean of the response variable,
/// quantile regression estimates the conditional median or other quantiles of the response variable.
/// This makes it robust to outliers and useful for modeling heterogeneous conditional distributions.
/// </para>
/// <para>
/// The algorithm uses gradient descent optimization to minimize the quantile loss function, which gives
/// different weights to positive and negative errors based on the specified quantile.
/// </para>
/// <para>
/// <b>For Beginners:</b> While standard regression tells you about the average relationship between variables, quantile regression
/// lets you explore different parts of the data distribution. For example, median regression (quantile=0.5)
/// tells you about the middle of the distribution, while quantile=0.9 tells you about the upper end.
/// This is useful when you suspect that the relationship between variables might be different for different
/// ranges of the outcome.
/// </para>
/// </remarks>
public class QuantileRegression<T> : RegressionBase<T>
{
    /// <summary>
    /// Configuration options for the quantile regression model.
    /// </summary>
    /// <value>
    /// Contains settings like the quantile to estimate, learning rate, and maximum iterations.
    /// </value>
    private readonly QuantileRegressionOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    /// <summary>
    /// Initializes a new instance of the QuantileRegression class with the specified options and regularization.
    /// </summary>
    /// <param name="options">Configuration options for the quantile regression model. If null, default options will be used.</param>
    /// <param name="regularization">Regularization method to prevent overfitting. If null, no regularization will be applied.</param>
    /// <remarks>
    /// <para>
    /// The constructor initializes the model with either the provided options or default settings.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This constructor sets up the quantile regression model with your specified settings or uses
    /// default settings if none are provided. Regularization is an optional technique to prevent the model
    /// from becoming too complex and overfitting to the training data.
    /// </para>
    /// </remarks>
    public QuantileRegression(QuantileRegressionOptions<T>? options = null, IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, regularization)
    {
        _options = options ?? new QuantileRegressionOptions<T>();
    }

    /// <summary>
    /// Trains the quantile regression model on the provided data.
    /// </summary>
    /// <param name="x">The input features matrix where each row is a training example and each column is a feature.</param>
    /// <param name="y">The target values vector corresponding to each training example.</param>
    /// <remarks>
    /// <para>
    /// This method implements gradient descent optimization to minimize the quantile loss function.
    /// The steps are:
    /// 1. Initialize coefficients and intercept
    /// 2. Apply regularization to the input matrix
    /// 3. For each iteration:
    ///    a. Calculate predictions and errors for all examples
    ///    b. Compute gradients based on the quantile loss function
    ///    c. Update coefficients and intercept using the gradients
    ///    d. Apply regularization to the coefficients
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Training is the process where the model learns from your data. The algorithm starts with initial guesses
    /// for the coefficients and then iteratively improves them. At each step, it calculates how far off its
    /// predictions are, but unlike standard regression, it penalizes over-predictions and under-predictions
    /// differently based on the quantile you specified. It then adjusts the coefficients to reduce these errors.
    /// </para>
    /// </remarks>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        int n = x.Rows;
        int p = x.Columns;

        // Initialize coefficients
        Coefficients = new Vector<T>(p);
        Intercept = NumOps.Zero;

        // Apply regularization to the input matrix
        x = Regularization.Regularize(x);

        // Gradient descent optimization
        for (int iter = 0; iter < _options.MaxIterations; iter++)
        {
            Vector<T> gradients = new(p);
            T interceptGradient = NumOps.Zero;

            for (int i = 0; i < n; i++)
            {
                T prediction = Predict(x.GetRow(i));
                T error = NumOps.Subtract(y[i], prediction);
                T gradient = NumOps.GreaterThan(error, NumOps.Zero)
                    ? NumOps.FromDouble(_options.Quantile)
                    : NumOps.FromDouble(_options.Quantile - 1);

                for (int j = 0; j < p; j++)
                {
                    gradients[j] = NumOps.Add(gradients[j], NumOps.Multiply(gradient, x[i, j]));
                }
                interceptGradient = NumOps.Add(interceptGradient, gradient);
            }

            // Update coefficients and intercept
            for (int j = 0; j < p; j++)
            {
                Coefficients[j] = NumOps.Add(Coefficients[j], NumOps.Multiply(NumOps.FromDouble(_options.LearningRate), gradients[j]));
            }
            Intercept = NumOps.Add(Intercept, NumOps.Multiply(NumOps.FromDouble(_options.LearningRate), interceptGradient));

            // Apply regularization to coefficients
            Coefficients = Regularization.Regularize(Coefficients);
        }
    }

    /// <summary>
    /// Makes predictions for the given input data.
    /// </summary>
    /// <param name="input">The input features matrix where each row is an example and each column is a feature.</param>
    /// <returns>A vector of predicted values for each input example.</returns>
    /// <remarks>
    /// <para>
    /// This method predicts the specified quantile of the conditional distribution for each input example.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> After training, this method is used to make predictions on new data. For each example in your input data,
    /// it calculates the predicted value at the quantile you specified. For instance, if you set quantile=0.5,
    /// it predicts the median value; if you set quantile=0.9, it predicts the value below which 90% of the
    /// observations would fall.
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> input)
    {
        var predictions = new Vector<T>(input.Rows);
        for (int i = 0; i < input.Rows; i++)
        {
            predictions[i] = Predict(input.GetRow(i));
        }

        return predictions;
    }

    /// <summary>
    /// Predicts the value for a single input vector.
    /// </summary>
    /// <param name="input">The input feature vector.</param>
    /// <returns>The predicted value at the specified quantile.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the dot product of the input vector and the coefficients, then adds the intercept.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is the core prediction function that calculates the predicted value for a single example.
    /// It multiplies each feature value by its corresponding coefficient, sums these products, and adds
    /// the intercept term to get the final prediction.
    /// </para>
    /// </remarks>
    private T Predict(Vector<T> input)
    {
        return NumOps.Add(Coefficients.DotProduct(input), Intercept);
    }

    /// <summary>
    /// Gets metadata about the model.
    /// </summary>
    /// <returns>A ModelMetadata object containing information about the model.</returns>
    /// <remarks>
    /// <para>
    /// This method returns metadata about the model, including its type and the quantile being estimated.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Model metadata provides information about the model itself, rather than the predictions it makes.
    /// For quantile regression, this includes which quantile the model is estimating (e.g., median, 90th percentile).
    /// This information can be useful for understanding and comparing different models.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = base.GetModelMetadata();
        metadata.AdditionalInfo["Quantile"] = _options.Quantile;

        return metadata;
    }

    /// <summary>
    /// Gets the type of the model.
    /// </summary>
    /// <returns>The model type identifier for quantile regression.</returns>
    /// <remarks>
    /// <para>
    /// This method is used for model identification and serialization purposes.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method simply returns an identifier that indicates this is a quantile regression model.
    /// It's used internally by the library to keep track of different types of models.
    /// </para>
    /// </remarks>
    protected override ModelType GetModelType()
    {
        return ModelType.QuantileRegression;
    }

    /// <summary>
    /// Serializes the model to a byte array.
    /// </summary>
    /// <returns>A byte array containing the serialized model data.</returns>
    /// <remarks>
    /// <para>
    /// This method serializes both the base class data and the quantile regression specific options,
    /// including the quantile, learning rate, and maximum iterations.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Serialization converts the model's internal state into a format that can be saved to disk or
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

        // Serialize QuantileRegression specific data
        writer.Write(_options.Quantile);
        writer.Write(_options.LearningRate);
        writer.Write(_options.MaxIterations);

        return ms.ToArray();
    }

    /// <summary>
    /// Deserializes the model from a byte array.
    /// </summary>
    /// <param name="modelData">The byte array containing the serialized model data.</param>
    /// <remarks>
    /// <para>
    /// This method deserializes both the base class data and the quantile regression specific options,
    /// reconstructing the model's state from the serialized data.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Deserialization is the opposite of serialization - it takes the saved model data and reconstructs
    /// the model's internal state. This allows you to load a previously trained model and use it to make
    /// predictions without having to retrain it. It's like loading a saved game to continue where you left off.
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

        // Deserialize QuantileRegression specific data
        _options.Quantile = reader.ReadDouble();
        _options.LearningRate = reader.ReadDouble();
        _options.MaxIterations = reader.ReadInt32();
    }

    /// <summary>
    /// Creates a new instance of the quantile regression model with the same options.
    /// </summary>
    /// <returns>A new instance of the quantile regression model with the same configuration but no trained parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new instance of the quantile regression model with the same configuration
    /// options and regularization method as the current instance, but without copying the trained parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a fresh copy of the model configuration without 
    /// any learned parameters.
    /// 
    /// Think of it like getting a blank notepad with the same paper quality and size, 
    /// but without any writing on it yet. The new model has the same:
    /// - Quantile setting (which part of the distribution you're estimating)
    /// - Learning rate (how quickly the model adjusts during training)
    /// - Maximum iterations (how long the model will train)
    /// - Regularization settings (safeguards against overfitting)
    /// 
    /// But it doesn't have any of the coefficient values that were learned from data.
    /// 
    /// This is mainly used internally when doing things like cross-validation or 
    /// creating ensembles of similar models with different training data.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        // Create a new instance with the same options and regularization
        return new QuantileRegression<T>(_options, Regularization);
    }
}
