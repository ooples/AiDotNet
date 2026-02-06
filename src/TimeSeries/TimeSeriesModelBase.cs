using AiDotNet.Autodiff;

namespace AiDotNet.TimeSeries;

/// <summary>
/// Provides a base class for all time series forecasting models in the library.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// This abstract class defines the common interface and functionality that all time series models share,
/// including training, prediction, evaluation, and serialization/deserialization capabilities.
/// </para>
/// <para>
/// Time series models capture temporal dependencies in data and use patterns learned from historical
/// observations to predict future values. This base class provides the foundation for implementing
/// various time series forecasting algorithms like ARIMA, Exponential Smoothing, TBATS, and more complex
/// machine learning approaches.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// A time series model helps predict future values based on past observations.
///
/// Think of a time series like a sequence of measurements taken over time - for example,
/// daily temperatures, monthly sales, or hourly website visits. These models analyze the patterns
/// in historical data to make predictions about what will happen next.
///
/// This base class is like a blueprint that all specific time series models follow.
/// It ensures that every model can:
/// - Be trained on historical data to learn patterns
/// - Make predictions for future periods based on what it learned
/// - Evaluate how accurate its predictions are compared to actual values
/// - Be saved to disk and loaded later without retraining
///
/// Time series models are used in many real-world applications, including:
/// - Weather forecasting
/// - Stock market prediction
/// - Demand planning for retail
/// - Energy consumption forecasting
/// - Website traffic prediction
/// </para>
/// </remarks>
public abstract class TimeSeriesModelBase<T> : ITimeSeriesModel<T>
{
    /// <summary>
    /// Configuration options for the time series model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// These options control the core behavior of the time series model, including how much
    /// historical data is considered, whether trends or seasonality are modeled, and how errors
    /// are handled.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// Think of these options as settings that determine how the model works:
    /// - LagOrder: How many past values to consider (like remembering the last 5 days to predict tomorrow)
    /// - IncludeTrend: Whether to account for ongoing trends (like sales steadily increasing over time)
    /// - SeasonalPeriod: Whether there are regular patterns (like retail sales spiking every December)
    /// - AutocorrelationCorrection: Whether to fix systematic errors in predictions
    /// </para>
    /// </remarks>
    protected TimeSeriesRegressionOptions<T> Options { get; private set; }

    /// <summary>
    /// Provides numeric operations for the specific type T.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property provides mathematical operations appropriate for the generic type T,
    /// allowing the algorithm to work consistently with different numeric types like
    /// float, double, or decimal.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This is a helper that knows how to do math (addition, multiplication, etc.) with
    /// your specific number type, whether that's a regular double, a precise decimal value,
    /// or something else. It allows the model to work with different types of numbers
    /// without changing its core logic.
    /// </para>
    /// </remarks>
    protected INumericOperations<T> NumOps { get; private set; }

    /// <summary>
    /// Gets the global execution engine for vector operations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property provides access to the execution engine (CPU or GPU) for performing
    /// vectorized operations. The engine is determined by the global AiDotNetEngine configuration
    /// and allows automatic fallback from GPU to CPU when GPU is not available.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This gives access to either CPU or GPU processing for faster computations.
    /// The system automatically chooses the best available option and falls back to CPU
    /// if GPU acceleration is not available.
    /// </para>
    /// </remarks>
    protected IEngine Engine => AiDotNetEngine.Current;

    /// <summary>
    /// Gets or sets the trained model parameters.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Contains the values that the model has learned during training, such as coefficients
    /// for different lags, trend components, and seasonal factors.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// These are the numerical values the model learns during training that tell it exactly
    /// how much influence each past observation should have on the prediction. They're like
    /// the recipe ingredients with specific measurements that the model has figured out work best.
    /// </para>
    /// </remarks>
    protected Vector<T> ModelParameters { get; set; }

    /// <summary>
    /// Indicates whether the model has been trained.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This flag is set to true after the model has been successfully trained on data.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This is like a switch that gets turned on once the model has learned from your data.
    /// It helps prevent errors by making sure you don't try to use the model for predictions
    /// before it's ready.
    /// </para>
    /// </remarks>
    protected bool IsTrained { get; private set; } = false;

    /// <summary>
    /// The default loss function used for gradient computation.
    /// </summary>
    private readonly ILossFunction<T> _defaultLossFunction;

    /// <summary>
    /// Gets the last computed error metrics when the model was evaluated.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Contains accuracy metrics calculated during model evaluation, such as MAE, RMSE, and MAPE.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// These numbers tell you how accurate the model's predictions are compared to actual values.
    /// Lower numbers mean better predictions. They're like a scorecard for the model's performance.
    /// </para>
    /// </remarks>
    protected Dictionary<string, T> LastEvaluationMetrics { get; private set; } = new Dictionary<string, T>();

    /// <summary>
    /// Initializes a new instance of the TimeSeriesModelBase class with the specified options.
    /// </summary>
    /// <param name="options">The configuration options for the time series model.</param>
    /// <exception cref="ArgumentNullException">Thrown when options is null.</exception>
    /// <exception cref="ArgumentException">Thrown when options contain invalid values.</exception>
    /// <remarks>
    /// <para>
    /// This constructor validates the provided options, initializes the model with the specified
    /// configuration, and sets up the numeric operations appropriate for the data type.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This constructor sets up the basic configuration for any time series model.
    ///
    /// It takes an options object that specifies important settings like:
    /// - How many past values to consider (lag order)
    /// - Whether to include a trend component (like steady growth or decline)
    /// - The length of seasonal patterns (e.g., 7 for weekly, 12 for monthly)
    /// - Whether to correct for autocorrelation in errors (systematic errors)
    ///
    /// It also checks that these settings make sense - for example, you can't have a negative
    /// number of past values or a seasonal period less than 2.
    /// </para>
    /// </remarks>
    protected TimeSeriesModelBase(TimeSeriesRegressionOptions<T> options)
    {
        // Validate options
        if (options == null)
        {
            throw new ArgumentNullException(nameof(options), "Time series options cannot be null.");
        }

        ValidateOptions(options);

        Options = options;
        NumOps = MathHelper.GetNumericOperations<T>();
        ModelParameters = new Vector<T>(0); // Initialize with empty vector
        _defaultLossFunction = options.LossFunction ?? new MeanSquaredErrorLoss<T>();
    }

    /// <summary>
    /// Validates the provided time series options to ensure they are within acceptable ranges.
    /// </summary>
    /// <param name="options">The options to validate.</param>
    /// <exception cref="ArgumentException">Thrown when any option is invalid.</exception>
    /// <remarks>
    /// <para>
    /// Checks that LagOrder is non-negative, SeasonalPeriod is either 0 (no seasonality) or at least 2,
    /// and that other parameters have reasonable values.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method makes sure the settings you've chosen for your model make logical sense.
    /// For example, you can't look back a negative number of time periods, and a seasonal
    /// pattern must repeat at least every 2 periods to be considered seasonal.
    /// </para>
    /// </remarks>
    protected virtual void ValidateOptions(TimeSeriesRegressionOptions<T> options)
    {
        if (options.LagOrder < 0)
        {
            throw new ArgumentException("Lag order must be non-negative.", nameof(options));
        }

        if (options.SeasonalPeriod < 0)
        {
            throw new ArgumentException("Seasonal period must be non-negative.", nameof(options));
        }

        if (options.SeasonalPeriod == 1)
        {
            throw new ArgumentException("Seasonal period must be at least 2 if seasonality is enabled.", nameof(options));
        }

        // Additional model-specific validation can be implemented in derived classes
    }

    /// <summary>
    /// Trains the time series model using the provided input data and target values.
    /// </summary>
    /// <param name="x">The input features matrix.</param>
    /// <param name="y">The target values vector.</param>
    /// <exception cref="ArgumentNullException">Thrown when x or y is null.</exception>
    /// <exception cref="ArgumentException">Thrown when the dimensions of x and y don't match or when the data is insufficient.</exception>
    /// <remarks>
    /// <para>
    /// This method validates the input data, prepares the model for training, performs the actual
    /// training algorithm, and sets the IsTrained flag once complete.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// Training is the process where the model learns patterns from historical data.
    ///
    /// During training, the model analyzes the relationship between:
    /// - Input features (x): These might include past values, time indicators, or external factors
    /// - Target values (y): The actual observed values we want to predict
    ///
    /// After training, the model will have learned parameters that capture the patterns
    /// in your data, which it can then use to make predictions for new inputs.
    ///
    /// This is an abstract method, meaning each specific model type (ARIMA, TBATS, etc.)
    /// will implement its own training algorithm.
    /// </para>
    /// </remarks>
    public void Train(Matrix<T> x, Vector<T> y)
    {
        // Input validation
        ValidateTrainingInputs(x, y);

        // Reset model state before training
        Reset();

        // Perform model-specific training (implemented by derived classes)
        TrainCore(x, y);

        // Mark the model as trained
        IsTrained = true;
    }

    /// <summary>
    /// Performs the model-specific training algorithm.
    /// </summary>
    /// <param name="x">The input features matrix.</param>
    /// <param name="y">The target values vector.</param>
    /// <remarks>
    /// <para>
    /// This abstract method must be implemented by derived classes to perform the actual model training.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This is where the specific math and algorithms for each type of time series model are implemented.
    /// Different models (like ARIMA, Exponential Smoothing, etc.) will have their own unique ways of
    /// finding patterns in the data.
    /// </para>
    /// </remarks>
    protected abstract void TrainCore(Matrix<T> x, Vector<T> y);

    /// <summary>
    /// Validates the training input data before proceeding with training.
    /// </summary>
    /// <param name="x">The input features matrix.</param>
    /// <param name="y">The target values vector.</param>
    /// <exception cref="ArgumentNullException">Thrown when x or y is null.</exception>
    /// <exception cref="ArgumentException">Thrown when the dimensions of x and y don't match or when the data is insufficient.</exception>
    /// <remarks>
    /// <para>
    /// This method verifies that the input data meets the requirements for model training,
    /// including checking dimensions, sample size, and consistency.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// Before the model starts learning, this method checks that your data is valid and properly formatted.
    /// It ensures that:
    /// - You have provided both input features and target values
    /// - The number of examples matches the number of target values
    /// - You have enough data points to train the model effectively
    /// - There are no obvious inconsistencies in your data structure
    /// </para>
    /// </remarks>
    protected virtual void ValidateTrainingInputs(Matrix<T> x, Vector<T> y)
    {
        if (x == null)
        {
            throw new ArgumentNullException(nameof(x), "Input features matrix cannot be null.");
        }

        if (y == null)
        {
            throw new ArgumentNullException(nameof(y), "Target values vector cannot be null.");
        }

        if (x.Rows != y.Length)
        {
            throw new ArgumentException(
                $"Number of rows in input matrix ({x.Rows}) must match the length of target vector ({y.Length}).");
        }

        if (x.Rows <= Options.LagOrder)
        {
            throw new ArgumentException(
                $"Number of samples ({x.Rows}) must be greater than lag order ({Options.LagOrder}).");
        }

        // Check for sufficient data to handle seasonality
        if (Options.SeasonalPeriod > 0 && x.Rows < 2 * Options.SeasonalPeriod)
        {
            throw new ArgumentException(
                $"For seasonal models, the number of samples ({x.Rows}) should be at least twice the seasonal period ({Options.SeasonalPeriod}).");
        }

        // Additional validation can be added in derived classes
    }

    /// <summary>
    /// Generates forecasts using the trained time series model.
    /// </summary>
    /// <param name="input">The input features matrix.</param>
    /// <returns>A vector of forecasted values.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the model has not been trained.</exception>
    /// <exception cref="ArgumentNullException">Thrown when input is null.</exception>
    /// <exception cref="ArgumentException">Thrown when input has incorrect dimensions.</exception>
    /// <remarks>
    /// <para>
    /// This method validates that the model is trained and the input data is valid, then
    /// generates predictions for each row in the input matrix using the model-specific
    /// prediction algorithm.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method uses the patterns learned during training to predict future values.
    ///
    /// The input matrix typically contains:
    /// - Past values of the time series
    /// - Time indicators (e.g., month, day of week)
    /// - Any external factors that might influence the forecast
    ///
    /// The output is a vector of predicted values, one for each row in the input matrix.
    /// Each prediction represents what the model thinks will happen at that future time point.
    /// </para>
    /// </remarks>
    public virtual Vector<T> Predict(Matrix<T> input)
    {
        // Check if model is trained
        if (!IsTrained)
        {
            throw new InvalidOperationException("The model must be trained before making predictions.");
        }

        // Validate input
        ValidatePredictionInput(input);

        // Create output vector for predictions
        var predictions = new Vector<T>(input.Rows);

        // Generate predictions for each input row
        for (int i = 0; i < input.Rows; i++)
        {
            predictions[i] = PredictSingle(input.GetRow(i));
        }

        return predictions;
    }

    /// <summary>
    /// Validates the input data for prediction.
    /// </summary>
    /// <param name="input">The input features matrix.</param>
    /// <exception cref="ArgumentNullException">Thrown when input is null.</exception>
    /// <exception cref="ArgumentException">Thrown when input has incorrect dimensions.</exception>
    /// <remarks>
    /// <para>
    /// This method verifies that the input data for prediction is valid and has the correct dimensions.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// Before making predictions, this method checks that your input data is properly formatted.
    /// It ensures that:
    /// - You have provided input features
    /// - The input has the correct structure (number of features/columns)
    /// - The data meets any model-specific requirements
    /// </para>
    /// </remarks>
    protected virtual void ValidatePredictionInput(Matrix<T> input)
    {
        if (input == null)
        {
            throw new ArgumentNullException(nameof(input), "Input features matrix cannot be null.");
        }

        // Additional validation can be added in derived classes
    }

    /// <summary>
    /// Generates a prediction for a single input vector.
    /// </summary>
    /// <param name="input">The input feature vector.</param>
    /// <returns>The predicted value.</returns>
    /// <remarks>
    /// <para>
    /// This abstract method must be implemented by derived classes to generate a prediction
    /// for a single input vector using the model-specific algorithm.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method takes a single row of input data (representing one time point) and
    /// calculates what the model predicts will happen at that point. Each type of
    /// time series model will have its own way of calculating this prediction based
    /// on the patterns it learned during training.
    /// </para>
    /// </remarks>
    public abstract T PredictSingle(Vector<T> input);

    /// <summary>
    /// Evaluates the performance of the trained model on test data.
    /// </summary>
    /// <param name="xTest">The input features matrix for testing.</param>
    /// <param name="yTest">The actual target values for testing.</param>
    /// <returns>A dictionary containing evaluation metrics.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the model has not been trained.</exception>
    /// <exception cref="ArgumentNullException">Thrown when xTest or yTest is null.</exception>
    /// <exception cref="ArgumentException">Thrown when the dimensions of xTest and yTest don't match.</exception>
    /// <remarks>
    /// <para>
    /// This method calculates various error metrics by comparing the model's predictions
    /// on the test data to the actual values, providing a quantitative assessment of
    /// model performance.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method tests how well the model performs by comparing its predictions to actual values.
    ///
    /// It works by:
    /// 1. Using the model to make predictions based on the test inputs
    /// 2. Comparing these predictions to the actual test values
    /// 3. Calculating various error metrics to quantify the accuracy
    ///
    /// Common metrics include:
    /// - Mean Absolute Error (MAE): Average of absolute differences between predictions and actual values
    /// - Root Mean Squared Error (RMSE): Square root of the average squared differences
    /// - Mean Absolute Percentage Error (MAPE): Average percentage differences
    ///
    /// These metrics help you understand how accurate your model is and compare different models.
    /// Lower values indicate better performance for all these metrics.
    /// </para>
    /// </remarks>
    public virtual Dictionary<string, T> EvaluateModel(Matrix<T> xTest, Vector<T> yTest)
    {
        // Check if model is trained
        if (!IsTrained)
        {
            throw new InvalidOperationException("The model must be trained before evaluation.");
        }

        // Validate inputs
        if (xTest == null)
        {
            throw new ArgumentNullException(nameof(xTest), "Test features matrix cannot be null.");
        }

        if (yTest == null)
        {
            throw new ArgumentNullException(nameof(yTest), "Test target vector cannot be null.");
        }

        if (xTest.Rows != yTest.Length)
        {
            throw new ArgumentException(
                $"Number of rows in test matrix ({xTest.Rows}) must match the length of test vector ({yTest.Length}).");
        }

        // Generate predictions
        Vector<T> predictions = Predict(xTest);

        // Calculate error metrics
        Dictionary<string, T> metrics = CalculateErrorMetrics(predictions, yTest);

        // Store metrics for later reference
        LastEvaluationMetrics = metrics;

        return metrics;
    }

    /// <summary>
    /// Calculates error metrics by comparing predictions to actual values.
    /// </summary>
    /// <param name="predictions">The predicted values.</param>
    /// <param name="actuals">The actual values.</param>
    /// <returns>A dictionary containing error metrics.</returns>
    /// <remarks>
    /// <para>
    /// This method computes standard error metrics for time series forecasting, including
    /// MAE, RMSE, MAPE, and others as appropriate for the model type.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method calculates how far off the model's predictions are from the actual values.
    /// It computes several different ways of measuring the prediction errors:
    ///
    /// - MAE (Mean Absolute Error): The average magnitude of errors, ignoring whether they're positive or negative
    /// - RMSE (Root Mean Squared Error): Emphasizes larger errors by squaring them before averaging
    /// - MAPE (Mean Absolute Percentage Error): Shows errors as percentages of the actual values
    ///
    /// These metrics help you understand not just how accurate the model is overall,
    /// but also what kinds of errors it tends to make.
    /// </para>
    /// </remarks>
    protected virtual Dictionary<string, T> CalculateErrorMetrics(Vector<T> predictions, Vector<T> actuals)
    {
        int n = predictions.Length;
        var metrics = new Dictionary<string, T>();

        // Calculate MAE (Mean Absolute Error)
        T sumAbsoluteError = NumOps.Zero;
        for (int i = 0; i < n; i++)
        {
            T error = NumOps.Subtract(predictions[i], actuals[i]);
            sumAbsoluteError = NumOps.Add(sumAbsoluteError, NumOps.Abs(error));
        }
        T mae = NumOps.Divide(sumAbsoluteError, NumOps.FromDouble(n));
        metrics["MAE"] = mae;

        // Calculate MSE (Mean Squared Error) and RMSE (Root Mean Squared Error)
        T sumSquaredError = NumOps.Zero;
        for (int i = 0; i < n; i++)
        {
            T error = NumOps.Subtract(predictions[i], actuals[i]);
            sumSquaredError = NumOps.Add(sumSquaredError, NumOps.Square(error));
        }
        T mse = NumOps.Divide(sumSquaredError, NumOps.FromDouble(n));
        T rmse = NumOps.Sqrt(mse);
        metrics["MSE"] = mse;
        metrics["RMSE"] = rmse;

        // Calculate MAPE (Mean Absolute Percentage Error)
        // Only if actuals don't contain zeros or very small values
        bool canCalculateMape = true;
        T sumAbsolutePercentageError = NumOps.Zero;
        for (int i = 0; i < n; i++)
        {
            if (NumOps.LessThan(NumOps.Abs(actuals[i]), NumOps.FromDouble(1e-10)))
            {
                canCalculateMape = false;
                break;
            }

            T percentageError = NumOps.Divide(
                NumOps.Abs(NumOps.Subtract(predictions[i], actuals[i])),
                NumOps.Abs(actuals[i])
            );
            sumAbsolutePercentageError = NumOps.Add(sumAbsolutePercentageError, percentageError);
        }

        if (canCalculateMape)
        {
            T mape = NumOps.Multiply(
                NumOps.Divide(sumAbsolutePercentageError, NumOps.FromDouble(n)),
                NumOps.FromDouble(100) // Convert to percentage
            );
            metrics["MAPE"] = mape;
        }

        return metrics;
    }

    /// <summary>
    /// Serializes the model to a byte array for storage or transmission.
    /// </summary>
    /// <returns>A byte array containing the serialized model.</returns>
    /// <remarks>
    /// <para>
    /// This method serializes the common components of the model (options, trained status, parameters)
    /// and then calls the model-specific serialization method to handle specialized data.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// Serialization converts the model's state into a format that can be saved to disk
    /// or transmitted over a network.
    ///
    /// This method:
    /// 1. Creates a memory stream to hold the serialized data
    /// 2. Writes the common configuration options shared by all models
    /// 3. Writes whether the model has been trained
    /// 4. Writes the model parameters learned during training
    /// 5. Calls the model-specific serialization method to write specialized data
    /// 6. Returns everything as a byte array
    ///
    /// This allows you to save a trained model and load it later without having to retrain it,
    /// which can save significant time for complex models trained on large datasets.
    /// </para>
    /// </remarks>
    public virtual byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Serialize common options
        writer.Write(Options.LagOrder);
        writer.Write(Options.IncludeTrend);
        writer.Write(Options.SeasonalPeriod);
        writer.Write(Options.AutocorrelationCorrection);
        writer.Write((int)Options.ModelType);

        // Serialize trained state
        writer.Write(IsTrained);

        // Serialize model parameters if trained
        if (IsTrained)
        {
            writer.Write(ModelParameters.Length);
            for (int i = 0; i < ModelParameters.Length; i++)
            {
                writer.Write(Convert.ToDouble(ModelParameters[i]));
            }

            // Serialize evaluation metrics
            writer.Write(LastEvaluationMetrics.Count);
            foreach (var kvp in LastEvaluationMetrics)
            {
                writer.Write(kvp.Key);
                writer.Write(Convert.ToDouble(kvp.Value));
            }
        }

        // Let derived classes serialize their specific data
        SerializeCore(writer);

        return ms.ToArray();
    }

    /// <summary>
    /// Deserializes the model from a byte array.
    /// </summary>
    /// <param name="data">The byte array containing the serialized model.</param>
    /// <exception cref="ArgumentNullException">Thrown when data is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when the serialized data is corrupted or incompatible.</exception>
    /// <remarks>
    /// <para>
    /// This method deserializes the common components of the model (options, trained status, parameters)
    /// and then calls the model-specific deserialization method to handle specialized data.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// Deserialization is the process of loading a previously saved model from a byte array.
    ///
    /// This method:
    /// 1. Creates a memory stream from the provided byte array
    /// 2. Reads the common configuration options shared by all models
    /// 3. Reads whether the model has been trained
    /// 4. Reads the model parameters learned during training
    /// 5. Calls the model-specific deserialization method to read specialized data
    ///
    /// After deserialization, the model is restored to the same state it was in when serialized,
    /// allowing you to make predictions without retraining the model.
    ///
    /// This is particularly useful for:
    /// - Deploying models to production environments
    /// - Sharing models between different applications
    /// - Saving computation time by not having to retrain complex models
    /// </para>
    /// </remarks>
    public virtual void Deserialize(byte[] data)
    {
        if (data == null)
        {
            throw new ArgumentNullException(nameof(data), "Serialized data cannot be null.");
        }

        try
        {
            using var ms = new MemoryStream(data);
            using var reader = new BinaryReader(ms);

            // Deserialize common options
            Options.LagOrder = reader.ReadInt32();
            Options.IncludeTrend = reader.ReadBoolean();
            Options.SeasonalPeriod = reader.ReadInt32();
            Options.AutocorrelationCorrection = reader.ReadBoolean();
            Options.ModelType = (TimeSeriesModelType)reader.ReadInt32();

            // Deserialize trained state
            IsTrained = reader.ReadBoolean();

            // Deserialize model parameters if trained
            if (IsTrained)
            {
                int parameterCount = reader.ReadInt32();
                ModelParameters = new Vector<T>(parameterCount);
                for (int i = 0; i < parameterCount; i++)
                {
                    ModelParameters[i] = NumOps.FromDouble(reader.ReadDouble());
                }

                // Deserialize evaluation metrics
                int metricsCount = reader.ReadInt32();
                LastEvaluationMetrics.Clear();
                for (int i = 0; i < metricsCount; i++)
                {
                    string key = reader.ReadString();
                    T value = NumOps.FromDouble(reader.ReadDouble());
                    LastEvaluationMetrics[key] = value;
                }
            }

            // Let derived classes deserialize their specific data
            DeserializeCore(reader);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException("Failed to deserialize model data. The data may be corrupted or incompatible with this model version.", ex);
        }
    }

    /// <summary>
    /// Serializes model-specific data to the binary writer.
    /// </summary>
    /// <param name="writer">The binary writer to write to.</param>
    /// <remarks>
    /// <para>
    /// This abstract method must be implemented by each specific model type to save
    /// its unique parameters and state.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method is responsible for saving the specific details that make each type of
    /// time series model unique. Different models have different internal structures and parameters
    /// that need to be saved separately from the common elements.
    ///
    /// For example:
    /// - An ARIMA model would save its AR, I, and MA coefficients
    /// - A TBATS model would save its level, trend, and seasonal components
    /// - A neural network model would save its weights and biases
    ///
    /// This separation allows the base class to handle common serialization tasks
    /// while each model type handles its specialized data.
    /// </para>
    /// </remarks>
    protected abstract void SerializeCore(BinaryWriter writer);

    /// <summary>
    /// Deserializes model-specific data from the binary reader.
    /// </summary>
    /// <param name="reader">The binary reader to read from.</param>
    /// <remarks>
    /// <para>
    /// This abstract method must be implemented by each specific model type to load
    /// its unique parameters and state.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method is responsible for loading the specific details that make each type of
    /// time series model unique. It reads exactly what was written by SerializeCore, in the
    /// same order, reconstructing the specialized parts of the model.
    ///
    /// It's the counterpart to SerializeCore and should read data in exactly the same
    /// order and format that it was written.
    ///
    /// This separation allows the base class to handle common deserialization tasks
    /// while each model type handles its specialized data.
    /// </para>
    /// </remarks>
    protected abstract void DeserializeCore(BinaryReader reader);

    /// <summary>
    /// Gets metadata about the time series model.
    /// </summary>
    /// <returns>A ModelMetaData object containing information about the model.</returns>
    /// <remarks>
    /// <para>
    /// This method provides comprehensive metadata about the model, including its type,
    /// configuration options, training status, evaluation metrics, and information about
    /// which features/lags are most important.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method provides important information about the model that can help you understand
    /// its characteristics and performance.
    ///
    /// The metadata includes:
    /// - The type of model (e.g., ARIMA, TBATS, Neural Network)
    /// - Configuration details (e.g., lag order, seasonality period)
    /// - Whether the model has been trained
    /// - Performance metrics from the last evaluation
    /// - Information about which features (time periods) are most influential
    ///
    /// This information is useful for documentation, model comparison, and debugging.
    /// It's like a complete summary of everything important about the model.
    /// </para>
    /// </remarks>
    public abstract ModelMetadata<T> GetModelMetadata();

    /// <summary>
    /// Gets the trainable parameters of the model as a vector.
    /// </summary>
    /// <returns>A vector containing all trainable parameters of the model.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the model has not been trained.</exception>
    /// <remarks>
    /// <para>
    /// This method returns all the parameters learned during training, combined into a single vector.
    /// These parameters determine how the model makes predictions based on input data.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method returns all the numerical values that the model has learned during training.
    ///
    /// For time series models, these parameters typically include:
    /// - Coefficients for each lag (how much each past value influences the prediction)
    /// - Trend coefficients (if trend is included)
    /// - Seasonal coefficients (if seasonality is included)
    /// - Error correction terms (if autocorrelation correction is enabled)
    ///
    /// These parameters can be:
    /// - Analyzed to understand what the model has learned
    /// - Saved for later use
    /// - Modified to adjust the model's behavior
    /// - Transferred to another model with the same structure
    /// </para>
    /// </remarks>
    public virtual Vector<T> GetParameters()
    {
        if (!IsTrained && (ModelParameters == null || ModelParameters.Length == 0))
        {
            throw new InvalidOperationException("Cannot get parameters for an untrained model.");
        }

        if (ModelParameters == null || ModelParameters.Length == 0)
        {
            throw new InvalidOperationException("Model parameters have not been initialized.");
        }

        return ModelParameters.Clone();
    }

    /// <summary>
    /// Creates a new model with the specified parameters.
    /// </summary>
    /// <param name="parameters">The vector of parameters to use for the new model.</param>
    /// <returns>A new model instance with the specified parameters.</returns>
    /// <exception cref="ArgumentNullException">Thrown when parameters is null.</exception>
    /// <exception cref="ArgumentException">Thrown when the parameters vector has incorrect length.</exception>
    /// <remarks>
    /// <para>
    /// This method creates a clone of the current model but replaces its parameters with the
    /// provided values. This allows for creating variations of a model without retraining.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method creates a copy of the current model but with different parameter values.
    ///
    /// This allows you to:
    /// - Create a model with manually specified parameters (e.g., from expert knowledge)
    /// - Make small adjustments to a trained model without full retraining
    /// - Implement ensemble models that combine multiple parameter sets
    /// - Perform what-if analysis by changing specific parameters
    ///
    /// The parameters must be in the same order and have the same meaning as those
    /// returned by the GetParameters method.
    /// </para>
    /// </remarks>
    public virtual IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        if (parameters == null)
        {
            throw new ArgumentNullException(nameof(parameters), "Parameters vector cannot be null.");
        }

        // Create a clone of the current model
        var newModel = (TimeSeriesModelBase<T>)this.Clone();

        // Apply the new parameters to the cloned model
        newModel.ApplyParameters(parameters);

        // Mark as trained since parameters have been specified
        newModel.IsTrained = true;

        return newModel;
    }

    /// <summary>
    /// Applies the provided parameters to the model.
    /// </summary>
    /// <param name="parameters">The vector of parameters to apply.</param>
    /// <exception cref="ArgumentException">Thrown when the parameters vector is invalid.</exception>
    /// <remarks>
    /// <para>
    /// This method applies the provided parameter values to the model, updating its internal state
    /// to reflect the new parameters. The implementation is model-specific and should be overridden
    /// by derived classes as needed.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method updates the model's internal parameters with new values.
    /// It's the counterpart to GetParameters and should understand the parameter
    /// vector in exactly the same way.
    ///
    /// For example, if the first 5 elements of the parameters vector represent
    /// lag coefficients, this method should apply them as lag coefficients in
    /// the model's internal structure.
    /// </para>
    /// </remarks>
    protected virtual void ApplyParameters(Vector<T> parameters)
    {
        if (parameters == null)
        {
            throw new ArgumentNullException(nameof(parameters), "Parameters vector cannot be null.");
        }

        // Store the parameters
        ModelParameters = parameters.Clone();

        // Derived classes should override this to apply parameters to their specific structures
    }

    /// <summary>
    /// Gets the indices of features (lags/time periods) actively used by the model.
    /// </summary>
    /// <returns>A collection of indices representing the active features.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the model has not been trained.</exception>
    /// <remarks>
    /// <para>
    /// This method identifies which input features (lags) have significant impact on the model's
    /// predictions, based on their corresponding parameter values.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method tells you which past time periods (lags) are most important for predictions.
    ///
    /// For example, if the result includes indices [1, 7, 12], this means:
    /// - The value from 1 period ago strongly influences the prediction
    /// - The value from 7 periods ago strongly influences the prediction (could be weekly seasonality)
    /// - The value from 12 periods ago strongly influences the prediction (could be yearly for monthly data)
    ///
    /// These active features are determined by the model's structure and learned parameters.
    /// For instance, in an ARIMA model, non-zero AR coefficients indicate active features.
    ///
    /// Understanding active features helps interpret how the model works and which
    /// historical points matter most for forecasting.
    /// </para>
    /// </remarks>
    public virtual IEnumerable<int> GetActiveFeatureIndices()
    {
        if (!IsTrained)
        {
            throw new InvalidOperationException("The model must be trained before getting active feature indices.");
        }

        List<int> activeIndices = new List<int>();

        // Consider common lag patterns based on model configuration
        for (int lag = 1; lag <= Options.LagOrder; lag++)
        {
            if (IsFeatureUsed(lag))
            {
                activeIndices.Add(lag);
            }
        }

        // If seasonal, also include seasonal lags
        if (Options.SeasonalPeriod > 0)
        {
            for (int s = 1; s <= 4; s++) // Consider up to 4 seasonal lags
            {
                int seasonalLag = s * Options.SeasonalPeriod;
                if (seasonalLag <= Options.LagOrder && IsFeatureUsed(seasonalLag))
                {
                    activeIndices.Add(seasonalLag);
                }
            }
        }

        return activeIndices;
    }

    /// <summary>
    /// Determines if a specific feature (lag) is actively used by the model.
    /// </summary>
    /// <param name="featureIndex">The index of the feature to check.</param>
    /// <returns>True if the feature is actively used; otherwise, false.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the model has not been trained.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when featureIndex is negative or exceeds the maximum lag order.</exception>
    /// <remarks>
    /// <para>
    /// This method determines whether a specific lag has a significant impact on the model's predictions,
    /// based on its corresponding parameter value. The threshold for significance is model-specific.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method checks if a specific past time period (lag) has a significant
    /// influence on the model's predictions.
    ///
    /// For example:
    /// - IsFeatureUsed(1) checks if the value from 1 period ago matters
    /// - IsFeatureUsed(7) checks if the value from 7 periods ago matters
    /// - IsFeatureUsed(12) checks if the value from 12 periods ago matters
    ///
    /// A feature is typically considered "used" if its coefficient or weight
    /// in the model is significantly different from zero.
    ///
    /// This information helps understand which historical points the model
    /// considers important when making predictions.
    /// </para>
    /// </remarks>
    public virtual bool IsFeatureUsed(int featureIndex)
    {
        if (!IsTrained)
        {
            throw new InvalidOperationException("The model must be trained before checking feature usage.");
        }

        if (featureIndex < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(featureIndex), "Feature index cannot be negative.");
        }

        if (featureIndex > Options.LagOrder)
        {
            // For indices beyond the lag order, check if it's a valid seasonal lag
            if (Options.SeasonalPeriod > 0 && featureIndex % Options.SeasonalPeriod == 0)
            {
                return NumOps.GreaterThan(GetFeatureImportance(featureIndex), NumOps.FromDouble(0.01));
            }

            return false;
        }

        // For standard lags, check if the feature importance exceeds a threshold
        T importance = GetFeatureImportance(featureIndex);
        return NumOps.GreaterThan(importance, NumOps.FromDouble(0.01));
    }

    /// <summary>
    /// Gets the importance of a specific feature (lag).
    /// </summary>
    /// <param name="featureIndex">The index of the feature.</param>
    /// <returns>A value indicating the feature's importance.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the model has not been trained.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when featureIndex is negative.</exception>
    /// <remarks>
    /// <para>
    /// This method calculates the importance of a specific lag in the model's predictions,
    /// based on its parameter value and the model's structure. The implementation is model-specific.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method estimates how important a specific past time period is
    /// for making predictions. Higher values indicate more influential features.
    ///
    /// For example, in many time series models:
    /// - Recent lags (like lag 1) often have higher importance
    /// - Seasonal lags (like lag 7 for weekly data) often have higher importance
    /// - Some lags may have near-zero importance, meaning they don't affect predictions much
    ///
    /// This information helps understand the model's internal logic and which past
    /// time periods it considers most predictive of future values.
    /// </para>
    /// </remarks>
    protected virtual T GetFeatureImportance(int featureIndex)
    {
        if (!IsTrained)
        {
            throw new InvalidOperationException("The model must be trained before getting feature importance.");
        }

        if (featureIndex < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(featureIndex), "Feature index cannot be negative.");
        }

        // Default implementation - derived classes should override with model-specific logic
        // For time series models, standard importance calculation might consider:
        // 1. The magnitude of coefficients for each lag
        // 2. The recency of the lag (more recent lags may be more important)
        // 3. Seasonal patterns (lags at seasonal intervals may be more important)

        // As a simple default, if the feature index is within the parameter range, use its absolute value
        if (featureIndex < ModelParameters.Length)
        {
            return NumOps.Abs(ModelParameters[featureIndex]);
        }

        // Otherwise, define some heuristic defaults
        if (featureIndex == 1)
        {
            // The most recent lag is usually important
            return NumOps.FromDouble(0.5);
        }
        else if (Options.SeasonalPeriod > 0 && featureIndex % Options.SeasonalPeriod == 0)
        {
            // Seasonal lags are usually important
            return NumOps.FromDouble(0.3);
        }
        else if (featureIndex <= 3)
        {
            // Recent lags are moderately important
            return NumOps.FromDouble(0.2);
        }

        // Default to very low importance for other lags
        return NumOps.FromDouble(0.01);
    }

    /// <summary>
    /// Sets the parameters for this model.
    /// </summary>
    /// <param name="parameters">A vector containing the model parameters.</param>
    /// <remarks>
    /// If the model is untrained (ModelParameters is empty), this method will
    /// resize ModelParameters to accept the incoming parameters. This allows
    /// optimizers to initialize untrained models with random parameters.
    /// </remarks>
    public virtual void SetParameters(Vector<T> parameters)
    {
        // If model is untrained (empty parameters), resize to accept the new parameters
        // This allows optimizers to initialize untrained models with random parameters
        if (ModelParameters.Length == 0 && parameters.Length > 0)
        {
            ModelParameters = new Vector<T>(parameters.Length);
        }

        if (parameters.Length != ModelParameters.Length)
        {
            throw new ArgumentException($"Expected {ModelParameters.Length} parameters, but got {parameters.Length}", nameof(parameters));
        }

        for (int i = 0; i < ModelParameters.Length; i++)
        {
            ModelParameters[i] = parameters[i];
        }
    }

    /// <summary>
    /// Sets the active feature indices for this model.
    /// </summary>
    /// <param name="featureIndices">The indices of features to activate.</param>
    public virtual void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
    {
        var activeSet = new HashSet<int>(featureIndices);

        for (int i = 0; i < ModelParameters.Length; i++)
        {
            if (!activeSet.Contains(i))
            {
                ModelParameters[i] = NumOps.Zero;
            }
        }
    }

    /// <summary>
    /// Gets the feature importance scores as a dictionary.
    /// </summary>
    /// <returns>A dictionary mapping feature names to their importance scores.</returns>
    public virtual Dictionary<string, T> GetFeatureImportance()
    {
        var result = new Dictionary<string, T>();

        for (int i = 0; i < ModelParameters.Length; i++)
        {
            string featureName = $"Lag_{i + 1}";
            result[featureName] = NumOps.Abs(ModelParameters[i]);
        }

        return result;
    }

    /// <summary>
    /// Creates a deep copy of the time series model.
    /// </summary>
    /// <returns>A new instance that is a deep copy of this model.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a completely independent copy of the model, with all parameters,
    /// options, and internal state duplicated. Modifications to the copy will not affect the
    /// original, and vice versa.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method creates a completely independent copy of the current model.
    ///
    /// A deep copy means that all components of the model are duplicated,
    /// including:
    /// - Configuration options
    /// - Learned parameters
    /// - Internal state variables
    ///
    /// This is useful when you need to:
    /// - Create multiple variations of a model for experimentation
    /// - Save a model at a specific point during training
    /// - Use the same model structure for different datasets
    ///
    /// Changes to the copy won't affect the original model and vice versa.
    /// </para>
    /// </remarks>
    public virtual IFullModel<T, Matrix<T>, Vector<T>> DeepCopy()
    {
        // Create a new instance through serialization/deserialization for a true deep copy
        byte[] serialized = this.Serialize();
        var newModel = (TimeSeriesModelBase<T>)CreateInstance();
        newModel.Deserialize(serialized);

        return newModel;
    }

    /// <summary>
    /// Creates a clone of the time series model.
    /// </summary>
    /// <returns>A new instance that is a clone of this model.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a copy of the model that shares the same options but has independent
    /// parameter values. It's a lighter-weight alternative to DeepCopy for cases where a complete
    /// independent copy is not needed.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method creates a copy of the current model with the same configuration
    /// and parameters.
    ///
    /// While DeepCopy creates a fully independent duplicate of everything in the model,
    /// Clone sometimes creates a more lightweight copy that might share some non-essential
    /// components with the original (depending on the specific model implementation).
    ///
    /// This is useful for:
    /// - Creating variations of a model for ensemble methods
    /// - Saving a snapshot of the model before making changes
    /// - Creating multiple instances for parallel training
    /// </para>
    /// </remarks>
    public virtual IFullModel<T, Matrix<T>, Vector<T>> Clone()
    {
        // Create a new instance
        var clone = (TimeSeriesModelBase<T>)CreateInstance();

        // Copy options (shallow copy is usually sufficient for options)
        clone.Options = this.Options;

        // Copy trained status
        clone.IsTrained = this.IsTrained;

        // Copy model parameters if trained
        if (this.IsTrained)
        {
            clone.ModelParameters = this.ModelParameters.Clone();

            // Copy evaluation metrics
            foreach (var kvp in this.LastEvaluationMetrics)
            {
                clone.LastEvaluationMetrics[kvp.Key] = kvp.Value;
            }
        }

        return clone;
    }

    /// <summary>
    /// Creates a new instance of the derived model class.
    /// </summary>
    /// <returns>A new instance of the same model type.</returns>
    /// <remarks>
    /// <para>
    /// This abstract factory method must be implemented by derived classes to create a new
    /// instance of their specific type. It's used by Clone and DeepCopy to ensure that
    /// the correct derived type is instantiated.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method creates a new, empty instance of the specific model type.
    /// It's used during cloning and deep copying to ensure that the copy
    /// is of the same specific type as the original.
    ///
    /// For example, if the original model is an ARIMA model, this method
    /// would create a new ARIMA model. If it's a TBATS model, it would
    /// create a new TBATS model.
    /// </para>
    /// </remarks>
    protected abstract IFullModel<T, Matrix<T>, Vector<T>> CreateInstance();

    /// <summary>
    /// Resets the model to its untrained state.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method clears all trained parameters and returns the model to its initial untrained state.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method erases all the patterns the model has learned.
    ///
    /// After calling this method:
    /// - All coefficients and learned parameters are cleared
    /// - The model behaves as if it was never trained
    /// - You would need to train it again before making predictions
    ///
    /// This is useful when you want to:
    /// - Experiment with different training data on the same model
    /// - Retrain a model from scratch with new parameters
    /// - Reset a model that might have been trained incorrectly
    /// </para>
    /// </remarks>
    public virtual void Reset()
    {
        // Clear model parameters
        ModelParameters = new Vector<T>(0);

        // Reset trained flag
        IsTrained = false;

        // Clear evaluation metrics
        LastEvaluationMetrics.Clear();

        // Derived classes should override this to reset any additional state
    }

    /// <summary>
    /// Clips a value to be within the specified range.
    /// </summary>
    /// <param name="value">The value to clip.</param>
    /// <param name="min">The minimum allowed value.</param>
    /// <param name="max">The maximum allowed value.</param>
    /// <returns>The clipped value.</returns>
    /// <remarks>
    /// <para>
    /// This utility method constrains a value to be within the specified range.
    /// If the value is less than the minimum, the minimum is returned.
    /// If the value is greater than the maximum, the maximum is returned.
    /// Otherwise, the original value is returned.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method ensures a value stays within a specified range (between min and max).
    /// It's like setting boundaries that a value cannot cross.
    ///
    /// For example, if you clip a value with min=0 and max=1:
    /// - If the value is -0.5, it returns 0 (the minimum)
    /// - If the value is 1.5, it returns 1 (the maximum)
    /// - If the value is 0.7, it returns 0.7 (unchanged, as it's within range)
    ///
    /// This is useful for:
    /// - Preventing parameters from taking extreme values
    /// - Constraining predictions to reasonable ranges
    /// - Implementing optimization algorithms that require bounded parameters
    /// </para>
    /// </remarks>
    protected T Clip(T value, T min, T max)
    {
        if (NumOps.LessThan(value, min))
        {
            return min;
        }

        if (NumOps.GreaterThan(value, max))
        {
            return max;
        }

        return value;
    }

    /// <summary>
    /// Generates a forecast for multiple steps ahead.
    /// </summary>
    /// <param name="history">The historical time series data.</param>
    /// <param name="steps">The number of steps to forecast.</param>
    /// <returns>A vector containing the forecasted values.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the model has not been trained.</exception>
    /// <exception cref="ArgumentNullException">Thrown when history is null.</exception>
    /// <exception cref="ArgumentException">Thrown when steps is not positive or history is insufficient.</exception>
    /// <remarks>
    /// <para>
    /// This method generates a multi-step forecast using the history data as the starting point.
    /// For each step, it makes a prediction and then updates the history with the predicted value
    /// to generate the next prediction.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method predicts multiple future values in sequence.
    ///
    /// For example, if you have daily data and want to forecast the next 7 days:
    /// 1. It first predicts day 1 using your historical data
    /// 2. Then it adds that prediction to the history
    /// 3. Then it predicts day 2 using the updated history (including the day 1 prediction)
    /// 4. And so on, until it has predicted all 7 days
    ///
    /// This approach lets you make predictions further into the future,
    /// but be aware that errors tend to accumulate with each step (predictions
    /// become less accurate the further ahead you forecast).
    /// </para>
    /// </remarks>
    public virtual Vector<T> Forecast(Vector<T> history, int steps)
    {
        if (!IsTrained)
        {
            throw new InvalidOperationException("The model must be trained before forecasting.");
        }

        if (history == null)
        {
            throw new ArgumentNullException(nameof(history), "History cannot be null.");
        }

        if (steps <= 0)
        {
            throw new ArgumentException("Number of forecast steps must be positive.", nameof(steps));
        }

        if (history.Length < Options.LagOrder)
        {
            throw new ArgumentException(
                $"History length ({history.Length}) must be at least equal to lag order ({Options.LagOrder}).",
                nameof(history));
        }

        // Create a working copy of the history that we can extend
        List<T> extendedHistory = new List<T>(history.Length + steps);
        for (int i = 0; i < history.Length; i++)
        {
            extendedHistory.Add(history[i]);
        }

        // Generate forecasts one step at a time
        Vector<T> forecasts = new Vector<T>(steps);
        for (int step = 0; step < steps; step++)
        {
            // Prepare input features for this forecast step
            Vector<T> features = PrepareForecastFeatures(extendedHistory, step);

            // Make prediction
            T forecast = PredictSingle(features);

            // Store forecast
            forecasts[step] = forecast;

            // Add forecast to extended history for next step
            extendedHistory.Add(forecast);
        }

        return forecasts;
    }

    /// <summary>
    /// Prepares input features for a forecast step using the extended history.
    /// </summary>
    /// <param name="extendedHistory">The historical data including any previous forecasts.</param>
    /// <param name="step">The current forecast step (0-based).</param>
    /// <returns>A vector of input features for the forecast.</returns>
    /// <remarks>
    /// <para>
    /// This method extracts the appropriate lags and constructs any additional features
    /// needed for the forecast, such as trend indicators or seasonal dummies.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method prepares the input data needed to make a forecast for a specific step.
    /// It typically extracts recent values, seasonal patterns, and trend indicators from
    /// the history (which may include previous predictions for multi-step forecasts).
    /// </para>
    /// </remarks>
    protected virtual Vector<T> PrepareForecastFeatures(List<T> extendedHistory, int step)
    {
        // This is a basic implementation that derived classes should override
        // to include model-specific feature preparation

        // For a simple AR model, we would just include the last LagOrder values
        int historyLength = extendedHistory.Count;
        int featureCount = Options.LagOrder;

        // Add space for trend if included
        if (Options.IncludeTrend)
        {
            featureCount += 1;
        }

        // Add space for seasonal dummies if seasonal
        if (Options.SeasonalPeriod > 0)
        {
            featureCount += Options.SeasonalPeriod;
        }

        Vector<T> features = new Vector<T>(featureCount);
        int featureIndex = 0;

        // Add lag features
        for (int lag = 1; lag <= Options.LagOrder; lag++)
        {
            if (historyLength - lag >= 0)
            {
                features[featureIndex++] = extendedHistory[historyLength - lag];
            }
            else
            {
                // Not enough history for this lag, use a default value
                features[featureIndex++] = NumOps.Zero;
            }
        }

        // Add trend feature if included
        if (Options.IncludeTrend)
        {
            features[featureIndex++] = NumOps.FromDouble(step + 1);
        }

        // Add seasonal dummies if seasonal
        if (Options.SeasonalPeriod > 0)
        {
            int season = (historyLength + step) % Options.SeasonalPeriod;
            for (int s = 0; s < Options.SeasonalPeriod; s++)
            {
                features[featureIndex++] = NumOps.FromDouble(s == season ? 1.0 : 0.0);
            }
        }

        return features;
    }

    public virtual int ParameterCount
    {
        get { return ModelParameters.Length; }
    }

    public virtual void SaveModel(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
            throw new ArgumentException("File path must not be null or empty.", nameof(filePath));

        try
        {
            var data = Serialize();
            var directory = Path.GetDirectoryName(filePath);
            if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
                Directory.CreateDirectory(directory);
            File.WriteAllBytes(filePath, data);
        }
        catch (IOException ex) { throw new InvalidOperationException($"Failed to save model to '{filePath}': {ex.Message}", ex); }
        catch (UnauthorizedAccessException ex) { throw new InvalidOperationException($"Access denied when saving model to '{filePath}': {ex.Message}", ex); }
        catch (System.Security.SecurityException ex) { throw new InvalidOperationException($"Security error when saving model to '{filePath}': {ex.Message}", ex); }
    }

    public virtual void LoadModel(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
            throw new ArgumentException("File path must not be null or empty.", nameof(filePath));

        try
        {
            var data = File.ReadAllBytes(filePath);
            Deserialize(data);
        }
        catch (FileNotFoundException ex) { throw new FileNotFoundException($"The specified model file does not exist: {filePath}", filePath, ex); }
        catch (IOException ex) { throw new InvalidOperationException($"File I/O error while loading model from '{filePath}': {ex.Message}", ex); }
        catch (UnauthorizedAccessException ex) { throw new InvalidOperationException($"Access denied when loading model from '{filePath}': {ex.Message}", ex); }
        catch (System.Security.SecurityException ex) { throw new InvalidOperationException($"Security error when loading model from '{filePath}': {ex.Message}", ex); }
        catch (Exception ex) { throw new InvalidOperationException($"Failed to deserialize model from file '{filePath}'. The file may be corrupted or incompatible: {ex.Message}", ex); }
    }

    public virtual ILossFunction<T> DefaultLossFunction => _defaultLossFunction;

    public virtual Vector<T> ComputeGradients(Matrix<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        var loss = lossFunction ?? DefaultLossFunction;
        var parameters = GetParameters();
        var gradients = new Vector<T>(parameters.Length);

        T epsilon = NumOps.FromDouble(1e-8);

        for (int i = 0; i < parameters.Length; i++)
        {
            var paramsPlus = parameters.Clone();
            paramsPlus[i] = NumOps.Add(paramsPlus[i], epsilon);

            var modelPlus = (TimeSeriesModelBase<T>)WithParameters(paramsPlus);
            var predPlus = modelPlus.Predict(input);
            var lossPlus = loss.CalculateLoss(predPlus, target);

            var paramsMinus = parameters.Clone();
            paramsMinus[i] = NumOps.Subtract(paramsMinus[i], epsilon);

            var modelMinus = (TimeSeriesModelBase<T>)WithParameters(paramsMinus);
            var predMinus = modelMinus.Predict(input);
            var lossMinus = loss.CalculateLoss(predMinus, target);

            var twoEpsilon = NumOps.Multiply(epsilon, NumOps.FromDouble(2.0));
            gradients[i] = NumOps.Divide(NumOps.Subtract(lossPlus, lossMinus), twoEpsilon);
        }

        return gradients;
    }

    public virtual void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        if (gradients == null)
            throw new ArgumentNullException(nameof(gradients));

        var parameters = GetParameters();

        if (gradients.Length != parameters.Length)
            throw new ArgumentException($"Gradient vector length ({gradients.Length}) must match parameter count ({parameters.Length}).");

        for (int i = 0; i < parameters.Length; i++)
        {
            parameters[i] = NumOps.Subtract(parameters[i], NumOps.Multiply(learningRate, gradients[i]));
        }

        SetParameters(parameters);
    }

    /// <summary>
    /// Saves the time series model's current state to a stream.
    /// </summary>
    /// <param name="stream">The stream to write the model state to.</param>
    /// <remarks>
    /// <para>
    /// This method serializes the time series model's parameters and configuration.
    /// It uses the existing Serialize method and writes the data to the provided stream.
    /// </para>
    /// <para><b>For Beginners:</b> This is like creating a snapshot of your trained time series model.
    ///
    /// When you call SaveState:
    /// - All learned parameters and trends are written to the stream
    /// - Model configuration and internal state are preserved
    ///
    /// This is particularly useful for:
    /// - Checkpointing during long training sessions
    /// - Saving the best model for forecasting
    /// - Knowledge distillation from time series models
    /// - Deploying forecasting models to production
    ///
    /// You can later use LoadState to restore the model.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException">Thrown when stream is null.</exception>
    /// <exception cref="IOException">Thrown when there's an error writing to the stream.</exception>
    public virtual void SaveState(Stream stream)
    {
        if (stream == null)
            throw new ArgumentNullException(nameof(stream));

        if (!stream.CanWrite)
            throw new ArgumentException("Stream must be writable.", nameof(stream));

        try
        {
            var data = this.Serialize();
            stream.Write(data, 0, data.Length);
            stream.Flush();
        }
        catch (IOException ex)
        {
            throw new IOException($"Failed to save time series model state to stream: {ex.Message}", ex);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Unexpected error while saving time series model state: {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Loads the time series model's state from a stream.
    /// </summary>
    /// <param name="stream">The stream to read the model state from.</param>
    /// <remarks>
    /// <para>
    /// This method deserializes a time series model that was previously saved with SaveState.
    /// It uses the existing Deserialize method after reading data from the stream.
    /// </para>
    /// <para><b>For Beginners:</b> This is like loading a saved snapshot of your time series model.
    ///
    /// When you call LoadState:
    /// - All parameters and trends are read from the stream
    /// - Model configuration and state are restored
    ///
    /// After loading, the model can:
    /// - Make forecasts using the restored parameters
    /// - Continue training from where it left off
    /// - Be deployed to production for time series prediction
    ///
    /// This is essential for:
    /// - Resuming interrupted training sessions
    /// - Loading the best model for forecasting
    /// - Deploying trained models to production
    /// - Knowledge distillation workflows
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException">Thrown when stream is null.</exception>
    /// <exception cref="IOException">Thrown when there's an error reading from the stream.</exception>
    /// <exception cref="InvalidOperationException">Thrown when the stream contains invalid or incompatible data.</exception>
    public virtual void LoadState(Stream stream)
    {
        if (stream == null)
            throw new ArgumentNullException(nameof(stream));

        if (!stream.CanRead)
            throw new ArgumentException("Stream must be readable.", nameof(stream));

        try
        {
            using var ms = new MemoryStream();
            stream.CopyTo(ms);
            var data = ms.ToArray();

            if (data.Length == 0)
                throw new InvalidOperationException("Stream contains no data.");

            this.Deserialize(data);
        }
        catch (IOException ex)
        {
            throw new IOException($"Failed to read time series model state from stream: {ex.Message}", ex);
        }
        catch (InvalidOperationException)
        {
            // Re-throw InvalidOperationException from Deserialize
            throw;
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException(
                $"Failed to deserialize time series model state. The stream may contain corrupted or incompatible data: {ex.Message}", ex);
        }
    }

    #region IJitCompilable Implementation

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// Time series models support JIT compilation for accelerated inference.
    /// The computation graph represents the linear time series model formula.
    /// </para>
    /// <para><b>For Beginners:</b> JIT (Just-In-Time) compilation optimizes time series models for faster predictions.
    ///
    /// Time series models often involve computing weighted sums of past observations and features.
    /// JIT compilation:
    /// - Analyzes the model's structure
    /// - Optimizes the mathematical operations
    /// - Generates specialized native code
    /// - Results in 3-7x faster predictions
    ///
    /// This is especially beneficial for:
    /// - Real-time forecasting systems
    /// - High-frequency time series (e.g., financial tick data)
    /// - Large-scale forecasting (predicting many series simultaneously)
    ///
    /// Note: JIT compilation works best for linear time series models (AR, ARMA, etc.).
    /// More complex models (e.g., those with non-linear transformations) may have
    /// limited JIT support.
    /// </para>
    /// </remarks>
    public virtual bool SupportsJitCompilation
    {
        get
        {
            // Check if model is trained and has parameters
            return IsTrained && ModelParameters != null && ModelParameters.Length > 0;
        }
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// Exports the time series model as a computation graph for JIT compilation.
    /// The graph represents the linear model formula: output = input @ model_parameters
    /// </para>
    /// <para><b>For Beginners:</b> This method converts the time series model into a computation graph.
    ///
    /// A computation graph is like a recipe that describes:
    /// 1. Take input features (past observations, seasonal indicators, etc.)
    /// 2. Multiply by learned model parameters (weights)
    /// 3. Return prediction
    ///
    /// The JIT compiler uses this graph to:
    /// - Optimize the operations
    /// - Combine steps where possible
    /// - Generate fast native code
    ///
    /// For time series models:
    /// - Input: [lag_1, lag_2, ..., lag_p, seasonal_features, trend_features]
    /// - Parameters: [φ₁, φ₂, ..., φ_p, seasonal_coeffs, trend_coeffs]
    /// - Output: prediction = sum(input[i] * parameters[i])
    ///
    /// This is similar to linear regression but specifically structured for time series data.
    /// </para>
    /// </remarks>
    public virtual ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        // Validation: Ensure inputNodes is not null
        if (inputNodes == null)
        {
            throw new ArgumentNullException(nameof(inputNodes), "Input nodes list cannot be null.");
        }

        // Validation: Ensure model is trained
        if (!IsTrained)
        {
            throw new InvalidOperationException("Cannot export computation graph: Model has not been trained yet.");
        }

        if (ModelParameters == null || ModelParameters.Length == 0)
        {
            throw new InvalidOperationException("Cannot export computation graph: Model has no parameters.");
        }

        // Create input node (placeholder for input features)
        // Time series input shape: [1, feature_count]
        // Features typically include: lag values, seasonal indicators, trend components
        var featureCount = ModelParameters.Length;
        var inputShape = new int[] { 1, featureCount };
        var inputTensor = new Tensor<T>(inputShape);
        var inputNode = new ComputationNode<T>(inputTensor);
        inputNodes.Add(inputNode);

        // Convert model parameters Vector<T> to Tensor<T>
        // Shape: [feature_count, 1] for matrix multiplication
        var paramShape = new int[] { featureCount, 1 };
        var paramData = new T[featureCount];
        for (int i = 0; i < featureCount; i++)
        {
            paramData[i] = ModelParameters[i];
        }
        var paramTensor = new Tensor<T>(paramShape, new Vector<T>(paramData));
        var paramNode = new ComputationNode<T>(paramTensor);

        // MatMul: input @ parameters
        // Result shape: [1, 1] (single prediction)
        var outputNode = TensorOperations<T>.MatrixMultiply(inputNode, paramNode);

        // Note: Most time series models don't have an explicit intercept term
        // as it's often absorbed into the parameters or handled during preprocessing.
        // If your specific model has an intercept, override this method to add it.

        return outputNode;
    }

    #endregion
}
