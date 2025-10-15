namespace AiDotNet.TimeSeries;

using AiDotNet.Interpretability;

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
public abstract class TimeSeriesModelBase<T> : ITimeSeriesModel<T>, IGradientTransformable<T, Matrix<T>, Vector<T>>
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
    /// Set of feature indices that have been explicitly marked as active.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This set contains feature indices that have been explicitly set as active through
    /// the SetActiveFeatureIndices method, overriding the automatic determination based
    /// on feature importance.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This tracks which past time periods (lags) have been manually selected as
    /// important for the model, regardless of their calculated importance based on
    /// the model's parameters.
    /// </para>
    /// </remarks>
    private HashSet<int>? _explicitlySetActiveFeatures;

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
    protected Dictionary<string, T> LastEvaluationMetrics { get; private set; } = [];

    protected Random Random = new();

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
        ModelParameters = Vector<T>.Empty();
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
    /// Base implementation of IGradientTransformable for time series models.
    /// </summary>
    /// <param name="input">The input matrix.</param>
    /// <param name="predictionGradient">The gradient with respect to predictions.</param>
    /// <returns>The gradient with respect to model parameters.</returns>
    /// <remarks>
    /// <para>
    /// This base implementation provides a standard gradient transformation for time series models.
    /// More complex models like ProphetModel should override this method with their specific implementations.
    /// </para>
    /// <para><b>For Beginners:</b> This method transforms gradients from predictions (one per time point)
    /// to parameters (one per model coefficient). It's like translating "how wrong each prediction is"
    /// into "how much to adjust each model parameter".
    /// </para>
    /// </remarks>
    public virtual Vector<T> TransformGradient(Matrix<T> input, Vector<T> predictionGradient)
    {
        // Get current parameters
        Vector<T> parameters = GetParameters();
        int paramCount = parameters.Length;
        var parameterGradient = new Vector<T>(paramCount);

        // For simple linear time series models: X^T * prediction_gradient
        Matrix<T> transposedInput = input.Transpose();

        // Create prediction gradient as a column vector
        Matrix<T> predictionGradientMatrix = Matrix<T>.FromVector(predictionGradient);

        // Matrix multiplication gives us parameter gradients for coefficients
        Matrix<T> paramGradMatrix = transposedInput * predictionGradientMatrix;

        // Extract the column vector
        Vector<T> coefficientGradients = paramGradMatrix.GetColumn(0);

        // Copy coefficient gradients
        int coefficientCount = Math.Min(coefficientGradients.Length, paramCount);
        for (int i = 0; i < coefficientCount; i++)
        {
            parameterGradient[i] = coefficientGradients[i];
        }

        // Additional parameters (beyond coefficients) get zero gradients
        // Derived classes should override this method to handle these properly
        for (int i = coefficientCount; i < paramCount; i++)
        {
            parameterGradient[i] = NumOps.Zero;
        }

        return parameterGradient;
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
    /// <returns>A ModelMetadata object containing information about the model.</returns>
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
        if (!IsTrained)
        {
            throw new InvalidOperationException("Cannot get parameters for an untrained model.");
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
        var activeIndices = new List<int>();

        if (!IsTrained)
        {
            // If not trained, return potential features based on configuration
            for (int lag = 1; lag <= Options.LagOrder; lag++)
            {
                activeIndices.Add(lag);
            }

            if (Options.SeasonalPeriod > 0)
            {
                for (int s = 1; s <= 4; s++)
                {
                    int seasonalLag = s * Options.SeasonalPeriod;
                    if (seasonalLag <= Options.LagOrder)
                    {
                        activeIndices.Add(seasonalLag);
                    }
                }
            }
        }
        else
        {
            // Original trained model logic
            for (int lag = 1; lag <= Options.LagOrder; lag++)
            {
                if (IsFeatureUsed(lag))
                {
                    activeIndices.Add(lag);
                }
            }

            if (Options.SeasonalPeriod > 0)
            {
                for (int s = 1; s <= 4; s++)
                {
                    int seasonalLag = s * Options.SeasonalPeriod;
                    if (seasonalLag <= Options.LagOrder && IsFeatureUsed(seasonalLag))
                    {
                        activeIndices.Add(seasonalLag);
                    }
                }
            }
        }

        return activeIndices.Distinct().OrderBy(i => i);
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
        // Check if the feature is explicitly set as active first
        if (_explicitlySetActiveFeatures != null && _explicitlySetActiveFeatures.Contains(featureIndex))
        {
            return true;
        }

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
        var extendedHistory = new List<T>(history.Length + steps);
        for (int i = 0; i < history.Length; i++)
        {
            extendedHistory.Add(history[i]);
        }
        
        // Generate forecasts one step at a time
        var forecasts = new Vector<T>(steps);
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
        
        var features = new Vector<T>(featureCount);
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

    /// <summary>
    /// Sets which features (lags) should be considered active in the model.
    /// </summary>
    /// <param name="featureIndices">The indices of features to mark as active.</param>
    /// <exception cref="ArgumentNullException">Thrown when featureIndices is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any feature index is negative.</exception>
    /// <remarks>
    /// <para>
    /// This method explicitly specifies which features (lags) should be considered active
    /// in the model, overriding the automatic determination based on feature importance.
    /// Any features not included in the provided collection will be considered inactive,
    /// regardless of their importance based on model parameters.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method lets you manually select which past time periods (lags) the model
    /// should consider when making predictions. For example, if you know from domain
    /// expertise that values from 1, 7, and 30 days ago are most important for predicting
    /// your time series, you can explicitly set these lags as active using this method,
    /// regardless of what the model would automatically determine.
    /// </para>
    /// </remarks>
    public void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
    {
        if (featureIndices == null)
        {
            throw new ArgumentNullException(nameof(featureIndices), "Feature indices cannot be null.");
        }

        // Initialize the set if it doesn't exist
        _explicitlySetActiveFeatures ??= [];

        // Clear existing explicitly set features
        _explicitlySetActiveFeatures.Clear();

        // Add the new feature indices
        foreach (var index in featureIndices)
        {
            if (index < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(featureIndices), $"Feature index {index} cannot be negative.");
            }

            _explicitlySetActiveFeatures.Add(index);
        }
    }

    /// <summary>
    /// Sets the parameters of the model.
    /// </summary>
    /// <param name="parameters">The parameters to set.</param>
    /// <remarks>
    /// <para>
    /// This method sets the model parameters from a vector. It delegates to
    /// the protected ApplyParameters method which derived classes can override
    /// to handle their specific parameter structures.
    /// </para>
    /// </remarks>
    public virtual void SetParameters(Vector<T> parameters)
    {
        ApplyParameters(parameters);
    }

    #region IInterpretableModel Implementation

    protected readonly HashSet<InterpretationMethod> _enabledMethods = new();
    protected Vector<int> _sensitiveFeatures;
    protected readonly List<FairnessMetric> _fairnessMetrics = new();
    protected IModel<Matrix<T>, Vector<T>, ModelMetadata<T>> _baseModel;

    /// <summary>
    /// Gets the global feature importance across all predictions.
    /// </summary>
    public virtual async Task<Dictionary<int, T>> GetGlobalFeatureImportanceAsync()
    {
        return await InterpretableModelHelper.GetGlobalFeatureImportanceAsync(this, _enabledMethods);
    }

    /// <summary>
    /// Gets the local feature importance for a specific input.
    /// </summary>
    public virtual async Task<Dictionary<int, T>> GetLocalFeatureImportanceAsync(Matrix<T> input)
    {
        return await InterpretableModelHelper.GetLocalFeatureImportanceAsync(this, _enabledMethods, input);
    }

    /// <summary>
    /// Gets SHAP values for the given inputs.
    /// </summary>
    public virtual async Task<Matrix<T>> GetShapValuesAsync(Matrix<T> inputs)
    {
        return await InterpretableModelHelper.GetShapValuesAsync(this, _enabledMethods);
    }

    /// <summary>
    /// Gets LIME explanation for a specific input.
    /// </summary>
    public virtual async Task<LimeExplanation<T>> GetLimeExplanationAsync(Matrix<T> input, int numFeatures = 10)
    {
        return await InterpretableModelHelper.GetLimeExplanationAsync<T>(_enabledMethods, numFeatures);
    }

    /// <summary>
    /// Gets partial dependence data for specified features.
    /// </summary>
    public virtual async Task<PartialDependenceData<T>> GetPartialDependenceAsync(Vector<int> featureIndices, int gridResolution = 20)
    {
        return await InterpretableModelHelper.GetPartialDependenceAsync<T>(_enabledMethods, featureIndices, gridResolution);
    }

    /// <summary>
    /// Gets counterfactual explanation for a given input and desired output.
    /// </summary>
    public virtual async Task<CounterfactualExplanation<T>> GetCounterfactualAsync(Matrix<T> input, Vector<T> desiredOutput, int maxChanges = 5)
    {
        return await InterpretableModelHelper.GetCounterfactualAsync<T>(_enabledMethods, maxChanges);
    }

    /// <summary>
    /// Gets model-specific interpretability information.
    /// </summary>
    public virtual async Task<Dictionary<string, object>> GetModelSpecificInterpretabilityAsync()
    {
        return await InterpretableModelHelper.GetModelSpecificInterpretabilityAsync(this);
    }

    /// <summary>
    /// Generates a text explanation for a prediction.
    /// </summary>
    public virtual async Task<string> GenerateTextExplanationAsync(Matrix<T> input, Vector<T> prediction)
    {
        return await InterpretableModelHelper.GenerateTextExplanationAsync(this, input, prediction);
    }

    /// <summary>
    /// Gets feature interaction effects between two features.
    /// </summary>
    public virtual async Task<T> GetFeatureInteractionAsync(int feature1Index, int feature2Index)
    {
        return await InterpretableModelHelper.GetFeatureInteractionAsync<T>(_enabledMethods, feature1Index, feature2Index);
    }

    /// <summary>
    /// Validates fairness metrics for the given inputs.
    /// </summary>
    public virtual async Task<FairnessMetrics<T>> ValidateFairnessAsync(Matrix<T> inputs, int sensitiveFeatureIndex)
    {
        return await InterpretableModelHelper.ValidateFairnessAsync<T>(_fairnessMetrics);
    }

    /// <summary>
    /// Gets anchor explanation for a given input.
    /// </summary>
    public virtual async Task<AnchorExplanation<T>> GetAnchorExplanationAsync(Matrix<T> input, T threshold)
    {
        return await InterpretableModelHelper.GetAnchorExplanationAsync(_enabledMethods, threshold);
    }

    /// <summary>
    /// Sets the base model for interpretability analysis.
    /// </summary>
    public virtual void SetBaseModel(IModel<Matrix<T>, Vector<T>, ModelMetadata<T>> model)
    {
        _baseModel = model ?? throw new ArgumentNullException(nameof(model));
    }

    /// <summary>
    /// Enables specific interpretation methods.
    /// </summary>
    public virtual void EnableMethod(params InterpretationMethod[] methods)
    {
        foreach (var method in methods)
        {
            _enabledMethods.Add(method);
        }
    }

    /// <summary>
    /// Configures fairness evaluation settings.
    /// </summary>
    public virtual void ConfigureFairness(Vector<int> sensitiveFeatures, params FairnessMetric[] fairnessMetrics)
    {
        _sensitiveFeatures = sensitiveFeatures ?? throw new ArgumentNullException(nameof(sensitiveFeatures));
        _fairnessMetrics.Clear();
        _fairnessMetrics.AddRange(fairnessMetrics);
    }

    #endregion
}