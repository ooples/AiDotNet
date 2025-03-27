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
/// <b>For Beginners:</b>
/// A time series model helps predict future values based on past observations.
/// 
/// This base class is like a blueprint that all specific time series models follow.
/// It ensures that every model can:
/// - Be trained on historical data
/// - Make predictions for future periods
/// - Evaluate how accurate its predictions are
/// - Be saved to disk and loaded later
/// 
/// Think of it as defining the basic capabilities that any time forecasting model
/// must have, regardless of whether it's a simple moving average or a complex
/// neural network.
/// </para>
/// </remarks>
public abstract class TimeSeriesModelBase<T> : ITimeSeriesModel<T>
{
    /// <summary>
    /// Configuration options for the time series model.
    /// </summary>
    protected TimeSeriesRegressionOptions<T> _options;
    
    /// <summary>
    /// Provides numeric operations for the specific type T.
    /// </summary>
    protected INumericOperations<T> NumOps;

    /// <summary>
    /// Initializes a new instance of the TimeSeriesModelBase class with the specified options.
    /// </summary>
    /// <param name="options">The configuration options for the time series model.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This constructor sets up the basic configuration for any time series model.
    /// 
    /// It takes an options object that specifies important settings like:
    /// - How many past values to consider (lag order)
    /// - Whether to include a trend component
    /// - The length of seasonal patterns (e.g., 7 for weekly, 12 for monthly)
    /// - Whether to correct for autocorrelation in errors
    /// 
    /// It also initializes the numeric operations appropriate for the data type being used
    /// (like addition, multiplication, etc. for double, float, or decimal values).
    /// </para>
    /// </remarks>
    protected TimeSeriesModelBase(TimeSeriesRegressionOptions<T> options)
    {
        _options = options;
        NumOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Trains the time series model using the provided input data and target values.
    /// </summary>
    /// <param name="x">The input features matrix.</param>
    /// <param name="y">The target values vector.</param>
    /// <remarks>
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
    public abstract void Train(Matrix<T> x, Vector<T> y);

    /// <summary>
    /// Generates forecasts using the trained time series model.
    /// </summary>
    /// <param name="input">The input features matrix.</param>
    /// <returns>A vector of forecasted values.</returns>
    /// <remarks>
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
    /// 
    /// This is an abstract method, so each specific model type will implement its own
    /// prediction algorithm based on how it represents and processes time series data.
    /// </para>
    /// </remarks>
    public abstract Vector<T> Predict(Matrix<T> input);

    /// <summary>
    /// Evaluates the performance of the trained model on test data.
    /// </summary>
    /// <param name="xTest">The input features matrix for testing.</param>
    /// <param name="yTest">The actual target values for testing.</param>
    /// <returns>A dictionary containing evaluation metrics.</returns>
    /// <remarks>
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
    /// - Mean Absolute Error (MAE): Average of absolute differences
    /// - Root Mean Squared Error (RMSE): Square root of the average squared differences
    /// - Mean Absolute Percentage Error (MAPE): Average percentage differences
    /// 
    /// These metrics help you understand how accurate your model is and compare different models.
    /// </para>
    /// </remarks>
    public abstract Dictionary<string, T> EvaluateModel(Matrix<T> xTest, Vector<T> yTest);

    /// <summary>
    /// Serializes the model to a byte array for storage or transmission.
    /// </summary>
    /// <returns>A byte array containing the serialized model.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// Serialization converts the model's state into a format that can be saved to disk
    /// or transmitted over a network.
    /// 
    /// This method:
    /// 1. Creates a memory stream to hold the serialized data
    /// 2. Writes the common configuration options shared by all models
    /// 3. Calls the model-specific serialization method to write specialized data
    /// 4. Returns everything as a byte array
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
        writer.Write(_options.LagOrder);
        writer.Write(_options.IncludeTrend);
        writer.Write(_options.SeasonalPeriod);
        writer.Write(_options.AutocorrelationCorrection);
        writer.Write((int)_options.ModelType);

        // Let derived classes serialize their specific data
        SerializeCore(writer);

        return ms.ToArray();
    }

    /// <summary>
    /// Deserializes the model from a byte array.
    /// </summary>
    /// <param name="data">The byte array containing the serialized model.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// Deserialization is the process of loading a previously saved model from a byte array.
    /// 
    /// This method:
    /// 1. Creates a memory stream from the provided byte array
    /// 2. Reads the common configuration options shared by all models
    /// 3. Calls the model-specific deserialization method to read specialized data
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
        using var ms = new MemoryStream(data);
        using var reader = new BinaryReader(ms);

        // Deserialize common options
        _options.LagOrder = reader.ReadInt32();
        _options.IncludeTrend = reader.ReadBoolean();
        _options.SeasonalPeriod = reader.ReadInt32();
        _options.AutocorrelationCorrection = reader.ReadBoolean();
        _options.ModelType = (TimeSeriesModelType)reader.ReadInt32();

        // Let derived classes deserialize their specific data
        DeserializeCore(reader);
    }

    /// <summary>
    /// Serializes model-specific data to the binary writer.
    /// </summary>
    /// <param name="writer">The binary writer to write to.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This abstract method must be implemented by each specific model type to save
    /// its unique parameters and state.
    /// 
    /// For example:
    /// - An ARIMA model would save its AR and MA coefficients
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
    /// <b>For Beginners:</b>
    /// This abstract method must be implemented by each specific model type to load
    /// its unique parameters and state.
    /// 
    /// It's the counterpart to SerializeCore and should read data in exactly the same
    /// order and format that it was written.
    /// 
    /// Each model type knows what specific data it needs to restore its state completely,
    /// so this method allows that specialized loading to happen while the base class
    /// handles the common deserialization tasks.
    /// </para>
    /// </remarks>
    protected abstract void DeserializeCore(BinaryReader reader);
}