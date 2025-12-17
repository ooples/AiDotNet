namespace AiDotNet.Regression;

/// <summary>
/// Represents a time series regression model that incorporates temporal dependencies, trends, and seasonality.
/// </summary>
/// <remarks>
/// <para>
/// The TimeSeriesRegression class extends basic regression by accounting for the temporal structure of the data.
/// It can model autoregressive components (past values affecting future values), trend components (long-term
/// directional movement), and seasonal components (recurring patterns at fixed intervals).
/// </para>
/// <para><b>For Beginners:</b> This class helps predict future values based on patterns in time-based data.
///
/// Think of it like weather forecasting:
/// - It looks at past weather patterns to predict future weather
/// - It can recognize long-term trends (like gradual warming)
/// - It can detect seasonal patterns (like winter being colder than summer)
/// - It accounts for how recent weather affects tomorrow's weather
///
/// This is useful for any data that changes over time, such as stock prices, website traffic,
/// energy consumption, or sales figures.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class TimeSeriesRegression<T> : RegressionBase<T>
{
    /// <summary>
    /// The options that configure this time series regression model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the configuration settings for the time series model, including lag order,
    /// seasonality settings, trend inclusion, and autocorrelation correction preferences.
    /// </para>
    /// <para><b>For Beginners:</b> This stores the settings that control how the model works:
    ///
    /// - How far back in time to look for patterns
    /// - Whether to look for long-term trends
    /// - Whether to account for seasonal patterns
    /// - How to handle special time-based relationships in the data
    ///
    /// These settings shape how the model analyzes and learns from your data.
    /// </para>
    /// </remarks>
    private readonly TimeSeriesRegressionOptions<T> _options;

    /// <summary>
    /// The underlying time series model that handles the core prediction logic.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field holds the specific time series model implementation (such as ARIMA, AR, MA) that
    /// is responsible for the core prediction functionality.
    /// </para>
    /// <para><b>For Beginners:</b> This is the actual prediction engine inside the regression model.
    /// 
    /// Different prediction engines work better for different types of data:
    /// - Some are good at data with short-term patterns
    /// - Others excel with long-term trends
    /// - Some balance both
    /// 
    /// The type of engine is determined by the options you provide when creating the model.
    /// </para>
    /// </remarks>
    private ITimeSeriesModel<T> _timeSeriesModel;

    /// <summary>
    /// The regularization strategy used to prevent overfitting.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the regularization method that helps prevent the model from overfitting
    /// to the training data, which can improve generalization to new data.
    /// </para>
    /// <para><b>For Beginners:</b> This helps prevent the model from becoming too specialized
    /// to your training data.
    /// 
    /// It's like adding guardrails that keep the model from going to extremes when learning patterns.
    /// Without regularization, the model might perform perfectly on your training data but poorly
    /// on new data because it learned the noise rather than the true patterns.
    /// </para>
    /// </remarks>
    private readonly IRegularization<T, Matrix<T>, Vector<T>> _regularization;

    /// <summary>
    /// Initializes a new instance of the TimeSeriesRegression class with specified options and optional regularization.
    /// </summary>
    /// <param name="options">Configuration options for the time series regression model.</param>
    /// <param name="regularization">Optional regularization method to prevent overfitting. If not provided, no regularization is applied.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new TimeSeriesRegression model with the specified options and regularization.
    /// It initializes the time series model according to the model type specified in the options.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up your time series prediction model with your chosen settings.
    /// 
    /// When creating a time series model:
    /// - The options parameter determines how the model will look for patterns in your data
    /// - The regularization parameter (optional) helps prevent the model from becoming too specialized to your training data
    /// 
    /// Think of it like configuring a new prediction tool before you start using it. The settings
    /// you choose will affect how well it works for your specific type of data.
    /// </para>
    /// </remarks>
    public TimeSeriesRegression(TimeSeriesRegressionOptions<T> options, IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, regularization)
    {
        _options = options;
        _regularization = regularization ?? new NoRegularization<T, Matrix<T>, Vector<T>>();
        _timeSeriesModel = TimeSeriesModelFactory<T, Matrix<T>, Vector<T>>.CreateModel(options.ModelType, options);
    }

    /// <summary>
    /// Trains the time series regression model on the provided data.
    /// </summary>
    /// <param name="x">Input feature matrix where each row is an observation and each column is a feature.</param>
    /// <param name="y">Target vector containing the values to predict.</param>
    /// <remarks>
    /// <para>
    /// This method prepares the input data by adding lagged features, trend, and seasonal components as specified
    /// in the options. It then trains the time series model on this prepared data and applies regularization
    /// and autocorrelation correction if configured.
    /// </para>
    /// <para><b>For Beginners:</b> This method teaches the model to recognize patterns in your data.
    /// 
    /// During training:
    /// - The data is prepared by adding time-related features
    /// - The model learns which patterns are most important for making predictions
    /// - Regularization may be applied to prevent the model from memorizing noise
    /// - Special corrections may be made for time-based patterns in the prediction errors
    /// 
    /// After training is complete, the model is ready to make predictions on new data.
    /// </para>
    /// </remarks>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        // Prepare the data
        Matrix<T> preparedX = PrepareInputData(x, y);
        Vector<T> preparedY = PrepareTargetData(y);

        // Apply regularization to the prepared input data
        if (Regularization != null)
        {
            preparedX = Regularization.Regularize(preparedX);
        }

        // Train the time series model
        _timeSeriesModel.Train(preparedX, preparedY);

        // Extract coefficients and apply regularization
        ExtractCoefficients();
        ApplyRegularization();

        if (_options.AutocorrelationCorrection)
        {
            ApplyAutocorrelationCorrection(preparedX, preparedY);
        }
    }

    /// <summary>
    /// Applies regularization to the model coefficients.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method applies the configured regularization technique to the model coefficients to prevent overfitting.
    /// Regularization helps the model generalize better to new, unseen data.
    /// </para>
    /// <para><b>For Beginners:</b> This method helps keep the model's predictions reasonable.
    /// 
    /// Regularization:
    /// - Prevents the model from relying too heavily on any single feature
    /// - Reduces the risk of the model becoming too specialized to your training data
    /// - Makes the model more robust when facing new data
    /// 
    /// It's similar to encouraging a student to solve problems using general principles 
    /// rather than memorizing specific answers.
    /// </para>
    /// </remarks>
    private void ApplyRegularization()
    {
        if (Coefficients != null && Regularization != null)
        {
            Coefficients = Regularization.Regularize(Coefficients);
        }
    }

    /// <summary>
    /// Applies autocorrelation correction to improve the model's handling of time-dependent error patterns.
    /// </summary>
    /// <param name="x">The prepared input feature matrix.</param>
    /// <param name="y">The prepared target vector.</param>
    /// <remarks>
    /// <para>
    /// This method implements the Cochrane-Orcutt procedure to correct for autocorrelation in the residuals.
    /// It iteratively estimates the autocorrelation coefficient and transforms the data accordingly until
    /// convergence or a maximum number of iterations is reached.
    /// </para>
    /// <para><b>For Beginners:</b> This method improves predictions by accounting for patterns in the prediction errors.
    /// 
    /// In time series data:
    /// - If today's prediction is too high, tomorrow's might also tend to be too high
    /// - This creates a pattern in the errors that can be corrected
    /// 
    /// This method:
    /// - Looks for patterns in how the predictions are wrong
    /// - Adjusts the model to account for these patterns
    /// - Repeats the process until the adjustments stop making significant improvements
    /// 
    /// It's like learning not just from the data itself, but also from your mistakes in predicting it.
    /// </para>
    /// </remarks>
    private void ApplyAutocorrelationCorrection(Matrix<T> x, Vector<T> y)
    {
        const int maxIterations = 20;
        const double convergenceThreshold = 1e-5;
        T previousAutocorrelation = NumOps.Zero;
        T autocorrelation = NumOps.Zero;

        for (int iteration = 0; iteration < maxIterations; iteration++)
        {
            // Calculate residuals using the time series model
            Vector<T> predictions = _timeSeriesModel.Predict(x);
            Vector<T> residuals = y.Subtract(predictions);

            // Calculate autocorrelation
            autocorrelation = CalculateAutocorrelation(residuals);

            // Check for convergence
            if (NumOps.LessThan(NumOps.Abs(NumOps.Subtract(autocorrelation, previousAutocorrelation)), NumOps.FromDouble(convergenceThreshold)))
            {
                break;
            }

            // Apply Cochrane-Orcutt transformation
            Matrix<T> correctedX = new(x.Rows - 1, x.Columns);
            Vector<T> correctedY = new(y.Length - 1);

            for (int i = 1; i < x.Rows; i++)
            {
                for (int j = 0; j < x.Columns; j++)
                {
                    correctedX[i - 1, j] = NumOps.Subtract(x[i, j], NumOps.Multiply(autocorrelation, x[i - 1, j]));
                }
                correctedY[i - 1] = NumOps.Subtract(y[i], NumOps.Multiply(autocorrelation, y[i - 1]));
            }

            // Retrain the time series model with corrected data
            _timeSeriesModel.Train(correctedX, correctedY);

            previousAutocorrelation = autocorrelation;
        }

        // Apply final correction to the original data
        if (!MathHelper.AlmostEqual(autocorrelation, NumOps.Zero))
        {
            Matrix<T> finalCorrectedX = new(x.Rows, x.Columns);
            Vector<T> finalCorrectedY = new(y.Length);

            for (int j = 0; j < x.Columns; j++)
            {
                finalCorrectedX[0, j] = x[0, j];
            }
            finalCorrectedY[0] = y[0];

            for (int i = 1; i < x.Rows; i++)
            {
                for (int j = 0; j < x.Columns; j++)
                {
                    finalCorrectedX[i, j] = NumOps.Subtract(x[i, j], NumOps.Multiply(autocorrelation, x[i - 1, j]));
                }
                finalCorrectedY[i] = NumOps.Subtract(y[i], NumOps.Multiply(autocorrelation, y[i - 1]));
            }

            // Final retraining of the time series model with the fully corrected data
            _timeSeriesModel.Train(finalCorrectedX, finalCorrectedY);
        }
    }

    /// <summary>
    /// Prepares the input data by adding lagged features, trend, and seasonal components.
    /// </summary>
    /// <param name="x">The original input feature matrix.</param>
    /// <param name="y">The target vector.</param>
    /// <returns>A transformed matrix with additional time-based features.</returns>
    /// <remarks>
    /// <para>
    /// This method transforms the input data to include lagged values of both the features and target,
    /// as well as trend and seasonal components as specified in the options. This enriched feature set
    /// allows the model to capture temporal dependencies and patterns.
    /// </para>
    /// <para><b>For Beginners:</b> This method enriches your data with time-related information.
    /// 
    /// It adds new features to your data that help the model recognize time patterns:
    /// - Past values (lags) that help find patterns like "today's value often depends on yesterday's"
    /// - Trend features that help identify long-term increases or decreases
    /// - Seasonal features that help capture repeating patterns (daily, weekly, monthly, etc.)
    /// 
    /// For example, if predicting store sales:
    /// - Lag features might show that sales typically drop the day after a big sale
    /// - Trend features might capture that sales are growing 5% each month
    /// - Seasonal features might show that sales always spike on weekends
    /// 
    /// These additional features make the model much better at predicting time-based data.
    /// </para>
    /// </remarks>
    private Matrix<T> PrepareInputData(Matrix<T> x, Vector<T> y)
    {
        int n = y.Length;
        int laggedFeatures = _options.LagOrder * (x.Columns + 1); // +1 for lagged y
        int trendFeatures = _options.IncludeTrend ? 1 : 0;
        int seasonalFeatures = _options.SeasonalPeriod > 0 ? _options.SeasonalPeriod - 1 : 0;
        int totalFeatures = x.Columns + laggedFeatures + trendFeatures + seasonalFeatures;

        Matrix<T> preparedX = new(n - _options.LagOrder, totalFeatures);

        // Add original features
        for (int i = _options.LagOrder; i < n; i++)
        {
            for (int j = 0; j < x.Columns; j++)
            {
                preparedX[i - _options.LagOrder, j] = x[i, j];
            }
        }

        // Add lagged features
        int column = x.Columns;
        for (int lag = 1; lag <= _options.LagOrder; lag++)
        {
            for (int i = _options.LagOrder; i < n; i++)
            {
                for (int j = 0; j < x.Columns; j++)
                {
                    preparedX[i - _options.LagOrder, column] = x[i - lag, j];
                    column++;
                }
                preparedX[i - _options.LagOrder, column] = y[i - lag];
                column++;
            }
        }

        // Add trend feature
        if (_options.IncludeTrend)
        {
            for (int i = 0; i < preparedX.Rows; i++)
            {
                preparedX[i, column] = NumOps.FromDouble(i + 1);
            }
            column++;
        }

        // Add seasonal features
        if (_options.SeasonalPeriod > 0)
        {
            for (int i = 0; i < preparedX.Rows; i++)
            {
                for (int s = 1; s < _options.SeasonalPeriod; s++)
                {
                    preparedX[i, column] = NumOps.FromDouble((i + _options.LagOrder) % _options.SeasonalPeriod == s ? 1 : 0);
                    column++;
                }
            }
        }

        // Apply regularization
        if (_regularization != null)
        {
            preparedX = _regularization.Regularize(preparedX);
        }

        return preparedX;
    }

    /// <summary>
    /// Prepares the target data by adjusting for the lag order.
    /// </summary>
    /// <param name="y">The original target vector.</param>
    /// <returns>A transformed target vector aligned with the prepared input data.</returns>
    /// <remarks>
    /// <para>
    /// This method adjusts the target vector to account for the lag order specified in the options,
    /// ensuring that the targets are properly aligned with the prepared input features.
    /// </para>
    /// <para><b>For Beginners:</b> This method prepares the values you want to predict.
    /// 
    /// When working with time series data and lags:
    /// - We need to align the target values with the input features
    /// - If we're using past values to predict future ones, we need to adjust when those future values occur
    /// 
    /// This ensures that when we ask "what happens next?" the model properly understands what "next" means
    /// in relation to the time points in our data.
    /// </para>
    /// </remarks>
    private Vector<T> PrepareTargetData(Vector<T> y)
    {
        return new Vector<T>([.. y.Skip(_options.LagOrder)]);
    }

    /// <summary>
    /// Extracts the relevant coefficients from the model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method extracts the main feature coefficients from the model, excluding the trend and seasonal
    /// coefficients which are handled separately.
    /// </para>
    /// <para><b>For Beginners:</b> This method organizes the model's learned patterns.
    /// 
    /// After training:
    /// - The model has learned what features are most important for predictions
    /// - This method separates the regular feature importance from trend and seasonal patterns
    /// - This separation makes the model easier to interpret and use
    /// 
    /// Think of it like organizing your tools after a project - putting the core tools in one box
    /// and the specialized tools in separate containers.
    /// </para>
    /// </remarks>
    private void ExtractCoefficients()
    {
        int originalFeatures = Coefficients.Length - (_options.LagOrder * (Coefficients.Length + 1) + (_options.IncludeTrend ? 1 : 0) + (_options.SeasonalPeriod > 0 ? _options.SeasonalPeriod - 1 : 0));

        // Remove trend and seasonal coefficients from the main Coefficients vector
        Coefficients = new Vector<T>([.. Coefficients.Take(originalFeatures)]);
    }

    /// <summary>
    /// Calculates the first-order autocorrelation coefficient for the given residuals.
    /// </summary>
    /// <param name="residuals">The prediction errors (residuals) from the model.</param>
    /// <returns>The first-order autocorrelation coefficient.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the first-order autocorrelation coefficient, which measures the correlation
    /// between consecutive error terms. A high positive value indicates that errors tend to be followed by
    /// errors of the same sign, while a high negative value indicates alternating error patterns.
    /// </para>
    /// <para><b>For Beginners:</b> This method measures how much today's prediction error relates to yesterday's error.
    /// 
    /// In time series predictions:
    /// - If you tend to predict too high today when you predicted too high yesterday, that's positive autocorrelation
    /// - If you tend to predict too high today when you predicted too low yesterday, that's negative autocorrelation
    /// - A value near zero means today's errors don't follow a pattern related to yesterday's errors
    /// 
    /// Finding this pattern helps improve predictions by adjusting for these systematic errors.
    /// </para>
    /// </remarks>
    private T CalculateAutocorrelation(Vector<T> residuals)
    {
        T numerator = NumOps.Zero;
        T denominator = NumOps.Zero;

        for (int i = 1; i < residuals.Length; i++)
        {
            numerator = NumOps.Add(numerator, NumOps.Multiply(residuals[i], residuals[i - 1]));
            denominator = NumOps.Add(denominator, NumOps.Multiply(residuals[i - 1], residuals[i - 1]));
        }

        return NumOps.Divide(numerator, denominator);
    }

    /// <summary>
    /// Extracts the trend coefficients from the model if trend was included in the options.
    /// </summary>
    /// <returns>A vector containing the trend coefficients, or an empty vector if trend was not included.</returns>
    /// <remarks>
    /// <para>
    /// This method extracts the trend coefficients from the model, which capture the long-term directional
    /// movement in the data. These coefficients are separate from the main feature coefficients.
    /// </para>
    /// <para><b>For Beginners:</b> This method gets the part of the model that measures steady increases or decreases over time.
    /// 
    /// The trend component:
    /// - Captures gradual, consistent changes in your data over time
    /// - For example, a steady 2% increase in sales each month or a gradual decline in temperature
    /// - Helps make predictions that account for these ongoing changes
    /// 
    /// If you didn't include trend in your model options, this returns an empty result.
    /// </para>
    /// </remarks>
    private Vector<T> ExtractTrendCoefficients()
    {
        if (_options.IncludeTrend)
        {
            int trendIndex = Coefficients.Length - 1;
            if (_options.SeasonalPeriod > 0)
            {
                trendIndex -= _options.SeasonalPeriod - 1;
            }

            return new Vector<T>([Coefficients[trendIndex]]);
        }

        return new Vector<T>(0);
    }

    /// <summary>
    /// Extracts the seasonal coefficients from the model if seasonality was included in the options.
    /// </summary>
    /// <returns>A vector containing the seasonal coefficients, or an empty vector if seasonality was not included.</returns>
    /// <remarks>
    /// <para>
    /// This method extracts the seasonal coefficients from the model, which capture recurring patterns
    /// at fixed intervals in the data. These coefficients are separate from the main feature coefficients.
    /// </para>
    /// <para><b>For Beginners:</b> This method gets the part of the model that captures repeating patterns.
    /// 
    /// The seasonal component:
    /// - Identifies patterns that repeat at regular intervals
    /// - Examples include daily patterns (busier in mornings/evenings), weekly patterns (busier on weekends), 
    ///   or yearly patterns (higher sales during holidays)
    /// - Helps make predictions that account for these cyclical changes
    /// 
    /// If you didn't include seasonality in your model options, this returns an empty result.
    /// </para>
    /// </remarks>
    private Vector<T> ExtractSeasonalCoefficients()
    {
        if (_options.SeasonalPeriod > 0)
        {
            return new Vector<T>(Coefficients.Skip(Coefficients.Length - (_options.SeasonalPeriod - 1)).ToArray());
        }

        return new Vector<T>(0);
    }

    /// <summary>
    /// Predicts target values for the given input features.
    /// </summary>
    /// <param name="input">Input feature matrix for which predictions should be made.</param>
    /// <returns>A vector containing the predicted values.</returns>
    /// <remarks>
    /// <para>
    /// This method generates predictions for the given input features by first preparing the input data
    /// with lagged features, trend, and seasonal components, and then applying the trained model.
    /// The trend and seasonal components are added to the base predictions to produce the final result.
    /// </para>
    /// <para><b>For Beginners:</b> This method makes predictions based on new data.
    /// 
    /// When making predictions:
    /// - The new data is prepared the same way as during training
    /// - The model applies what it learned to generate base predictions
    /// - Trend and seasonal effects are added to improve accuracy
    /// - The final predictions account for all time-related patterns the model found
    /// 
    /// For example, predicting store sales might combine:
    /// - Base prediction using features like promotion and price
    /// - Plus the ongoing trend of 5% growth per month
    /// - Plus the seasonal effect of higher weekend sales
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> input)
    {
        Matrix<T> preparedInput = PrepareInputData(input, new Vector<T>(input.Rows)); // Dummy y vector
        Vector<T> predictions = base.Predict(preparedInput);

        Vector<T> trendCoefficients = ExtractTrendCoefficients();
        Vector<T> seasonalCoefficients = ExtractSeasonalCoefficients();

        // Add trend and seasonality components
        for (int i = 0; i < predictions.Length; i++)
        {
            if (_options.IncludeTrend)
            {
                predictions[i] = NumOps.Add(predictions[i], NumOps.Multiply(trendCoefficients[0], NumOps.FromDouble(i + 1)));
            }

            if (_options.SeasonalPeriod > 0)
            {
                int seasonIndex = i % _options.SeasonalPeriod;
                if (seasonIndex > 0)
                {
                    predictions[i] = NumOps.Add(predictions[i], seasonalCoefficients[seasonIndex - 1]);
                }
            }
        }

        return predictions;
    }

    /// <summary>
    /// Returns the type of this regression model.
    /// </summary>
    /// <returns>Always returns ModelType.TimeSeriesRegression.</returns>
    /// <remarks>
    /// <para>
    /// This method identifies the type of this regression model as a time series regression, which
    /// helps with model type checking and serialization.
    /// </para>
    /// <para><b>For Beginners:</b> This method simply identifies what kind of model this is.
    /// 
    /// It's like a name tag for the model that helps the system distinguish between different types
    /// of models (like time series regression vs. linear regression). This is useful when saving or
    /// loading models, or when deciding how to process them.
    /// </para>
    /// </remarks>
    protected override ModelType GetModelType() => ModelType.TimeSeriesRegression;

    /// <summary>
    /// Converts the model into a byte array that can be stored or transmitted.
    /// </summary>
    /// <returns>A byte array representation of the model.</returns>
    /// <remarks>
    /// <para>
    /// This method serializes the time series regression model, including its base class data and
    /// specific configuration options, into a byte array. This allows the model to be saved to disk,
    /// transmitted over a network, or otherwise persisted.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves your trained model to a format that can be stored or shared.
    /// 
    /// Serialization:
    /// - Converts your trained model into simple bytes that can be saved
    /// - Preserves all the patterns and relationships the model has learned
    /// - Includes all settings and configuration options
    /// 
    /// It's like taking a snapshot of the model that can be saved to a file or database.
    /// Later, you can use Deserialize to recreate the exact same model without retraining.
    /// </para>
    /// </remarks>
    public override byte[] Serialize()
    {
        using (MemoryStream ms = new MemoryStream())
        using (BinaryWriter writer = new BinaryWriter(ms))
        {
            // Serialize base class data
            byte[] baseData = base.Serialize();
            writer.Write(baseData.Length);
            writer.Write(baseData);

            // Serialize TimeSeriesRegression specific data
            writer.Write(_options.LagOrder);
            writer.Write(_options.IncludeTrend);
            writer.Write(_options.SeasonalPeriod);
            writer.Write(_options.AutocorrelationCorrection);
            writer.Write((int)_options.ModelType);

            // Serialize the time series model
            byte[] modelData = _timeSeriesModel.Serialize();
            writer.Write(modelData.Length);
            writer.Write(modelData);

            return ms.ToArray();
        }
    }

    /// <summary>
    /// Restores the model state from a byte array previously created by the Serialize method.
    /// </summary>
    /// <param name="modelData">The byte array containing the serialized model.</param>
    /// <remarks>
    /// <para>
    /// This method deserializes a time series regression model from a byte array, reconstructing the
    /// base class data, configuration options, and time series model. This allows a previously saved
    /// model to be restored without retraining.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads a previously saved model.
    /// 
    /// Deserialization:
    /// - Takes the bytes created by Serialize and converts them back into a working model
    /// - Restores all the learned patterns and relationships
    /// - Recreates the exact same model configuration
    /// 
    /// It's like restoring a snapshot of the model, allowing you to use a trained model
    /// without having to retrain it each time. This saves time and ensures consistent predictions.
    /// </para>
    /// </remarks>
    public override void Deserialize(byte[] modelData)
    {
        using (MemoryStream ms = new MemoryStream(modelData))
        using (BinaryReader reader = new BinaryReader(ms))
        {
            // Deserialize base class data
            int baseDataLength = reader.ReadInt32();
            byte[] baseData = reader.ReadBytes(baseDataLength);
            base.Deserialize(baseData);

            // Deserialize TimeSeriesRegression specific data
            _options.LagOrder = reader.ReadInt32();
            _options.IncludeTrend = reader.ReadBoolean();
            _options.SeasonalPeriod = reader.ReadInt32();
            _options.AutocorrelationCorrection = reader.ReadBoolean();
            _options.ModelType = (TimeSeriesModelType)reader.ReadInt32();

            // Deserialize the time series model
            int modelDataLength = reader.ReadInt32();
            byte[] timeSeriesModelData = reader.ReadBytes(modelDataLength);
            _timeSeriesModel = TimeSeriesModelFactory<T, Matrix<T>, Vector<T>>.CreateModel(_options.ModelType, _options);
            _timeSeriesModel.Deserialize(timeSeriesModelData);
        }
    }

    /// <summary>
    /// Creates a new instance of the time series regression model with the same configuration.
    /// </summary>
    /// <returns>
    /// A new instance of <see cref="TimeSeriesRegression{T}"/> with the same configuration as the current instance.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method creates a new time series regression model that has the same configuration as the current instance.
    /// It's used for model persistence, cloning, and transferring the model's configuration to new instances.
    /// </para>
    /// <para><b>For Beginners:</b> This method makes a fresh copy of the current model with the same settings.
    /// 
    /// It's like creating a blueprint copy of your model that can be used to:
    /// - Save your model's settings
    /// - Create a new identical model
    /// - Transfer your model's configuration to another system
    /// 
    /// This is useful when you want to:
    /// - Create multiple similar models
    /// - Save a model's configuration for later use
    /// - Reset a model while keeping its settings
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        // Create and return a new instance with the same configuration
        return new TimeSeriesRegression<T>(_options, _regularization);
    }
}
