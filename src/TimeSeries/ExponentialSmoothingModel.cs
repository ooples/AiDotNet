namespace AiDotNet.TimeSeries;

/// <summary>
/// Represents a model that implements exponential smoothing for time series forecasting.
/// </summary>
/// <remarks>
/// <para>
/// Exponential smoothing is a time series forecasting method that assigns exponentially decreasing weights
/// to past observations, giving more importance to recent data while still considering older observations.
/// This model supports simple, double (with trend), and triple (with trend and seasonality) exponential smoothing.
/// </para>
/// <para><b>For Beginners:</b> Exponential smoothing helps predict future values based on past data.
/// 
/// Think of it like predicting tomorrow's weather:
/// - Recent weather (yesterday, today) is more important than weather from weeks ago
/// - You can identify trends (getting warmer over time)
/// - You can account for seasons (summer is usually warmer than winter)
/// 
/// For example, if you're forecasting daily sales:
/// - Simple smoothing: Uses a weighted average of past values, giving more weight to recent sales
/// - Double smoothing: Also captures if sales are trending up or down
/// - Triple smoothing: Adds seasonal patterns (e.g., higher sales on weekends)
/// 
/// Exponential smoothing is called "exponential" because the weight given to older data
/// decreases exponentially as the data gets older.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class ExponentialSmoothingModel<T> : TimeSeriesModelBase<T>
{
    private T _alpha; // Smoothing factor for level
    private T _beta;  // Smoothing factor for trend (if applicable)
    private T _gamma; // Smoothing factor for seasonality (if applicable)
    private Vector<T> _initialValues;

    /// <summary>
    /// Initializes a new instance of the <see cref="ExponentialSmoothingModel{T}"/> class with the specified options.
    /// </summary>
    /// <param name="options">The configuration options for the exponential smoothing model.</param>
    /// <remarks>
    /// <para>
    /// This constructor initializes the exponential smoothing model with the provided configuration options.
    /// The options specify parameters such as smoothing factors, whether to include trend and seasonality
    /// components, and the seasonal period if applicable.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up your forecasting model with your chosen settings.
    /// 
    /// When creating the model, you can specify:
    /// - Alpha: How much weight to give to recent data vs. older data (0-1)
    /// - Beta: How much to consider trends in your data (0-1)
    /// - Gamma: How much to consider seasonal patterns (0-1)
    /// - Whether to include trend analysis
    /// - Whether to include seasonal analysis
    /// 
    /// Higher values (closer to 1) make the model respond quickly to changes.
    /// Lower values (closer to 0) make the model more stable and less reactive to short-term fluctuations.
    /// </para>
    /// </remarks>
    public ExponentialSmoothingModel(ExponentialSmoothingOptions<T> options) : base(options)
    {
        _alpha = NumOps.FromDouble(options.InitialAlpha);
        _beta = options.UseTrend ? NumOps.FromDouble(options.InitialBeta) : NumOps.Zero;
        _gamma = options.UseSeasonal ? NumOps.FromDouble(options.InitialGamma) : NumOps.Zero;
        _initialValues = Vector<T>.Empty();
    }

    /// <summary>
    /// Trains the exponential smoothing model on the provided input and output data.
    /// </summary>
    /// <param name="x">The input features matrix (typically time indicators or related variables).</param>
    /// <param name="y">The target values vector (the time series data to forecast).</param>
    /// <remarks>
    /// <para>
    /// This method trains the exponential smoothing model by finding optimal values for the smoothing
    /// parameters (alpha, beta, gamma) using a grid search approach. It also estimates initial values
    /// for the level, trend, and seasonal components based on the training data.
    /// </para>
    /// <para><b>For Beginners:</b> This teaches the model patterns in your data.
    /// 
    /// When training the model:
    /// - It tries different combinations of alpha, beta, and gamma values
    /// - It keeps the combination that gives the most accurate predictions
    /// - It sets up initial estimates for level, trend, and seasonality
    /// 
    /// Think of it like adjusting the dials on a radio until you get the clearest signal.
    /// The model tests many different settings automatically and picks the best ones.
    /// </para>
    /// </remarks>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        // Implement Exponential Smoothing training logic
        (_alpha, _beta, _gamma) = EstimateParametersGridSearch(y);
        _initialValues = EstimateInitialValues(y);
    }

    /// <summary>
    /// Estimates optimal smoothing parameters using a grid search approach.
    /// </summary>
    /// <param name="y">The target values vector (the time series data).</param>
    /// <returns>A tuple containing the optimal alpha, beta, and gamma values.</returns>
    /// <remarks>
    /// <para>
    /// This method performs a grid search to find the optimal smoothing parameters. It tests various
    /// combinations of alpha, beta, and gamma values and selects the combination that minimizes the
    /// mean squared error between the forecasted and actual values.
    /// </para>
    /// <para><b>For Beginners:</b> This finds the best settings for your forecasting model.
    /// 
    /// The grid search works by:
    /// - Testing many combinations of alpha, beta, and gamma values
    /// - For each combination, making predictions and measuring how accurate they are
    /// - Selecting the combination that gives the smallest errors
    /// 
    /// It's like trying on different pairs of glasses and picking the ones that give you
    /// the clearest vision.
    /// </para>
    /// </remarks>
    private (T alpha, T beta, T gamma) EstimateParametersGridSearch(Vector<T> y)
    {
        T bestAlpha = NumOps.Zero, bestBeta = NumOps.Zero, bestGamma = NumOps.Zero;
        T bestMSE = NumOps.MaxValue;

        for (double a = 0; a <= 1; a += 0.1)
        {
            for (double b = 0; b <= 1; b += 0.1)
            {
                for (double g = 0; g <= 1; g += 0.1)
                {
                    T alpha = NumOps.FromDouble(a);
                    T beta = NumOps.FromDouble(b);
                    T gamma = NumOps.FromDouble(g);

                    T mse = CalculateMSE(y, alpha, beta, gamma);

                    if (NumOps.LessThan(mse, bestMSE))
                    {
                        bestMSE = mse;
                        bestAlpha = alpha;
                        bestBeta = beta;
                        bestGamma = gamma;
                    }
                }
            }
        }

        return (bestAlpha, bestBeta, bestGamma);
    }

    /// <summary>
    /// Calculates the Mean Squared Error (MSE) for a given set of smoothing parameters.
    /// </summary>
    /// <param name="y">The target values vector (the time series data).</param>
    /// <param name="alpha">The smoothing factor for level.</param>
    /// <param name="beta">The smoothing factor for trend.</param>
    /// <param name="gamma">The smoothing factor for seasonality.</param>
    /// <returns>The mean squared error between the forecasted and actual values.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the Mean Squared Error (MSE) between forecasted values and actual values
    /// using the provided smoothing parameters. The MSE is a common metric for evaluating forecast accuracy,
    /// where lower values indicate better performance.
    /// </para>
    /// <para><b>For Beginners:</b> This measures how accurate the model is with specific settings.
    /// 
    /// Mean Squared Error (MSE):
    /// - Calculates the difference between predicted and actual values
    /// - Squares these differences (to make all values positive)
    /// - Takes the average of these squared differences
    /// 
    /// Lower MSE means more accurate predictions.
    /// For example, if your model predicts 10 and the actual value is 12, the error is 2.
    /// Squaring this gives 4, and averaging all these squared errors gives the MSE.
    /// </para>
    /// </remarks>
    private T CalculateMSE(Vector<T> y, T alpha, T beta, T gamma)
    {
        Vector<T> predictions = ForecastWithParameters(y, alpha, beta, gamma);
        return StatisticsHelper<T>.CalculateMeanSquaredError(predictions, y);
    }

    /// <summary>
    /// Generates forecasts using specified smoothing parameters.
    /// </summary>
    /// <param name="y">The target values vector (the time series data).</param>
    /// <param name="alpha">The smoothing factor for level.</param>
    /// <param name="beta">The smoothing factor for trend.</param>
    /// <param name="gamma">The smoothing factor for seasonality.</param>
    /// <returns>A vector containing the forecasted values.</returns>
    /// <remarks>
    /// <para>
    /// This method generates forecasts for the given time series using the specified smoothing parameters.
    /// It implements the exponential smoothing algorithm by iteratively updating the level, trend, and
    /// seasonal components based on the observed values and the smoothing parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This makes predictions using specific alpha, beta, and gamma values.
    /// 
    /// The forecasting process:
    /// - Starts with initial estimates for level, trend, and seasonality
    /// - For each time period, makes a prediction based on these components
    /// - Updates the components based on the observed value and the smoothing parameters
    /// 
    /// The formula works by:
    /// - Level: Weighted average of the current observation and the previous level
    /// - Trend: Weighted average of the change in level and the previous trend
    /// - Seasonality: Weighted average of the seasonal factor and the previous seasonal factor
    /// 
    /// These updates allow the model to adapt to changes in the data over time.
    /// </para>
    /// </remarks>
    private Vector<T> ForecastWithParameters(Vector<T> y, T alpha, T beta, T gamma)
    {
        Vector<T> forecasts = new(y.Length);
        T level = y[0];
        T trend = _options.IncludeTrend ? NumOps.Subtract(y[1], y[0]) : NumOps.Zero;
        Vector<T> seasonalFactors = _options.SeasonalPeriod > 0 ? EstimateInitialSeasonalFactors(y) : Vector<T>.Empty();

        for (int i = 0; i < y.Length; i++)
        {
            T forecast;
            if (_options.SeasonalPeriod > 0)
            {
                forecast = NumOps.Multiply(NumOps.Add(level, trend), seasonalFactors[i % _options.SeasonalPeriod]);
            }
            else
            {
                forecast = NumOps.Add(level, trend);
            }

            forecasts[i] = forecast;

            if (i < y.Length - 1)
            {
                T observation = y[i + 1];
                T oldLevel = level;

                // Update level
                level = NumOps.Add(
                    NumOps.Multiply(alpha, NumOps.Divide(observation, _options.SeasonalPeriod > 0 ? seasonalFactors[i % _options.SeasonalPeriod] : NumOps.One)),
                    NumOps.Multiply(NumOps.Subtract(NumOps.One, alpha), NumOps.Add(oldLevel, trend))
                );

                // Update trend
                if (_options.IncludeTrend)
                {
                    trend = NumOps.Add(
                        NumOps.Multiply(beta, NumOps.Subtract(level, oldLevel)),
                        NumOps.Multiply(NumOps.Subtract(NumOps.One, beta), trend)
                    );
                }

                // Update seasonal factors
                if (_options.SeasonalPeriod > 0)
                {
                    int seasonIndex = i % _options.SeasonalPeriod;
                    seasonalFactors[seasonIndex] = NumOps.Add(
                        NumOps.Multiply(gamma, NumOps.Divide(observation, level)),
                        NumOps.Multiply(NumOps.Subtract(NumOps.One, gamma), seasonalFactors[seasonIndex])
                    );
                }
            }
        }

        return forecasts;
    }

    /// <summary>
    /// Estimates the initial values for level, trend, and seasonal components.
    /// </summary>
    /// <param name="y">The target values vector (the time series data).</param>
    /// <returns>A vector containing the initial values for level, trend, and seasonal components.</returns>
    /// <remarks>
    /// <para>
    /// This method estimates the initial values for the level, trend, and seasonal components based on
    /// the observed time series data. These initial values are important for starting the exponential
    /// smoothing algorithm and can significantly impact forecast accuracy.
    /// </para>
    /// <para><b>For Beginners:</b> This determines the starting point for the forecasting model.
    /// 
    /// The method sets up:
    /// - Initial level: The base value (like the average)
    /// - Initial trend: How much the value is increasing or decreasing per period
    /// - Initial seasonal factors: How much each season typically differs from the average
    /// 
    /// This is like establishing a baseline before making predictions.
    /// For example, if forecasting ice cream sales, you might start with:
    /// - Base level of 100 sales per day
    /// - Upward trend of +5 sales per day
    /// - Summer months having 50% higher sales than average
    /// </para>
    /// </remarks>
    private Vector<T> EstimateInitialValues(Vector<T> y)
    {
        Vector<T> initialValues = new Vector<T>(_options.SeasonalPeriod > 0 ? _options.SeasonalPeriod + 2 : 2);
        
        // Initial level
        initialValues[0] = y[0];

        // Initial trend
        if (_options.IncludeTrend)
        {
            initialValues[1] = NumOps.Subtract(y[1], y[0]);
        }
        else
        {
            initialValues[1] = NumOps.Zero;
        }

        // Initial seasonal factors
        if (_options.SeasonalPeriod > 0)
        {
            Vector<T> seasonalFactors = EstimateInitialSeasonalFactors(y);
            for (int i = 0; i < _options.SeasonalPeriod; i++)
            {
                initialValues[i + 2] = seasonalFactors[i];
            }
        }

        return initialValues;
    }

    /// <summary>
    /// Estimates the initial seasonal factors for time series with seasonality.
    /// </summary>
    /// <param name="y">The target values vector (the time series data).</param>
    /// <returns>A vector containing the initial seasonal factors.</returns>
    /// <remarks>
    /// <para>
    /// This method estimates the initial seasonal factors by averaging the observations for each season
    /// across multiple seasonal cycles. The seasonal factors are then normalized so that they sum to the
    /// seasonal period, ensuring they represent relative deviations from the level.
    /// </para>
    /// <para><b>For Beginners:</b> This identifies the typical seasonal patterns in your data.
    /// 
    /// The method:
    /// - Groups data by season (e.g., all January values, all February values)
    /// - Calculates the average value for each season
    /// - Adjusts these averages so they represent how much each season differs from normal
    /// 
    /// For example, in retail sales data:
    /// - December might have a factor of 1.5 (50% higher than normal)
    /// - January might have a factor of 0.8 (20% lower than normal)
    /// - These factors help the model account for predictable seasonal variations
    /// </para>
    /// </remarks>
    private Vector<T> EstimateInitialSeasonalFactors(Vector<T> y)
    {
        Vector<T> seasonalFactors = new Vector<T>(_options.SeasonalPeriod);
        int seasons = y.Length / _options.SeasonalPeriod;

        for (int i = 0; i < _options.SeasonalPeriod; i++)
        {
            T sum = NumOps.Zero;
            for (int j = 0; j < seasons; j++)
            {
                sum = NumOps.Add(sum, y[i + j * _options.SeasonalPeriod]);
            }
            seasonalFactors[i] = NumOps.Divide(sum, NumOps.FromDouble(seasons));
        }

        // Normalize seasonal factors
        T seasonalSum = seasonalFactors.Sum();
        for (int i = 0; i < _options.SeasonalPeriod; i++)
        {
            seasonalFactors[i] = NumOps.Divide(NumOps.Multiply(seasonalFactors[i], NumOps.FromDouble(_options.SeasonalPeriod)), seasonalSum);
        }

        return seasonalFactors;
    }

    /// <summary>
    /// Generates predictions for the given input data.
    /// </summary>
    /// <param name="input">The input features matrix.</param>
    /// <returns>A vector containing the predicted values.</returns>
    /// <remarks>
    /// <para>
    /// This method generates forecasts for future time periods based on the trained model parameters.
    /// It applies the exponential smoothing algorithm using the optimized smoothing parameters (alpha, beta, gamma)
    /// and the initial values for level, trend, and seasonal components.
    /// </para>
    /// <para><b>For Beginners:</b> This makes predictions for future time periods.
    /// 
    /// After the model is trained:
    /// - It uses the best alpha, beta, and gamma values it found
    /// - It starts with the established level, trend, and seasonal patterns
    /// - It calculates predictions for each requested future time period
    /// - It updates its components after each prediction to stay accurate
    /// 
    /// This is like a weather forecast that uses current conditions and patterns
    /// to predict what will happen in the coming days.
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> input)
    {
        Vector<T> predictions = new Vector<T>(input.Rows);
        T level = _initialValues[0];
        T trend = _options.IncludeTrend ? _initialValues[1] : NumOps.Zero;
        Vector<T> seasonalFactors = _options.SeasonalPeriod > 0 ? new Vector<T>([.. _initialValues.Skip(2)]) : Vector<T>.Empty();

        for (int i = 0; i < predictions.Length; i++)
        {
            T prediction;
            if (_options.SeasonalPeriod > 0)
            {
                prediction = NumOps.Multiply(NumOps.Add(level, trend), seasonalFactors[i % _options.SeasonalPeriod]);
            }
            else
            {
                prediction = NumOps.Add(level, trend);
            }

            predictions[i] = prediction;

            // Update level, trend, and seasonal factors
            T oldLevel = level;
            level = NumOps.Add(NumOps.Multiply(_alpha, prediction), NumOps.Multiply(NumOps.Subtract(NumOps.One, _alpha), NumOps.Add(oldLevel, trend)));

            if (_options.IncludeTrend)
            {
                trend = NumOps.Add(NumOps.Multiply(_beta, NumOps.Subtract(level, oldLevel)), NumOps.Multiply(NumOps.Subtract(NumOps.One, _beta), trend));
            }

            if (_options.SeasonalPeriod > 0)
            {
                int seasonIndex = i % _options.SeasonalPeriod;
                seasonalFactors[seasonIndex] = NumOps.Add(
                    NumOps.Multiply(_gamma, NumOps.Divide(prediction, NumOps.Add(oldLevel, trend))),
                    NumOps.Multiply(NumOps.Subtract(NumOps.One, _gamma), seasonalFactors[seasonIndex])
                );
            }
        }

        return predictions;
    }

    /// <summary>
    /// Evaluates the model's performance on test data.
    /// </summary>
    /// <param name="xTest">The test input features matrix.</param>
    /// <param name="yTest">The test target values vector.</param>
    /// <returns>A dictionary containing various evaluation metrics.</returns>
    /// <remarks>
    /// <para>
    /// This method evaluates the model's performance on test data by generating predictions and
    /// calculating various error metrics. The returned metrics include Mean Squared Error (MSE),
    /// Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE).
    /// </para>
    /// <para><b>For Beginners:</b> This measures how accurate the model's predictions are.
    /// 
    /// The evaluation:
    /// - Makes predictions for data the model hasn't seen before
    /// - Compares these predictions to the actual values
    /// - Calculates different types of error measurements:
    ///   - MSE (Mean Squared Error): Average of squared differences
    ///   - RMSE (Root Mean Squared Error): Square root of MSE, in the same units as your data
    ///   - MAPE (Mean Absolute Percentage Error): Average percentage difference
    /// 
    /// Lower values indicate better performance. For example, a MAPE of 5% means
    /// predictions are off by 5% on average.
    /// </para>
    /// </remarks>
    public override Dictionary<string, T> EvaluateModel(Matrix<T> xTest, Vector<T> yTest)
    {
        Vector<T> predictions = Predict(xTest);
        Dictionary<string, T> metrics = [];

        // Calculate MSE
        T mse = StatisticsHelper<T>.CalculateMeanSquaredError(predictions, yTest);
        metrics["MSE"] = mse;

        // Calculate RMSE
        T rmse = NumOps.Sqrt(mse);
        metrics["RMSE"] = rmse;

        // Calculate MAPE
        T mape = StatisticsHelper<T>.CalculateMeanAbsolutePercentageError(predictions, yTest);
        metrics["MAPE"] = mape;

        return metrics;
    }

    /// <summary>
    /// Serializes the model's core parameters to a binary writer.
    /// </summary>
    /// <param name="writer">The binary writer to write to.</param>
    /// <remarks>
    /// <para>
    /// This method writes the model's essential parameters to a binary stream, allowing the model
    /// to be saved to a file or database. The serialized parameters include the smoothing factors
    /// (alpha, beta, gamma) and the initial values for level, trend, and seasonal components.
    /// </para>
    /// <para><b>For Beginners:</b> This saves the model so you can use it later.
    /// 
    /// The method:
    /// - Converts the model's parameters to a format that can be saved
    /// - Writes these values to a file or database
    /// - Includes all the information needed to recreate the model exactly
    /// 
    /// This is like saving a document so you can open it again later without
    /// having to start from scratch.
    /// </para>
    /// </remarks>
    protected override void SerializeCore(BinaryWriter writer)
    {
        writer.Write(Convert.ToDouble(_alpha));
        writer.Write(Convert.ToDouble(_beta));
        writer.Write(Convert.ToDouble(_gamma));
        writer.Write(_initialValues.Length);

        foreach (var value in _initialValues)
        {
            writer.Write(Convert.ToDouble(value));
        }
    }

    /// <summary>
    /// Deserializes the model's core parameters from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader to read from.</param>
    /// <remarks>
    /// <para>
    /// This method reads the model's essential parameters from a binary stream, allowing a previously
    /// saved model to be loaded from a file or database. The deserialized parameters include the smoothing
    /// factors (alpha, beta, gamma) and the initial values for level, trend, and seasonal components.
    /// </para>
    /// <para><b>For Beginners:</b> This loads a previously saved model.
    /// 
    /// The method:
    /// - Reads the saved model data from a file or database
    /// - Converts this data back into the model's parameters
    /// - Reconstructs the model exactly as it was when saved
    /// 
    /// This is like opening a document you previously saved, allowing you
    /// to continue using the model without having to train it again.
    /// </para>
    /// </remarks>
    protected override void DeserializeCore(BinaryReader reader)
    {
        _alpha = NumOps.FromDouble(reader.ReadDouble());
        _beta = NumOps.FromDouble(reader.ReadDouble());
        _gamma = NumOps.FromDouble(reader.ReadDouble());
        int initialValuesLength = reader.ReadInt32();
        _initialValues = new Vector<T>(initialValuesLength);

        for (int i = 0; i < initialValuesLength; i++)
        {
            _initialValues[i] = NumOps.FromDouble(reader.ReadDouble());
        }
    }
}