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
    /// <summary>
    /// The smoothing factor for the level component (alpha).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Alpha controls how much weight is given to recent observations versus historical level values.
    /// Values range from 0 to 1, where higher values give more weight to recent observations.
    /// </para>
    /// <para><b>For Beginners:</b> Think of alpha as a "sensitivity dial" for your forecast.
    /// 
    /// - A high alpha (close to 1) means the model responds quickly to changes in your data.
    ///   It's like saying "what happened yesterday matters a lot more than what happened last month."
    /// 
    /// - A low alpha (close to 0) makes the model more stable and less reactive to sudden changes.
    ///   It's like saying "let's look at the long-term average and not overreact to yesterday's data."
    /// 
    /// For example, if you're forecasting ice cream sales:
    /// - High alpha (0.9): If yesterday was unusually hot and sales spiked, your forecast will predict high sales today too
    /// - Low alpha (0.1): Even if yesterday was unusually hot, your forecast will only adjust slightly upward
    /// </para>
    /// </remarks>
    private T _alpha; // Smoothing factor for level

    /// <summary>
    /// The smoothing factor for the trend component (beta).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Beta controls how much weight is given to recent trend changes versus historical trend values.
    /// This is only used when trend analysis is enabled in the model options.
    /// Values range from 0 to 1, where higher values make the trend more responsive to recent changes.
    /// </para>
    /// <para><b>For Beginners:</b> Beta controls how quickly your model adapts to changing trends.
    /// 
    /// - A high beta (close to 1) means the model quickly adjusts its trend estimates when the data direction changes.
    ///   For example, if sales have been growing by 5 units per day but suddenly start growing by 10 units per day,
    ///   the model will quickly adjust to the new growth rate.
    /// 
    /// - A low beta (close to 0) means the model is slower to adjust its trend estimates.
    ///   It will hold onto the established trend and only gradually change if new data consistently shows a different trend.
    /// 
    /// Think of it like steering a ship:
    /// - High beta: Responsive steering that quickly changes direction
    /// - Low beta: Steady steering that maintains course with only gradual adjustments
    /// </para>
    /// </remarks>
    private T _beta;  // Smoothing factor for trend (if applicable)

    /// <summary>
    /// The smoothing factor for the seasonal component (gamma).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Gamma controls how much weight is given to recent seasonal observations versus historical seasonal patterns.
    /// This is only used when seasonal analysis is enabled in the model options.
    /// Values range from 0 to 1, where higher values make the seasonal pattern more responsive to recent observations.
    /// </para>
    /// <para><b>For Beginners:</b> Gamma determines how quickly your model updates its understanding of seasonal patterns.
    /// 
    /// - A high gamma (close to 1) means the model quickly adjusts its seasonal estimates based on recent observations.
    ///   If this December's holiday sales are much higher than previous Decembers, the model will quickly adapt.
    /// 
    /// - A low gamma (close to 0) means the model sticks more closely to established seasonal patterns.
    ///   It assumes that this December will be similar to previous Decembers, even if early sales figures suggest otherwise.
    /// 
    /// For example, in retail sales forecasting:
    /// - High gamma: "This holiday season seems different from past years, let's adjust our expectations"
    /// - Low gamma: "Holiday patterns are pretty consistent year to year, let's not overreact to early signals"
    /// </para>
    /// </remarks>
    private T _gamma; // Smoothing factor for seasonality (if applicable)

    /// <summary>
    /// The initial values for level, trend, and seasonal components.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This vector stores the starting values for the exponential smoothing algorithm:
    /// - Index 0: Initial level value
    /// - Index 1: Initial trend value (if trend is enabled)
    /// - Index 2 and above: Initial seasonal factors (if seasonality is enabled)
    /// 
    /// These values are estimated from the training data and provide the foundation for the forecasting process.
    /// </para>
    /// <para><b>For Beginners:</b> These are the starting points for your forecasting model.
    /// 
    /// Think of these as the "first day" values that the model needs to get started:
    /// 
    /// - Initial level: The baseline value of your time series (like average daily sales)
    /// - Initial trend: How much your values are typically increasing or decreasing per period
    /// - Initial seasonal factors: How each season typically differs from the average
    /// 
    /// For example, if forecasting monthly ice cream sales, your initial values might include:
    /// - A base level of 1000 sales per month
    /// - An upward trend of +50 sales per month
    /// - Seasonal factors showing summer months at 150% of average and winter months at 50% of average
    /// 
    /// The model uses these as starting points and then updates them as it processes more data.
    /// </para>
    /// </remarks>
    private Vector<T> _initialValues;

    /// <summary>
    /// The level value at the end of training, used as the starting point for forecasting.
    /// </summary>
    private T _trainedLevel;

    /// <summary>
    /// The trend value at the end of training, used as the starting point for forecasting.
    /// </summary>
    private T _trainedTrend;

    /// <summary>
    /// The seasonal factors at the end of training, used as the starting point for forecasting.
    /// </summary>
    private Vector<T> _trainedSeasonalFactors;

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
        _trainedLevel = NumOps.Zero;
        _trainedTrend = NumOps.Zero;
        _trainedSeasonalFactors = Vector<T>.Empty();
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
        T trend = Options.IncludeTrend ? NumOps.Subtract(y[1], y[0]) : NumOps.Zero;
        Vector<T> seasonalFactors = Options.SeasonalPeriod > 0 ? EstimateInitialSeasonalFactors(y) : Vector<T>.Empty();

        for (int i = 0; i < y.Length; i++)
        {
            T forecast;
            if (Options.SeasonalPeriod > 0)
            {
                forecast = NumOps.Multiply(NumOps.Add(level, trend), seasonalFactors[i % Options.SeasonalPeriod]);
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
                    NumOps.Multiply(alpha, NumOps.Divide(observation, Options.SeasonalPeriod > 0 ? seasonalFactors[i % Options.SeasonalPeriod] : NumOps.One)),
                    NumOps.Multiply(NumOps.Subtract(NumOps.One, alpha), NumOps.Add(oldLevel, trend))
                );

                // Update trend
                if (Options.IncludeTrend)
                {
                    trend = NumOps.Add(
                        NumOps.Multiply(beta, NumOps.Subtract(level, oldLevel)),
                        NumOps.Multiply(NumOps.Subtract(NumOps.One, beta), trend)
                    );
                }

                // Update seasonal factors
                if (Options.SeasonalPeriod > 0)
                {
                    int seasonIndex = i % Options.SeasonalPeriod;
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
        Vector<T> initialValues = new Vector<T>(Options.SeasonalPeriod > 0 ? Options.SeasonalPeriod + 2 : 2);

        // Initial level
        initialValues[0] = y[0];

        // Initial trend
        if (Options.IncludeTrend)
        {
            initialValues[1] = NumOps.Subtract(y[1], y[0]);
        }
        else
        {
            initialValues[1] = NumOps.Zero;
        }

        // Initial seasonal factors
        if (Options.SeasonalPeriod > 0)
        {
            Vector<T> seasonalFactors = EstimateInitialSeasonalFactors(y);
            for (int i = 0; i < Options.SeasonalPeriod; i++)
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
        Vector<T> seasonalFactors = new Vector<T>(Options.SeasonalPeriod);
        int seasons = y.Length / Options.SeasonalPeriod;

        if (seasons <= 0)
        {
            throw new InvalidOperationException(
                $"Cannot estimate initial seasonal factors: " +
                $"time series length ({y.Length}) is shorter than one seasonal period ({Options.SeasonalPeriod}).");
        }

        for (int i = 0; i < Options.SeasonalPeriod; i++)
        {
            T sum = NumOps.Zero;
            for (int j = 0; j < seasons; j++)
            {
                sum = NumOps.Add(sum, y[i + j * Options.SeasonalPeriod]);
            }
            seasonalFactors[i] = NumOps.Divide(sum, NumOps.FromDouble(seasons));
        }

        // Normalize seasonal factors
        T seasonalSum = Engine.Sum(seasonalFactors);
        // VECTORIZED: Normalize seasonal factors using Engine operations
        var periodScalar = NumOps.FromDouble(Options.SeasonalPeriod);
        var periodVec = new Vector<T>(Options.SeasonalPeriod);
        for (int i = 0; i < Options.SeasonalPeriod; i++) periodVec[i] = periodScalar;
        seasonalFactors = (Vector<T>)Engine.Multiply(seasonalFactors, periodVec);

        var sumVec = new Vector<T>(Options.SeasonalPeriod);
        for (int i = 0; i < Options.SeasonalPeriod; i++) sumVec[i] = seasonalSum;
        seasonalFactors = (Vector<T>)Engine.Divide(seasonalFactors, sumVec);

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
        T trend = Options.IncludeTrend ? _initialValues[1] : NumOps.Zero;
        Vector<T> seasonalFactors = Options.SeasonalPeriod > 0 ? new Vector<T>([.. _initialValues.Skip(2)]) : Vector<T>.Empty();

        for (int i = 0; i < predictions.Length; i++)
        {
            T prediction;
            if (Options.SeasonalPeriod > 0)
            {
                prediction = NumOps.Multiply(NumOps.Add(level, trend), seasonalFactors[i % Options.SeasonalPeriod]);
            }
            else
            {
                prediction = NumOps.Add(level, trend);
            }

            predictions[i] = prediction;

            // Update level, trend, and seasonal factors
            T oldLevel = level;
            level = NumOps.Add(NumOps.Multiply(_alpha, prediction), NumOps.Multiply(NumOps.Subtract(NumOps.One, _alpha), NumOps.Add(oldLevel, trend)));

            if (Options.IncludeTrend)
            {
                trend = NumOps.Add(NumOps.Multiply(_beta, NumOps.Subtract(level, oldLevel)), NumOps.Multiply(NumOps.Subtract(NumOps.One, _beta), trend));
            }

            if (Options.SeasonalPeriod > 0)
            {
                int seasonIndex = i % Options.SeasonalPeriod;
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
        Dictionary<string, T> metrics = new Dictionary<string, T>();

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

    /// <summary>
    /// Resets the model to its initial state.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the exponential smoothing model to its initial state, clearing any learned
    /// parameters and returning to the initial smoothing factors provided in the options.
    /// </para>
    /// <para><b>For Beginners:</b> This resets the model to start fresh.
    /// 
    /// Resetting the model:
    /// - Clears any learning that has happened
    /// - Returns to the initial settings you provided
    /// - Allows you to train the model again from scratch
    /// 
    /// This is useful if you want to re-train the model with different data
    /// or if you want to compare different training approaches.
    /// </para>
    /// </remarks>
    public override void Reset()
    {
        _alpha = NumOps.Zero;
        _beta = NumOps.Zero;
        _gamma = NumOps.Zero;
        _initialValues = Vector<T>.Empty();
    }

    /// <summary>
    /// Creates a new instance of the exponential smoothing model with the same options.
    /// </summary>
    /// <returns>A new instance of the exponential smoothing model.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new instance of the exponential smoothing model with the same configuration
    /// options as the current instance. This is useful for creating copies or clones of the model.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new copy of the model with the same settings.
    /// 
    /// Creating a new instance:
    /// - Makes a fresh copy of the model with the same configuration
    /// - The new copy hasn't been trained yet
    /// - You can train and use the copy independently from the original
    /// 
    /// This is helpful when you want to experiment with different training data
    /// while preserving your original model.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateInstance()
    {
        return new ExponentialSmoothingModel<T>((ExponentialSmoothingOptions<T>)Options);
    }

    /// <summary>
    /// Returns metadata about the model, including its type, parameters, and configuration.
    /// </summary>
    /// <returns>A ModelMetaData object containing information about the model.</returns>
    /// <remarks>
    /// <para>
    /// This method returns detailed metadata about the exponential smoothing model, including its type,
    /// current parameters (alpha, beta, gamma, initial values), and configuration options. This metadata
    /// can be used for model selection, comparison, and documentation purposes.
    /// </para>
    /// <para><b>For Beginners:</b> This provides information about your model's settings and state.
    /// 
    /// The metadata includes:
    /// - The type of model (Exponential Smoothing)
    /// - Current smoothing parameters (alpha, beta, gamma)
    /// - Initial values for level, trend, and seasonal components
    /// - Configuration settings from when you created the model
    /// - A serialized version of the entire model
    /// 
    /// This information is useful for:
    /// - Keeping track of different models you've created
    /// - Comparing model configurations
    /// - Documenting which settings worked best
    /// - Sharing model information with others
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var esOptions = (ExponentialSmoothingOptions<T>)Options;
        var metadata = new ModelMetadata<T>
        {
            ModelType = ModelType.ExponentialSmoothingModel,
            AdditionalInfo = new Dictionary<string, object>
            {
                // Include the actual model state variables
                { "Alpha", Convert.ToDouble(_alpha) },
                { "Beta", Convert.ToDouble(_beta) },
                { "Gamma", Convert.ToDouble(_gamma) },
                { "InitialValues", _initialValues },
            
                // Include model configuration as well
                { "InitialAlpha", esOptions.InitialAlpha },
                { "InitialBeta", esOptions.InitialBeta },
                { "InitialGamma", esOptions.InitialGamma },
                { "UseTrend", esOptions.UseTrend },
                { "UseSeasonal", esOptions.UseSeasonal },
                { "SeasonalPeriod", esOptions.SeasonalPeriod }
            },
            ModelData = this.Serialize()
        };

        return metadata;
    }

    /// <summary>
    /// Implements the core training logic for the exponential smoothing model.
    /// </summary>
    /// <param name="x">The input features matrix (typically time indicators or related variables).</param>
    /// <param name="y">The target values vector (the time series data to forecast).</param>
    /// <remarks>
    /// <para>
    /// This method contains the implementation details of the training process for the exponential
    /// smoothing model. It estimates the optimal smoothing parameters and initial values using the
    /// provided time series data.
    /// </para>
    /// <para><b>For Beginners:</b> This is the engine that powers the training process.
    /// 
    /// While the public Train method handles input validation and high-level logic,
    /// this method does the actual work of:
    /// - Finding the best alpha, beta, and gamma values through grid search
    /// - Calculating the initial level, trend, and seasonal components
    /// 
    /// It's like the detailed work that happens in a car engine when you press
    /// the accelerator pedal.
    /// </para>
    /// </remarks>
    protected override void TrainCore(Matrix<T> x, Vector<T> y)
    {
        // Estimate optimal alpha, beta, and gamma parameters
        (_alpha, _beta, _gamma) = EstimateParametersGridSearch(y);

        // Estimate initial values for level, trend, and seasonal components
        _initialValues = EstimateInitialValues(y);

        // If we're using the triple exponential smoothing model with seasonality,
        // validate and prepare the seasonal factors
        if (Options.SeasonalPeriod > 0 && y.Length >= Options.SeasonalPeriod)
        {
            // Make sure we have enough data for at least one full seasonal cycle
            Vector<T> seasonalFactors = EstimateInitialSeasonalFactors(y);

            // Ensure the seasonal factors are properly included in the initial values
            for (int i = 0; i < Options.SeasonalPeriod && i + 2 < _initialValues.Length; i++)
            {
                _initialValues[i + 2] = seasonalFactors[i];
            }
        }

        // Run through the training data to compute the final level/trend/seasonal state
        SaveTrainedState(y);
    }

    /// <summary>
    /// Runs through the training data with the optimized parameters to compute
    /// the final level, trend, and seasonal state for use in forecasting.
    /// </summary>
    private void SaveTrainedState(Vector<T> y)
    {
        T level = _initialValues[0];
        T trend = Options.IncludeTrend ? _initialValues[1] : NumOps.Zero;
        Vector<T> seasonalFactors = Options.SeasonalPeriod > 0
            ? new Vector<T>([.. _initialValues.Skip(2)])
            : Vector<T>.Empty();

        for (int i = 0; i < y.Length; i++)
        {
            T observation = y[i];
            T oldLevel = level;

            // Update level using actual observation
            if (Options.SeasonalPeriod > 0)
            {
                T seasonFactor = seasonalFactors[i % Options.SeasonalPeriod];
                T deseasonalized = NumOps.Divide(observation, seasonFactor);
                level = NumOps.Add(
                    NumOps.Multiply(_alpha, deseasonalized),
                    NumOps.Multiply(NumOps.Subtract(NumOps.One, _alpha), NumOps.Add(oldLevel, trend))
                );
            }
            else
            {
                level = NumOps.Add(
                    NumOps.Multiply(_alpha, observation),
                    NumOps.Multiply(NumOps.Subtract(NumOps.One, _alpha), NumOps.Add(oldLevel, trend))
                );
            }

            // Update trend
            if (Options.IncludeTrend)
            {
                trend = NumOps.Add(
                    NumOps.Multiply(_beta, NumOps.Subtract(level, oldLevel)),
                    NumOps.Multiply(NumOps.Subtract(NumOps.One, _beta), trend)
                );
            }

            // Update seasonal factors
            if (Options.SeasonalPeriod > 0)
            {
                int seasonIndex = i % Options.SeasonalPeriod;
                seasonalFactors[seasonIndex] = NumOps.Add(
                    NumOps.Multiply(_gamma, NumOps.Divide(observation, level)),
                    NumOps.Multiply(NumOps.Subtract(NumOps.One, _gamma), seasonalFactors[seasonIndex])
                );
            }
        }

        // Save the final state
        _trainedLevel = level;
        _trainedTrend = trend;
        _trainedSeasonalFactors = seasonalFactors.Length > 0 ? seasonalFactors.Clone() : Vector<T>.Empty();
    }

    /// <summary>
    /// Predicts a single value based on the input features vector.
    /// </summary>
    /// <param name="input">The input features vector.</param>
    /// <returns>The predicted value.</returns>
    /// <remarks>
    /// <para>
    /// This method generates a single forecast value based on the provided input features.
    /// It uses the trained model parameters (alpha, beta, gamma) and the established level,
    /// trend, and seasonal components to make the prediction.
    /// </para>
    /// <para><b>For Beginners:</b> This is a shortcut for getting just one prediction.
    /// 
    /// Instead of making multiple predictions at once, this method:
    /// - Takes a single set of input features
    /// - Uses the model's learned patterns
    /// - Returns a single predicted value
    /// 
    /// It's like asking for tomorrow's forecast specifically, rather than
    /// getting the forecast for the whole week ahead.
    /// </para>
    /// </remarks>
    public override T PredictSingle(Vector<T> input)
    {
        // Create a single-row matrix from the input vector
        Matrix<T> singleRowMatrix = new Matrix<T>(1, input.Length);
        for (int i = 0; i < input.Length; i++)
        {
            singleRowMatrix[0, i] = input[i];
        }

        // Use the existing Predict method to get a vector of predictions
        // (in this case, a vector with just one value)
        Vector<T> predictions = Predict(singleRowMatrix);

        // Return the single prediction value
        if (predictions.Length > 0)
        {
            return predictions[0];
        }

        // If for some reason we couldn't make a prediction, return a default value
        // This should never happen in normal operation, but provides a fallback
        return NumOps.Zero;
    }

    /// <summary>
    /// Forecasts future values using the trained end-of-training state instead of
    /// resetting to initial values on each call.
    /// </summary>
    /// <param name="history">The historical time series values.</param>
    /// <param name="steps">The number of future steps to forecast.</param>
    /// <returns>A vector of forecasted values.</returns>
    public override Vector<T> Forecast(Vector<T> history, int steps)
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

        // Start from the trained end-of-training state
        T level = _trainedLevel;
        T trend = _trainedTrend;
        Vector<T> seasonalFactors = _trainedSeasonalFactors.Length > 0
            ? _trainedSeasonalFactors.Clone()
            : Vector<T>.Empty();

        // The seasonal index continues from where training left off
        // Training processed y.Length observations, so the next season index is y.Length % period
        // But we don't know training length here, so use history length as approximation
        int seasonStartIndex = history.Length;

        Vector<T> forecasts = new Vector<T>(steps);
        for (int i = 0; i < steps; i++)
        {
            T forecast;
            if (Options.SeasonalPeriod > 0 && seasonalFactors.Length > 0)
            {
                int seasonIdx = (seasonStartIndex + i) % Options.SeasonalPeriod;
                forecast = NumOps.Multiply(NumOps.Add(level, trend), seasonalFactors[seasonIdx]);
            }
            else
            {
                forecast = NumOps.Add(level, trend);
            }

            forecasts[i] = forecast;

            // Update level and trend for the next step using the forecast value
            // (since we don't have actual observations for future steps)
            T oldLevel = level;
            if (Options.SeasonalPeriod > 0 && seasonalFactors.Length > 0)
            {
                int seasonIdx = (seasonStartIndex + i) % Options.SeasonalPeriod;
                T deseasonalized = NumOps.Divide(forecast, seasonalFactors[seasonIdx]);
                level = NumOps.Add(
                    NumOps.Multiply(_alpha, deseasonalized),
                    NumOps.Multiply(NumOps.Subtract(NumOps.One, _alpha), NumOps.Add(oldLevel, trend))
                );
            }
            else
            {
                level = NumOps.Add(
                    NumOps.Multiply(_alpha, forecast),
                    NumOps.Multiply(NumOps.Subtract(NumOps.One, _alpha), NumOps.Add(oldLevel, trend))
                );
            }

            if (Options.IncludeTrend)
            {
                trend = NumOps.Add(
                    NumOps.Multiply(_beta, NumOps.Subtract(level, oldLevel)),
                    NumOps.Multiply(NumOps.Subtract(NumOps.One, _beta), trend)
                );
            }

            if (Options.SeasonalPeriod > 0 && seasonalFactors.Length > 0)
            {
                int seasonIdx = (seasonStartIndex + i) % Options.SeasonalPeriod;
                seasonalFactors[seasonIdx] = NumOps.Add(
                    NumOps.Multiply(_gamma, NumOps.Divide(forecast, NumOps.Add(oldLevel, trend))),
                    NumOps.Multiply(NumOps.Subtract(NumOps.One, _gamma), seasonalFactors[seasonIdx])
                );
            }
        }

        return forecasts;
    }
}
