namespace AiDotNet.TimeSeries;

/// <summary>
/// Implements a Seasonal Autoregressive Integrated Moving Average (SARIMA) model for time series forecasting.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// The SARIMA model extends the ARIMA model by incorporating seasonal components, making it suitable for 
/// data with seasonal patterns. It combines autoregressive (AR), integrated (I), and moving average (MA) 
/// components for both seasonal and non-seasonal parts of the time series.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// SARIMA is used to predict future values in a time series (data collected over time) that has seasonal patterns.
/// Think of it like predicting ice cream sales throughout the year - there's a general trend (maybe increasing over years)
/// and a seasonal pattern (higher in summer, lower in winter). SARIMA can capture both these patterns.
/// 
/// The model has several parameters:
/// - p: How many previous values influence the current value
/// - d: How many times we need to subtract consecutive values to make the data stable
/// - q: How many previous prediction errors influence the current prediction
/// - P, D, Q: The same as above, but for seasonal patterns
/// - m: The length of the seasonal cycle (e.g., 12 for monthly data with yearly patterns)
/// </para>
/// </remarks>
public class SARIMAModel<T> : TimeSeriesModelBase<T>
{
    /// <summary>
    /// Stores the configuration options for the SARIMA model.
    /// </summary>
    private readonly SARIMAOptions<T> _sarimaOptions;

    /// <summary>
    /// Coefficients for the non-seasonal autoregressive (AR) component.
    /// </summary>
    private Vector<T> _arCoefficients;

    /// <summary>
    /// Coefficients for the non-seasonal moving average (MA) component.
    /// </summary>
    private Vector<T> _maCoefficients;

    /// <summary>
    /// Coefficients for the seasonal autoregressive (SAR) component.
    /// </summary>
    private Vector<T> _sarCoefficients;

    /// <summary>
    /// Coefficients for the seasonal moving average (SMA) component.
    /// </summary>
    private Vector<T> _smaCoefficients;

    /// <summary>
    /// The constant term in the SARIMA model.
    /// </summary>
    private T _constant;

    /// <summary>
    /// Order of the non-seasonal autoregressive component.
    /// </summary>
    private readonly int _p;

    /// <summary>
    /// Order of the non-seasonal moving average component.
    /// </summary>
    private readonly int _q;

    /// <summary>
    /// Order of the non-seasonal differencing.
    /// </summary>
    private readonly int _d;

    /// <summary>
    /// The seasonal period (e.g., 12 for monthly data with yearly seasonality).
    /// </summary>
    private readonly int _m;

    /// <summary>
    /// Order of the seasonal autoregressive component.
    /// </summary>
    private readonly int _P;

    /// <summary>
    /// Order of the seasonal moving average component.
    /// </summary>
    private readonly int _Q;

    /// <summary>
    /// Order of the seasonal differencing.
    /// </summary>
    private readonly int _D;

    /// <summary>
    /// Initializes a new instance of the SARIMAModel class with the specified options.
    /// </summary>
    /// <param name="options">The configuration options for the SARIMA model.</param>
    public SARIMAModel(SARIMAOptions<T> options) : base(options)
    {
        _sarimaOptions = options;
        _constant = NumOps.Zero;
        _arCoefficients = Vector<T>.Empty();
        _maCoefficients = Vector<T>.Empty();
        _sarCoefficients = Vector<T>.Empty();
        _smaCoefficients = Vector<T>.Empty();
        _p = _sarimaOptions.P;
        _q = _sarimaOptions.Q;
        _d = _sarimaOptions.D;
        _m = _sarimaOptions.SeasonalPeriod;
        _P = _sarimaOptions.SeasonalP;
        _Q = _sarimaOptions.SeasonalQ;
        _D = _sarimaOptions.SeasonalD;
    }

    /// <summary>
    /// Gets the non-seasonal autoregressive (AR) parameters of the model.
    /// </summary>
    /// <returns>A vector containing the AR parameters.</returns>
    /// <remarks>
    /// <para>
    /// AR parameters represent how much the current value depends on its previous values.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// Autoregressive (AR) parameters tell us how much each previous value affects the current value.
    /// For example, if we're predicting today's temperature, these parameters tell us how much
    /// yesterday's temperature, the day before, etc. influence today's temperature.
    /// </para>
    /// </remarks>
    public Vector<T> GetARParameters()
    {
        return new Vector<T>(_p);
    }

    /// <summary>
    /// Gets the non-seasonal moving average (MA) parameters of the model.
    /// </summary>
    /// <returns>A vector containing the MA parameters.</returns>
    /// <remarks>
    /// <para>
    /// MA parameters represent how much the current value depends on previous prediction errors.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// Moving Average (MA) parameters tell us how much each previous prediction error affects the current value.
    /// A prediction error is the difference between what we predicted and what actually happened.
    /// These parameters help the model learn from its mistakes.
    /// </para>
    /// </remarks>
    public Vector<T> GetMAParameters()
    {
        return new Vector<T>(_q);
    }

    /// <summary>
    /// Gets the seasonal autoregressive (SAR) parameters of the model.
    /// </summary>
    /// <returns>A vector containing the seasonal AR parameters.</returns>
    /// <remarks>
    /// <para>
    /// Seasonal AR parameters represent how much the current value depends on values from previous seasons.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// Seasonal Autoregressive parameters are similar to regular AR parameters, but they look at values from
    /// previous seasons rather than just previous time points. For example, if we're predicting July's ice cream sales,
    /// these parameters tell us how much last July's sales influence this July's sales.
    /// </para>
    /// </remarks>
    public Vector<T> GetSeasonalARParameters()
    {
        return new Vector<T>(_P);
    }

    /// <summary>
    /// Gets the seasonal moving average (SMA) parameters of the model.
    /// </summary>
    /// <returns>A vector containing the seasonal MA parameters.</returns>
    /// <remarks>
    /// <para>
    /// Seasonal MA parameters represent how much the current value depends on prediction errors from previous seasons.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// Seasonal Moving Average parameters are similar to regular MA parameters, but they look at prediction errors from
    /// previous seasons. They help the model learn from seasonal patterns in its past mistakes.
    /// </para>
    /// </remarks>
    public Vector<T> GetSeasonalMAParameters()
    {
        return new Vector<T>(_Q);
    }

    /// <summary>
    /// Gets the seasonal period used in the model.
    /// </summary>
    /// <returns>The seasonal period (e.g., 12 for monthly data with yearly seasonality).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// The seasonal period is the number of time points after which the pattern repeats.
    /// For example, for monthly data with yearly patterns, the seasonal period is 12.
    /// For weekly data with yearly patterns, it would be 52.
    /// </para>
    /// </remarks>
    public int GetSeasonalPeriod()
    {
        return _m;
    }

    /// <summary>
    /// Applies both seasonal and non-seasonal differencing to the input series.
    /// </summary>
    /// <param name="y">The original time series data.</param>
    /// <returns>The differenced time series.</returns>
    /// <remarks>
    /// <para>
    /// Differencing is a technique to make a time series stationary by computing
    /// the differences between consecutive observations.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// Differencing is like looking at the changes between values rather than the values themselves.
    /// For example, instead of looking at the actual temperature each day, we look at how much it
    /// changed from the previous day. This helps remove trends and make the data more stable.
    /// 
    /// Seasonal differencing compares values to the same time in the previous season
    /// (like comparing this January to last January).
    /// </para>
    /// </remarks>
    private Vector<T> ApplyDifferencing(Vector<T> y)
    {
        Vector<T> result = y;

        // Apply seasonal differencing
        for (int i = 0; i < _D; i++)
        {
            result = SeasonalDifference(result, _m);
        }

        // Apply non-seasonal differencing
        result = TimeSeriesHelper<T>.DifferenceSeries(result, _d);

        return result;
    }

    /// <summary>
    /// Applies seasonal differencing to the input series.
    /// </summary>
    /// <param name="y">The time series data.</param>
    /// <param name="period">The seasonal period to use for differencing.</param>
    /// <returns>The seasonally differenced time series.</returns>
    private Vector<T> SeasonalDifference(Vector<T> y, int period)
    {
        Vector<T> result = new Vector<T>(y.Length - period);
        for (int i = period; i < y.Length; i++)
        {
            result[i - period] = NumOps.Subtract(y[i], y[i - period]);
        }
        return result;
    }

    /// <summary>
    /// Estimates the seasonal autoregressive (SAR) coefficients.
    /// </summary>
    /// <param name="y">The differenced time series data.</param>
    /// <returns>A vector of seasonal AR coefficients.</returns>
    private Vector<T> EstimateSeasonalARCoefficients(Vector<T> y)
    {
        Matrix<T> X = new Matrix<T>(y.Length - _P * _m, _P);
        Vector<T> Y = new Vector<T>(y.Length - _P * _m);

        for (int i = _P * _m; i < y.Length; i++)
        {
            for (int j = 0; j < _P; j++)
            {
                X[i - _P * _m, j] = y[i - (j + 1) * _m];
            }
            Y[i - _P * _m] = y[i];
        }

        return MatrixSolutionHelper.SolveLinearSystem(X, Y, MatrixDecompositionType.Qr);
    }

    /// <summary>
    /// Calculates residuals after applying AR and seasonal AR components.
    /// </summary>
    /// <param name="y">The differenced time series data.</param>
    /// <returns>A vector of residuals.</returns>
    /// <remarks>
    /// <para>
    /// Residuals are the differences between the observed values and the values
    /// predicted by the AR and seasonal AR components.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// Residuals are the errors in our predictions - the difference between what we predicted
    /// and what actually happened. By analyzing these errors, we can improve our model.
    /// </para>
    /// </remarks>
    private Vector<T> CalculateARSARResiduals(Vector<T> y)
    {
        int n = y.Length;
        int maxLag = Math.Max(_p, _P * _m);
        Vector<T> residuals = new Vector<T>(n - maxLag);

        for (int i = maxLag; i < n; i++)
        {
            T predicted = NumOps.Zero;

            // Non-seasonal AR component
            for (int j = 0; j < _p; j++)
            {
                predicted = NumOps.Add(predicted, NumOps.Multiply(_arCoefficients[j], y[i - j - 1]));
            }

            // Seasonal AR component
            for (int j = 0; j < _P; j++)
            {
                predicted = NumOps.Add(predicted, NumOps.Multiply(_sarCoefficients[j], y[i - (j + 1) * _m]));
            }

            residuals[i - maxLag] = NumOps.Subtract(y[i], predicted);
        }

        return residuals;
    }


    /// <summary>
    /// Estimates the seasonal moving average (SMA) coefficients.
    /// </summary>
    /// <param name="residuals">The residuals after applying AR and seasonal AR components.</param>
    /// <returns>A vector of seasonal MA coefficients.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method calculates how much the prediction errors from previous seasons affect the current prediction.
    /// It helps the model account for seasonal patterns in the errors, improving forecast accuracy.
    /// </para>
    /// </remarks>
    private Vector<T> EstimateSeasonalMACoefficients(Vector<T> residuals)
    {
        Vector<T> smaCoefficients = new Vector<T>(_Q);
        for (int i = 0; i < _Q; i++)
        {
            smaCoefficients[i] = TimeSeriesHelper<T>.CalculateAutoCorrelation(residuals, (i + 1) * _m);
        }

        return smaCoefficients;
    }

    /// <summary>
    /// Estimates the constant term for the SARIMA model.
    /// </summary>
    /// <param name="y">The differenced time series data.</param>
    /// <returns>The estimated constant term.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// The constant term is like a baseline value for our predictions. It represents the average value
    /// after accounting for all the patterns and trends in the data. Think of it as the starting point
    /// before we add the effects of past values and errors.
    /// </para>
    /// </remarks>
    private T EstimateConstant(Vector<T> y)
    {
        T mean = y.Average();
        // VECTORIZED: Use Engine.Sum() for coefficient summation
        T arSum = Engine.Sum(_arCoefficients);
        T sarSum = Engine.Sum(_sarCoefficients);

        return NumOps.Multiply(mean, NumOps.Subtract(NumOps.One, NumOps.Add(arSum, sarSum)));
    }

    /// <summary>
    /// Generates predictions using the trained SARIMA model.
    /// </summary>
    /// <param name="input">The input features matrix for which predictions are to be made.</param>
    /// <returns>A vector of predicted values.</returns>
    /// <remarks>
    /// <para>
    /// The prediction process combines:
    /// - Non-seasonal AR component
    /// - Seasonal AR component
    /// - Non-seasonal MA component
    /// - Seasonal MA component
    /// - The constant term
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method makes predictions for future values based on what the model has learned.
    /// It combines several components:
    /// - The effect of recent values (AR component)
    /// - The effect of values from the same season in previous years (seasonal AR)
    /// - The effect of recent prediction errors (MA component)
    /// - The effect of prediction errors from the same season in previous years (seasonal MA)
    /// - A baseline value (constant term)
    /// 
    /// The result is a forecast that accounts for both recent trends and seasonal patterns.
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> input)
    {
        Vector<T> predictions = new(input.Rows);
        int maxLag = Math.Max(_p, _P * _m);
        Vector<T> lastObservedValues = new(maxLag);
        Vector<T> lastErrors = new(Math.Max(_q, _Q * _m));

        for (int i = 0; i < predictions.Length; i++)
        {
            T prediction = _constant;

            // VECTORIZED: Add non-seasonal AR component using dot product
            if (_p > 0)
            {
                var arValues = lastObservedValues.Slice(0, _p);
                prediction = NumOps.Add(prediction, Engine.DotProduct(_arCoefficients, arValues));
            }

            // VECTORIZED: Add seasonal AR component
            if (_P > 0)
            {
                var sarValues = new Vector<T>(_P);
                for (int j = 0; j < _P; j++)
                    sarValues[j] = lastObservedValues[(j + 1) * _m - 1];
                prediction = NumOps.Add(prediction, Engine.DotProduct(_sarCoefficients, sarValues));
            }

            // VECTORIZED: Add non-seasonal MA component using dot product
            if (_q > 0)
            {
                var maValues = lastErrors.Slice(0, _q);
                prediction = NumOps.Add(prediction, Engine.DotProduct(_maCoefficients, maValues));
            }

            // VECTORIZED: Add seasonal MA component
            if (_Q > 0)
            {
                var smaValues = new Vector<T>(_Q);
                for (int j = 0; j < _Q; j++)
                    smaValues[j] = lastErrors[(j + 1) * _m - 1];
                prediction = NumOps.Add(prediction, Engine.DotProduct(_smaCoefficients, smaValues));
            }

            predictions[i] = prediction;

            // Update last observed values and errors for next prediction
            for (int j = lastObservedValues.Length - 1; j > 0; j--)
            {
                lastObservedValues[j] = lastObservedValues[j - 1];
            }
            lastObservedValues[0] = prediction;

            for (int j = lastErrors.Length - 1; j > 0; j--)
            {
                lastErrors[j] = lastErrors[j - 1];
            }
            lastErrors[0] = NumOps.Zero; // Assume zero error for future predictions
        }

        return predictions;
    }

    /// <summary>
    /// Evaluates the performance of the trained model on test data.
    /// </summary>
    /// <param name="xTest">The input features matrix for testing.</param>
    /// <param name="yTest">The actual target values for testing.</param>
    /// <returns>A dictionary containing evaluation metrics (MSE, RMSE, MAE).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method tests how well our model performs by comparing its predictions to actual values.
    /// It calculates several error metrics:
    /// - MSE (Mean Squared Error): The average of the squared differences between predictions and actual values
    /// - RMSE (Root Mean Squared Error): The square root of MSE, which gives an error in the same units as the data
    /// - MAE (Mean Absolute Error): The average of the absolute differences between predictions and actual values
    /// 
    /// Lower values for these metrics indicate better model performance.
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

        // Calculate MAE
        T mae = StatisticsHelper<T>.CalculateMeanAbsoluteError(predictions, yTest);
        metrics["MAE"] = mae;

        return metrics;
    }

    /// <summary>
    /// Serializes the model's core parameters to a binary writer.
    /// </summary>
    /// <param name="writer">The binary writer to write to.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// Serialization is the process of converting the model's state into a format that can be saved to disk.
    /// This allows you to save a trained model and load it later without having to retrain it.
    /// </para>
    /// </remarks>
    protected override void SerializeCore(BinaryWriter writer)
    {
        // Serialize SARIMA-specific options
        writer.Write(_sarimaOptions.P);
        writer.Write(_sarimaOptions.D);
        writer.Write(_sarimaOptions.Q);
        writer.Write(_sarimaOptions.SeasonalP);
        writer.Write(_sarimaOptions.SeasonalD);
        writer.Write(_sarimaOptions.SeasonalQ);
        writer.Write(_sarimaOptions.MaxIterations);
        writer.Write(Convert.ToDouble(_sarimaOptions.Tolerance));

        // Serialize coefficients
        SerializationHelper<T>.SerializeVector(writer, _arCoefficients);
        SerializationHelper<T>.SerializeVector(writer, _maCoefficients);
        SerializationHelper<T>.SerializeVector(writer, _sarCoefficients);
        SerializationHelper<T>.SerializeVector(writer, _smaCoefficients);
        writer.Write(Convert.ToDouble(_constant));
    }

    /// <summary>
    /// Deserializes the model's core parameters from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader to read from.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// Deserialization is the process of loading a previously saved model from disk.
    /// This method reads the model's parameters from a file and reconstructs the model
    /// exactly as it was when it was saved.
    /// </para>
    /// </remarks>
    protected override void DeserializeCore(BinaryReader reader)
    {
        // Deserialize SARIMA-specific options
        _sarimaOptions.P = reader.ReadInt32();
        _sarimaOptions.D = reader.ReadInt32();
        _sarimaOptions.Q = reader.ReadInt32();
        _sarimaOptions.SeasonalP = reader.ReadInt32();
        _sarimaOptions.SeasonalD = reader.ReadInt32();
        _sarimaOptions.SeasonalQ = reader.ReadInt32();
        _sarimaOptions.MaxIterations = reader.ReadInt32();
        _sarimaOptions.Tolerance = Convert.ToDouble(reader.ReadDouble());

        // Deserialize coefficients
        _arCoefficients = SerializationHelper<T>.DeserializeVector(reader);
        _maCoefficients = SerializationHelper<T>.DeserializeVector(reader);
        _sarCoefficients = SerializationHelper<T>.DeserializeVector(reader);
        _smaCoefficients = SerializationHelper<T>.DeserializeVector(reader);
        _constant = NumOps.FromDouble(reader.ReadDouble());
    }

    /// <summary>
    /// Core implementation of the training logic for the SARIMA model.
    /// </summary>
    /// <param name="x">The input features matrix.</param>
    /// <param name="y">The target values vector.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This is the core method that performs the actual training of the SARIMA model.
    /// It applies differencing to make the data stationary, then estimates the various
    /// components of the model (AR, MA, seasonal AR, seasonal MA) to capture patterns in the data.
    /// </para>
    /// </remarks>
    protected override void TrainCore(Matrix<T> x, Vector<T> y)
    {
        // Validate input data
        if (y == null || y.Length <= 0)
        {
            throw new ArgumentException("Target vector cannot be null or empty", nameof(y));
        }

        // Check if we have enough data for the seasonal component
        int minRequiredLength = Math.Max(_d + _m * _D, _p + _P * _m) + _m * Math.Max(_P, _Q);
        if (y.Length < minRequiredLength)
        {
            throw new ArgumentException(
                $"Time series is too short (length: {y.Length}) for the specified model parameters. " +
                $"Minimum required length: {minRequiredLength}.", nameof(y));
        }

        // Step 1: Apply seasonal and non-seasonal differencing
        Vector<T> diffY = ApplyDifferencing(y);

        // Step 2: Estimate non-seasonal AR coefficients
        _arCoefficients = TimeSeriesHelper<T>.EstimateARCoefficients(diffY, _p, MatrixDecompositionType.Qr);

        // Step 3: Estimate seasonal AR coefficients
        _sarCoefficients = EstimateSeasonalARCoefficients(diffY);

        // Step 4: Calculate residuals after AR and SAR
        Vector<T> arResiduals = CalculateARSARResiduals(diffY);

        // Step 5: Estimate non-seasonal MA coefficients
        _maCoefficients = TimeSeriesHelper<T>.EstimateMACoefficients(arResiduals, _q);

        // Step 6: Estimate seasonal MA coefficients
        _smaCoefficients = EstimateSeasonalMACoefficients(arResiduals);

        // Step 7: Estimate constant term
        _constant = EstimateConstant(diffY);
    }

    /// <summary>
    /// Predicts a single value based on the input vector.
    /// </summary>
    /// <param name="input">The input vector containing historical values.</param>
    /// <returns>The predicted value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method predicts a single future value based on historical data provided in the input.
    /// The input should contain the recent history needed to make a prediction.
    /// 
    /// For a SARIMA model, the input should include:
    /// - At least p values for the non-seasonal AR component
    /// - At least P*m values for the seasonal AR component
    /// - Recent errors for the MA components (though these are often set to zero if unknown)
    /// </para>
    /// </remarks>
    public override T PredictSingle(Vector<T> input)
    {
        // Validate input
        if (input == null)
        {
            throw new ArgumentNullException(nameof(input), "Input vector cannot be null");
        }

        int minRequiredInputLength = Math.Max(_p, _P * _m);
        if (input.Length < minRequiredInputLength)
        {
            throw new ArgumentException(
                $"Input vector is too short. Length: {input.Length}, required: {minRequiredInputLength}",
                nameof(input));
        }

        // Start with the constant term
        T prediction = _constant;

        // Add non-seasonal AR component
        for (int j = 0; j < _p && j < input.Length; j++)
        {
            prediction = NumOps.Add(prediction, NumOps.Multiply(_arCoefficients[j], input[j]));
        }

        // Add seasonal AR component
        for (int j = 0; j < _P && (j + 1) * _m - 1 < input.Length; j++)
        {
            prediction = NumOps.Add(prediction,
                NumOps.Multiply(_sarCoefficients[j], input[(j + 1) * _m - 1]));
        }

        // Note: For single-point prediction, we typically assume zero errors for MA terms
        // since the actual errors are unknown until after the prediction is made and
        // compared to the actual value.

        // If the input contains error terms (optional additional data), we could use them:
        if (input.Length > minRequiredInputLength)
        {
            int errorStartIndex = minRequiredInputLength;
            int errorLength = input.Length - errorStartIndex;

            // Add non-seasonal MA component if errors are provided
            for (int j = 0; j < _q && j < errorLength; j++)
            {
                prediction = NumOps.Add(prediction,
                    NumOps.Multiply(_maCoefficients[j], input[errorStartIndex + j]));
            }

            // Add seasonal MA component if errors are provided
            for (int j = 0; j < _Q && j * _m < errorLength; j++)
            {
                int index = errorStartIndex + j * _m;
                if (index < input.Length)
                {
                    prediction = NumOps.Add(prediction,
                        NumOps.Multiply(_smaCoefficients[j], input[index]));
                }
            }
        }

        return prediction;
    }

    /// <summary>
    /// Forecasts future values using the trained SARIMA model, properly handling both
    /// regular and seasonal differencing.
    /// </summary>
    /// <param name="history">The historical time series values.</param>
    /// <param name="steps">The number of future steps to forecast.</param>
    /// <returns>A vector of forecasted values in the original (undifferenced) scale.</returns>
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

        int minRequiredLength = Math.Max(_p, _P * _m) + _d + _D * _m;
        if (history.Length < minRequiredLength)
        {
            throw new ArgumentException(
                $"History length ({history.Length}) is too short for the configured SARIMA parameters. " +
                $"Minimum required length: {minRequiredLength}.",
                nameof(history));
        }

        // Apply the same differencing as in TrainCore
        Vector<T> diffHistory = ApplyDifferencing(history);

        // Build a working list of differenced values
        List<T> extendedDiff = new List<T>(diffHistory.Length + steps);
        for (int i = 0; i < diffHistory.Length; i++)
        {
            extendedDiff.Add(diffHistory[i]);
        }

        // Generate forecasts on the differenced scale
        Vector<T> diffForecasts = new Vector<T>(steps);
        for (int step = 0; step < steps; step++)
        {
            T prediction = _constant;

            // Non-seasonal AR component
            for (int j = 0; j < _p && j < extendedDiff.Count; j++)
            {
                prediction = NumOps.Add(prediction, NumOps.Multiply(
                    _arCoefficients[j], extendedDiff[extendedDiff.Count - 1 - j]));
            }

            // Seasonal AR component
            for (int j = 0; j < _P; j++)
            {
                int lagIdx = extendedDiff.Count - (j + 1) * _m;
                if (lagIdx >= 0)
                {
                    prediction = NumOps.Add(prediction, NumOps.Multiply(
                        _sarCoefficients[j], extendedDiff[lagIdx]));
                }
            }

            // MA/SMA components are assumed zero for future predictions

            diffForecasts[step] = prediction;
            extendedDiff.Add(prediction);
        }

        // Undifference: first undo regular differencing, then seasonal differencing
        // (reverse order of ApplyDifferencing which does seasonal first, then regular)
        var currentForecasts = new List<T>(steps);
        for (int i = 0; i < steps; i++)
        {
            currentForecasts.Add(diffForecasts[i]);
        }

        UndoRegularDifferencing(currentForecasts, history);
        UndoSeasonalDifferencing(currentForecasts, history);

        Vector<T> result = new Vector<T>(steps);
        for (int i = 0; i < steps; i++)
        {
            result[i] = currentForecasts[i];
        }

        return result;
    }

    /// <summary>
    /// Undoes regular (non-seasonal) differencing by computing tail values at each
    /// integration level and cumulatively summing in reverse order.
    /// </summary>
    private void UndoRegularDifferencing(List<T> forecasts, Vector<T> history)
    {
        // Compute the series after seasonal differencing but before regular differencing
        Vector<T> afterSeasonalDiff = history;
        for (int i = 0; i < _D; i++)
        {
            afterSeasonalDiff = SeasonalDifference(afterSeasonalDiff, _m);
        }

        // Compute tail value at each regular differencing level
        Vector<T> tempSeries = afterSeasonalDiff;
        var regularTailValues = new T[_d];
        for (int level = 0; level < _d; level++)
        {
            regularTailValues[level] = tempSeries[tempSeries.Length - 1];
            Vector<T> nextLevel = new Vector<T>(tempSeries.Length - 1);
            for (int i = 1; i < tempSeries.Length; i++)
            {
                nextLevel[i - 1] = NumOps.Subtract(tempSeries[i], tempSeries[i - 1]);
            }
            tempSeries = nextLevel;
        }

        // Undo regular differencing in reverse
        for (int level = _d - 1; level >= 0; level--)
        {
            T lastVal = regularTailValues[level];
            for (int i = 0; i < forecasts.Count; i++)
            {
                T undiff = NumOps.Add(forecasts[i], lastVal);
                forecasts[i] = undiff;
                lastVal = undiff;
            }
        }
    }

    /// <summary>
    /// Undoes seasonal differencing D times by reconstructing original-scale values
    /// from the seasonal tail of the history at each level.
    /// </summary>
    private void UndoSeasonalDifferencing(List<T> forecasts, Vector<T> history)
    {
        for (int level = 0; level < _D; level++)
        {
            // Compute the series at this seasonal differencing level
            Vector<T> seriesAtLevel = history;
            for (int d2 = 0; d2 < _D - 1 - level; d2++)
            {
                seriesAtLevel = SeasonalDifference(seriesAtLevel, _m);
            }

            // Get the last m values from this series
            var seasonalTail = new List<T>();
            for (int i = Math.Max(0, seriesAtLevel.Length - _m); i < seriesAtLevel.Length; i++)
            {
                seasonalTail.Add(seriesAtLevel[i]);
            }

            // Undo seasonal differencing: forecast[i] = diffForecast[i] + value[i - m]
            var combined = new List<T>(seasonalTail);
            for (int i = 0; i < forecasts.Count; i++)
            {
                int refIdx = combined.Count - _m;
                T refVal = refIdx >= 0 ? combined[refIdx] : NumOps.Zero;
                T undiff = NumOps.Add(forecasts[i], refVal);
                forecasts[i] = undiff;
                combined.Add(undiff);
            }
        }
    }

    /// <summary>
    /// Gets metadata about the model, including its type, parameters, and configuration.
    /// </summary>
    /// <returns>A ModelMetaData object containing information about the model.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method provides information about the model's configuration and parameters.
    /// It's useful for:
    /// - Comparing different models
    /// - Documenting the model's specifications
    /// - Tracking which parameters worked best for different datasets
    /// - Saving model metadata along with predictions
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            ModelType = ModelType.SARIMAModel,
            AdditionalInfo = new Dictionary<string, object>
            {
                // SARIMA parameters
                { "P", _p },
                { "D", _d },
                { "Q", _q },
                { "SeasonalP", _P },
                { "SeasonalD", _D },
                { "SeasonalQ", _Q },
                { "SeasonalPeriod", _m },
            
                // Model coefficients
                { "ARCoefficientsCount", _arCoefficients?.Length ?? 0 },
                { "MACoefficientsCount", _maCoefficients?.Length ?? 0 },
                { "SARCoefficientsCount", _sarCoefficients?.Length ?? 0 },
                { "SMACoefficientsCount", _smaCoefficients?.Length ?? 0 },
                { "Constant", Convert.ToDouble(_constant) },
            
                // Coefficient values if not too large
                { "ARCoefficients", _arCoefficients?.Select(c => Convert.ToDouble(c)).ToArray() ?? [] },
                { "MACoefficients", _maCoefficients?.Select(c => Convert.ToDouble(c)).ToArray() ?? [] },
                { "SARCoefficients", _sarCoefficients ?.Select(c => Convert.ToDouble(c)).ToArray() ??[] },
                { "SMACoefficients", _smaCoefficients ?.Select(c => Convert.ToDouble(c)).ToArray() ??[] },
            
                // Additional settings from options
                { "MaxIterations", _sarimaOptions.MaxIterations },
                { "Tolerance", _sarimaOptions.Tolerance }
            },
            ModelData = this.Serialize()
        };

        return metadata;
    }

    /// <summary>
    /// Creates a new instance of the SARIMA model with the same options.
    /// </summary>
    /// <returns>A new instance of the SARIMA model.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method creates a fresh copy of the model with the same settings but no trained coefficients.
    /// It's useful when you want to:
    /// - Train the same model structure on different data
    /// - Create multiple versions of the same model for ensemble methods
    /// - Start fresh with the same configuration after experimenting
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateInstance()
    {
        // Create a deep copy of the options
        var newOptions = new SARIMAOptions<T>
        {
            P = _sarimaOptions.P,
            D = _sarimaOptions.D,
            Q = _sarimaOptions.Q,
            SeasonalP = _sarimaOptions.SeasonalP,
            SeasonalD = _sarimaOptions.SeasonalD,
            SeasonalQ = _sarimaOptions.SeasonalQ,
            SeasonalPeriod = _sarimaOptions.SeasonalPeriod,
            MaxIterations = _sarimaOptions.MaxIterations,
            Tolerance = _sarimaOptions.Tolerance
        };

        // Create a new instance with the copied options
        return new SARIMAModel<T>(newOptions);
    }
}
