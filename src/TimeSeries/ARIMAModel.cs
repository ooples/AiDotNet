namespace AiDotNet.TimeSeries;

/// <summary>
/// Implements an ARIMA (AutoRegressive Integrated Moving Average) model for time series forecasting.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double, decimal).</typeparam>
/// <remarks>
/// <para>
/// ARIMA models are widely used for time series forecasting. The model combines three components:
/// - AR (AutoRegressive): Uses the dependent relationship between an observation and a number of lagged observations
/// - I (Integrated): Uses differencing of observations to make the time series stationary
/// - MA (Moving Average): Uses the dependency between an observation and residual errors from a moving average model
/// </para>
/// 
/// <para><b>For Beginners:</b>
/// ARIMA is a popular technique for analyzing and forecasting time series data (data collected over time,
/// like stock prices, temperature readings, or monthly sales figures).</para>
/// <para>Think of ARIMA as combining three different approaches:</para>
/// <list type="number">
/// <item>AutoRegressive (AR): Looks at past values to predict future values. For example, today's
///    temperature might be related to yesterday's temperature.</item>
/// <item>Integrated (I): Transforms the data to make it easier to analyze by removing trends.
///    For example, instead of looking at temperatures directly, we might look at how they
///    change from day to day.</item>
/// <item>Moving Average (MA): Looks at past prediction errors to improve future predictions.
///    For example, if we consistently underestimate temperature, we can adjust for that.</item>
/// </list>
/// <para>The model has three key parameters (p, d, q):</para>
/// <list type="bullet">
/// <item>p: How many past values to look at (AR component)</item>
/// <item>d: How many times to difference the data (I component)</item>
/// <item>q: How many past prediction errors to consider (MA component)</item>
/// </list>
/// </remarks>
public class ARIMAModel<T> : TimeSeriesModelBase<T>
{
    /// <summary>
    /// Options specific to the ARIMA model, including p, d, and q parameters.
    /// </summary>
    private ARIMAOptions<T> _arimaOptions;

    /// <summary>
    /// Coefficients for the autoregressive (AR) component of the model.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// These coefficients determine how much each past value influences the prediction.
    /// For example, if the coefficient for yesterday's value is 0.7, it means yesterday's
    /// value has a strong influence on today's prediction.</para>
    /// </remarks>
    private Vector<T> _arCoefficients;

    /// <summary>
    /// Coefficients for the moving average (MA) component of the model.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// These coefficients determine how much each past prediction error influences the prediction.
    /// They help the model learn from its mistakes. For example, if the model consistently
    /// underpredicts, these coefficients help correct that bias.</para>
    /// </remarks>
    private Vector<T> _maCoefficients;

    /// <summary>
    /// The constant term in the ARIMA equation.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// This is like the "baseline" value in the prediction, before considering the effects
    /// of past values and errors. It's similar to the y-intercept in a linear equation.</para>
    /// </remarks>
    private T _constant;

    /// <summary>
    /// The anomaly detection threshold computed during training.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// This value represents the cutoff for determining if a prediction error is large enough
    /// to be considered an anomaly. It's computed from the training data residuals.</para>
    /// </remarks>
    private T _anomalyThreshold;

    /// <summary>
    /// The standard deviation of residuals computed during training.
    /// </summary>
    private T _residualStdDev;

    /// <summary>
    /// The mean of residuals computed during training.
    /// </summary>
    private T _residualMean;

    /// <summary>
    /// Stores the last d values from the original (undifferenced) training data,
    /// needed for undifferencing forecasts.
    /// </summary>
    private Vector<T> _lastOriginalValues;

    /// <summary>
    /// Creates a new ARIMA model with the specified options.
    /// </summary>
    /// <param name="options">Options for the ARIMA model, including p, d, and q parameters. If null, default options are used.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// This constructor creates a new ARIMA model. You can customize the model by providing options:</para>
    /// <list type="bullet">
    /// <item>p: How many past values to consider (AR order)</item>
    /// <item>d: How many times to difference the data to remove trends</item>
    /// <item>q: How many past prediction errors to consider (MA order)</item>
    /// </list>
    /// <para>If you don't provide options, default values will be used, but it's usually best
    /// to choose values that make sense for your specific data.</para>
    /// </remarks>
    public ARIMAModel(ARIMAOptions<T>? options = null) : base(options ?? new())
    {
        _arimaOptions = options ?? new();
        _constant = NumOps.Zero;
        _arCoefficients = Vector<T>.Empty();
        _maCoefficients = Vector<T>.Empty();
        _anomalyThreshold = NumOps.Zero;
        _residualStdDev = NumOps.Zero;
        _residualMean = NumOps.Zero;
        _lastOriginalValues = Vector<T>.Empty();
    }

    /// <summary>
    /// Estimates the constant term for the ARIMA model.
    /// </summary>
    /// <param name="y">The differenced time series.</param>
    /// <param name="arCoefficients">The AR coefficients.</param>
    /// <param name="maCoefficients">The MA coefficients.</param>
    /// <returns>The estimated constant term.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// This private method calculates the "baseline" value for predictions.</para>
    /// <para>The constant term represents the average value of the time series after
    /// accounting for the influence of the AR components. It ensures that
    /// the model's predictions center around the correct average value.</para>
    /// </remarks>
    private T EstimateConstant(Vector<T> y, Vector<T> arCoefficients, Vector<T> maCoefficients)
    {
        T mean = y.Average();
        // VECTORIZED: Use Engine.Sum() for AR coefficient summation
        T arSum = Engine.Sum(arCoefficients);

        return NumOps.Multiply(mean, NumOps.Subtract(NumOps.One, arSum));
    }

    /// <summary>
    /// Makes predictions using the trained ARIMA model.
    /// </summary>
    /// <param name="input">Input matrix for prediction (typically just time indices for future periods).</param>
    /// <returns>A vector of predicted values.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// This method uses the trained ARIMA model to forecast future values.</para>
    /// <para>The prediction process:</para>
    /// <list type="number">
    /// <item>Starts with the constant term as a base value</item>
    /// <item>Adds the effects of past observations (AR component)</item>
    /// <item>Adds the effects of past prediction errors (MA component)</item>
    /// <item>For each prediction, updates the history used for the next prediction</item>
    /// </list>
    /// <para>Note: For pure time series forecasting, the input parameter might just indicate
    /// how many future periods to predict.</para>
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> input)
    {
        Vector<T> predictions = new(input.Rows);
        // Use _arCoefficients.Length (which is P) for the AR component, not LagOrder
        // This ensures the dot product vectors have matching lengths
        Vector<T> lastObservedValues = new(_arCoefficients.Length);
        Vector<T> lastErrors = new(_maCoefficients.Length);

        for (int i = 0; i < predictions.Length; i++)
        {
            T prediction = _constant;

            // VECTORIZED: Add AR component using dot product
            if (_arCoefficients.Length > 0)
            {
                prediction = NumOps.Add(prediction, Engine.DotProduct(_arCoefficients, lastObservedValues));
            }

            // VECTORIZED: Add MA component using dot product
            if (_maCoefficients.Length > 0)
            {
                prediction = NumOps.Add(prediction, Engine.DotProduct(_maCoefficients, lastErrors));
            }

            predictions[i] = prediction;

            // VECTORIZED: Shift last observed values using slice and copy
            if (lastObservedValues.Length > 1)
            {
                var shifted = lastObservedValues.Slice(0, lastObservedValues.Length - 1);
                for (int j = 1; j < lastObservedValues.Length; j++)
                {
                    lastObservedValues[j] = shifted[j - 1];
                }
            }
            // Only set observed value if there are AR coefficients (P > 0)
            if (lastObservedValues.Length > 0)
            {
                lastObservedValues[0] = prediction;
            }

            // VECTORIZED: Shift last errors using slice and copy
            if (lastErrors.Length > 1)
            {
                var shiftedErrors = lastErrors.Slice(0, lastErrors.Length - 1);
                for (int j = 1; j < lastErrors.Length; j++)
                {
                    lastErrors[j] = shiftedErrors[j - 1];
                }
            }
            // Only set error if there are MA coefficients (Q > 0)
            if (lastErrors.Length > 0)
            {
                lastErrors[0] = NumOps.Zero; // Assume zero error for future predictions
            }
        }

        return predictions;
    }

    /// <summary>
    /// Evaluates the model's performance on test data.
    /// </summary>
    /// <param name="xTest">Feature matrix for testing.</param>
    /// <param name="yTest">Actual target values for testing.</param>
    /// <returns>A dictionary of evaluation metrics (MSE, RMSE, MAE).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// This method measures how well the model performs by comparing its predictions
    /// against actual values from a test dataset.</para>
    /// <para>It calculates several common error metrics:</para>
    /// <list type="bullet">
    /// <item>MSE (Mean Squared Error): The average of squared differences between predictions and actual values</item>
    /// <item>RMSE (Root Mean Squared Error): The square root of MSE, which is in the same units as the original data</item>
    /// <item>MAE (Mean Absolute Error): The average of absolute differences between predictions and actual values</item>
    /// </list>
    /// <para>Lower values for all these metrics indicate better performance.</para>
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
    /// Serializes the model's state to a binary stream.
    /// </summary>
    /// <param name="writer">The binary writer to write to.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// This private method saves the model's internal state to a file or stream.</para>
    /// <para>Serialization allows you to:</para>
    /// <list type="number">
    /// <item>Save a trained model to disk</item>
    /// <item>Load it later without having to retrain</item>
    /// <item>Share the model with others</item>
    /// </list>
    /// <para>The method saves all the essential parameters: the p, d, q values,
    /// the constant term, and the AR and MA coefficients.</para>
    /// </remarks>
    protected override void SerializeCore(BinaryWriter writer)
    {
        // Write ARIMA-specific options
        writer.Write(_arimaOptions.P);
        writer.Write(_arimaOptions.D);
        writer.Write(_arimaOptions.Q);

        // Write constant
        writer.Write(Convert.ToDouble(_constant));

        // Write AR coefficients
        writer.Write(_arCoefficients.Length);
        for (int i = 0; i < _arCoefficients.Length; i++)
        {
            writer.Write(Convert.ToDouble(_arCoefficients[i]));
        }

        // Write MA coefficients
        writer.Write(_maCoefficients.Length);
        for (int i = 0; i < _maCoefficients.Length; i++)
        {
            writer.Write(Convert.ToDouble(_maCoefficients[i]));
        }
    }

    /// <summary>
    /// Deserializes the model's state from a binary stream.
    /// </summary>
    /// <param name="reader">The binary reader to read from.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// This private method loads a previously saved model from a file or stream.</para>
    /// <para>Deserialization allows you to:</para>
    /// <list type="number">
    /// <item>Load a previously trained model</item>
    /// <item>Use it immediately without retraining</item>
    /// <item>Apply the exact same model to new data</item>
    /// </list>
    /// <para>The method loads all the parameters that were saved during serialization:
    /// the p, d, q values, the constant term, and the AR and MA coefficients.</para>
    /// </remarks>
    protected override void DeserializeCore(BinaryReader reader)
    {
        // Read ARIMA-specific options
        int p = reader.ReadInt32();
        int d = reader.ReadInt32();
        int q = reader.ReadInt32();
        _arimaOptions = new ARIMAOptions<T>
        {
            P = p,
            D = d,
            Q = q
        };

        // Read constant
        _constant = NumOps.FromDouble(reader.ReadDouble());

        // Read AR coefficients
        int arLength = reader.ReadInt32();
        _arCoefficients = new Vector<T>(arLength);
        for (int i = 0; i < arLength; i++)
        {
            _arCoefficients[i] = NumOps.FromDouble(reader.ReadDouble());
        }

        // Read MA coefficients
        int maLength = reader.ReadInt32();
        _maCoefficients = new Vector<T>(maLength);
        for (int i = 0; i < maLength; i++)
        {
            _maCoefficients[i] = NumOps.FromDouble(reader.ReadDouble());
        }
    }

    /// <summary>
    /// Core implementation of the training logic for the ARIMA model.
    /// </summary>
    /// <param name="x">Feature matrix (typically just time indices for ARIMA models).</param>
    /// <param name="y">Target vector (the time series values to be modeled).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// This method contains the core implementation of the training process. It:</para>
    /// <list type="number">
    /// <item>Differences the data to remove trends (the "I" in ARIMA)</item>
    /// <item>Estimates the AR coefficients that capture how past values affect future values</item>
    /// <item>Calculates residuals and uses them to estimate the MA coefficients</item>
    /// <item>Estimates the constant term that serves as the baseline prediction</item>
    /// </list>
    /// <para>This implementation follows the same process as the public Train method but
    /// provides the actual mechanism that fits the model to your data.</para>
    /// </remarks>
    protected override void TrainCore(Matrix<T> x, Vector<T> y)
    {
        int p = _arimaOptions.P; // AR order
        int d = _arimaOptions.D; // Differencing order
        int q = _arimaOptions.Q; // MA order

        // Step 1: Difference the series to make it stationary
        Vector<T> diffY = TimeSeriesHelper<T>.DifferenceSeries(y, d);

        // Step 2: Estimate AR coefficients using least squares or Yule-Walker equations
        _arCoefficients = TimeSeriesHelper<T>.EstimateARCoefficients(diffY, p, MatrixDecompositionType.Qr);

        // Step 3: Calculate AR residuals and use them to estimate MA coefficients
        Vector<T> arResiduals = TimeSeriesHelper<T>.CalculateARResiduals(diffY, _arCoefficients);
        _maCoefficients = TimeSeriesHelper<T>.EstimateMACoefficients(arResiduals, q);

        // Step 4: Estimate constant term for the model
        _constant = EstimateConstant(diffY, _arCoefficients, _maCoefficients);

        // Step 5: Store the last d values from the original series for undifferencing forecasts
        if (d > 0)
        {
            _lastOriginalValues = new Vector<T>(d);
            for (int i = 0; i < d; i++)
            {
                _lastOriginalValues[i] = y[y.Length - d + i];
            }
        }
        else
        {
            _lastOriginalValues = Vector<T>.Empty();
        }

        // Step 6: If anomaly detection is enabled, compute threshold from residuals
        if (_arimaOptions.EnableAnomalyDetection)
        {
            ComputeAnomalyThreshold(arResiduals);
        }
    }

    /// <summary>
    /// Computes the anomaly detection threshold from training residuals.
    /// </summary>
    /// <param name="residuals">The residuals from training.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// This method calculates what counts as "normal" variation in your data by looking at how
    /// far off the model's predictions were during training. It then sets a threshold so that
    /// only unusually large errors get flagged as anomalies.</para>
    /// </remarks>
    private void ComputeAnomalyThreshold(Vector<T> residuals)
    {
        if (residuals.Length == 0)
        {
            _residualMean = NumOps.Zero;
            _residualStdDev = NumOps.One;
            _anomalyThreshold = NumOps.FromDouble(_arimaOptions.AnomalyThresholdSigma);
            return;
        }

        // Compute mean of absolute residuals
        T sum = NumOps.Zero;
        for (int i = 0; i < residuals.Length; i++)
        {
            sum = NumOps.Add(sum, NumOps.Abs(residuals[i]));
        }
        _residualMean = NumOps.Divide(sum, NumOps.FromDouble(residuals.Length));

        // Compute standard deviation of absolute residuals
        T sumSquaredDiff = NumOps.Zero;
        for (int i = 0; i < residuals.Length; i++)
        {
            T diff = NumOps.Subtract(NumOps.Abs(residuals[i]), _residualMean);
            sumSquaredDiff = NumOps.Add(sumSquaredDiff, NumOps.Multiply(diff, diff));
        }
        _residualStdDev = NumOps.Sqrt(NumOps.Divide(sumSquaredDiff, NumOps.FromDouble(residuals.Length)));

        // If std dev is zero (all residuals are the same), use a small default
        if (NumOps.Equals(_residualStdDev, NumOps.Zero))
        {
            _residualStdDev = NumOps.FromDouble(0.001);
        }

        // Threshold = mean + (sigma * stddev)
        _anomalyThreshold = NumOps.Add(
            _residualMean,
            NumOps.Multiply(NumOps.FromDouble(_arimaOptions.AnomalyThresholdSigma), _residualStdDev)
        );
    }

    /// <summary>
    /// Predicts a single value based on the input vector.
    /// </summary>
    /// <param name="input">Input vector containing features for prediction.</param>
    /// <returns>The predicted value.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// This method generates a single prediction based on your input data.</para>
    /// <para>The prediction process:</para>
    /// <list type="number">
    /// <item>Starts with the constant term as the baseline value</item>
    /// <item>Adds the influence of past observations (AR component)</item>
    /// <item>Adds the influence of past prediction errors (MA component)</item>
    /// </list>
    /// <para>This is useful when you need just one prediction rather than a whole series.
    /// For example, if you want to predict tomorrow's temperature specifically,
    /// rather than temperatures for the next week.</para>
    /// </remarks>
    public override T PredictSingle(Vector<T> input)
    {
        // Start with the constant term as the baseline prediction
        T prediction = _constant;

        // Check if model has been trained
        if (_arCoefficients.Length == 0 && _maCoefficients.Length == 0)
        {
            throw new InvalidOperationException("Model must be trained before making predictions.");
        }

        // For ARIMA models, we typically use recent observations from the time series
        // rather than arbitrary input features.
        // This implementation assumes input contains recent observations in reverse order
        // (most recent first)

        // Add AR component - influence of past observations
        for (int j = 0; j < _arCoefficients.Length && j < input.Length; j++)
        {
            prediction = NumOps.Add(prediction, NumOps.Multiply(_arCoefficients[j], input[j]));
        }

        // Since we can't know the actual errors for future predictions,
        // the MA component is often excluded when predicting a single value
        // or we assume errors of zero for simplicity

        return prediction;
    }

    /// <summary>
    /// Forecasts future values using the trained ARIMA model, properly handling differencing.
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

        int d = _arimaOptions.D;

        if (d > 0 && history.Length < d + 1)
        {
            throw new ArgumentException(
                $"History length ({history.Length}) must be at least {d + 1} to support differencing order {d}.",
                nameof(history));
        }

        // If no differencing, fall back to base class behavior
        if (d == 0)
        {
            return base.Forecast(history, steps);
        }

        // Difference the history to get the series that the AR/MA coefficients were trained on
        Vector<T> diffHistory = TimeSeriesHelper<T>.DifferenceSeries(history, d);

        // Build a working list of differenced values we can extend
        List<T> extendedDiffHistory = new List<T>(diffHistory.Length + steps);
        for (int i = 0; i < diffHistory.Length; i++)
        {
            extendedDiffHistory.Add(diffHistory[i]);
        }

        // Generate forecasts on the differenced scale
        Vector<T> diffForecasts = new Vector<T>(steps);
        for (int step = 0; step < steps; step++)
        {
            T prediction = _constant;

            // AR component: use the most recent p differenced values
            for (int j = 0; j < _arCoefficients.Length && j < extendedDiffHistory.Count; j++)
            {
                prediction = NumOps.Add(prediction, NumOps.Multiply(
                    _arCoefficients[j], extendedDiffHistory[extendedDiffHistory.Count - 1 - j]));
            }

            // MA component is assumed zero for future predictions (no actual errors available)

            diffForecasts[step] = prediction;
            extendedDiffHistory.Add(prediction);
        }

        // Undifference the forecasts back to the original scale
        return UndifferenceForecasts(diffForecasts, history, d, steps);
    }

    /// <summary>
    /// Undifferences forecasted values back to the original scale by reversing the differencing
    /// that was applied during training.
    /// </summary>
    /// <param name="diffForecasts">The forecasts on the differenced scale.</param>
    /// <param name="history">The original (undifferenced) history to derive tail values from.</param>
    /// <param name="d">The differencing order.</param>
    /// <param name="steps">The number of forecast steps.</param>
    /// <returns>A vector of forecasts on the original (undifferenced) scale.</returns>
    private Vector<T> UndifferenceForecasts(Vector<T> diffForecasts, Vector<T> history, int d, int steps)
    {
        // Compute tail values at each integration level from the history parameter
        // Level 0: original history -> last value = history[n-1]
        // Level 1: first-differenced -> last value = history[n-1] - history[n-2]
        // etc.
        var tailValues = new T[d];
        var tempTail = new List<T>();
        int tailStart = Math.Max(0, history.Length - d);
        for (int i = tailStart; i < history.Length; i++)
        {
            tempTail.Add(history[i]);
        }

        for (int level = 0; level < d; level++)
        {
            tailValues[level] = tempTail[tempTail.Count - 1];
            var newTail = new List<T>();
            for (int i = 1; i < tempTail.Count; i++)
            {
                newTail.Add(NumOps.Subtract(tempTail[i], tempTail[i - 1]));
            }
            tempTail = newTail;
        }

        // Undifference d times (in reverse order of differencing)
        var currentForecasts = new List<T>(steps);
        for (int i = 0; i < steps; i++)
        {
            currentForecasts.Add(diffForecasts[i]);
        }

        for (int level = d - 1; level >= 0; level--)
        {
            T lastVal = tailValues[level];
            for (int i = 0; i < currentForecasts.Count; i++)
            {
                T undiff = NumOps.Add(currentForecasts[i], lastVal);
                currentForecasts[i] = undiff;
                lastVal = undiff;
            }
        }

        Vector<T> result = new Vector<T>(steps);
        for (int i = 0; i < steps; i++)
        {
            result[i] = currentForecasts[i];
        }

        return result;
    }

    /// <summary>
    /// Gets metadata about the model, including its type, parameters, and configuration.
    /// </summary>
    /// <returns>A ModelMetaData object containing information about the model.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// This method provides a summary of your model's settings and what it has learned.</para>
    /// <para>The metadata includes:</para>
    /// <list type="bullet">
    /// <item>The type of model (ARIMA)</item>
    /// <item>The p, d, and q parameters that define the model structure</item>
    /// <item>The AR and MA coefficients that were learned during training</item>
    /// <item>The constant term that serves as the baseline prediction</item>
    /// </list>
    /// <para>This information is useful for:</para>
    /// <list type="bullet">
    /// <item>Documenting your model for future reference</item>
    /// <item>Comparing different models to see which performs best</item>
    /// <item>Understanding what patterns the model has identified in your data</item>
    /// </list>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            ModelType = ModelType.ARIMAModel,
            AdditionalInfo = new Dictionary<string, object>
            {
                // ARIMA-specific parameters
                { "P", _arimaOptions.P },
                { "D", _arimaOptions.D },
                { "Q", _arimaOptions.Q },
            
                // Model coefficients
                { "ARCoefficientsCount", _arCoefficients.Length },
                { "MACoefficientsCount", _maCoefficients.Length },
                { "Constant", Convert.ToDouble(_constant) },
            
                // Additional settings from options
                { "LagOrder", _arimaOptions.LagOrder }
            },
            ModelData = this.Serialize()
        };

        return metadata;
    }

    /// <summary>
    /// Creates a new instance of the ARIMA model with the same options.
    /// </summary>
    /// <returns>A new instance of the ARIMA model.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// This method creates a fresh copy of the model with the same settings.</para>
    /// <para>The new copy:</para>
    /// <list type="bullet">
    /// <item>Has the same p, d, and q parameters as the original model</item>
    /// <item>Has the same configuration options</item>
    /// <item>Is untrained (doesn't have coefficients yet)</item>
    /// </list>
    /// <para>This is useful when you want to:</para>
    /// <list type="bullet">
    /// <item>Train multiple versions of the same model on different data</item>
    /// <item>Create ensemble models that combine predictions from multiple similar models</item>
    /// <item>Reset a model to start fresh while keeping the same structure</item>
    /// </list>
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateInstance()
    {
        // Create a new instance with the same options
        return new ARIMAModel<T>(_arimaOptions);
    }

    /// <summary>
    /// Detects anomalies in a time series by comparing predictions to actual values.
    /// </summary>
    /// <param name="timeSeries">The time series data to analyze for anomalies.</param>
    /// <returns>A boolean array where true indicates an anomaly at that position.</returns>
    /// <remarks>
    /// <para>
    /// This method uses the ARIMA model to predict each point in the time series based on
    /// previous values, then flags points where the prediction error exceeds the anomaly threshold
    /// computed during training.
    /// </para>
    /// <para><b>For Beginners:</b> This method goes through your time series and identifies
    /// points that are "unusual" compared to what the model would expect. A point is considered
    /// an anomaly if the difference between the actual value and the predicted value is larger
    /// than the threshold learned during training.
    ///
    /// Example use case: If you have daily sales data, this method can identify days where
    /// sales were abnormally high or low compared to the typical pattern.
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the model hasn't been trained yet or when anomaly detection wasn't enabled during training.
    /// </exception>
    public bool[] DetectAnomalies(Vector<T> timeSeries)
    {
        if (!IsTrained)
        {
            throw new InvalidOperationException("Model must be trained before detecting anomalies.");
        }

        if (!_arimaOptions.EnableAnomalyDetection)
        {
            throw new InvalidOperationException("Anomaly detection was not enabled during training. " +
                "Set EnableAnomalyDetection = true in ARIMAOptions and retrain the model.");
        }

        var scores = ComputeAnomalyScores(timeSeries);
        bool[] anomalies = new bool[scores.Length];

        for (int i = 0; i < scores.Length; i++)
        {
            anomalies[i] = NumOps.GreaterThan(scores[i], _anomalyThreshold);
        }

        return anomalies;
    }

    /// <summary>
    /// Computes anomaly scores for each point in a time series.
    /// </summary>
    /// <param name="timeSeries">The time series data to analyze.</param>
    /// <returns>A vector of anomaly scores (absolute prediction errors) for each point.</returns>
    /// <remarks>
    /// <para>
    /// The anomaly score is the absolute difference between the actual value and the predicted value.
    /// Higher scores indicate more anomalous points. The first few points (up to the lag order)
    /// will have a score of zero since there isn't enough history to make predictions.
    /// </para>
    /// <para><b>For Beginners:</b> Instead of just saying "anomaly or not", this method tells you
    /// exactly how unusual each point is. A score of 0 means the value matches the prediction perfectly.
    /// Higher scores mean the value was more unexpected.
    ///
    /// You can use these scores to:
    /// - Rank anomalies by severity (higher score = more unusual)
    /// - Set your own custom threshold
    /// - Visualize the anomaly intensity over time
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">Thrown when the model hasn't been trained yet.</exception>
    public Vector<T> ComputeAnomalyScores(Vector<T> timeSeries)
    {
        if (!IsTrained)
        {
            throw new InvalidOperationException("Model must be trained before computing anomaly scores.");
        }

        int lagOrder = _arimaOptions.LagOrder;
        if (timeSeries.Length <= lagOrder)
        {
            throw new ArgumentException($"Time series must have more than {lagOrder} points (the lag order).");
        }

        Vector<T> scores = new Vector<T>(timeSeries.Length);

        // First lagOrder points can't have scores computed (not enough history)
        for (int i = 0; i < lagOrder; i++)
        {
            scores[i] = NumOps.Zero;
        }

        // Compute scores for the rest of the series
        for (int i = lagOrder; i < timeSeries.Length; i++)
        {
            // Create input vector from previous lagOrder values in reverse order
            // (most recent first, as expected by PredictSingle)
            Vector<T> input = new Vector<T>(lagOrder);
            for (int j = 0; j < lagOrder; j++)
            {
                input[j] = timeSeries[i - 1 - j];
            }

            T prediction = PredictSingle(input);
            T actual = timeSeries[i];
            scores[i] = NumOps.Abs(NumOps.Subtract(actual, prediction));
        }

        return scores;
    }

    /// <summary>
    /// Detects anomalies and returns detailed information about each detected anomaly.
    /// </summary>
    /// <param name="timeSeries">The time series data to analyze.</param>
    /// <returns>A list of tuples containing (index, actual value, predicted value, score) for each anomaly.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method not only tells you which points are anomalies,
    /// but also provides additional context:
    /// - Index: The position of the anomaly in the time series
    /// - Actual: What the value actually was
    /// - Predicted: What the model expected the value to be
    /// - Score: How far off the prediction was
    ///
    /// This extra information helps you understand why each point was flagged as an anomaly.
    /// </para>
    /// </remarks>
    public List<(int Index, T Actual, T Predicted, T Score)> DetectAnomaliesDetailed(Vector<T> timeSeries)
    {
        if (!IsTrained)
        {
            throw new InvalidOperationException("Model must be trained before detecting anomalies.");
        }

        if (!_arimaOptions.EnableAnomalyDetection)
        {
            throw new InvalidOperationException("Anomaly detection was not enabled during training.");
        }

        int lagOrder = _arimaOptions.LagOrder;
        var anomalies = new List<(int Index, T Actual, T Predicted, T Score)>();

        for (int i = lagOrder; i < timeSeries.Length; i++)
        {
            // Create input vector in reverse order (most recent first, as expected by PredictSingle)
            Vector<T> input = new Vector<T>(lagOrder);
            for (int j = 0; j < lagOrder; j++)
            {
                input[j] = timeSeries[i - 1 - j];
            }

            T prediction = PredictSingle(input);
            T actual = timeSeries[i];
            T score = NumOps.Abs(NumOps.Subtract(actual, prediction));

            if (NumOps.GreaterThan(score, _anomalyThreshold))
            {
                anomalies.Add((i, actual, prediction, score));
            }
        }

        return anomalies;
    }

    /// <summary>
    /// Gets the current anomaly detection threshold.
    /// </summary>
    /// <returns>The anomaly threshold computed during training.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This tells you the current cutoff value used to decide
    /// whether a prediction error is large enough to be an anomaly. Values above this threshold
    /// are considered anomalies.
    /// </para>
    /// </remarks>
    public T GetAnomalyThreshold()
    {
        if (!_arimaOptions.EnableAnomalyDetection)
        {
            throw new InvalidOperationException("Anomaly detection was not enabled during training.");
        }
        return _anomalyThreshold;
    }

    /// <summary>
    /// Sets a custom anomaly detection threshold.
    /// </summary>
    /// <param name="threshold">The new threshold value.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> If the automatic threshold is flagging too many or too few
    /// anomalies, you can set your own. A higher threshold means fewer anomalies will be detected
    /// (only more extreme values). A lower threshold means more anomalies will be detected.
    /// </para>
    /// </remarks>
    public void SetAnomalyThreshold(T threshold)
    {
        _anomalyThreshold = threshold;
    }
}
