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
/// <para>
/// For Beginners:
/// ARIMA is a popular technique for analyzing and forecasting time series data (data collected over time, 
/// like stock prices, temperature readings, or monthly sales figures).
/// 
/// Think of ARIMA as combining three different approaches:
/// 1. AutoRegressive (AR): Looks at past values to predict future values. For example, today's
///    temperature might be related to yesterday's temperature.
/// 2. Integrated (I): Transforms the data to make it easier to analyze by removing trends.
///    For example, instead of looking at temperatures directly, we might look at how they
///    change from day to day.
/// 3. Moving Average (MA): Looks at past prediction errors to improve future predictions.
///    For example, if we consistently underestimate temperature, we can adjust for that.
/// 
/// The model has three key parameters (p, d, q):
/// - p: How many past values to look at (AR component)
/// - d: How many times to difference the data (I component)
/// - q: How many past prediction errors to consider (MA component)
/// </para>
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
    /// For Beginners:
    /// These coefficients determine how much each past value influences the prediction.
    /// For example, if the coefficient for yesterday's value is 0.7, it means yesterday's
    /// value has a strong influence on today's prediction.
    /// </remarks>
    private Vector<T> _arCoefficients;

    /// <summary>
    /// Coefficients for the moving average (MA) component of the model.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// These coefficients determine how much each past prediction error influences the prediction.
    /// They help the model learn from its mistakes. For example, if the model consistently
    /// underpredicts, these coefficients help correct that bias.
    /// </remarks>
    private Vector<T> _maCoefficients;

    /// <summary>
    /// The constant term in the ARIMA equation.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// This is like the "baseline" value in the prediction, before considering the effects
    /// of past values and errors. It's similar to the y-intercept in a linear equation.
    /// </remarks>
    private T _constant;

    /// <summary>
    /// Creates a new ARIMA model with the specified options.
    /// </summary>
    /// <param name="options">Options for the ARIMA model, including p, d, and q parameters. If null, default options are used.</param>
    /// <remarks>
    /// For Beginners:
    /// This constructor creates a new ARIMA model. You can customize the model by providing options:
    /// - p: How many past values to consider (AR order)
    /// - d: How many times to difference the data to remove trends
    /// - q: How many past prediction errors to consider (MA order)
    /// 
    /// If you don't provide options, default values will be used, but it's usually best
    /// to choose values that make sense for your specific data.
    /// </remarks>
    public ARIMAModel(ARIMAOptions<T>? options = null) : base(options ?? new())
    {
        _arimaOptions = options ?? new();
        _constant = NumOps.Zero;
        _arCoefficients = Vector<T>.Empty();
        _maCoefficients = Vector<T>.Empty();
    }

    /// <summary>
    /// Trains the ARIMA model on the provided data.
    /// </summary>
    /// <param name="x">Feature matrix (typically just time indices for ARIMA models).</param>
    /// <param name="y">Target vector (the time series values to be modeled).</param>
    /// <remarks>
    /// For Beginners:
    /// This method "teaches" the ARIMA model using your historical data. The training process:
    /// 1. Differences the data (if needed) to remove trends
    /// 2. Estimates how past values influence future values (AR coefficients)
    /// 3. Estimates how past prediction errors influence future values (MA coefficients)
    /// 4. Calculates a constant term for the model
    /// 
    /// After training, the model can be used to make predictions.
    /// 
    /// Note: For ARIMA models, the x parameter is often just a placeholder as the model primarily
    /// uses the time series values themselves (y) for prediction.
    /// </remarks>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        int p = _arimaOptions.P; // AR order
        int d = _arimaOptions.D;
        int q = _arimaOptions.Q;

        // Step 1: Difference the series
        Vector<T> diffY = TimeSeriesHelper<T>.DifferenceSeries(y, d);

        // Step 2: Estimate AR coefficients
        _arCoefficients = TimeSeriesHelper<T>.EstimateARCoefficients(diffY, p, MatrixDecompositionType.Qr);

        // Step 3: Estimate MA coefficients
        Vector<T> arResiduals = TimeSeriesHelper<T>.CalculateARResiduals(diffY, _arCoefficients);
        _maCoefficients = TimeSeriesHelper<T>.EstimateMACoefficients(arResiduals, q);

        // Step 4: Estimate constant term
        _constant = EstimateConstant(diffY, _arCoefficients, _maCoefficients);
    }

    /// <summary>
    /// Estimates the constant term for the ARIMA model.
    /// </summary>
    /// <param name="y">The differenced time series.</param>
    /// <param name="arCoefficients">The AR coefficients.</param>
    /// <param name="maCoefficients">The MA coefficients.</param>
    /// <returns>The estimated constant term.</returns>
    /// <remarks>
    /// For Beginners:
    /// This private method calculates the "baseline" value for predictions.
    /// 
    /// The constant term represents the average value of the time series after
    /// accounting for the influence of the AR components. It ensures that
    /// the model's predictions center around the correct average value.
    /// </remarks>
    private T EstimateConstant(Vector<T> y, Vector<T> arCoefficients, Vector<T> maCoefficients)
    {
        T mean = y.Average();
        T arSum = NumOps.Zero;
        for (int i = 0; i < arCoefficients.Length; i++)
        {
            arSum = NumOps.Add(arSum, arCoefficients[i]);
        }

        return NumOps.Multiply(mean, NumOps.Subtract(NumOps.One, arSum));
    }

    /// <summary>
    /// Makes predictions using the trained ARIMA model.
    /// </summary>
    /// <param name="input">Input matrix for prediction (typically just time indices for future periods).</param>
    /// <returns>A vector of predicted values.</returns>
    /// <remarks>
    /// For Beginners:
    /// This method uses the trained ARIMA model to forecast future values.
    /// 
    /// The prediction process:
    /// 1. Starts with the constant term as a base value
    /// 2. Adds the effects of past observations (AR component)
    /// 3. Adds the effects of past prediction errors (MA component)
    /// 4. For each prediction, updates the history used for the next prediction
    /// 
    /// Note: For pure time series forecasting, the input parameter might just indicate
    /// how many future periods to predict.
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> input)
    {
        Vector<T> predictions = new(input.Rows);
        Vector<T> lastObservedValues = new(_options.LagOrder);
        Vector<T> lastErrors = new(_maCoefficients.Length);

        for (int i = 0; i < predictions.Length; i++)
        {
            T prediction = _constant;

            // Add AR component
            for (int j = 0; j < _arCoefficients.Length; j++)
            {
                prediction = NumOps.Add(prediction, NumOps.Multiply(_arCoefficients[j], lastObservedValues[j]));
            }

            // Add MA component
            for (int j = 0; j < _maCoefficients.Length; j++)
            {
                prediction = NumOps.Add(prediction, NumOps.Multiply(_maCoefficients[j], lastErrors[j]));
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
    /// Evaluates the model's performance on test data.
    /// </summary>
    /// <param name="xTest">Feature matrix for testing.</param>
    /// <param name="yTest">Actual target values for testing.</param>
    /// <returns>A dictionary of evaluation metrics (MSE, RMSE, MAE).</returns>
    /// <remarks>
    /// For Beginners:
    /// This method measures how well the model performs by comparing its predictions
    /// against actual values from a test dataset.
    /// 
    /// It calculates several common error metrics:
    /// - MSE (Mean Squared Error): The average of squared differences between predictions and actual values
    /// - RMSE (Root Mean Squared Error): The square root of MSE, which is in the same units as the original data
    /// - MAE (Mean Absolute Error): The average of absolute differences between predictions and actual values
    /// 
    /// Lower values for all these metrics indicate better performance.
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
    /// For Beginners:
    /// This private method saves the model's internal state to a file or stream.
    /// 
    /// Serialization allows you to:
    /// 1. Save a trained model to disk
    /// 2. Load it later without having to retrain
    /// 3. Share the model with others
    /// 
    /// The method saves all the essential parameters: the p, d, q values,
    /// the constant term, and the AR and MA coefficients.
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
    /// For Beginners:
    /// This private method loads a previously saved model from a file or stream.
    /// 
    /// Deserialization allows you to:
    /// 1. Load a previously trained model
    /// 2. Use it immediately without retraining
    /// 3. Apply the exact same model to new data
    /// 
    /// The method loads all the parameters that were saved during serialization:
    /// the p, d, q values, the constant term, and the AR and MA coefficients.
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
}