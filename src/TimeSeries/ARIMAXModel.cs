namespace AiDotNet.TimeSeries;

/// <summary>
/// Implements an ARIMAX (AutoRegressive Integrated Moving Average with eXogenous variables) model for time series forecasting.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double, decimal).</typeparam>
/// <remarks>
/// <para>
/// ARIMAX extends the ARIMA model by including external (exogenous) variables that might influence the time series.
/// The model combines:
/// - AR (AutoRegressive): Uses the dependent relationship between an observation and lagged observations
/// - I (Integrated): Uses differencing to make the time series stationary
/// - MA (Moving Average): Uses the dependency between an observation and residual errors
/// - X (eXogenous): Incorporates external variables that may influence the time series
/// </para>
/// 
/// <para>
/// For Beginners:
/// ARIMAX is an advanced technique for forecasting time series data (data collected over time like
/// daily temperatures, stock prices, or monthly sales) that takes into account both the history of
/// the series itself AND external factors that might influence it.
/// 
/// Think of it like this:
/// - Basic forecasting might just look at past sales to predict future sales
/// - ARIMAX also considers things like holidays, promotions, or economic indicators that might affect sales
/// 
/// The model has four components:
/// 1. AutoRegressive (AR): Uses past values of the series itself (like yesterday's temperature to predict today's)
/// 2. Integrated (I): Transforms the data by looking at differences between values to remove trends
/// 3. Moving Average (MA): Looks at past prediction errors to improve future predictions
/// 4. eXogenous (X): Includes external factors that might affect the series (like whether it's a holiday)
/// 
/// The "X" is what makes ARIMAX different from ARIMA - it can include information from outside the time series itself.
/// </para>
/// </remarks>
public class ARIMAXModel<T> : TimeSeriesModelBase<T>
{
    /// <summary>
    /// Options specific to the ARIMAX model including AR order, MA order, differencing order, and exogenous variables count.
    /// </summary>
    private ARIMAXModelOptions<T> _arimaxOptions;

    /// <summary>
    /// Coefficients for the autoregressive (AR) component of the model.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// These coefficients determine how much each past value influences the prediction.
    /// For example, if the AR order is 2, there will be two coefficients that define how strongly
    /// yesterday's value and the day before's value affect today's prediction.
    /// Larger coefficients mean stronger influence from that time period.
    /// </remarks>
    private Vector<T> _arCoefficients;

    /// <summary>
    /// Coefficients for the moving average (MA) component of the model.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// These coefficients determine how much each past prediction error influences the forecast.
    /// For example, if we consistently underpredicted in the past, these coefficients help the
    /// model learn to adjust future predictions upward. They help the model correct systematic
    /// errors in its forecasts.
    /// </remarks>
    private Vector<T> _maCoefficients;

    /// <summary>
    /// Coefficients for the exogenous variables in the model.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// These coefficients determine how much each external factor affects the prediction.
    /// For example, if one of your exogenous variables is "is_holiday" (1 if it's a holiday, 0 otherwise),
    /// its coefficient might be negative for a workplace attendance model (fewer people come to work on holidays)
    /// or positive for a retail sales model (more people shop on holidays).
    /// </remarks>
    private Vector<T> _exogenousCoefficients;

    /// <summary>
    /// Stores values needed to reverse the differencing operation during prediction.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// Differencing is a technique that transforms the data by calculating the change from one period to the next,
    /// rather than using the absolute values. This helps remove trends from the data.
    /// 
    /// For example, instead of using temperatures [68, 70, 73, 71], differencing would transform this to [2, 3, -2].
    /// 
    /// To convert predictions back to the original scale, we need to "undo" this differencing,
    /// and these stored values help with that process.
    /// </remarks>
    private Vector<T> _differenced;

    /// <summary>
    /// The constant term (intercept) in the ARIMAX equation.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// The intercept is like the "baseline" value in the prediction, before considering the effects of
    /// past values, errors, and external factors. It's similar to the y-intercept in a linear equation.
    /// 
    /// If all other factors were zero, the prediction would equal this intercept value.
    /// </remarks>
    private T _intercept;

    /// <summary>
    /// Creates a new ARIMAX model with the specified options.
    /// </summary>
    /// <param name="options">Options for the ARIMAX model, including AR order, MA order, differencing order, and exogenous variables. 
    /// If null, default options are used.</param>
    /// <remarks>
    /// For Beginners:
    /// This constructor creates a new ARIMAX model. You can customize the model by providing options:
    /// - AROrder: How many past values to consider (like using yesterday and the day before to predict today)
    /// - MAOrder: How many past prediction errors to consider
    /// - DifferenceOrder: How many times to difference the data to remove trends
    /// - ExogenousVariables: How many external factors to include in the model
    /// 
    /// If you don't provide options, default values will be used, but it's usually best
    /// to choose values that make sense for your specific data.
    /// </remarks>
    public ARIMAXModel(ARIMAXModelOptions<T>? options = null) : base(options ?? new ARIMAXModelOptions<T>())
    {
        _arimaxOptions = options ?? new();
        _arCoefficients = new Vector<T>(_arimaxOptions.AROrder);
        _maCoefficients = new Vector<T>(_arimaxOptions.MAOrder);
        _exogenousCoefficients = new Vector<T>(_arimaxOptions.ExogenousVariables);
        _differenced = new Vector<T>(0);
        _intercept = NumOps.Zero;
    }

    /// <summary>
    /// Trains the ARIMAX model on the provided data.
    /// </summary>
    /// <param name="x">Matrix of exogenous variables (external factors that may influence the time series).</param>
    /// <param name="y">Vector of time series values to be modeled and predicted.</param>
    /// <remarks>
    /// For Beginners:
    /// This method "teaches" the ARIMAX model using your historical data. The training process:
    /// 1. Differences the data (if needed) to remove trends and make it easier to analyze
    /// 2. Fits the ARIMAX model by:
    ///    - Estimating how external factors influence the time series
    ///    - Estimating how past values influence future values (AR coefficients)
    ///    - Estimating how past prediction errors influence future values (MA coefficients)
    ///    - Calculating a baseline value (intercept)
    /// 3. Updates and finalizes the model parameters
    /// 
    /// After training, the model can be used to make predictions.
    /// 
    /// Unlike ARIMA, ARIMAX requires the x parameter to contain the external factors
    /// that might influence your time series.
    /// </remarks>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        // Step 1: Perform differencing if necessary
        Vector<T> diffY = DifferenceTimeSeries(y, _arimaxOptions.DifferenceOrder);

        // Step 2: Fit ARIMAX model
        FitARIMAXModel(x, diffY);

        // Step 3: Update model parameters
        UpdateModelParameters();
    }

    /// <summary>
    /// Makes predictions using the trained ARIMAX model.
    /// </summary>
    /// <param name="xNew">Matrix of exogenous variables for the periods to be predicted.</param>
    /// <returns>A vector of predicted values.</returns>
    /// <remarks>
    /// For Beginners:
    /// This method uses the trained model to forecast future values of your time series.
    /// The prediction process:
    /// 1. Starts with the intercept (baseline value)
    /// 2. Adds the effects of external factors (like holidays, promotions, etc.)
    /// 3. Adds the effects of past observations (AR component)
    /// 4. Adds the effects of past prediction errors (MA component)
    /// 5. If differencing was used in training, "undoes" the differencing to get predictions in the original scale
    /// 
    /// The xNew parameter must contain the external factors for the future periods you want to predict.
    /// If you don't know these future external factors, you would need to predict them separately or
    /// make reasonable assumptions.
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> xNew)
    {
        Vector<T> predictions = new Vector<T>(xNew.Rows);

        for (int t = 0; t < xNew.Rows; t++)
        {
            T prediction = _intercept;

            // Apply exogenous component
            for (int i = 0; i < xNew.Columns; i++)
            {
                prediction = NumOps.Add(prediction, NumOps.Multiply(xNew[t, i], _exogenousCoefficients[i]));
            }

            // Apply AR component
            for (int p = 0; p < _arimaxOptions.AROrder; p++)
            {
                if (t - p - 1 >= 0)
                {
                    prediction = NumOps.Add(prediction, NumOps.Multiply(_arCoefficients[p], NumOps.Subtract(predictions[t - p - 1], _intercept)));
                }
            }

            // Apply MA component
            for (int q = 0; q < _arimaxOptions.MAOrder; q++)
            {
                if (t - q - 1 >= 0)
                {
                    T error = NumOps.Subtract(predictions[t - q - 1], xNew[t - q - 1, 0]); // Assuming the first column is the target variable
                    prediction = NumOps.Add(prediction, NumOps.Multiply(_maCoefficients[q], error));
                }
            }

            predictions[t] = prediction;
        }

        // Reverse differencing if necessary
        if (_arimaxOptions.DifferenceOrder > 0)
        {
            predictions = InverseDifferenceTimeSeries(predictions, _differenced);
        }

        return predictions;
    }

    /// <summary>
    /// Applies differencing to the time series to make it stationary.
    /// </summary>
    /// <param name="y">The original time series.</param>
    /// <param name="order">The number of times to apply differencing.</param>
    /// <returns>The differenced time series.</returns>
    /// <remarks>
    /// For Beginners:
    /// Differencing is a technique that transforms the data to remove trends. Instead of looking at
    /// absolute values, it looks at changes from one period to the next.
    /// 
    /// For example:
    /// - Original data: [10, 15, 14, 18]
    /// - After first-order differencing: [5, -1, 4] (the differences between consecutive values)
    /// - After second-order differencing: [-6, 5] (the differences of the differences)
    /// 
    /// This helps make the data "stationary," which means its statistical properties don't change over time.
    /// Many time series models work better with stationary data.
    /// 
    /// The "order" parameter tells the method how many times to apply this differencing operation.
    /// </remarks>
    private Vector<T> DifferenceTimeSeries(Vector<T> y, int order)
    {
        Vector<T> diffY = y;
        for (int d = 0; d < order; d++)
        {
            Vector<T> temp = new Vector<T>(diffY.Length - 1);
            for (int i = 0; i < temp.Length; i++)
            {
                temp[i] = NumOps.Subtract(diffY[i + 1], diffY[i]);
            }
            _differenced = new Vector<T>(order);
            for (int i = 0; i < order; i++)
            {
                _differenced[i] = diffY[i];
            }
            diffY = temp;
        }

        return diffY;
    }

    /// <summary>
    /// Reverses the differencing process to convert predictions back to the original scale.
    /// </summary>
    /// <param name="diffY">The differenced predictions.</param>
    /// <param name="original">The original values needed to reverse the differencing.</param>
    /// <returns>Predictions in the original scale.</returns>
    /// <remarks>
    /// For Beginners:
    /// This method "undoes" the differencing that was applied during training. Since differencing
    /// looks at changes rather than absolute values, we need the original starting values to
    /// convert back.
    /// 
    /// For example:
    /// - If original data was [10, 15, 14, 18]
    /// - And differenced data was [5, -1, 4]
    /// - To predict the next value in the original scale, we need both the prediction in the differenced scale
    ///   (let's say 2) and the last value of the original series (18)
    /// - The prediction in the original scale would be 18 + 2 = 20
    /// 
    /// This method handles this conversion process, potentially through multiple levels of differencing.
    /// </remarks>
    private Vector<T> InverseDifferenceTimeSeries(Vector<T> diffY, Vector<T> original)
    {
        Vector<T> y = diffY;
        for (int d = _arimaxOptions.DifferenceOrder - 1; d >= 0; d--)
        {
            Vector<T> temp = new Vector<T>(y.Length + 1);
            temp[0] = original[d];
            for (int i = 1; i < temp.Length; i++)
            {
                temp[i] = NumOps.Add(temp[i - 1], y[i - 1]);
            }
            y = temp;
        }
        return y;
    }

    /// <summary>
    /// Fits the ARIMAX model to the provided data.
    /// </summary>
    /// <param name="x">Matrix of exogenous variables.</param>
    /// <param name="y">Vector of differenced time series values.</param>
    /// <remarks>
    /// For Beginners:
    /// This private method does the core work of training the model. It:
    /// 
    /// 1. Estimates how the external factors (like holidays, promotions) affect the time series
    /// 2. Calculates the "residuals" - the part of the time series that can't be explained by external factors
    /// 3. Fits an ARMA model to these residuals to capture the time-dependent patterns
    /// 4. Calculates an intercept (baseline value) for the model
    /// 
    /// The resulting coefficients tell us:
    /// - How each external factor influences the time series
    /// - How past values influence future values (AR coefficients)
    /// - How past prediction errors influence future values (MA coefficients)
    /// </remarks>
    private void FitARIMAXModel(Matrix<T> x, Vector<T> y)
    {
        // Implement ARIMAX model fitting
        // This is a simplified version and may need to be expanded for more accurate results

        // Fit exogenous variables
        Matrix<T> xT = x.Transpose();
        Matrix<T> xTx = xT * x;
        Vector<T> xTy = xT * y;

        _exogenousCoefficients = MatrixSolutionHelper.SolveLinearSystem(xTx, xTy, _arimaxOptions.DecompositionType);

        // Extract residuals
        Vector<T> residuals = y - (x * _exogenousCoefficients);

        // Fit ARMA model to residuals
        FitARMAModel(residuals);

        _intercept = NumOps.Divide(y.Sum(), NumOps.FromDouble(y.Length));
    }

    /// <summary>
    /// Fits an ARMA model to the residuals after accounting for exogenous variables.
    /// </summary>
    /// <param name="residuals">The residuals after removing the effect of exogenous variables.</param>
    /// <remarks>
    /// For Beginners:
    /// This private method fits the AR and MA components of the model to the residuals.
    /// Residuals are what's left of the time series after removing the effect of external factors.
    /// 
    /// The method:
    /// 1. Calculates autocorrelations - how values at different time lags are related to each other
    /// 2. Uses these autocorrelations to estimate the AR coefficients using the Yule-Walker equations
    /// 3. Uses a simplified approach to estimate the MA coefficients
    /// 
    /// These coefficients help the model capture patterns that depend on time, like:
    /// - Seasonal patterns (e.g., higher sales every weekend)
    /// - Momentum effects (e.g., if sales have been rising, they might continue to rise)
    /// - Reversion to the mean (e.g., if today was unusually hot, tomorrow might be cooler)
    /// </remarks>
    private void FitARMAModel(Vector<T> residuals)
    {
        int p = _arimaxOptions.AROrder;
        int q = _arimaxOptions.MAOrder;

        // Calculate autocorrelations
        T[] autocorrelations = CalculateAutocorrelations(residuals, Math.Max(p, q));

        // Update AR coefficients using Yule-Walker equations
        Matrix<T> R = new Matrix<T>(p, p);
        Vector<T> r = new Vector<T>(p);

        for (int i = 0; i < p; i++)
        {
            r[i] = autocorrelations[i + 1];
            for (int j = 0; j < p; j++)
            {
                R[i, j] = autocorrelations[Math.Abs(i - j)];
            }
        }

        // Solve Yule-Walker equations
        _arCoefficients = MatrixSolutionHelper.SolveLinearSystem(R, r, _arimaxOptions.DecompositionType);

        // Update MA coefficients using a simple method
        for (int i = 0; i < q; i++)
        {
            _maCoefficients[i] = NumOps.Multiply(NumOps.FromDouble(0.5), autocorrelations[i + 1]);
        }
    }

    /// <summary>
    /// Calculates autocorrelations of a time series up to a specified lag.
    /// </summary>
    /// <param name="y">The time series.</param>
    /// <param name="maxLag">The maximum lag to calculate autocorrelations for.</param>
    /// <returns>Array of autocorrelations from lag 0 to maxLag.</returns>
    /// <remarks>
    /// For Beginners:
    /// Autocorrelation measures how similar a time series is to a delayed version of itself.
    /// It helps identify patterns and dependencies over time.
    /// 
    /// For example:
    /// - An autocorrelation of 0.8 at lag 1 means values tend to be very similar to the previous day's values
    /// - An autocorrelation of -0.4 at lag 7 means values tend to be opposite of values from a week ago
    /// - An autocorrelation near 0 means there's no relationship between values at that time lag
    /// 
    /// The result is an array where:
    /// - Position 0 contains the autocorrelation at lag 0 (always 1.0)
    /// - Position 1 contains the autocorrelation at lag 1 (correlation with previous value)
    /// - Position 2 contains the autocorrelation at lag 2 (correlation with value from 2 periods ago)
    /// And so on...
    /// 
    /// These autocorrelations are used to determine the AR and MA coefficients in the model.
    /// </remarks>
    private T[] CalculateAutocorrelations(Vector<T> y, int maxLag)
    {
        T[] autocorrelations = new T[maxLag + 1];
        T mean = StatisticsHelper<T>.CalculateMean(y);
        T variance = StatisticsHelper<T>.CalculateVariance(y);

        for (int lag = 0; lag <= maxLag; lag++)
        {
            T sum = NumOps.Zero;
            int n = y.Length - lag;

            for (int t = 0; t < n; t++)
            {
                T diff1 = NumOps.Subtract(y[t], mean);
                T diff2 = NumOps.Subtract(y[t + lag], mean);
                sum = NumOps.Add(sum, NumOps.Multiply(diff1, diff2));
            }

            autocorrelations[lag] = NumOps.Divide(sum, NumOps.Multiply(NumOps.FromDouble(n), variance));
        }

        return autocorrelations;
    }

    /// <summary>
    /// Updates model parameters after fitting.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// This private method can be used to apply constraints or transformations to the model parameters
    /// after they've been estimated. This might include:
    /// 
    /// - Enforcing stability conditions (ensuring predictions won't explode)
    /// - Applying regularization (preventing the model from becoming too complex)
    /// - Making final adjustments based on domain knowledge
    /// 
    /// In this implementation, it's a placeholder for potential future enhancements.
    /// </remarks>
    private void UpdateModelParameters()
    {
        // Implement any necessary parameter updates or constraints
    }

    /// <summary>
    /// Evaluates the model's performance on test data.
    /// </summary>
    /// <param name="xTest">Matrix of exogenous variables for testing.</param>
    /// <param name="yTest">Actual values for testing.</param>
    /// <returns>A dictionary of evaluation metrics.</returns>
    /// <remarks>
    /// For Beginners:
    /// This method measures how well the model performs by comparing its predictions
    /// against actual values from a test dataset.
    /// 
    /// It calculates several common error metrics:
    /// - MSE (Mean Squared Error): Average of squared differences between predictions and actual values.
    ///   Lower is better, but squares the errors, so large errors have a bigger impact.
    /// 
    /// - RMSE (Root Mean Squared Error): Square root of MSE, which gives errors in the same units as the original data.
    ///   For example, if your data is in dollars, RMSE is also in dollars.
    /// 
    /// - MAE (Mean Absolute Error): Average of absolute differences between predictions and actual values.
    ///   Easier to interpret than MSE and treats all sizes of errors equally.
    /// 
    /// - MAPE (Mean Absolute Percentage Error): Average of percentage differences between predictions and actual values.
    ///   Useful for understanding the relative size of errors compared to the actual values.
    /// 
    /// Lower values for all these metrics indicate better performance.
    /// </remarks>
    public override Dictionary<string, T> EvaluateModel(Matrix<T> xTest, Vector<T> yTest)
    {
        Vector<T> predictions = Predict(xTest);
        Dictionary<string, T> metrics = new Dictionary<string, T>
        {
            ["MSE"] = StatisticsHelper<T>.CalculateMeanSquaredError(yTest, predictions),
            ["RMSE"] = StatisticsHelper<T>.CalculateRootMeanSquaredError(yTest, predictions),
            ["MAE"] = StatisticsHelper<T>.CalculateMeanAbsoluteError(yTest, predictions),
            ["MAPE"] = StatisticsHelper<T>.CalculateMeanAbsolutePercentageError(yTest, predictions)
        };

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
    /// The method saves all essential components: AR coefficients, MA coefficients,
    /// exogenous coefficients, differencing information, intercept value, and
    /// model options. This allows the model to be fully reconstructed later.
    /// </remarks>
    protected override void SerializeCore(BinaryWriter writer)
    {
        writer.Write(_arCoefficients.Length);
        for (int i = 0; i < _arCoefficients.Length; i++)
            writer.Write(Convert.ToDouble(_arCoefficients[i]));

        writer.Write(_maCoefficients.Length);
        for (int i = 0; i < _maCoefficients.Length; i++)
            writer.Write(Convert.ToDouble(_maCoefficients[i]));

        writer.Write(_exogenousCoefficients.Length);
        for (int i = 0; i < _exogenousCoefficients.Length; i++)
            writer.Write(Convert.ToDouble(_exogenousCoefficients[i]));

        writer.Write(_differenced.Length);
        for (int i = 0; i < _differenced.Length; i++)
            writer.Write(Convert.ToDouble(_differenced[i]));

        writer.Write(Convert.ToDouble(_intercept));

        writer.Write(JsonConvert.SerializeObject(_arimaxOptions));
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
    /// The method loads all components that were saved during serialization:
    /// AR coefficients, MA coefficients, exogenous coefficients, differencing information,
    /// intercept value, and model options. This fully reconstructs the model exactly
    /// as it was when saved.
    /// </remarks>
    protected override void DeserializeCore(BinaryReader reader)
    {
        int arCoefficientsLength = reader.ReadInt32();
        _arCoefficients = new Vector<T>(arCoefficientsLength);
        for (int i = 0; i < arCoefficientsLength; i++)
            _arCoefficients[i] = NumOps.FromDouble(reader.ReadDouble());

        int maCoefficientsLength = reader.ReadInt32();
        _maCoefficients = new Vector<T>(maCoefficientsLength);
        for (int i = 0; i < maCoefficientsLength; i++)
            _maCoefficients[i] = NumOps.FromDouble(reader.ReadDouble());

        int exogenousCoefficientsLength = reader.ReadInt32();
        _exogenousCoefficients = new Vector<T>(exogenousCoefficientsLength);
        for (int i = 0; i < exogenousCoefficientsLength; i++)
            _exogenousCoefficients[i] = NumOps.FromDouble(reader.ReadDouble());

        int differencedLength = reader.ReadInt32();
        _differenced = new Vector<T>(differencedLength);
        for (int i = 0; i < differencedLength; i++)
            _differenced[i] = NumOps.FromDouble(reader.ReadDouble());

        _intercept = NumOps.FromDouble(reader.ReadDouble());

        string optionsJson = reader.ReadString();
        _arimaxOptions = JsonConvert.DeserializeObject<ARIMAXModelOptions<T>>(optionsJson) ?? new();
    }
}