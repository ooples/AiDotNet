using Newtonsoft.Json;

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

            // Apply exogenous component - vectorized with Engine.DotProduct
            var exogRow = new Vector<T>(xNew.Columns);
            for (int i = 0; i < xNew.Columns; i++)
            {
                exogRow[i] = xNew[t, i];
            }
            T exogContribution = Engine.DotProduct(exogRow, _exogenousCoefficients);
            prediction = NumOps.Add(prediction, exogContribution);

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
        // Validate inputs
        if (x == null)
            throw new ArgumentNullException(nameof(x), "Exogenous variables matrix cannot be null");

        if (y == null)
            throw new ArgumentNullException(nameof(y), "Time series vector cannot be null");

        if (x.Rows != y.Length)
            throw new ArgumentException($"Number of rows in exogenous variables matrix ({x.Rows}) must match the length of the time series vector ({y.Length})");

        if (x.Columns != _arimaxOptions.ExogenousVariables)
            throw new ArgumentException($"Number of columns in exogenous variables matrix ({x.Columns}) must match the number of exogenous variables ({_arimaxOptions.ExogenousVariables})");

        // Fit exogenous variables using the linear regression model
        Matrix<T> xT = x.Transpose();
        Matrix<T> xTx = xT * x;
        Vector<T> xTy = xT * y;

        // Add small regularization to diagonal elements to avoid singularity issues
        // This is more efficient than using try-catch for potential singularity
        T regularizationFactor = NumOps.FromDouble(1e-6);
        for (int i = 0; i < xTx.Rows; i++)
        {
            xTx[i, i] = NumOps.Add(xTx[i, i], regularizationFactor);
        }

        // Solve the linear system for exogenous coefficients
        _exogenousCoefficients = MatrixSolutionHelper.SolveLinearSystem(xTx, xTy, _arimaxOptions.DecompositionType);

        // Extract residuals - vectorized with Engine.Subtract
        Vector<T> fitted = x * _exogenousCoefficients;
        Vector<T> residuals = (Vector<T>)Engine.Subtract(y, fitted);

        // Replace any invalid values in residuals with zeros
        // This is more efficient than checking each value with try-catch
        for (int i = 0; i < residuals.Length; i++)
        {
            double val = Convert.ToDouble(residuals[i]);
            if (double.IsNaN(val) || double.IsInfinity(val))
            {
                residuals[i] = NumOps.Zero;
            }
        }

        // Fit ARMA model to residuals
        FitARMAModel(residuals);

        // Calculate intercept efficiently - vectorized with Engine.Sum
        T sum = Engine.Sum(y);
        int validCount = y.Length;

        // Check for invalid values and adjust count
        for (int i = 0; i < y.Length; i++)
        {
            double val = Convert.ToDouble(y[i]);
            if (double.IsNaN(val) || double.IsInfinity(val))
            {
                validCount--;
            }
        }

        _intercept = validCount > 0
            ? NumOps.Divide(sum, NumOps.FromDouble(validCount))
            : NumOps.Zero;
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

    /// <summary>
    /// Creates a new instance of the ARIMAX model.
    /// </summary>
    /// <returns>A new ARIMAX model with the same options as this one.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method creates a blank copy of this ARIMAX model.
    /// It's used internally by methods like DeepCopy and Clone to create a new model
    /// with the same configuration but without copying the trained parameters.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateInstance()
    {
        var arimaxOptions = (ARIMAXModelOptions<T>)Options;
        // Create a clone of the options
        var optionsClone = new ARIMAXModelOptions<T>
        {
            AROrder = arimaxOptions.AROrder,
            MAOrder = arimaxOptions.MAOrder,
            DifferenceOrder = arimaxOptions.DifferenceOrder,
            ExogenousVariables = arimaxOptions.ExogenousVariables,
            DecompositionType = arimaxOptions.DecompositionType,
            // Copy base options
            LagOrder = Options.LagOrder,
            IncludeTrend = Options.IncludeTrend,
            SeasonalPeriod = Options.SeasonalPeriod,
            AutocorrelationCorrection = Options.AutocorrelationCorrection,
            ModelType = Options.ModelType
        };

        return new ARIMAXModel<T>(optionsClone);
    }

    /// <summary>
    /// Applies the provided parameters to the model.
    /// </summary>
    /// <param name="parameters">The vector of parameters to apply.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method updates all the numerical values that determine
    /// how the model behaves. It's like replacing all the settings that control how
    /// the model makes predictions.
    /// </para>
    /// </remarks>
    protected override void ApplyParameters(Vector<T> parameters)
    {
        int index = 0;

        // Get the number of base parameters (if any)
        int baseParamCount = 0;
        base.ApplyParameters(parameters);

        // Skip base parameters
        index += baseParamCount;

        // Extract the intercept
        _intercept = parameters[index++];

        // Extract the AR coefficients
        for (int i = 0; i < _arCoefficients.Length; i++)
        {
            _arCoefficients[i] = parameters[index++];
        }

        // Extract the MA coefficients
        for (int i = 0; i < _maCoefficients.Length; i++)
        {
            _maCoefficients[i] = parameters[index++];
        }

        // Extract the exogenous coefficients
        for (int i = 0; i < _exogenousCoefficients.Length; i++)
        {
            _exogenousCoefficients[i] = parameters[index++];
        }
    }

    /// <summary>
    /// Gets metadata about the ARIMAX model.
    /// </summary>
    /// <returns>A ModelMetaData object containing information about the model.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method provides important information about the ARIMAX model
    /// that can help you understand its characteristics and behavior.
    /// 
    /// The metadata includes:
    /// - The type of model (ARIMAX)
    /// - The actual coefficients and parameters the model has learned
    /// - State information needed for predictions
    /// 
    /// This complete picture of the model is useful for analysis, debugging, and potentially
    /// transferring the model's knowledge to other systems.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var arimaxOptions = (ARIMAXModelOptions<T>)Options;

        var metadata = new ModelMetadata<T>
        {
            ModelType = ModelType.ARIMAXModel,
            AdditionalInfo = new Dictionary<string, object>
            {
                // Include the actual model state variables
                { "ARCoefficients", _arCoefficients },
                { "MACoefficients", _maCoefficients },
                { "ExogenousCoefficients", _exogenousCoefficients },
                { "Differenced", _differenced },
                { "Intercept", Convert.ToDouble(_intercept) },
            
                // Include model configuration as well
                { "AROrder", arimaxOptions.AROrder },
                { "MAOrder", arimaxOptions.MAOrder },
                { "DifferenceOrder", arimaxOptions.DifferenceOrder },
                { "ExogenousVariables", arimaxOptions.ExogenousVariables },
                { "DecompositionType", arimaxOptions.DecompositionType }
            },
            ModelData = this.Serialize()
        };

        return metadata;
    }

    /// <summary>
    /// Gets the indices of exogenous features actively used by the model.
    /// </summary>
    /// <returns>A collection of indices representing the active features.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method tells you which external factors have the strongest
    /// influence on the model's predictions.
    /// </para>
    /// </remarks>
    public override IEnumerable<int> GetActiveFeatureIndices()
    {
        // Select features with coefficients significantly different from zero
        List<int> activeIndices = new List<int>();

        // Define a threshold for "active" features
        T threshold = NumOps.FromDouble(0.01);

        // Check exogenous coefficients
        for (int i = 0; i < _exogenousCoefficients.Length; i++)
        {
            if (NumOps.GreaterThan(NumOps.Abs(_exogenousCoefficients[i]), threshold))
            {
                activeIndices.Add(i);
            }
        }

        return activeIndices;
    }

    /// <summary>
    /// Determines if a specific exogenous feature is actively used by the model.
    /// </summary>
    /// <param name="featureIndex">The index of the feature to check.</param>
    /// <returns>True if the feature is actively used; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method tells you whether a specific external factor
    /// has a meaningful influence on the model's predictions.
    /// </para>
    /// </remarks>
    public override bool IsFeatureUsed(int featureIndex)
    {
        // Check if the feature index is valid
        if (featureIndex < 0 || featureIndex >= _exogenousCoefficients.Length)
        {
            return false;
        }

        // Define a threshold for "active" features
        T threshold = NumOps.FromDouble(0.01);

        // Check if the coefficient's absolute value exceeds the threshold
        return NumOps.GreaterThan(NumOps.Abs(_exogenousCoefficients[featureIndex]), threshold);
    }

    /// <summary>
    /// Makes predictions using the trained ARIMAX model with tensor input.
    /// </summary>
    /// <param name="input">Tensor containing exogenous variables for the periods to be predicted.</param>
    /// <returns>A tensor of predicted values.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This overload allows the model to work with tensor data formats.
    /// It converts the tensor to a matrix and then uses the standard prediction method.
    /// </para>
    /// </remarks>
    public Tensor<T> Predict(Tensor<T> input)
    {
        // Convert tensor to matrix
        if (input.Rank != 2)
        {
            throw new ArgumentException("Input tensor must be 2-dimensional (equivalent to a matrix)");
        }

        var matrix = new Matrix<T>(input.Shape[0], input.Shape[1]);
        for (int i = 0; i < input.Shape[0]; i++)
        {
            for (int j = 0; j < input.Shape[1]; j++)
            {
                matrix[i, j] = input[i, j];
            }
        }

        // Use the matrix-based predict method
        var predictions = Predict(matrix);

        // Convert predictions to tensor
        var resultTensor = Tensor<T>.FromVector(predictions);
        for (int i = 0; i < predictions.Length; i++)
        {
            resultTensor[i] = predictions[i];
        }

        return resultTensor;
    }

    /// <summary>
    /// Trains the ARIMAX model with tensor input data.
    /// </summary>
    /// <param name="input">Tensor containing exogenous variables.</param>
    /// <param name="expectedOutput">Tensor containing time series values to model.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method allows the model to be trained with tensor data formats.
    /// It converts the tensors to matrix and vector formats and then uses the standard training method.
    /// </para>
    /// </remarks>
    public void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // Convert input tensor to matrix
        if (input.Rank != 2)
        {
            throw new ArgumentException("Input tensor must be 2-dimensional (equivalent to a matrix)");
        }

        var matrix = new Matrix<T>(input.Shape[0], input.Shape[1]);
        for (int i = 0; i < input.Shape[0]; i++)
        {
            for (int j = 0; j < input.Shape[1]; j++)
            {
                matrix[i, j] = input[i, j];
            }
        }

        // Convert output tensor to vector
        if (expectedOutput.Rank != 1)
        {
            throw new ArgumentException("Expected output tensor must be 1-dimensional (equivalent to a vector)");
        }

        var vector = new Vector<T>(expectedOutput.Shape[0]);
        for (int i = 0; i < expectedOutput.Shape[0]; i++)
        {
            vector[i] = expectedOutput[i];
        }

        // Use the matrix-based train method
        Train(matrix, vector);
    }

    /// <summary>
    /// Creates a deep copy of the current model.
    /// </summary>
    /// <returns>A new instance of the ARIMAX model with the same state and parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a complete copy of the model, including its configuration and trained parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates an exact duplicate of your trained model.
    /// 
    /// Unlike CreateInstance(), which creates a blank model with the same settings,
    /// Clone() creates a complete copy including:
    /// - The model configuration (AR order, MA order, etc.)
    /// - All trained coefficients (AR, MA, exogenous)
    /// - Differencing information and intercept value
    /// 
    /// This is useful for:
    /// - Creating a backup before experimenting with a model
    /// - Using the same trained model in multiple scenarios
    /// - Creating ensemble models that use variations of the same base model
    /// </para>
    /// </remarks>
    public override IFullModel<T, Matrix<T>, Vector<T>> Clone()
    {
        var clone = (ARIMAXModel<T>)CreateInstance();

        // Copy AR coefficients
        for (int i = 0; i < _arCoefficients.Length; i++)
        {
            clone._arCoefficients[i] = _arCoefficients[i];
        }

        // Copy MA coefficients
        for (int i = 0; i < _maCoefficients.Length; i++)
        {
            clone._maCoefficients[i] = _maCoefficients[i];
        }

        // Copy exogenous coefficients
        for (int i = 0; i < _exogenousCoefficients.Length; i++)
        {
            clone._exogenousCoefficients[i] = _exogenousCoefficients[i];
        }

        // Copy differenced values
        clone._differenced = new Vector<T>(_differenced.Length);
        for (int i = 0; i < _differenced.Length; i++)
        {
            clone._differenced[i] = _differenced[i];
        }

        // Copy intercept
        clone._intercept = _intercept;

        return clone;
    }

    /// <summary>
    /// Resets the model to its untrained state.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method clears all trained parameters, effectively resetting the model to its initial state.
    /// </para>
    /// <para><b>For Beginners:</b> This method erases all the learned patterns from your model.
    /// 
    /// After calling this method:
    /// - All coefficients are reset to zero
    /// - The intercept is reset to zero
    /// - The differencing information is cleared
    /// 
    /// The model behaves as if it was never trained, and you would need to train it again before
    /// making predictions. This is useful when you want to:
    /// - Experiment with different training data on the same model
    /// - Retrain a model from scratch with new parameters
    /// - Reset a model that might have been trained incorrectly
    /// </para>
    /// </remarks>
    public override void Reset()
    {
        // Reset AR coefficients
        _arCoefficients = new Vector<T>(_arimaxOptions.AROrder);

        // Reset MA coefficients
        _maCoefficients = new Vector<T>(_arimaxOptions.MAOrder);

        // Reset exogenous coefficients
        _exogenousCoefficients = new Vector<T>(_arimaxOptions.ExogenousVariables);

        // Reset differencing information
        _differenced = new Vector<T>(0);

        // Reset intercept
        _intercept = NumOps.Zero;
    }

    /// <summary>
    /// Implements the core training algorithm for the ARIMAX model.
    /// </summary>
    /// <param name="x">Matrix of exogenous variables (external factors that may influence the time series).</param>
    /// <param name="y">Vector of time series values to be modeled and predicted.</param>
    /// <remarks>
    /// <para>
    /// This method contains the implementation details of the training process, handling differencing,
    /// model fitting, and parameter updates.
    /// </para>
    /// <para><b>For Beginners:</b> This is the engine room of the training process.
    /// 
    /// While the public Train method provides a high-level interface, this method does the actual work:
    /// 1. Differences the data (if needed) to remove trends
    /// 2. Fits the ARIMAX model components (external factors, AR, and MA)
    /// 3. Updates model parameters as needed
    /// 
    /// Think of it as the detailed step-by-step recipe that the chef follows when you order a meal.
    /// </para>
    /// </remarks>
    protected override void TrainCore(Matrix<T> x, Vector<T> y)
    {
        // Step 1: Perform differencing if necessary
        Vector<T> diffY = DifferenceTimeSeries(y, _arimaxOptions.DifferenceOrder);

        // Step 2: Fit ARIMAX model
        FitARIMAXModel(x, diffY);

        // Step 3: Update model parameters
        UpdateModelParameters();
    }

    /// <summary>
    /// Predicts a single value based on a single vector of exogenous variables.
    /// </summary>
    /// <param name="input">Vector of exogenous variables for a single time point.</param>
    /// <returns>The predicted value for that time point.</returns>
    /// <remarks>
    /// <para>
    /// This method provides a convenient way to get a prediction for a single time point without
    /// having to create a matrix with a single row.
    /// </para>
    /// <para><b>For Beginners:</b> This is a shortcut for getting just one prediction.
    /// 
    /// Instead of providing a table of inputs for multiple time periods, you can provide
    /// just one set of external factors and get back a single prediction.
    /// 
    /// For example, if you want to predict tomorrow's sales based on tomorrow's weather, 
    /// promotions, and other factors, this method lets you do that directly.
    /// 
    /// Under the hood, it:
    /// 1. Takes your single set of external factors
    /// 2. Creates a small table with just one row
    /// 3. Gets a prediction using the main prediction engine
    /// 4. Returns that single prediction to you
    /// </para>
    /// </remarks>
    public override T PredictSingle(Vector<T> input)
    {
        // Validate input dimensions
        if (input.Length != _exogenousCoefficients.Length)
        {
            throw new ArgumentException(
                $"Input vector length ({input.Length}) must match the number of exogenous variables ({_exogenousCoefficients.Length}).",
                nameof(input));
        }

        // Create a matrix with a single row
        Matrix<T> singleRowMatrix = new Matrix<T>(1, input.Length);
        for (int i = 0; i < input.Length; i++)
        {
            singleRowMatrix[0, i] = input[i];
        }

        // Use the existing Predict method
        Vector<T> predictions = Predict(singleRowMatrix);

        // Return the single prediction
        return predictions[0];
    }
}
