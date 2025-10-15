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
    private ARIMAOptions<T> _arimaOptions = default!;

    /// <summary>
    /// Coefficients for the autoregressive (AR) component of the model.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// These coefficients determine how much each past value influences the prediction.
    /// For example, if the coefficient for yesterday's value is 0.7, it means yesterday's
    /// value has a strong influence on today's prediction.
    /// </remarks>
    private Vector<T> _arCoefficients = default!;

    /// <summary>
    /// Coefficients for the moving average (MA) component of the model.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// These coefficients determine how much each past prediction error influences the prediction.
    /// They help the model learn from its mistakes. For example, if the model consistently
    /// underpredicts, these coefficients help correct that bias.
    /// </remarks>
    private Vector<T> _maCoefficients = default!;

    /// <summary>
    /// The constant term in the ARIMA equation.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// This is like the "baseline" value in the prediction, before considering the effects
    /// of past values and errors. It's similar to the y-intercept in a linear equation.
    /// </remarks>
    private T _constant = default!;

    /// <summary>
    /// Last values of original time series before differencing, used for undifferencing predictions.
    /// </summary>
    private Vector<T> _originalLastValues = default!;

    /// <summary>
    /// Last values of the differenced series, used for initializing predictions.
    /// </summary>
    private Vector<T> _lastDiffValues = default!;

    /// <summary>
    /// Last residuals from the AR process, used for initializing MA component.
    /// </summary>
    private Vector<T> _lastResiduals = default!;

    /// <summary>
    /// Coefficients for exogenous variables in the model.
    /// </summary>
    private Vector<T> _exogCoefficients = default!;

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
        // Ensure LagOrder is at least equal to P for AR components
        _arimaOptions.LagOrder = Math.Max(_arimaOptions.LagOrder, _arimaOptions.P);
        _constant = NumOps.Zero;
        _arCoefficients = Vector<T>.Empty();
        _maCoefficients = Vector<T>.Empty();
        _originalLastValues = Vector<T>.Empty(); // Store historical values for undifferencing
        _lastDiffValues = Vector<T>.Empty();
        _lastResiduals = Vector<T>.Empty();
        _exogCoefficients = Vector<T>.Empty();
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
        // Validate model is trained
        if (_arCoefficients.Length == 0 || _maCoefficients.Length == 0)
        {
            throw new InvalidOperationException("Model must be trained before making predictions.");
        }

        // Validate input
        if (input.Rows == 0)
        {
            return new Vector<T>(0);
        }

        Vector<T> predictions = new(input.Rows);

        // Initialize with proper historical values from training
        Vector<T> lastObservedValues = _lastDiffValues != null && _lastDiffValues.Length > 0
            ? new Vector<T>(_lastDiffValues)
            : new Vector<T>(_arimaOptions.P);

        Vector<T> lastErrors = _lastResiduals != null && _lastResiduals.Length > 0
            ? new Vector<T>(_lastResiduals)
            : new Vector<T>(_arimaOptions.Q);

        // If we don't have stored values and input has data, try to use it
        if ((_lastDiffValues == null || _lastDiffValues.Length == 0) && input.Columns > 0 && input.Rows > 0)
        {
            // Initialize with actual values from input if available (first column is assumed to be the target series)
            for (int j = 0; j < Math.Min(_arimaOptions.P, input.Rows); j++)
            {
                if (j < input.Rows)
                {
                    lastObservedValues[j] = input[input.Rows - 1 - j, 0];
                }
            }
        }

        // Initialize lastErrors with zeros if not already set
        for (int j = 0; j < lastErrors.Length; j++)
        {
            if (j >= (_lastResiduals?.Length ?? 0))
            {
                lastErrors[j] = NumOps.Zero;
            }
        }

        for (int i = 0; i < predictions.Length; i++)
        {
            T prediction = _constant;

            // Add AR component - make sure we don't exceed array bounds
            for (int j = 0; j < Math.Min(_arCoefficients.Length, lastObservedValues.Length); j++)
            {
                prediction = NumOps.Add(prediction, NumOps.Multiply(_arCoefficients[j], lastObservedValues[j]));
            }

            // Add MA component - make sure we don't exceed array bounds
            for (int j = 0; j < Math.Min(_maCoefficients.Length, lastErrors.Length); j++)
            {
                prediction = NumOps.Add(prediction, NumOps.Multiply(_maCoefficients[j], lastErrors[j]));
            }

            // Add exogenous variables effect if present (columns after the first one)
            if (input.Columns > 1 && _exogCoefficients != null && _exogCoefficients.Length > 0)
            {
                for (int j = 1; j < input.Columns && (j - 1) < _exogCoefficients.Length; j++)
                {
                    prediction = NumOps.Add(prediction, NumOps.Multiply(_exogCoefficients[j - 1], input[i, j]));
                }
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

            // For forecasting we assume zero error
            lastErrors[0] = NumOps.Zero;
        }

        // Undifference the predictions if needed
        if (_arimaOptions.D > 0 && _originalLastValues != null && _originalLastValues.Length > 0)
        {
            predictions = UndifferenceSeries(predictions, _originalLastValues, _arimaOptions.D);
        }

        return predictions;
    }

    /// <summary>
    /// Helper method to undifference a series based on original values.
    /// </summary>
    private Vector<T> UndifferenceSeries(Vector<T> diffSeries, Vector<T> originalLastValues, int d)
    {
        if (d <= 0 || diffSeries.Length == 0 || originalLastValues.Length < d)
        {
            return diffSeries;
        }

        Vector<T> result = new Vector<T>(diffSeries.Length);

        // For first-order differencing
        if (d == 1)
        {
            T lastValue = originalLastValues[0];
            for (int i = 0; i < diffSeries.Length; i++)
            {
                result[i] = NumOps.Add(lastValue, diffSeries[i]);
                lastValue = result[i];
            }
        }
        // For higher-order differencing, apply undifferencing recursively
        else
        {
            // First undifference to d-1 order
            Vector<T> undiffD1 = UndifferenceSeries(diffSeries, new Vector<T>(originalLastValues.Skip(1)), d - 1);
            // Then apply first-order undifferencing
            Vector<T> lastValues = new Vector<T>(1) { [0] = originalLastValues[0] };
            result = UndifferenceSeries(undiffD1, lastValues, 1);
        }

        return result;
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

        // Write exogenous coefficients
        writer.Write(_exogCoefficients != null ? _exogCoefficients.Length : 0);
        if (_exogCoefficients != null)
        {
            for (int i = 0; i < _exogCoefficients.Length; i++)
            {
                writer.Write(Convert.ToDouble(_exogCoefficients[i]));
            }
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

        // Read exogenous coefficients
        int exogLength = reader.ReadInt32();
        _exogCoefficients = new Vector<T>(exogLength);
        for (int i = 0; i < exogLength; i++)
        {
            _exogCoefficients[i] = NumOps.FromDouble(reader.ReadDouble());
        }
    }

    /// <summary>
    /// Core implementation of the training logic for the ARIMA model.
    /// </summary>
    /// <param name="x">Feature matrix (typically just time indices for ARIMA models).</param>
    /// <param name="y">Target vector (the time series values to be modeled).</param>
    /// <remarks>
    /// For Beginners:
    /// This method contains the core implementation of the training process. It:
    /// 
    /// 1. Differences the data to remove trends (the "I" in ARIMA)
    /// 2. Estimates the AR coefficients that capture how past values affect future values
    /// 3. Calculates residuals and uses them to estimate the MA coefficients
    /// 4. Estimates the constant term that serves as the baseline prediction
    /// 
    /// This implementation follows the same process as the public Train method but
    /// provides the actual mechanism that fits the model to your data.
    /// </remarks>
    protected override void TrainCore(Matrix<T> x, Vector<T> y)
    {
        int p = _arimaOptions.P; // AR order
        int d = _arimaOptions.D; // Differencing order
        int q = _arimaOptions.Q; // MA order
        int numExogVars = x.Columns > 1 ? x.Columns - 1 : 0; // Assuming first column is time and rest are exogenous

        // Store last values of original series for later undifferencing
        _originalLastValues = new Vector<T>(d);
        for (int i = 0; i < d && i < y.Length; i++)
        {
            _originalLastValues[i] = y[y.Length - 1 - i];
        }

        // Step 1: Difference the series to make it stationary
        Vector<T> diffY = TimeSeriesHelper<T>.DifferenceSeries(y, d);

        // Step 2: If we have exogenous variables, prepare them
        Matrix<T> exogMatrix = Matrix<T>.Empty();
        if (numExogVars > 0)
        {
            // Extract exogenous variables (columns after the first one)
            exogMatrix = new Matrix<T>(x.Rows, numExogVars);
            for (int i = 0; i < x.Rows; i++)
            {
                for (int j = 0; j < numExogVars; j++)
                {
                    exogMatrix[i, j] = x[i, j + 1]; // Skip the first column (time)
                }
            }

            // Difference exogenous variables if differencing is applied to target
            if (d > 0)
            {
                Matrix<T> diffExog = new Matrix<T>(diffY.Length, numExogVars);
                for (int j = 0; j < numExogVars; j++)
                {
                    Vector<T> column = new Vector<T>(x.Rows);
                    for (int i = 0; i < x.Rows; i++)
                    {
                        column[i] = x[i, j + 1];
                    }

                    Vector<T> diffColumn = TimeSeriesHelper<T>.DifferenceSeries(column, d);
                    for (int i = 0; i < diffY.Length; i++)
                    {
                        diffExog[i, j] = diffColumn[i];
                    }
                }

                exogMatrix = diffExog;
            }
        }

        // Step 3: Estimate AR coefficients first
        _arCoefficients = TimeSeriesHelper<T>.EstimateARCoefficients(diffY, p, MatrixDecompositionType.Qr);

        // Step 4: Now estimate exogenous variable coefficients
        if (numExogVars > 0)
        {
            // Calculate AR predictions
            Vector<T> arPredictions = new Vector<T>(diffY.Length - p);
            for (int i = p; i < diffY.Length; i++)
            {
                T predicted = NumOps.Zero;
                for (int j = 0; j < p; j++)
                {
                    predicted = NumOps.Add(predicted, NumOps.Multiply(_arCoefficients[j], diffY[i - j - 1]));
                }
                arPredictions[i - p] = predicted;
            }

            // Calculate residuals (target - AR predictions)
            Vector<T> arResiduals = new Vector<T>(diffY.Length - p);
            for (int i = 0; i < arResiduals.Length; i++)
            {
                arResiduals[i] = NumOps.Subtract(diffY[i + p], arPredictions[i]);
            }

            // Create exogenous design matrix for the same period
            Matrix<T> exogDesign = new Matrix<T>(diffY.Length - p, numExogVars);
            for (int i = 0; i < exogDesign.Rows; i++)
            {
                for (int j = 0; j < numExogVars; j++)
                {
                    exogDesign[i, j] = exogMatrix[i + p, j];
                }
            }

            // Estimate exogenous coefficients using linear regression on residuals
            _exogCoefficients = MatrixSolutionHelper.SolveLinearSystem<T>(exogDesign, arResiduals, MatrixDecompositionType.Qr);
        }
        else
        {
            _exogCoefficients = new Vector<T>(0);
        }

        // Step 5: Calculate complete residuals (including exog effects) for MA
        Vector<T> fullResiduals;
        if (numExogVars > 0 && _exogCoefficients.Length > 0)
        {
            fullResiduals = new Vector<T>(diffY.Length - p);
            for (int i = p; i < diffY.Length; i++)
            {
                T predicted = NumOps.Zero;

                // Add AR component
                for (int j = 0; j < p; j++)
                {
                    predicted = NumOps.Add(predicted, NumOps.Multiply(_arCoefficients[j], diffY[i - j - 1]));
                }

                // Add exogenous component
                for (int j = 0; j < numExogVars; j++)
                {
                    predicted = NumOps.Add(predicted, NumOps.Multiply(_exogCoefficients[j], exogMatrix[i, j]));
                }

                fullResiduals[i - p] = NumOps.Subtract(diffY[i], predicted);
            }
        }
        else
        {
            fullResiduals = TimeSeriesHelper<T>.CalculateARResiduals(diffY, _arCoefficients);
        }

        // Step 6: Estimate MA coefficients using the full residuals
        _maCoefficients = TimeSeriesHelper<T>.EstimateMACoefficients(fullResiduals, q);

        // Step 7: Estimate constant term
        _constant = EstimateConstant(diffY, _arCoefficients, _maCoefficients);

        // Store last values for initialization in Predict
        _lastDiffValues = new Vector<T>(p);
        for (int i = 0; i < p && i < diffY.Length; i++)
        {
            _lastDiffValues[i] = diffY[diffY.Length - 1 - i];
        }

        _lastResiduals = new Vector<T>(q);
        for (int i = 0; i < q && i < fullResiduals.Length; i++)
        {
            _lastResiduals[i] = fullResiduals[fullResiduals.Length - 1 - i];
        }
    }

    /// <summary>
    /// Predicts a single value based on the input vector.
    /// </summary>
    /// <param name="input">Input vector containing features for prediction.</param>
    /// <returns>The predicted value.</returns>
    /// <remarks>
    /// For Beginners:
    /// This method generates a single prediction based on your input data.
    /// 
    /// The prediction process:
    /// 1. Starts with the constant term as the baseline value
    /// 2. Adds the influence of past observations (AR component)
    /// 3. Adds the influence of past prediction errors (MA component)
    /// 
    /// This is useful when you need just one prediction rather than a whole series.
    /// For example, if you want to predict tomorrow's temperature specifically,
    /// rather than temperatures for the next week.
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
    /// Gets metadata about the model, including its type, parameters, and configuration.
    /// </summary>
    /// <returns>A ModelMetadata object containing information about the model.</returns>
    /// <remarks>
    /// For Beginners:
    /// This method provides a summary of your model's settings and what it has learned.
    /// 
    /// The metadata includes:
    /// - The type of model (ARIMA)
    /// - The p, d, and q parameters that define the model structure
    /// - The AR and MA coefficients that were learned during training
    /// - The constant term that serves as the baseline prediction
    /// 
    /// This information is useful for:
    /// - Documenting your model for future reference
    /// - Comparing different models to see which performs best
    /// - Understanding what patterns the model has identified in your data
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
    /// For Beginners:
    /// This method creates a fresh copy of the model with the same settings.
    /// 
    /// The new copy:
    /// - Has the same p, d, and q parameters as the original model
    /// - Has the same configuration options
    /// - Is untrained (doesn't have coefficients yet)
    /// 
    /// This is useful when you want to:
    /// - Train multiple versions of the same model on different data
    /// - Create ensemble models that combine predictions from multiple similar models
    /// - Reset a model to start fresh while keeping the same structure
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateInstance()
    {
        // Create a new instance with the same options
        return new ARIMAModel<T>(_arimaOptions);
    }
}