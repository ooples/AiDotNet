namespace AiDotNet.TimeSeries;

/// <summary>
/// Implements an AR (AutoRegressive) model for time series forecasting.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double, decimal).</typeparam>
/// <remarks>
/// <para>
/// The AR model is a time series forecasting method that uses the relationship between 
/// an observation and a number of lagged observations to predict future values.
/// </para>
/// 
/// <para>
/// For Beginners:
/// The AR (AutoRegressive) model is one of the simplest and most intuitive time series
/// forecasting methods. It's similar to how we naturally predict things in everyday life.
/// 
/// Think of it this way: If you want to guess tomorrow's temperature, you might look at today's
/// temperature. If it's hot today, it's likely to be hot tomorrow. That's essentially what an
/// AR model does - it uses past values to predict future values.
/// 
/// For example, if you want to predict today's stock price, an AR model might look at the 
/// prices from the last few days. If the stock has been trending upward, the model will likely
/// predict that it continues to rise.
/// 
/// The key parameter is the "AR order" (p), which determines how many past values to consider.
/// For example:
/// - AR(1): Only looks at the previous value (yesterday to predict today)
/// - AR(2): Looks at the previous two values (yesterday and the day before to predict today)
/// - AR(7): Looks at values from the past week to make predictions
/// 
/// Unlike more complex models like ARMA or ARIMA, the AR model only contains the autoregressive
/// component and doesn't account for moving average errors or trends that require differencing.
/// </para>
/// </remarks>
public class ARModel<T> : TimeSeriesModelBase<T>
{
    /// <summary>
    /// Coefficients for the autoregressive (AR) component of the model.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// These coefficients determine how much each past value influences the prediction.
    /// 
    /// For example, if you have an AR(2) model (looking at the past 2 values):
    /// - A coefficient of 0.7 for the first lag means yesterday's value has a strong influence
    /// - A coefficient of 0.2 for the second lag means the day before yesterday has a weaker influence
    /// 
    /// Larger coefficients mean stronger influence from that time period.
    /// These values are learned during training to best fit your historical data.
    /// </remarks>
    private Vector<T> _arCoefficients;

    /// <summary>
    /// The time series values from training, used for in-sample predictions via Predict(Matrix).
    /// </summary>
    private Vector<T> _trainedSeries;

    /// <summary>
    /// The number of past observations to consider (the AR order).
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// This determines how far back in time the model looks when considering past values.
    /// 
    /// For example:
    /// - AR order = 1: Only considers yesterday's value to predict today
    /// - AR order = 7: Considers values from the past week to predict today
    /// 
    /// Choosing the right AR order is important:
    /// - Too small: The model might miss important patterns
    /// - Too large: The model might overfit or become unnecessarily complex
    /// 
    /// The appropriate order often depends on the natural cycle of your data
    /// (daily, weekly, monthly, etc.).
    /// </remarks>
    private int _arOrder;

    /// <summary>
    /// The step size for gradient descent during training.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// The learning rate controls how quickly the model updates its coefficients during training.
    /// 
    /// - A small learning rate means the model takes small, cautious steps, which may
    ///   take longer to train but is less likely to overshoot the optimal solution
    /// - A large learning rate means the model takes larger steps, which may train faster
    ///   but risks overshooting and not finding the best solution
    /// 
    /// Think of it like adjusting a shower's temperature: a small learning rate is like
    /// making tiny adjustments to get exactly the right temperature, while a large learning
    /// rate is like making big adjustments that might go from too cold to too hot.
    /// </remarks>
    private readonly double _learningRate;

    /// <summary>
    /// The maximum number of iterations for the training algorithm.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// This sets a limit on how many attempts the model makes to improve its coefficients.
    /// 
    /// If the model hasn't converged (found a stable solution) after this many iterations,
    /// it will stop anyway. This prevents the training from running indefinitely.
    /// 
    /// Higher values give the model more chances to improve but increase training time.
    /// </remarks>
    private readonly int _maxIterations;

    /// <summary>
    /// The convergence threshold for training.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// This determines how small the changes to the model coefficients need to be
    /// before training stops.
    /// 
    /// When adjustments to coefficients become smaller than this value, the model
    /// is considered "good enough" and training finishes.
    /// 
    /// Smaller values may lead to more precise models but require more training time.
    /// </remarks>
    private readonly double _tolerance;

    /// <summary>
    /// Creates a new AR model with the specified options.
    /// </summary>
    /// <param name="options">Options for the AR model, including AR order and training parameters.</param>
    /// <remarks>
    /// For Beginners:
    /// This constructor creates a new AR model. You need to provide options to customize the model:
    /// 
    /// - AROrder: How many past values to consider (like using yesterday and the day before to predict today)
    /// - LearningRate: How quickly the model adjusts during training
    /// - MaxIterations: The maximum number of training attempts
    /// - Tolerance: How precise the model needs to be before training stops
    /// 
    /// Choosing good values for these options depends on your specific data and requirements.
    /// </remarks>
    public ARModel(ARModelOptions<T> options) : base(options)
    {
        _arOrder = options.AROrder;
        _learningRate = options.LearningRate;
        _maxIterations = options.MaxIterations;
        _tolerance = options.Tolerance;
        _arCoefficients = Vector<T>.Empty();
        _trainedSeries = Vector<T>.Empty();
    }

    /// <summary>
    /// Calculates the residuals (prediction errors) for the current model.
    /// </summary>
    /// <param name="y">The time series values.</param>
    /// <returns>A vector of residuals.</returns>
    /// <remarks>
    /// For Beginners:
    /// This private method calculates how far off the model's predictions are from the actual values.
    /// These differences (called "residuals") are used for two purposes:
    /// 
    /// 1. To evaluate how well the current model is performing
    /// 2. To improve the model by adjusting its coefficients
    /// 
    /// For each time point, the residual is calculated as:
    ///    residual = actual value - predicted value
    /// 
    /// A positive residual means the model underpredicted (predicted too low).
    /// A negative residual means the model overpredicted (predicted too high).
    /// 
    /// Ideally, residuals should be small and should not follow any pattern.
    /// </remarks>
    private Vector<T> CalculateResiduals(Vector<T> y)
    {
        Vector<T> residuals = new Vector<T>(y.Length);
        for (int t = _arOrder; t < y.Length; t++)
        {
            T yHat = Predict(y, t);
            residuals[t] = NumOps.Subtract(y[t], yHat);
        }

        return residuals;
    }

    /// <summary>
    /// Calculates the gradients for adjusting the AR coefficients.
    /// </summary>
    /// <param name="y">The time series values.</param>
    /// <param name="residuals">The current residuals (prediction errors).</param>
    /// <returns>Gradients for the AR coefficients.</returns>
    /// <remarks>
    /// For Beginners:
    /// This private method calculates how to adjust the model's coefficients to improve predictions.
    /// 
    /// The "gradient" is the direction and amount by which to change each coefficient to
    /// reduce prediction errors. Think of it like a compass that points in the direction
    /// that will improve the model the most.
    /// 
    /// For each coefficient:
    /// - A positive gradient means decreasing the coefficient will improve the model
    /// - A negative gradient means increasing the coefficient will improve the model
    /// - A gradient near zero means the coefficient is already close to optimal
    /// 
    /// The method returns a set of gradients for adjusting coefficients that determine 
    /// how past values affect predictions.
    /// </remarks>
    private Vector<T> CalculateGradients(Vector<T> y, Vector<T> residuals)
    {
        Vector<T> gradAR = new Vector<T>(_arOrder);

        // VECTORIZED: Process lagged y values as vectors for each timestep
        for (int t = _arOrder; t < y.Length; t++)
        {
            var laggedY = new Vector<T>(_arOrder);
            for (int i = 0; i < _arOrder; i++)
            {
                laggedY[i] = y[t - i - 1];
            }

            var contribution = (Vector<T>)Engine.Multiply(laggedY, residuals[t]);
            gradAR = (Vector<T>)Engine.Add(gradAR, contribution);
        }

        return gradAR;
    }

    /// <summary>
    /// Checks if the training process has converged (reached a stable solution).
    /// </summary>
    /// <param name="gradAR">Current AR gradients.</param>
    /// <param name="prevGradAR">Previous AR gradients.</param>
    /// <returns>True if the model has converged, false otherwise.</returns>
    /// <remarks>
    /// For Beginners:
    /// This private method determines when to stop training the model.
    /// 
    /// Training stops when the adjustments to the model become very small,
    /// indicating that further training would only make minimal improvements.
    /// 
    /// The method compares current gradients to previous gradients to see
    /// how much they've changed. When the change is smaller than the tolerance
    /// value, training is considered complete.
    /// 
    /// This is like adjusting a microscope's focus - eventually, further adjustments
    /// make such a small difference that it's not worth continuing.
    /// </remarks>
    private bool CheckConvergence(Vector<T> gradAR, Vector<T> prevGradAR)
    {
        var diffARVec = (Vector<T>)Engine.Subtract(gradAR, prevGradAR);
        T diffAR = diffARVec.Norm();
        return NumOps.LessThan(diffAR, NumOps.FromDouble(_tolerance));
    }

    /// <summary>
    /// Makes predictions using the trained AR model.
    /// </summary>
    /// <param name="input">Input matrix for prediction (typically just time indices for future periods).</param>
    /// <returns>A vector of predicted values.</returns>
    /// <remarks>
    /// For Beginners:
    /// This method uses the trained AR model to forecast future values.
    /// 
    /// For each time point in the input, it calls the prediction helper method to generate
    /// a forecast based on the AR component (effect of past values).
    /// 
    /// The result is a vector of predicted values, one for each time point in the input.
    /// 
    /// Note: For pure time series forecasting, the input parameter might just indicate
    /// how many future periods to predict, and the actual values used will be from the
    /// training data or previous predictions.
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> input)
    {
        // Use the stored training series for in-sample predictions.
        // The matrix rows indicate how many predictions to make.
        if (_trainedSeries.Length == 0)
        {
            throw new InvalidOperationException(
                "Model has not been trained. Call Train() before Predict().");
        }

        var series = _trainedSeries;
        int horizon = input.Rows;
        Vector<T> predictions = new Vector<T>(horizon);
        for (int t = 0; t < horizon; t++)
        {
            if (t < series.Length)
            {
                predictions[t] = Predict(series, t);
            }
            else
            {
                // Out-of-sample: predict using available history including prior predictions
                var extended = new Vector<T>(t + 1);
                for (int j = 0; j < series.Length; j++)
                    extended[j] = series[j];
                for (int j = series.Length; j < t; j++)
                    extended[j] = predictions[j];
                predictions[t] = Predict(extended, t);
            }
        }

        return predictions;
    }

    /// <summary>
    /// Helper method that predicts a single value at a specific time point.
    /// </summary>
    /// <param name="y">Vector of time series values.</param>
    /// <param name="t">The time index to predict.</param>
    /// <returns>The predicted value.</returns>
    /// <remarks>
    /// For Beginners:
    /// This private method calculates a single prediction for a specific time point.
    /// 
    /// The prediction uses the AR component, which is the effect of past values.
    /// For example, if yesterday's temperature was high, today's might also be high.
    /// 
    /// The prediction is calculated as:
    /// prediction = (coefficient1 × value1) + (coefficient2 × value2) + ... + (coefficientp × valuep)
    /// 
    /// Where:
    /// - coefficientn is the importance of each past value
    /// - valuen is the actual value at that past time point
    /// 
    /// The method handles cases where we don't have enough history (e.g., at the beginning
    /// of the series) by only using the available information.
    /// </remarks>
    private T Predict(Vector<T> y, int t)
    {
        // VECTORIZED: Use dot product for AR prediction
        int availableHistory = Math.Min(_arOrder, t);
        if (availableHistory == 0)
        {
            return NumOps.Zero;
        }

        // Build vector of past values
        T[] pastValues = new T[availableHistory];
        for (int i = 0; i < availableHistory; i++)
        {
            pastValues[i] = y[t - i - 1];
        }

        Vector<T> pastVector = new Vector<T>(pastValues);
        Vector<T> coeffSlice = availableHistory < _arOrder
            ? _arCoefficients.Slice(0, availableHistory)
            : _arCoefficients;

        return Engine.DotProduct(coeffSlice, pastVector);
    }

    /// <summary>
    /// Evaluates the model's performance on test data.
    /// </summary>
    /// <param name="xTest">Feature matrix for testing.</param>
    /// <param name="yTest">Actual target values for testing.</param>
    /// <returns>A dictionary of evaluation metrics (MSE, RMSE, MAE, MAPE).</returns>
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
    /// This protected method saves the model's internal state to a file or stream.
    /// 
    /// Serialization allows you to:
    /// 1. Save a trained model to disk
    /// 2. Load it later without having to retrain
    /// 3. Share the model with others
    /// 
    /// The method saves the essential components of the model:
    /// - The AR order (how many coefficients)
    /// - The AR coefficients (how past values affect predictions)
    /// 
    /// This allows the model to be fully reconstructed later.
    /// </remarks>
    protected override void SerializeCore(BinaryWriter writer)
    {
        writer.Write(_arOrder);
        for (int i = 0; i < _arOrder; i++)
        {
            writer.Write(Convert.ToDouble(_arCoefficients[i]));
        }

        // Serialize training series for in-sample prediction support
        writer.Write(_trainedSeries.Length);
        for (int i = 0; i < _trainedSeries.Length; i++)
        {
            writer.Write(Convert.ToDouble(_trainedSeries[i]));
        }
    }

    /// <summary>
    /// Deserializes the model's state from a binary stream.
    /// </summary>
    /// <param name="reader">The binary reader to read from.</param>
    /// <remarks>
    /// For Beginners:
    /// This protected method loads a previously saved model from a file or stream.
    /// 
    /// Deserialization allows you to:
    /// 1. Load a previously trained model
    /// 2. Use it immediately without retraining
    /// 3. Apply the exact same model to new data
    /// 
    /// The method loads:
    /// - The AR order (how many coefficients)
    /// - The AR coefficients (how past values affect predictions)
    /// 
    /// After deserialization, the model is ready to make predictions as if it had just been trained.
    /// </remarks>
    protected override void DeserializeCore(BinaryReader reader)
    {
        _arOrder = reader.ReadInt32();
        _arCoefficients = new Vector<T>(_arOrder);
        for (int i = 0; i < _arOrder; i++)
        {
            _arCoefficients[i] = NumOps.FromDouble(reader.ReadDouble());
        }

        // Deserialize training series if available (backward-compatible)
        _trainedSeries = Vector<T>.Empty();
        try
        {
            int seriesLength = reader.ReadInt32();
            if (seriesLength > 0)
            {
                _trainedSeries = new Vector<T>(seriesLength);
                for (int i = 0; i < seriesLength; i++)
                {
                    _trainedSeries[i] = NumOps.FromDouble(reader.ReadDouble());
                }
            }
        }
        catch (EndOfStreamException)
        {
            // Older serialized models don't include training series — leave empty
        }
    }

    /// <summary>
    /// Creates a new instance of the AR model with the same options.
    /// </summary>
    /// <returns>A new instance of the AR model.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new, uninitialized instance of the AR model with the same configuration options.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a fresh copy of the model with the same settings.
    /// 
    /// Think of this like creating a new blank notebook with the same paper quality, size, and number of pages
    /// as another notebook, but without copying any of the written content.
    /// 
    /// This is used internally by the framework to create new model instances when needed,
    /// such as when cloning a model or creating ensemble models.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateInstance()
    {
        return new ARModel<T>((ARModelOptions<T>)Options);
    }

    /// <summary>
    /// Gets metadata about the trained model, including its type, coefficients, and configuration.
    /// </summary>
    /// <returns>A ModelMetaData object containing information about the model.</returns>
    /// <remarks>
    /// <para>
    /// This method provides comprehensive information about the model, including its type, parameters, coefficients,
    /// and serialized state. This metadata can be used for model inspection, selection, or persistence.
    /// </para>
    /// <para><b>For Beginners:</b> This method returns a complete description of your trained model.
    /// 
    /// The metadata includes:
    /// - The type of model (AR in this case)
    /// - The trained coefficients that determine predictions
    /// - The configuration options you specified when creating the model
    /// - A serialized version of the entire model that can be saved
    /// 
    /// This is useful for:
    /// - Comparing different models to choose the best one
    /// - Documenting what model was used for a particular analysis
    /// - Saving model details for future reference
    /// 
    /// Think of it like creating a detailed ID card for your model that contains
    /// all the important information about how it works and was configured.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var arOptions = (ARModelOptions<T>)Options;
        var metadata = new ModelMetadata<T>
        {
            ModelType = ModelType.ARModel,
            AdditionalInfo = new Dictionary<string, object>
            {
                // Include the actual model state variables
                { "ARCoefficients", _arCoefficients },
            
                // Include model configuration as well
                { "AROrder", arOptions.AROrder },
                { "LearningRate", arOptions.LearningRate },
                { "MaxIterations", arOptions.MaxIterations },
                { "Tolerance", arOptions.Tolerance }
            },
            ModelData = this.Serialize()
        };
        return metadata;
    }

    /// <summary>
    /// Predicts future values based on a history of time series data.
    /// </summary>
    /// <param name="history">The historical time series data.</param>
    /// <param name="horizon">The number of future periods to predict.</param>
    /// <returns>A vector of predicted values for future periods.</returns>
    /// <remarks>
    /// <para>
    /// This method generates forecasts for future time periods based on the trained model and historical data.
    /// It uses a rolling prediction approach where each prediction becomes part of the history for subsequent predictions.
    /// </para>
    /// <para><b>For Beginners:</b> This method forecasts future values based on past data.
    /// 
    /// Given:
    /// - A series of past observations (the "history")
    /// - The number of future periods to predict (the "horizon")
    /// 
    /// This method:
    /// 1. Uses the model to predict the next value after the history
    /// 2. Adds that prediction to the extended history
    /// 3. Uses the extended history to predict the next value
    /// 4. Repeats until it has made predictions for the entire horizon
    /// 
    /// For example, if you have sales data for Jan-Jun and want to predict Jul-Sep (horizon=3),
    /// this method will predict Jul, then use Jan-Jul to predict Aug, then use Jan-Aug to predict Sep.
    /// 
    /// This approach is common in time series forecasting as it allows each prediction to build on previous predictions.
    /// </para>
    /// </remarks>
    public override Vector<T> Forecast(Vector<T> history, int horizon)
    {
        if (horizon <= 0)
        {
            throw new ArgumentException("Forecast horizon must be greater than zero.", nameof(horizon));
        }

        if (_arCoefficients.Length == 0)
        {
            throw new InvalidOperationException("Model must be trained before making forecasts.");
        }

        // Create an extended history that will include predictions
        Vector<T> extendedHistory = new Vector<T>(history.Length + horizon);
        for (int i = 0; i < history.Length; i++)
        {
            extendedHistory[i] = history[i];
        }

        // Generate predictions one by one
        for (int t = history.Length; t < extendedHistory.Length; t++)
        {
            extendedHistory[t] = Predict(extendedHistory, t);
        }

        // Extract just the forecasted values
        Vector<T> forecast = new Vector<T>(horizon);
        for (int i = 0; i < horizon; i++)
        {
            forecast[i] = extendedHistory[history.Length + i];
        }

        return forecast;
    }

    /// <summary>
    /// Returns a string representation of the AR model.
    /// </summary>
    /// <returns>A string describing the model and its parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method provides a human-readable description of the AR model, including its order and coefficients.
    /// </para>
    /// <para><b>For Beginners:</b> This method gives you a text description of your model.
    /// 
    /// The returned string includes:
    /// - The type of model (AR)
    /// - The AR order (how many past values the model uses)
    /// - The actual coefficient values (if the model has been trained)
    /// 
    /// This is useful for printing or logging model details, or for quickly seeing
    /// the structure of the model without examining all its properties individually.
    /// </para>
    /// </remarks>
    public override string ToString()
    {
        StringBuilder sb = new StringBuilder();
        sb.AppendLine($"AR({_arOrder}) Model");

        if (_arCoefficients.Length > 0)
        {
            sb.AppendLine("AR Coefficients:");
            for (int i = 0; i < _arOrder; i++)
            {
                sb.AppendLine($"  AR[{i + 1}] = {Convert.ToDouble(_arCoefficients[i]):F4}");
            }
        }

        return sb.ToString();
    }

    /// <summary>
    /// Resets the model to its untrained state.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method clears the trained coefficients, effectively resetting the model to its initial state.
    /// </para>
    /// <para><b>For Beginners:</b> This method erases all the learned patterns from your model.
    /// 
    /// After calling this method:
    /// - All coefficients are cleared
    /// - The model behaves as if it was never trained
    /// - You would need to train it again before making predictions
    /// 
    /// This is useful when you want to:
    /// - Experiment with different training data on the same model
    /// - Retrain a model from scratch with new parameters
    /// - Reset a model that might have been trained incorrectly
    /// 
    /// Think of it like erasing a whiteboard so you can start fresh with new calculations.
    /// </para>
    /// </remarks>
    public override void Reset()
    {
        _arCoefficients = Vector<T>.Empty();
        _trainedSeries = Vector<T>.Empty();
    }

    /// <summary>
    /// Creates a deep copy of the current model.
    /// </summary>
    /// <returns>A new instance of the AR model with the same state and parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a complete copy of the model, including its configuration and trained coefficients.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates an exact duplicate of your trained model.
    /// 
    /// Unlike CreateInstance(), which creates a blank model with the same settings,
    /// Clone() creates a complete copy including:
    /// - The model configuration (AR order, etc.)
    /// - All trained coefficients and internal state
    /// 
    /// This is useful for:
    /// - Creating a backup before experimenting with a model
    /// - Using the same trained model in multiple scenarios
    /// - Creating ensemble models that use variations of the same base model
    /// 
    /// Think of it like photocopying a completed notebook - you get all the written content
    /// as well as the structure of the notebook itself.
    /// </para>
    /// </remarks>
    public override IFullModel<T, Matrix<T>, Vector<T>> Clone()
    {
        var clone = new ARModel<T>((ARModelOptions<T>)Options);

        // Copy trained coefficients
        if (_arCoefficients.Length > 0)
        {
            clone._arCoefficients = new Vector<T>(_arCoefficients.Length);
            for (int i = 0; i < _arCoefficients.Length; i++)
            {
                clone._arCoefficients[i] = _arCoefficients[i];
            }
        }

        // Copy stored training series
        if (_trainedSeries.Length > 0)
        {
            clone._trainedSeries = new Vector<T>(_trainedSeries.Length);
            for (int i = 0; i < _trainedSeries.Length; i++)
            {
                clone._trainedSeries[i] = _trainedSeries[i];
            }
        }

        return clone;
    }

    /// <summary>
    /// Implements the core training algorithm for the AR model.
    /// </summary>
    /// <param name="x">Feature matrix (typically just time indices for AR models).</param>
    /// <param name="y">Target vector (the time series values to be modeled).</param>
    /// <remarks>
    /// <para>
    /// This method contains the implementation details of the training process, handling coefficient
    /// initialization, gradient calculation, and parameter updates.
    /// </para>
    /// <para><b>For Beginners:</b> This is the engine room of the training process.
    /// 
    /// While the public Train method provides a high-level interface, this method does the actual work:
    /// 1. Initializes the model coefficients for each past time period
    /// 2. Calculates prediction errors with the current model
    /// 3. Determines how to adjust coefficients to improve predictions
    /// 4. Updates the coefficients accordingly
    /// 5. Repeats until the improvements become very small
    /// 
    /// Think of it as the detailed step-by-step recipe that the chef follows when you order a meal.
    /// </para>
    /// </remarks>
    protected override void TrainCore(Matrix<T> x, Vector<T> y)
    {
        // Store defensive copy of training series for in-sample predictions via Predict(Matrix<T>)
        _trainedSeries = new Vector<T>(y.Length);
        for (int i = 0; i < y.Length; i++)
        {
            _trainedSeries[i] = y[i];
        }

        // Initialize coefficients
        _arCoefficients = new Vector<T>(_arOrder);

        Vector<T> prevGradAR = new Vector<T>(_arOrder);

        for (int iter = 0; iter < _maxIterations; iter++)
        {
            Vector<T> residuals = CalculateResiduals(y);
            Vector<T> gradAR = CalculateGradients(y, residuals);

            // Normalize gradient by number of data points used
            int nSamples = y.Length - _arOrder;
            if (nSamples > 0)
            {
                var invN = NumOps.FromDouble(1.0 / nSamples);
                gradAR = (Vector<T>)Engine.Multiply(gradAR, invN);
            }

            // Gradient descent update: θ := θ - α * ∂L/∂θ.
            // The gradient is pre-negated (grad = -∂L/∂φ) during computation above, so we ADD here.
            var learningRateT = NumOps.FromDouble(_learningRate);
            var update = (Vector<T>)Engine.Multiply(gradAR, learningRateT);
            _arCoefficients = (Vector<T>)Engine.Add(_arCoefficients, update);

            // Check for convergence
            if (CheckConvergence(gradAR, prevGradAR))
            {
                break;
            }

            prevGradAR = gradAR;
        }
    }

    /// <summary>
    /// Predicts a single value based on a single input vector.
    /// </summary>
    /// <param name="input">The input vector for a single time point.</param>
    /// <returns>The predicted value for that time point.</returns>
    /// <remarks>
    /// <para>
    /// This method provides a convenient way to get a prediction for a single time point without
    /// having to create a matrix with a single row.
    /// </para>
    /// <para><b>For Beginners:</b> This is a shortcut for getting just one prediction.
    /// 
    /// Instead of providing a table of inputs for multiple time periods, you can provide
    /// just one set of inputs and get back a single prediction.
    /// 
    /// For AR models, the input typically represents a sequence of past values. This method
    /// looks at those past values and predicts what the next value will be, based on the
    /// patterns the model learned during training.
    /// 
    /// Under the hood, it:
    /// 1. Takes your single input vector
    /// 2. Creates a small table with just one row
    /// 3. Gets a prediction using the main prediction engine
    /// 4. Returns that single prediction to you
    /// </para>
    /// </remarks>
    public override T PredictSingle(Vector<T> input)
    {
        if (input == null)
        {
            throw new ArgumentNullException(nameof(input));
        }

        if (input.Length < _arOrder)
        {
            throw new ArgumentException(
                $"Input vector must contain at least {_arOrder} elements for an AR({_arOrder}) model.",
                nameof(input));
        }

        // Interpret the input as the complete history up to the current time
        // and predict the next value using the vectorized helper
        return Predict(input, input.Length);
    }
}
