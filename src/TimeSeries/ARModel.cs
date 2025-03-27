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
    }

    /// <summary>
    /// Trains the AR model on the provided data.
    /// </summary>
    /// <param name="x">Feature matrix (typically just time indices for AR models).</param>
    /// <param name="y">Target vector (the time series values to be modeled).</param>
    /// <remarks>
    /// For Beginners:
    /// This method "teaches" the AR model using your historical data. The training process:
    /// 
    /// 1. Initializes the model coefficients (for each past time period)
    /// 2. Repeatedly:
    ///    - Calculates how well the current model fits the data
    ///    - Determines how to adjust the coefficients to fit better
    ///    - Updates the coefficients
    ///    - Checks if the improvements are small enough to stop
    /// 
    /// The method uses a technique called "gradient descent" to find the best coefficients.
    /// Think of it like walking downhill to find the lowest point - at each step, you look
    /// around and move in the direction that goes downhill the fastest.
    /// 
    /// After training, the model can be used to make predictions.
    /// 
    /// Note: For AR models, the x parameter is often just a placeholder as the model primarily
    /// uses the time series values themselves (y) for prediction.
    /// </remarks>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        // Initialize coefficients
        _arCoefficients = new Vector<T>(_arOrder);

        Vector<T> prevGradAR = new Vector<T>(_arOrder);

        for (int iter = 0; iter < _maxIterations; iter++)
        {
            Vector<T> residuals = CalculateResiduals(y);
            Vector<T> gradAR = CalculateGradients(y, residuals);

            // Update coefficients using gradient descent
            for (int i = 0; i < _arOrder; i++)
            {
                _arCoefficients[i] = NumOps.Subtract(_arCoefficients[i], NumOps.Multiply(NumOps.FromDouble(_learningRate), gradAR[i]));
            }

            // Check for convergence
            if (CheckConvergence(gradAR, prevGradAR))
            {
                break;
            }

            prevGradAR = gradAR;
        }
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

        for (int t = _arOrder; t < y.Length; t++)
        {
            for (int i = 0; i < _arOrder; i++)
            {
                gradAR[i] = NumOps.Add(gradAR[i], NumOps.Multiply(residuals[t], y[t - i - 1]));
            }
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
        T diffAR = gradAR.Subtract(prevGradAR).Norm();
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
        Vector<T> predictions = new Vector<T>(input.Rows);
        for (int t = 0; t < input.Rows; t++)
        {
            predictions[t] = Predict(input.GetRow(t), t);
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
    /// - coefficientⁿ is the importance of each past value
    /// - valueⁿ is the actual value at that past time point
    /// 
    /// The method handles cases where we don't have enough history (e.g., at the beginning
    /// of the series) by only using the available information.
    /// </remarks>
    private T Predict(Vector<T> y, int t)
    {
        T prediction = NumOps.Zero;
        for (int i = 0; i < _arOrder && t - i - 1 >= 0; i++)
        {
            prediction = NumOps.Add(prediction, NumOps.Multiply(_arCoefficients[i], y[t - i - 1]));
        }

        return prediction;
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
    }
}