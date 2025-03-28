namespace AiDotNet.TimeSeries;

/// <summary>
/// Implements an ARMA (AutoRegressive Moving Average) model for time series forecasting.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double, decimal).</typeparam>
/// <remarks>
/// <para>
/// ARMA models combine two components to forecast time series data:
/// - AR (AutoRegressive): Uses the relationship between an observation and a number of lagged observations
/// - MA (Moving Average): Uses the relationship between an observation and residual errors from moving average model
/// </para>
/// 
/// <para>
/// For Beginners:
/// The ARMA model is a popular method for analyzing and forecasting time series data
/// (data collected over time, like daily temperatures, stock prices, or monthly sales).
/// 
/// Think of ARMA as combining two different approaches:
/// 1. AutoRegressive (AR): This component predicts future values based on past values.
///    For example, tomorrow's temperature might be related to today's temperature.
///    If it's hot today, it's likely to be hot tomorrow as well.
/// 
/// 2. Moving Average (MA): This component predicts future values based on past prediction errors.
///    For example, if we consistently underestimate temperature, the MA component
///    helps adjust our future predictions upward.
/// 
/// The model has two key parameters:
/// - p: The AR order - how many past values to consider
/// - q: The MA order - how many past prediction errors to consider
/// 
/// Unlike ARIMA, ARMA doesn't include the differencing (I) component, so it works
/// best with time series data that is already stationary (doesn't have strong trends).
/// </para>
/// </remarks>
public class ARMAModel<T> : TimeSeriesModelBase<T>
{
    /// <summary>
    /// Coefficients for the autoregressive (AR) component of the model.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// These coefficients determine how much each past value influences the prediction.
    /// For example, if the coefficient for yesterday's value is 0.7, it means yesterday's
    /// value has a strong influence on today's prediction.
    /// 
    /// These values are determined during training to best fit your historical data.
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
    /// 
    /// These values are determined during training to best fit your historical data.
    /// </remarks>
    private Vector<T> _maCoefficients;

    /// <summary>
    /// The number of past observations to consider (the AR order).
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// This determines how far back in time the model looks when considering past values.
    /// For example, an AR order of 3 means the model considers values from the past 3 time periods.
    /// 
    /// If you're working with daily data, AR order = 7 means the model will look at data from the past week.
    /// </remarks>
    private int _arOrder;

    /// <summary>
    /// The number of past prediction errors to consider (the MA order).
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// This determines how far back in time the model looks when considering past prediction errors.
    /// For example, an MA order of 2 means the model considers prediction errors from the past 2 time periods.
    /// 
    /// Higher values can help the model adjust to systematic errors in prediction but may
    /// also lead to overfitting if set too high.
    /// </remarks>
    private int _maOrder;

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
    /// Creates a new ARMA model with the specified options.
    /// </summary>
    /// <param name="options">Options for the ARMA model, including AR order, MA order, and training parameters.</param>
    /// <remarks>
    /// For Beginners:
    /// This constructor creates a new ARMA model. You need to provide options to customize the model:
    /// 
    /// - AROrder: How many past values to consider (like using yesterday and the day before to predict today)
    /// - MAOrder: How many past prediction errors to consider
    /// - LearningRate: How quickly the model adjusts during training
    /// - MaxIterations: The maximum number of training attempts
    /// - Tolerance: How precise the model needs to be before training stops
    /// 
    /// Choosing good values for these options depends on your specific data and requirements.
    /// </remarks>
    public ARMAModel(ARMAOptions<T> options) : base(options)
    {
        _arOrder = options.AROrder;
        _maOrder = options.MAOrder;
        _learningRate = options.LearningRate;
        _maxIterations = options.MaxIterations;
        _tolerance = options.Tolerance;
        _arCoefficients = Vector<T>.Empty();
        _maCoefficients = Vector<T>.Empty();
    }

    /// <summary>
    /// Trains the ARMA model on the provided data.
    /// </summary>
    /// <param name="x">Feature matrix (typically just time indices for ARMA models).</param>
    /// <param name="y">Target vector (the time series values to be modeled).</param>
    /// <remarks>
    /// For Beginners:
    /// This method "teaches" the ARMA model using your historical data. The training process:
    /// 
    /// 1. Initializes the model coefficients (AR and MA components)
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
    /// Note: For ARMA models, the x parameter is often just a placeholder as the model primarily
    /// uses the time series values themselves (y) for prediction.
    /// </remarks>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        // Initialize coefficients
        _arCoefficients = new Vector<T>(_arOrder);
        _maCoefficients = new Vector<T>(_maOrder);

        Vector<T> prevGradAR = new Vector<T>(_arOrder);
        Vector<T> prevGradMA = new Vector<T>(_maOrder);

        for (int iter = 0; iter < _maxIterations; iter++)
        {
            Vector<T> residuals = CalculateResiduals(y);
            (Vector<T> gradAR, Vector<T> gradMA) = CalculateGradients(y, residuals);

            // Update coefficients using gradient descent
            for (int i = 0; i < _arOrder; i++)
            {
                _arCoefficients[i] = NumOps.Subtract(_arCoefficients[i], NumOps.Multiply(NumOps.FromDouble(_learningRate), gradAR[i]));
            }
            for (int i = 0; i < _maOrder; i++)
            {
                _maCoefficients[i] = NumOps.Subtract(_maCoefficients[i], NumOps.Multiply(NumOps.FromDouble(_learningRate), gradMA[i]));
            }

            // Check for convergence
            if (CheckConvergence(gradAR, gradMA, prevGradAR, prevGradMA))
            {
                break;
            }

            prevGradAR = gradAR;
            prevGradMA = gradMA;
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
        for (int t = Math.Max(_arOrder, _maOrder); t < y.Length; t++)
        {
            T yHat = Predict(y, t);
            residuals[t] = NumOps.Subtract(y[t], yHat);
        }

        return residuals;
    }

    /// <summary>
    /// Calculates the gradients for adjusting the AR and MA coefficients.
    /// </summary>
    /// <param name="y">The time series values.</param>
    /// <param name="residuals">The current residuals (prediction errors).</param>
    /// <returns>Gradients for the AR and MA coefficients.</returns>
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
    /// The method returns two sets of gradients:
    /// - gradAR: For adjusting coefficients that determine how past values affect predictions
    /// - gradMA: For adjusting coefficients that determine how past errors affect predictions
    /// </remarks>
    private (Vector<T>, Vector<T>) CalculateGradients(Vector<T> y, Vector<T> residuals)
    {
        Vector<T> gradAR = new Vector<T>(_arOrder);
        Vector<T> gradMA = new Vector<T>(_maOrder);

        for (int t = Math.Max(_arOrder, _maOrder); t < y.Length; t++)
        {
            for (int i = 0; i < _arOrder; i++)
            {
                gradAR[i] = NumOps.Add(gradAR[i], NumOps.Multiply(residuals[t], y[t - i - 1]));
            }
            for (int i = 0; i < _maOrder; i++)
            {
                gradMA[i] = NumOps.Add(gradMA[i], NumOps.Multiply(residuals[t], residuals[t - i - 1]));
            }
        }

        return (gradAR, gradMA);
    }

    /// <summary>
    /// Checks if the training process has converged (reached a stable solution).
    /// </summary>
    /// <param name="gradAR">Current AR gradients.</param>
    /// <param name="gradMA">Current MA gradients.</param>
    /// <param name="prevGradAR">Previous AR gradients.</param>
    /// <param name="prevGradMA">Previous MA gradients.</param>
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
    private bool CheckConvergence(Vector<T> gradAR, Vector<T> gradMA, Vector<T> prevGradAR, Vector<T> prevGradMA)
    {
        T diffAR = gradAR.Subtract(prevGradAR).Norm();
        T diffMA = gradMA.Subtract(prevGradMA).Norm();

        return NumOps.LessThan(diffAR, NumOps.FromDouble(_tolerance)) && NumOps.LessThan(diffMA, NumOps.FromDouble(_tolerance));
    }

    /// <summary>
    /// Makes predictions using the trained ARMA model.
    /// </summary>
    /// <param name="input">Input matrix for prediction (typically just time indices for future periods).</param>
    /// <returns>A vector of predicted values.</returns>
    /// <remarks>
    /// For Beginners:
    /// This method uses the trained ARMA model to forecast future values.
    /// 
    /// For each time point in the input, it calls the prediction helper method to generate
    /// a forecast based on:
    /// 1. The AR component (effect of past values)
    /// 2. The MA component (effect of past prediction errors)
    /// 
    /// The result is a vector of predicted values, one for each time point in the input.
    /// 
    /// Note: For pure time series forecasting, the input parameter might just indicate
    /// how many future periods to predict.
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
    /// The prediction combines:
    /// 1. AR component: The effect of past values
    ///    - For example, if yesterday's temperature was high, today's might also be high
    /// 
    /// 2. MA component: The effect of past prediction errors
    ///    - For example, if we've been consistently underpredicting, we might adjust upward
    /// 
    /// The method handles cases where we don't have enough history (e.g., at the beginning
    /// of the series) by only using the available information.
    /// 
    /// This recursive approach allows the model to build up predictions one step at a time,
    /// using both actual values and predictions as needed.
    /// </remarks>
    private T Predict(Vector<T> y, int t)
    {
        T prediction = NumOps.Zero;
        for (int i = 0; i < _arOrder && t - i - 1 >= 0; i++)
        {
            prediction = NumOps.Add(prediction, NumOps.Multiply(_arCoefficients[i], y[t - i - 1]));
        }
        for (int i = 0; i < _maOrder && t - i - 1 >= 0; i++)
        {
            T residual = NumOps.Subtract(y[t - i - 1], Predict(y, t - i - 1));
            prediction = NumOps.Add(prediction, NumOps.Multiply(_maCoefficients[i], residual));
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
        Dictionary<string, T> metrics = new()
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
    /// - The AR and MA orders (how many coefficients)
    /// - The AR coefficients (how past values affect predictions)
    /// - The MA coefficients (how past errors affect predictions)
    /// 
    /// This allows the model to be fully reconstructed later.
    /// </remarks>
    protected override void SerializeCore(BinaryWriter writer)
    {
        writer.Write(_arOrder);
        writer.Write(_maOrder);
        for (int i = 0; i < _arOrder; i++)
        {
            writer.Write(Convert.ToDouble(_arCoefficients[i]));
        }
        for (int i = 0; i < _maOrder; i++)
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
    /// This protected method loads a previously saved model from a file or stream.
    /// 
    /// Deserialization allows you to:
    /// 1. Load a previously trained model
    /// 2. Use it immediately without retraining
    /// 3. Apply the exact same model to new data
    /// 
    /// The method loads:
    /// - The AR and MA orders (how many coefficients)
    /// - The AR coefficients (how past values affect predictions)
    /// - The MA coefficients (how past errors affect predictions)
    /// 
    /// After deserialization, the model is ready to make predictions as if it had just been trained.
    /// </remarks>
    protected override void DeserializeCore(BinaryReader reader)
    {
        _arOrder = reader.ReadInt32();
        _maOrder = reader.ReadInt32();
        _arCoefficients = new Vector<T>(_arOrder);
        _maCoefficients = new Vector<T>(_maOrder);
        for (int i = 0; i < _arOrder; i++)
        {
            _arCoefficients[i] = NumOps.FromDouble(reader.ReadDouble());
        }
        for (int i = 0; i < _maOrder; i++)
        {
            _maCoefficients[i] = NumOps.FromDouble(reader.ReadDouble());
        }
    }
}