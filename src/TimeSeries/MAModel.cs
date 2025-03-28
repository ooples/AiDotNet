namespace AiDotNet.TimeSeries;

/// <summary>
/// Represents a Moving Average (MA) model for time series forecasting.
/// </summary>
/// <remarks>
/// <para>
/// The Moving Average (MA) model is a time series forecasting method that uses past forecast errors 
/// (residuals) to predict future values. It assumes that the time series can be modeled as a weighted 
/// sum of past residuals plus a constant term. The MA model is particularly useful for modeling 
/// time series with short-term dependencies and no persistent trends.
/// </para>
/// <para><b>For Beginners:</b> An MA model uses past prediction errors to forecast future values.
/// 
/// Think of it like adjusting your expectations based on recent mistakes:
/// - If you've consistently underestimated values recently, you might adjust your next prediction upward
/// - If you've consistently overestimated values recently, you might adjust your next prediction downward
/// 
/// For example, if forecasting daily restaurant visitors:
/// - You predicted 100 customers yesterday, but 110 showed up (error of +10)
/// - You predicted 120 today, but 105 showed up (error of -15)
/// - For tomorrow, you might adjust your prediction based on these recent errors
/// 
/// The MA model works well for data with short-term patterns but no long-term trends,
/// such as random fluctuations around a stable average.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class MAModel<T> : TimeSeriesModelBase<T>
{
    /// <summary>
    /// The Moving Average (MA) coefficients that weight the past forecast errors.
    /// </summary>
    /// <remarks>
    /// <para>
    /// These coefficients determine how much each past forecast error influences the current prediction.
    /// They are estimated during the training process to minimize the prediction error.
    /// </para>
    /// <para><b>For Beginners:</b> These determine how strongly each past error affects your prediction.
    /// 
    /// The MA coefficients:
    /// - Show how much weight to give to each past prediction error
    /// - Higher values mean stronger influence from that error
    /// - Lower values mean less influence
    /// 
    /// For example, if the coefficient for the most recent error is 0.7, and the coefficient for
    /// the error from two periods ago is 0.3, it means the most recent error has more than twice
    /// the influence on your prediction compared to the older error.
    /// </para>
    /// </remarks>
    private Vector<T> _maCoefficients;

    /// <summary>
    /// The order of the Moving Average model, indicating how many past errors are considered.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The MA order defines how many past forecast errors are included in the model. For example,
    /// an MA(2) model considers the errors from the two most recent periods. A higher order captures
    /// more complex error patterns but requires more data and risks overfitting.
    /// </para>
    /// <para><b>For Beginners:</b> This specifies how many past prediction errors the model considers.
    /// 
    /// The MA order:
    /// - Determines how far back in time the model "looks" for error patterns
    /// - An MA(1) model only looks at the most recent error
    /// - An MA(2) model looks at the two most recent errors
    /// - And so on for higher orders
    /// 
    /// Higher orders can capture more complex patterns but may "overfit" the data
    /// (learn patterns that don't generalize well to new data).
    /// </para>
    /// </remarks>
    private int _maOrder;

    /// <summary>
    /// The learning rate used in gradient descent optimization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The learning rate controls the step size in each iteration of the gradient descent optimization
    /// algorithm used to estimate the MA coefficients. A higher learning rate can speed up convergence
    /// but risks overshooting the optimal solution, while a lower learning rate provides more precise
    /// optimization but may require more iterations.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how quickly the model adjusts its parameters during training.
    /// 
    /// The learning rate:
    /// - Determines how big of adjustments to make when updating the model's coefficients
    /// - Higher values mean bigger, faster adjustments (but might miss the optimal solution)
    /// - Lower values mean smaller, more careful adjustments (but take longer to train)
    /// 
    /// Think of it like turning a dial to find the right radio station:
    /// - A high learning rate is like turning the dial quickly
    /// - A low learning rate is like turning the dial very slowly and carefully
    /// </para>
    /// </remarks>
    private readonly double _learningRate;

    /// <summary>
    /// The maximum number of iterations allowed in the optimization process.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This parameter limits the number of iterations in the gradient descent optimization algorithm.
    /// It prevents the algorithm from running indefinitely if convergence is slow or not achieved.
    /// </para>
    /// <para><b>For Beginners:</b> This sets a limit on how long the model will train.
    /// 
    /// The maximum iterations:
    /// - Acts as a safety limit on how many attempts the model makes to improve itself
    /// - Prevents the training process from running forever if it can't find the perfect solution
    /// - Usually set high enough to allow convergence but not so high that training takes too long
    /// 
    /// It's like telling someone "try to solve this puzzle, but give up after 1000 attempts
    /// if you haven't figured it out yet."
    /// </para>
    /// </remarks>
    private readonly int _maxIterations;

    /// <summary>
    /// The convergence tolerance for the optimization process.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The tolerance defines the threshold for determining when the optimization has converged.
    /// If the change in the gradient norm between consecutive iterations is less than this tolerance,
    /// the algorithm stops and considers the solution optimized.
    /// </para>
    /// <para><b>For Beginners:</b> This determines when the model decides it's "good enough" and stops training.
    /// 
    /// The tolerance:
    /// - Sets how small the improvement between training iterations must be before stopping
    /// - Smaller values require more precise optimization (may take longer)
    /// - Larger values allow the model to stop sooner when improvements become minimal
    /// 
    /// It's like saying "keep improving until the difference between attempts is less than 0.0001,
    /// then you can stop because further improvements would be negligible."
    /// </para>
    /// </remarks>
    private readonly double _tolerance;

    /// <summary>
    /// Initializes a new instance of the <see cref="MAModel{T}"/> class with the specified options.
    /// </summary>
    /// <param name="options">The configuration options for the Moving Average model.</param>
    /// <remarks>
    /// <para>
    /// This constructor initializes the Moving Average model with the provided configuration options.
    /// The options specify parameters such as the MA order, learning rate, maximum iterations,
    /// and convergence tolerance.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up your Moving Average model with your chosen settings.
    /// 
    /// When creating the model, you can specify:
    /// - How many past errors to consider (MA order)
    /// - How quickly the model should learn (learning rate)
    /// - How many training attempts to make (max iterations)
    /// - When to consider the model "good enough" (tolerance)
    /// 
    /// These settings control the balance between how accurate the model is,
    /// how long it takes to train, and how well it generalizes to new data.
    /// </para>
    /// </remarks>
    public MAModel(MAModelOptions<T> options) : base(options)
    {
        _maOrder = options.MAOrder;
        _learningRate = options.LearningRate;
        _maxIterations = options.MaxIterations;
        _tolerance = options.Tolerance;
        _maCoefficients = Vector<T>.Empty();
    }

    /// <summary>
    /// Trains the Moving Average model on the provided input and output data.
    /// </summary>
    /// <param name="x">The input features matrix (typically time indicators or related variables).</param>
    /// <param name="y">The target values vector (the time series data to forecast).</param>
    /// <remarks>
    /// <para>
    /// This method trains the Moving Average model using gradient descent optimization to find
    /// the optimal MA coefficients that minimize the prediction error. The training process iteratively
    /// updates the coefficients based on the gradient of the error with respect to each coefficient.
    /// </para>
    /// <para><b>For Beginners:</b> This teaches the model how to make predictions based on past errors.
    /// 
    /// The training process:
    /// 1. Starts with initial coefficient values (all zeros)
    /// 2. Makes predictions and calculates errors
    /// 3. Determines how to adjust each coefficient to reduce errors
    /// 4. Updates the coefficients and repeats the process
    /// 5. Stops when the improvements become very small or after reaching max iterations
    /// 
    /// Through this process, the model learns the optimal weights to give to each past error
    /// when making new predictions.
    /// </para>
    /// </remarks>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        // Initialize coefficients
        _maCoefficients = new Vector<T>(_maOrder);

        Vector<T> prevGradMA = new Vector<T>(_maOrder);
        Vector<T> errors = new Vector<T>(y.Length);

        for (int iter = 0; iter < _maxIterations; iter++)
        {
            Vector<T> residuals = CalculateResiduals(y, errors);
            Vector<T> gradMA = CalculateGradients(errors, residuals);

            // Update coefficients using gradient descent
            for (int i = 0; i < _maOrder; i++)
            {
                _maCoefficients[i] = NumOps.Subtract(_maCoefficients[i], NumOps.Multiply(NumOps.FromDouble(_learningRate), gradMA[i]));
            }

            // Check for convergence
            if (CheckConvergence(gradMA, prevGradMA))
            {
                break;
            }

            prevGradMA = gradMA;
        }
    }

    /// <summary>
    /// Calculates the residuals (errors) between the actual and predicted values.
    /// </summary>
    /// <param name="y">The target values vector.</param>
    /// <param name="errors">The vector to store the calculated errors.</param>
    /// <returns>A vector containing the residuals.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the residuals by comparing the actual time series values with the model's
    /// predictions. These residuals are both used to evaluate the model's performance and as inputs
    /// for making future predictions in the MA model.
    /// </para>
    /// <para><b>For Beginners:</b> This calculates how far off the model's predictions are from actual values.
    /// 
    /// The method:
    /// - Makes predictions for each time point
    /// - Compares each prediction to the actual value
    /// - Calculates the difference (the error or residual)
    /// - Stores these errors for both evaluation and future predictions
    /// 
    /// Since MA models use past errors to make predictions, these calculated errors
    /// are essential both for training and for making future forecasts.
    /// </para>
    /// </remarks>
    private Vector<T> CalculateResiduals(Vector<T> y, Vector<T> errors)
    {
        Vector<T> residuals = new Vector<T>(y.Length);
        for (int t = 0; t < y.Length; t++)
        {
            T yHat = Predict(errors, t);
            residuals[t] = NumOps.Subtract(y[t], yHat);
            errors[t] = residuals[t];
        }

        return residuals;
    }

    /// <summary>
    /// Calculates the gradients of the error with respect to the MA coefficients.
    /// </summary>
    /// <param name="errors">The vector of errors from previous predictions.</param>
    /// <param name="residuals">The vector of residuals from the current predictions.</param>
    /// <returns>A vector containing the gradients for each MA coefficient.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the gradients that indicate how the prediction error would change
    /// with respect to small changes in each MA coefficient. These gradients guide the direction
    /// and magnitude of the coefficient updates during gradient descent optimization.
    /// </para>
    /// <para><b>For Beginners:</b> This determines how to adjust each coefficient to improve predictions.
    /// 
    /// The gradient calculation:
    /// - Shows which direction to move each coefficient to reduce errors
    /// - Larger gradient values suggest bigger adjustments are needed
    /// - Smaller gradient values suggest the coefficient is closer to optimal
    /// 
    /// Think of gradients like a compass that shows which way to adjust each
    /// coefficient and how big of an adjustment to make.
    /// </para>
    /// </remarks>
    private Vector<T> CalculateGradients(Vector<T> errors, Vector<T> residuals)
    {
        Vector<T> gradMA = new Vector<T>(_maOrder);

        for (int t = _maOrder; t < errors.Length; t++)
        {
            for (int i = 0; i < _maOrder; i++)
            {
                gradMA[i] = NumOps.Add(gradMA[i], NumOps.Multiply(residuals[t], errors[t - i - 1]));
            }
        }

        return gradMA;
    }

    /// <summary>
    /// Checks if the optimization process has converged.
    /// </summary>
    /// <param name="gradMA">The current gradient vector.</param>
    /// <param name="prevGradMA">The gradient vector from the previous iteration.</param>
    /// <returns>True if convergence is achieved; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method determines if the optimization process has converged by comparing the norm of
    /// the difference between the current and previous gradients to the specified tolerance.
    /// If the difference is smaller than the tolerance, the process is considered to have converged.
    /// </para>
    /// <para><b>For Beginners:</b> This decides when the model is "good enough" and training can stop.
    /// 
    /// The convergence check:
    /// - Compares how much the adjustments have changed since the last iteration
    /// - If the change is smaller than the tolerance, it means the model has nearly stopped improving
    /// - This suggests that further training would yield minimal benefits
    /// 
    /// It's like checking if your last few attempts to improve a score are making
    /// less and less difference, suggesting you're approaching your best possible result.
    /// </para>
    /// </remarks>
    private bool CheckConvergence(Vector<T> gradMA, Vector<T> prevGradMA)
    {
        T diffMA = gradMA.Subtract(prevGradMA).Norm();
        return NumOps.LessThan(diffMA, NumOps.FromDouble(_tolerance));
    }

    /// <summary>
    /// Generates predictions for the given input data.
    /// </summary>
    /// <param name="input">The input features matrix.</param>
    /// <returns>A vector containing the predicted values.</returns>
    /// <remarks>
    /// <para>
    /// This method generates forecasts for each time point in the input matrix by applying the MA model.
    /// For each prediction, it uses the past forecast errors and the estimated MA coefficients.
    /// </para>
    /// <para><b>For Beginners:</b> This makes predictions for each time point in your data.
    /// 
    /// The prediction process:
    /// - For each time point, makes a separate prediction
    /// - Uses the past errors (the differences between actual and predicted values)
    /// - Applies the learned MA coefficients to weight these past errors
    /// - Combines these weighted errors to form the prediction
    /// 
    /// The model essentially says, "Based on how wrong my recent predictions were,
    /// here's what I think the next value will be."
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> input)
    {
        Vector<T> predictions = new Vector<T>(input.Rows);
        Vector<T> errors = new Vector<T>(input.Rows);
        for (int t = 0; t < input.Rows; t++)
        {
            predictions[t] = Predict(errors, t);
            if (t < input.Rows - 1)
            {
                errors[t] = NumOps.Subtract(input[t + 1, 0], predictions[t]);
            }
        }

        return predictions;
    }

    /// <summary>
    /// Predicts a single value at the specified index.
    /// </summary>
    /// <param name="errors">The vector of errors from previous predictions.</param>
    /// <param name="t">The index to predict.</param>
    /// <returns>The predicted value for the specified index.</returns>
    /// <remarks>
    /// <para>
    /// This method predicts a single value at the specified index by applying the MA model.
    /// It calculates a weighted sum of past forecast errors using the estimated MA coefficients.
    /// </para>
    /// <para><b>For Beginners:</b> This generates a prediction for a single time point.
    /// 
    /// The prediction formula:
    /// - Starts with a base value (usually zero in a pure MA model)
    /// - Adds weighted contributions from past errors
    /// - Each past error is multiplied by its corresponding MA coefficient
    /// - These weighted errors are summed to form the prediction
    /// 
    /// For example, with an MA(2) model with coefficients [0.7, 0.3]:
    /// - If the last two errors were +10 and -5
    /// - The prediction would be: 0 + (0.7 × 10) + (0.3 × -5) = 7 - 1.5 = 5.5
    /// </para>
    /// </remarks>
    private T Predict(Vector<T> errors, int t)
    {
        T prediction = NumOps.Zero;
        for (int i = 0; i < _maOrder && t - i - 1 >= 0; i++)
        {
            prediction = NumOps.Add(prediction, NumOps.Multiply(_maCoefficients[i], errors[t - i - 1]));
        }

        return prediction;
    }

    /// <summary>
    /// Evaluates the model's performance on test data.
    /// </summary>
    /// <param name="xTest">The test input features matrix.</param>
    /// <param name="yTest">The test target values vector.</param>
    /// <returns>A dictionary containing various evaluation metrics.</returns>
    /// <remarks>
    /// <para>
    /// This method evaluates the model's performance on test data by generating predictions and calculating
    /// various error metrics. The returned metrics include Mean Squared Error (MSE), Root Mean Squared Error (RMSE),
    /// Mean Absolute Error (MAE), and Mean Absolute Percentage Error (MAPE).
    /// </para>
    /// <para><b>For Beginners:</b> This measures how accurate the model's predictions are.
    /// 
    /// The evaluation:
    /// - Makes predictions for data the model hasn't seen before
    /// - Compares these predictions to the actual values
    /// - Calculates different types of error measurements:
    ///   - MSE (Mean Squared Error): Average of squared differences
    ///   - RMSE (Root Mean Squared Error): Square root of MSE, in the same units as your data
    ///   - MAE (Mean Absolute Error): Average of absolute differences
    ///   - MAPE (Mean Absolute Percentage Error): Average percentage difference
    /// 
    /// Lower values for all these metrics indicate better performance.
    /// Each metric provides a slightly different perspective on the model's accuracy.
    /// </para>
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
    /// Serializes the model's core parameters to a binary writer.
    /// </summary>
    /// <param name="writer">The binary writer to write to.</param>
    /// <remarks>
    /// <para>
    /// This method writes the model's essential parameters to a binary stream, allowing the model to be saved
    /// to a file or database. The serialized parameters include the MA order and the MA coefficients.
    /// </para>
    /// <para><b>For Beginners:</b> This saves the model so you can use it later.
    /// 
    /// The method:
    /// - Converts the model's parameters to a format that can be saved
    /// - Writes these values to a file or database
    /// - Includes all the information needed to recreate the model exactly
    /// 
    /// This allows you to:
    /// - Save a trained model for future use
    /// - Share the model with others
    /// - Use the model in other applications
    /// 
    /// It's like saving a document so you can open it again later without
    /// having to start from scratch.
    /// </para>
    /// </remarks>
    protected override void SerializeCore(BinaryWriter writer)
    {
        writer.Write(_maOrder);
        for (int i = 0; i < _maOrder; i++)
        {
            writer.Write(Convert.ToDouble(_maCoefficients[i]));
        }
    }

    /// <summary>
    /// Deserializes the model's core parameters from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader to read from.</param>
    /// <remarks>
    /// <para>
    /// This method reads the model's essential parameters from a binary stream, allowing a previously saved model
    /// to be loaded from a file or database. The deserialized parameters include the MA order and the MA coefficients.
    /// </para>
    /// <para><b>For Beginners:</b> This loads a previously saved model.
    /// 
    /// The method:
    /// - Reads the saved model data from a file or database
    /// - Converts this data back into the model's parameters
    /// - Reconstructs the model exactly as it was when saved
    /// 
    /// This is particularly useful when:
    /// - You want to use a model that took a long time to train
    /// - You want to ensure consistent results across different runs
    /// - You need to deploy the model in a production environment
    /// 
    /// Think of it like opening a document you previously saved, allowing you
    /// to continue using the model without having to train it again.
    /// </para>
    /// </remarks>
    protected override void DeserializeCore(BinaryReader reader)
    {
        _maOrder = reader.ReadInt32();
        _maCoefficients = new Vector<T>(_maOrder);
        for (int i = 0; i < _maOrder; i++)
        {
            _maCoefficients[i] = NumOps.FromDouble(reader.ReadDouble());
        }
    }
}