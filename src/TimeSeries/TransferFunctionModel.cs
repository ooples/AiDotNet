namespace AiDotNet.TimeSeries;

/// <summary>
/// Implements a Transfer Function Model for time series analysis, which combines ARIMA modeling with
/// external input variables to capture dynamic relationships between multiple time series.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// The Transfer Function Model extends traditional ARIMA models by incorporating the effects of
/// external input variables. It models the relationship between an output time series and one or more
/// input time series, accounting for both immediate and lagged effects.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// A Transfer Function Model helps you understand how one time series affects another over time.
/// 
/// For example, you might want to know:
/// - How advertising spending affects sales (with delays of days or weeks)
/// - How temperature changes affect energy consumption
/// - How interest rate changes impact housing prices
/// 
/// This model captures both:
/// - The internal patterns of your target variable (like sales following their own seasonal patterns)
/// - The external influence of input variables (like how advertising boosts sales)
/// 
/// It's particularly useful when you know there are external factors influencing your target variable
/// and you want to quantify their effects, including any time delays in those effects.
/// </para>
/// </remarks>
public class TransferFunctionModel<T> : TimeSeriesModelBase<T>
{
    /// <summary>
    /// Configuration options specific to the Transfer Function Model.
    /// </summary>
    private readonly TransferFunctionOptions<T> _tfOptions;
    
    /// <summary>
    /// Autoregressive (AR) parameters that capture the dependency on past values of the output series.
    /// </summary>
    private Vector<T> _arParameters;
    
    /// <summary>
    /// Moving Average (MA) parameters that capture the dependency on past error terms.
    /// </summary>
    private Vector<T> _maParameters;
    
    /// <summary>
    /// Parameters that capture the effect of input variables at different lags.
    /// </summary>
    private Vector<T> _inputLags;
    
    /// <summary>
    /// Parameters that capture the effect of output variables at different lags.
    /// </summary>
    private Vector<T> _outputLags;
    
    /// <summary>
    /// Residuals (errors) from the model fit.
    /// </summary>
    private Vector<T> _residuals;
    
    /// <summary>
    /// Fitted values from the model.
    /// </summary>
    private Vector<T> _fitted;
    
    /// <summary>
    /// The original output time series data.
    /// </summary>
    private Vector<T> _y;
    
    /// <summary>
    /// The optimization algorithm used to estimate model parameters.
    /// </summary>
    private readonly IOptimizer<T> _optimizer;

    /// <summary>
    /// Initializes a new instance of the TransferFunctionModel class with optional configuration options.
    /// </summary>
    /// <param name="options">The configuration options for the Transfer Function Model. If null, default options are used.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// When you create a Transfer Function Model, you can customize it with various options:
    /// 
    /// - AR Order: How many past values of the output series to consider
    /// - MA Order: How many past error terms to consider
    /// - Input Lag Order: How many past values of the input series to consider
    /// - Output Lag Order: How many additional past values of the output to consider
    /// - Optimizer: The algorithm used to find the best parameter values
    /// 
    /// The constructor initializes all the parameters that will be estimated during training.
    /// These parameters start empty and will be filled with values during the training process.
    /// </para>
    /// </remarks>
    public TransferFunctionModel(TransferFunctionOptions<T>? options = null) : base(options ?? new())
    {
        _tfOptions = options ?? new TransferFunctionOptions<T>();
        _optimizer = _tfOptions.Optimizer ?? new LBFGSOptimizer<T>();
        _y = Vector<T>.Empty();
        _arParameters = Vector<T>.Empty();
        _maParameters = Vector<T>.Empty();
        _inputLags = Vector<T>.Empty();
        _outputLags = Vector<T>.Empty();
        _residuals = Vector<T>.Empty();
        _fitted = Vector<T>.Empty();
    }

    /// <summary>
    /// Trains the Transfer Function Model using the provided input data and target values.
    /// </summary>
    /// <param name="x">The input features matrix containing external variables.</param>
    /// <param name="y">The output time series data to model.</param>
    /// <exception cref="ArgumentException">Thrown when the input matrix rows don't match the output vector length.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// Training a Transfer Function Model involves finding the best values for all its parameters
    /// to accurately capture the relationship between input and output time series.
    /// 
    /// The training process follows these steps:
    /// 
    /// 1. Validate that the input and output data have compatible dimensions
    /// 2. Initialize the model parameters with small random values
    /// 3. Use an optimization algorithm to find the parameter values that best fit the data
    /// 4. Compute the residuals (errors) between the model's predictions and actual values
    /// 
    /// After training, the model will have learned:
    /// - How the output series depends on its own past values
    /// - How the output series is influenced by the input series
    /// - The patterns in the error terms
    /// 
    /// These learned parameters can then be used to make predictions or analyze the relationship
    /// between the input and output variables.
    /// </para>
    /// </remarks>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        if (x.Rows != y.Length)
        {
            throw new ArgumentException("Input matrix rows must match output vector length.");
        }

        int n = y.Length;
        _y = y;

        InitializeParameters();
        OptimizeParameters(x, y);
        ComputeResiduals(x, y);
    }

    /// <summary>
    /// Initializes the model parameters with small random values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// Before we can optimize the model parameters, we need to give them initial values.
    /// 
    /// This method:
    /// 1. Creates vectors of the appropriate size for each type of parameter
    /// 2. Fills them with small random values (between 0 and 0.1)
    /// 
    /// Using small random values helps the optimization process start from a neutral point
    /// without making strong assumptions about the relationships in the data.
    /// 
    /// The parameters initialized are:
    /// - AR parameters: How past values of the output affect current values
    /// - MA parameters: How past prediction errors affect current values
    /// - Input lag parameters: How past values of the input affect current output
    /// - Output lag parameters: How additional past values of the output affect current values
    /// </para>
    /// </remarks>
    private void InitializeParameters()
    {
        int p = _tfOptions.AROrder;
        int q = _tfOptions.MAOrder;
        int r = _tfOptions.InputLagOrder;
        int s = _tfOptions.OutputLagOrder;

        _arParameters = new Vector<T>(p);
        _maParameters = new Vector<T>(q);
        _inputLags = new Vector<T>(r);
        _outputLags = new Vector<T>(s);

        // Initialize with small random values
        Random rand = new Random();
        for (int i = 0; i < p; i++) _arParameters[i] = NumOps.FromDouble(rand.NextDouble() * 0.1);
        for (int i = 0; i < q; i++) _maParameters[i] = NumOps.FromDouble(rand.NextDouble() * 0.1);
        for (int i = 0; i < r; i++) _inputLags[i] = NumOps.FromDouble(rand.NextDouble() * 0.1);
        for (int i = 0; i < s; i++) _outputLags[i] = NumOps.FromDouble(rand.NextDouble() * 0.1);
    }

    /// <summary>
    /// Optimizes the model parameters using the specified optimization algorithm.
    /// </summary>
    /// <param name="x">The input features matrix.</param>
    /// <param name="y">The output time series data.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method finds the best values for all model parameters to minimize prediction errors.
    /// 
    /// It works by:
    /// 1. Packaging the input and output data for the optimizer
    /// 2. Running the optimization algorithm (like L-BFGS) to find parameter values
    ///    that minimize the difference between predictions and actual values
    /// 3. Updating the model with these optimized parameter values
    /// 
    /// The optimization process is like turning knobs on a radio until you get the clearest signal.
    /// The algorithm systematically tries different parameter values, measuring how well each set
    /// performs, and gradually moves toward the best combination.
    /// </para>
    /// </remarks>
    private void OptimizeParameters(Matrix<T> x, Vector<T> y)
    {
        var inputData = new OptimizationInputData<T>
        {
            XTrain = x,
            YTrain = y
        };

        OptimizationResult<T> result = _optimizer.Optimize(inputData);
        UpdateModelParameters(result.BestSolution.Coefficients);
    }

    /// <summary>
    /// Updates the model parameters with the optimized values.
    /// </summary>
    /// <param name="optimizedParameters">The vector of optimized parameter values.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// After the optimization algorithm finds the best parameter values, this method
    /// updates the model with those values.
    /// 
    /// It works by:
    /// 1. Taking the flat vector of optimized parameters
    /// 2. Distributing these values to the appropriate parameter vectors (AR, MA, input lags, output lags)
    /// 
    /// This is like taking the results of a calculation and updating your spreadsheet with the new values.
    /// Each parameter now contains the value that best helps the model predict the output series
    /// based on its own past values and the input series.
    /// </para>
    /// </remarks>
    private void UpdateModelParameters(Vector<T> optimizedParameters)
    {
        int paramIndex = 0;

        // Update AR parameters
        for (int i = 0; i < _arParameters.Length; i++)
        {
            _arParameters[i] = optimizedParameters[paramIndex++];
        }

        // Update MA parameters
        for (int i = 0; i < _maParameters.Length; i++)
        {
            _maParameters[i] = optimizedParameters[paramIndex++];
        }

        // Update input lag parameters
        for (int i = 0; i < _inputLags.Length; i++)
        {
            _inputLags[i] = optimizedParameters[paramIndex++];
        }

        // Update output lag parameters
        for (int i = 0; i < _outputLags.Length; i++)
        {
            _outputLags[i] = optimizedParameters[paramIndex++];
        }
    }

    /// <summary>
    /// Computes the residuals (errors) between the model's predictions and the actual values.
    /// </summary>
    /// <param name="x">The input features matrix.</param>
    /// <param name="y">The output time series data.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// Residuals are the differences between what the model predicts and what actually happened.
    /// They tell us how well the model fits the data.
    /// 
    /// This method:
    /// 1. Uses the trained model to make predictions for the training data
    /// 2. Calculates the difference between each prediction and the actual value
    /// 3. Stores these differences as residuals
    /// 
    /// Analyzing residuals is important because:
    /// - If they're randomly distributed around zero, the model has captured the patterns well
    /// - If they show patterns, the model might be missing something important
    /// - They're used in the MA (Moving Average) part of the model for future predictions
    /// </para>
    /// </remarks>
    private void ComputeResiduals(Matrix<T> x, Vector<T> y)
    {
        _fitted = Predict(x);
        _residuals = new Vector<T>(y.Length);

        for (int i = 0; i < y.Length; i++)
        {
            _residuals[i] = NumOps.Subtract(y[i], _fitted[i]);
        }
    }

    /// <summary>
    /// Generates forecasts using the trained Transfer Function Model.
    /// </summary>
    /// <param name="input">The input features matrix containing external variables.</param>
    /// <returns>A vector of forecasted values.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method uses the trained model to predict values for new input data.
    /// 
    /// For each time point, it:
    /// 1. Considers the AR terms (past values of the output)
    /// 2. Adds the MA terms (past prediction errors)
    /// 3. Adds the effects of the input variables at various lags
    /// 4. Adds the effects of additional output lags
    /// 
    /// The result is a forecast that accounts for both:
    /// - The internal patterns of the output series
    /// - The influence of the input variables
    /// 
    /// This is useful for scenarios like predicting sales based on advertising spend,
    /// where you want to account for both the natural sales patterns and the boost from ads.
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> input)
    {
        int n = input.Rows;
        Vector<T> predictions = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            predictions[i] = PredictSingle(input, predictions, i);
        }

        return predictions;
    }

    /// <summary>
    /// Predicts a single value at the specified index.
    /// </summary>
    /// <param name="x">The input features matrix.</param>
    /// <param name="predictions">The vector of predictions made so far.</param>
    /// <param name="index">The index at which to make a prediction.</param>
    /// <returns>The predicted value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method calculates a single prediction by combining all the components of the model.
    /// 
    /// It works by adding up:
    /// 
    /// 1. AR terms: The effect of previous predictions
    ///    - Example: If sales yesterday were high, they might be high today too
    /// 
    /// 2. MA terms: The effect of previous prediction errors
    ///    - Example: If we consistently underestimated sales recently, we might adjust upward
    /// 
    /// 3. Input lag terms: The effect of the input variables at different time lags
    ///    - Example: Advertising from 3 days ago might still be boosting today's sales
    /// 
    /// 4. Output lag terms: The effect of additional past output values
    ///    - Example: Sales from two weeks ago might influence today due to customer return cycles
    /// 
    /// The result is a comprehensive prediction that accounts for all these factors.
    /// </para>
    /// </remarks>
    private T PredictSingle(Matrix<T> x, Vector<T> predictions, int index)
    {
                T prediction = NumOps.Zero;

        // Add AR terms
        for (int i = 0; i < _arParameters.Length; i++)
        {
            if (index - i - 1 >= 0)
            {
                prediction = NumOps.Add(prediction, NumOps.Multiply(_arParameters[i], predictions[index - i - 1]));
            }
        }

        // Add MA terms
        for (int i = 0; i < _maParameters.Length; i++)
        {
            if (index - i - 1 >= 0 && _residuals != null)
            {
                prediction = NumOps.Add(prediction, NumOps.Multiply(_maParameters[i], _residuals[index - i - 1]));
            }
        }

        // Add input lag terms
        for (int i = 0; i < _inputLags.Length; i++)
        {
            if (index - i >= 0)
            {
                prediction = NumOps.Add(prediction, NumOps.Multiply(_inputLags[i], x[index - i, 0]));
            }
        }

        // Add output lag terms
        for (int i = 0; i < _outputLags.Length; i++)
        {
            if (index - i - 1 >= 0)
            {
                prediction = NumOps.Add(prediction, NumOps.Multiply(_outputLags[i], _y[index - i - 1]));
            }
        }

        return prediction;
    }

    /// <summary>
    /// Evaluates the performance of the trained model on test data.
    /// </summary>
    /// <param name="xTest">The input features matrix for testing.</param>
    /// <param name="yTest">The actual target values for testing.</param>
    /// <returns>A dictionary containing evaluation metrics.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method tests how well the model performs on data it hasn't seen during training.
    /// 
    /// It works by:
    /// 1. Using the model to make predictions on the test data
    /// 2. Comparing these predictions to the actual test values
    /// 3. Calculating various error metrics to quantify the accuracy
    /// 
    /// The metrics calculated include:
    /// 
    /// - MAE (Mean Absolute Error): The average absolute difference between predictions and actual values
    ///   - Lower is better
    ///   - Easy to interpret: "On average, predictions are off by X units"
    /// 
    /// - MSE (Mean Squared Error): The average squared difference
    ///   - Lower is better
    ///   - Penalizes large errors more heavily
    /// 
    /// - RMSE (Root Mean Squared Error): The square root of MSE
    ///   - Lower is better
    ///   - In the same units as the original data
    /// 
    /// - R� (R-squared): The proportion of variance explained by the model
    ///   - Ranges from 0 to 1 (higher is better)
    ///   - 0.7 means the model explains 70% of the variation in the data
    /// 
    /// These metrics help you understand how accurate your model is and compare it to other models.
    /// </para>
    /// </remarks>
    public override Dictionary<string, T> EvaluateModel(Matrix<T> xTest, Vector<T> yTest)
    {
        Vector<T> predictions = Predict(xTest);
        Dictionary<string, T> metrics = new Dictionary<string, T>();

        // Mean Absolute Error (MAE)
        metrics["MAE"] = StatisticsHelper<T>.CalculateMeanAbsoluteError(yTest, predictions);

        // Mean Squared Error (MSE)
        metrics["MSE"] = StatisticsHelper<T>.CalculateMeanSquaredError(yTest, predictions);

        // Root Mean Squared Error (RMSE)
        metrics["RMSE"] = StatisticsHelper<T>.CalculateRootMeanSquaredError(yTest, predictions);

        // R-squared (R2)
        metrics["R2"] = StatisticsHelper<T>.CalculateR2(yTest, predictions);

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
    /// 
    /// This method saves:
    /// - The AR parameters (how past output values affect current values)
    /// - The MA parameters (how past errors affect current values)
    /// - The input lag parameters (how past input values affect current output)
    /// - The output lag parameters (how additional past output values affect current output)
    /// - The model configuration options (AR order, MA order, etc.)
    /// 
    /// After serializing, you can store the model and later deserialize it to make predictions
    /// without repeating the training process, which can save significant time for complex models.
    /// </para>
    /// </remarks>
    protected override void SerializeCore(BinaryWriter writer)
    {
        // Write model parameters
        SerializationHelper<T>.SerializeVector(writer, _arParameters);
        SerializationHelper<T>.SerializeVector(writer, _maParameters);
        SerializationHelper<T>.SerializeVector(writer, _inputLags);
        SerializationHelper<T>.SerializeVector(writer, _outputLags);

        // Write options
        writer.Write(_tfOptions.AROrder);
        writer.Write(_tfOptions.MAOrder);
        writer.Write(_tfOptions.InputLagOrder);
        writer.Write(_tfOptions.OutputLagOrder);
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
    /// 
    /// It reads:
    /// - The AR parameters
    /// - The MA parameters
    /// - The input lag parameters
    /// - The output lag parameters
    /// - The model configuration options
    /// 
    /// After deserialization, the model is ready to make predictions without needing to be retrained.
    /// This is particularly useful for:
    /// - Deploying models to production environments
    /// - Sharing models between different applications
    /// - Saving computation time by not having to retrain complex models
    /// </para>
    /// </remarks>
    protected override void DeserializeCore(BinaryReader reader)
    {
        // Read model parameters
        _arParameters = SerializationHelper<T>.DeserializeVector(reader);
        _maParameters = SerializationHelper<T>.DeserializeVector(reader);
        _inputLags = SerializationHelper<T>.DeserializeVector(reader);
        _outputLags = SerializationHelper<T>.DeserializeVector(reader);

        // Read options
        _tfOptions.AROrder = reader.ReadInt32();
        _tfOptions.MAOrder = reader.ReadInt32();
        _tfOptions.InputLagOrder = reader.ReadInt32();
        _tfOptions.OutputLagOrder = reader.ReadInt32();
    }
}