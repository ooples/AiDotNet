namespace AiDotNet.TimeSeries;

/// <summary>
/// Represents a model that analyzes and forecasts time series data with interventions or structural changes.
/// </summary>
/// <remarks>
/// <para>
/// Intervention analysis combines ARIMA (AutoRegressive Integrated Moving Average) modeling with the ability to 
/// account for external events or interventions that cause structural changes in the time series. These interventions 
/// can be temporary or permanent and can have various effects on the level, trend, or seasonality of the data.
/// </para>
/// <para><b>For Beginners:</b> Intervention analysis helps understand how specific events affect your data patterns.
/// 
/// Think of it like analyzing sales data:
/// - You've been tracking monthly sales that follow a regular pattern
/// - Then you run a major marketing campaign in July
/// - Sales jump significantly and stay higher for several months
/// 
/// This model helps you:
/// - Measure exactly how much the marketing campaign boosted sales
/// - Understand how long the effect lasted
/// - Make better predictions by accounting for these special events
/// 
/// Other examples of interventions include policy changes, natural disasters, product launches,
/// or any significant event that changes the normal pattern of your data.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class InterventionAnalysisModel<T> : TimeSeriesModelBase<T>
{
    /// <summary>
    /// The configuration options for the intervention analysis model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Contains the settings for the model, such as the orders of the AR and MA components,
    /// details about the interventions to consider, and the optimizer to use for parameter estimation.
    /// </para>
    /// <para><b>For Beginners:</b> This holds all the settings that define how the model works.
    /// 
    /// These options include:
    /// - How many past values to consider (AR order)
    /// - How many past errors to consider (MA order)
    /// - Information about when interventions occurred
    /// - How to find the best parameters for the model
    /// 
    /// It's like the recipe that tells the model how to analyze your data.
    /// </para>
    /// </remarks>
    private readonly InterventionAnalysisOptions<T> _iaOptions;

    /// <summary>
    /// The autoregressive (AR) parameters of the model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// These parameters determine how much the forecast depends on previous values in the time series.
    /// The number of parameters is determined by the AR order specified in the model options.
    /// </para>
    /// <para><b>For Beginners:</b> These determine how past values influence future predictions.
    /// 
    /// Autoregressive parameters:
    /// - Show how strongly each previous time period affects the current one
    /// - Higher values mean stronger influence from that particular lag
    /// - For example, if monthly sales are heavily influenced by what happened 
    ///   1 month ago, but less so by what happened 2 months ago
    /// 
    /// Think of these as "memory weights" - how strongly the model remembers
    /// and uses each past time period.
    /// </para>
    /// </remarks>
    private Vector<T> _arParameters;

    /// <summary>
    /// The moving average (MA) parameters of the model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// These parameters determine how much the forecast depends on previous forecast errors.
    /// The number of parameters is determined by the MA order specified in the model options.
    /// </para>
    /// <para><b>For Beginners:</b> These determine how past prediction errors influence future predictions.
    /// 
    /// Moving average parameters:
    /// - Show how strongly each previous prediction error affects the current prediction
    /// - Help the model correct itself based on recent mistakes
    /// - For example, if the model consistently under-predicted sales last month,
    ///   it will adjust this month's prediction upward
    /// 
    /// Think of these as "error correction weights" - how strongly the model
    /// adjusts based on each past prediction error.
    /// </para>
    /// </remarks>
    private Vector<T> _maParameters;

    /// <summary>
    /// The effects of interventions on the time series.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This list contains information about how each intervention affects the time series, including
    /// the start time, duration, and magnitude of the effect. These effects are estimated during model training.
    /// </para>
    /// <para><b>For Beginners:</b> These measure how much each special event affected your data.
    /// 
    /// Intervention effects:
    /// - Capture the size of the impact from each intervention
    /// - Record when the intervention started and how long it lasted
    /// - Allow the model to account for these special events when forecasting
    /// 
    /// For example, a marketing campaign might increase sales by 20% for two months,
    /// or a natural disaster might decrease production by 15% for three weeks.
    /// </para>
    /// </remarks>
    private List<InterventionEffect<T>> _interventionEffects;

    /// <summary>
    /// The residuals (errors) of the model predictions on the training data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// These are the differences between the actual observations and the model's predictions.
    /// Residuals are used both for model evaluation and as inputs to the moving average component.
    /// </para>
    /// <para><b>For Beginners:</b> These are the prediction errors from the model.
    /// 
    /// Residuals:
    /// - The differences between actual values and predicted values
    /// - Represent the "surprises" or unpredicted components
    /// - Used to evaluate how well the model fits the data
    /// - Also used as inputs for the MA part of the model
    /// 
    /// If residuals are large, the model isn't capturing the patterns well.
    /// If residuals look random (no pattern), the model is doing a good job.
    /// </para>
    /// </remarks>
    private Vector<T> _residuals;

    /// <summary>
    /// The fitted (predicted) values for the training data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// These are the model's predictions for the training data period. They are compared with
    /// the actual values to calculate residuals and evaluate model performance.
    /// </para>
    /// <para><b>For Beginners:</b> These are the model's predictions for past data points.
    /// 
    /// Fitted values:
    /// - The predictions made by the model for the same time periods we have actual data for
    /// - Used to see how well the model captures historical patterns
    /// - Help calculate how much error is in the model's predictions
    /// 
    /// By comparing fitted values to actual values, we can see how well
    /// the model would have predicted history if we had used it in the past.
    /// </para>
    /// </remarks>
    private Vector<T> _fitted;

    /// <summary>
    /// The optimizer used to find the best model parameters.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This component is responsible for finding the optimal values for the model parameters
    /// (AR, MA, and intervention effects) by minimizing the prediction error on the training data.
    /// </para>
    /// <para><b>For Beginners:</b> This is the tool that finds the best parameter values.
    /// 
    /// The optimizer:
    /// - Tries different combinations of parameter values
    /// - Measures how well each combination performs
    /// - Adjusts parameters to improve performance
    /// - Keeps going until it finds the best possible set of values
    /// 
    /// Think of it like tuning a radio dial to find the clearest signal.
    /// The optimizer automatically turns the dials until the reception is optimal.
    /// </para>
    /// </remarks>
    private readonly IOptimizer<T> _optimizer;

    /// <summary>
    /// The target values (observed time series data) used for training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This vector contains the original time series values used for training the model.
    /// It is stored for use in various calculations during and after training.
    /// </para>
    /// <para><b>For Beginners:</b> This is your original historical data.
    /// 
    /// The target values:
    /// - The actual observed values in your time series
    /// - What the model is trying to learn from and predict
    /// - For example, monthly sales figures, daily temperature readings, etc.
    /// 
    /// This is the ground truth that the model compares its predictions against.
    /// </para>
    /// </remarks>
    private Vector<T> _y;

    /// <summary>
    /// Initializes a new instance of the <see cref="InterventionAnalysisModel{T}"/> class with the specified options.
    /// </summary>
    /// <param name="options">The configuration options for the intervention analysis model. If null, default options are used.</param>
    /// <remarks>
    /// <para>
    /// This constructor initializes the intervention analysis model with the provided configuration options or default options if none
    /// are specified. The options determine parameters such as the AR and MA orders, the interventions to consider,
    /// and the optimizer to use for parameter estimation.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up your model with your chosen settings.
    /// 
    /// When creating the model, you can specify:
    /// - How many past values to use in predictions (AR order)
    /// - How many past errors to consider (MA order)
    /// - When interventions or special events occurred
    /// - What method to use to find the best parameters
    /// 
    /// If you don't provide options, the model uses sensible defaults.
    /// </para>
    /// </remarks>
    public InterventionAnalysisModel(InterventionAnalysisOptions<T>? options = null) : base(options ?? new())
    {
        _iaOptions = options ?? new InterventionAnalysisOptions<T>();
        _optimizer = _iaOptions.Optimizer ?? new LBFGSOptimizer<T>();
        _interventionEffects = [];
        _arParameters = Vector<T>.Empty();
        _maParameters = Vector<T>.Empty();
        _residuals = Vector<T>.Empty();
        _fitted = Vector<T>.Empty();
        _y = Vector<T>.Empty();
    }

    /// <summary>
    /// Trains the intervention analysis model on the provided input and output data.
    /// </summary>
    /// <param name="x">The input features matrix (typically time indicators or related variables).</param>
    /// <param name="y">The target values vector (the time series data to forecast).</param>
    /// <exception cref="ArgumentException">Thrown when the input matrix rows do not match the output vector length.</exception>
    /// <remarks>
    /// <para>
    /// This method trains the intervention analysis model by initializing parameters with random values,
    /// optimizing these parameters to minimize prediction error, and then computing residuals. The
    /// training process accounts for both the time series patterns and the effects of interventions.
    /// </para>
    /// <para><b>For Beginners:</b> This teaches the model patterns in your data, including the effects of special events.
    /// 
    /// The training process follows these steps:
    /// 1. Check that your input data is formatted correctly
    /// 2. Set initial parameter values
    /// 3. Find the best parameters by testing different combinations
    /// 4. Calculate how well the model fits your historical data
    /// 
    /// After training, the model will have learned:
    /// - The regular patterns in your data
    /// - How much each intervention affected the data
    /// - How to make forecasts that account for both
    /// </para>
    /// </remarks>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        if (x.Rows != y.Length)
        {
            throw new ArgumentException("Input matrix rows must match output vector length.");
        }

        _y = y;

        InitializeParameters();
        OptimizeParameters(x, _y);
        ComputeResiduals(x, _y);
    }

    /// <summary>
    /// Initializes the model parameters with small random values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method initializes the AR and MA parameters and the intervention effects with small random values
    /// before optimization. This provides a starting point for the optimization process.
    /// </para>
    /// <para><b>For Beginners:</b> This sets starting values for the model before fine-tuning.
    /// 
    /// The method:
    /// - Creates arrays of the right size for AR and MA parameters
    /// - Fills them with small random values
    /// - Sets up initial estimates for how much each intervention affected the data
    /// 
    /// These random starting values give the optimization process a place to begin.
    /// They're like initial guesses that will be refined during training.
    /// </para>
    /// </remarks>
    private void InitializeParameters()
    {
        int p = _iaOptions.AROrder;
        int q = _iaOptions.MAOrder;

        _arParameters = new Vector<T>(p);
        _maParameters = new Vector<T>(q);

        // Initialize with small random values
        Random rand = new();
        for (int i = 0; i < p; i++) _arParameters[i] = NumOps.FromDouble(rand.NextDouble() * 0.1);
        for (int i = 0; i < q; i++) _maParameters[i] = NumOps.FromDouble(rand.NextDouble() * 0.1);

        // Initialize intervention effects
        foreach (var intervention in _iaOptions.Interventions)
        {
            _interventionEffects.Add(new InterventionEffect<T>
            {
                StartIndex = intervention.StartIndex,
                Duration = intervention.Duration,
                Effect = rand.NextDouble() * 0.1
            });
        }
    }

    /// <summary>
    /// Optimizes the model parameters to minimize prediction error.
    /// </summary>
    /// <param name="x">The input features matrix.</param>
    /// <param name="y">The target values vector.</param>
    /// <remarks>
    /// <para>
    /// This method uses the specified optimizer to find the optimal values for the AR parameters,
    /// MA parameters, and intervention effects that minimize the prediction error on the training data.
    /// </para>
    /// <para><b>For Beginners:</b> This finds the best settings for your model.
    /// 
    /// The optimization process:
    /// - Packages your data for the optimizer
    /// - Lets the optimizer try different parameter values
    /// - Finds the combination that best explains your historical data
    /// - Updates the model with these optimal values
    /// 
    /// It's like a chef adjusting a recipe through trial and error
    /// until they find the perfect balance of ingredients.
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
    /// This method updates the model's AR parameters, MA parameters, and intervention effects
    /// with the optimized values found by the optimizer. The parameters are extracted from
    /// the single vector of optimized values in the correct order.
    /// </para>
    /// <para><b>For Beginners:</b> This applies the best parameter values found by the optimizer.
    /// 
    /// The method:
    /// - Takes the optimized values from the optimizer
    /// - Separates them into the different parameter types
    /// - Updates the AR parameters (which capture patterns in the values)
    /// - Updates the MA parameters (which capture patterns in the errors)
    /// - Updates the intervention effects (which measure the impact of special events)
    /// 
    /// It's like sorting groceries after shopping - putting each item in its proper place
    /// so they can be used effectively.
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

        // Update intervention effects
        for (int i = 0; i < _interventionEffects.Count; i++)
        {
            _interventionEffects[i].Effect = Convert.ToDouble(optimizedParameters[paramIndex++]);
        }
    }

    /// <summary>
    /// Computes the residuals between the actual and predicted values.
    /// </summary>
    /// <param name="x">The input features matrix.</param>
    /// <param name="y">The target values vector.</param>
    /// <remarks>
    /// <para>
    /// This method calculates the differences between the actual time series values and the model's predictions.
    /// These residuals are used for model evaluation and as inputs to the moving average component during prediction.
    /// </para>
    /// <para><b>For Beginners:</b> This calculates how far off the model's predictions are from actual values.
    /// 
    /// The method:
    /// - Gets predictions for the historical data points
    /// - Compares each prediction to the actual value
    /// - Calculates the difference (the error or residual)
    /// - Stores these errors for later use
    /// 
    /// Residuals help evaluate how well the model fits the data and are also
    /// used as inputs for future predictions (in the MA component).
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
    /// Generates predictions for the given input data.
    /// </summary>
    /// <param name="input">The input features matrix.</param>
    /// <returns>A vector containing the predicted values.</returns>
    /// <remarks>
    /// <para>
    /// This method generates forecasts for each time point in the input matrix. It calls PredictSingle
    /// for each row in the input matrix, which applies the AR and MA terms and considers intervention effects.
    /// </para>
    /// <para><b>For Beginners:</b> This makes predictions for each time point in your data.
    /// 
    /// The prediction process:
    /// - Takes the input features for each time point
    /// - Makes a separate prediction for each time point
    /// - Each prediction takes into account:
    ///   - Past values (AR component)
    ///   - Past errors (MA component)
    ///   - Any interventions active at that time point
    /// 
    /// The result is a forecast that considers both the natural patterns in your data
    /// and the effects of special events or interventions.
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> input)
    {
        int n = input.Rows;
        Vector<T> predictions = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            predictions[i] = PredictSingle(predictions, i);
        }

        return predictions;
    }

    /// <summary>
    /// Predicts a single value at the specified index.
    /// </summary>
    /// <param name="predictions">The vector of predictions made so far.</param>
    /// <param name="index">The index to predict.</param>
    /// <returns>The predicted value for the specified index.</returns>
    /// <remarks>
    /// <para>
    /// This method predicts a single value at the specified index by applying the AR and MA terms
    /// and accounting for any active interventions. It uses previously computed predictions and residuals
    /// as needed for the AR and MA components.
    /// </para>
    /// <para><b>For Beginners:</b> This generates a prediction for a single time point.
    /// 
    /// The method builds the prediction in three parts:
    /// 1. Autoregressive (AR) part: Uses past predicted values
    ///    - For example, if current sales depend on recent sales trends
    /// 
    /// 2. Moving Average (MA) part: Uses past prediction errors
    ///    - For example, if the model consistently under-predicted recently,
    ///      it will adjust the new prediction upward
    /// 
    /// 3. Intervention effects: Adds the impact of any active interventions
    ///    - For example, if a marketing campaign is active at this time point,
    ///      it adds the estimated sales boost from that campaign
    /// 
    /// The final prediction combines all these components.
    /// </para>
    /// </remarks>
    private T PredictSingle(Vector<T> predictions, int index)
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

        // Add intervention effects
        foreach (var effect in _interventionEffects)
        {
            if (index >= effect.StartIndex && (effect.Duration == 0 || index < effect.StartIndex + effect.Duration))
            {
                prediction = NumOps.Add(prediction, NumOps.FromDouble(effect.Effect));
            }
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
    /// various error metrics. The returned metrics include Mean Absolute Error (MAE), Mean Squared Error (MSE),
    /// Root Mean Squared Error (RMSE), and R-squared (R2).
    /// </para>
    /// <para><b>For Beginners:</b> This measures how accurate the model's predictions are.
    /// 
    /// The evaluation:
    /// - Makes predictions for data the model hasn't seen before
    /// - Compares these predictions to the actual values
    /// - Calculates different types of error measurements:
    ///   - MAE (Mean Absolute Error): Average of absolute differences
    ///   - MSE (Mean Squared Error): Average of squared differences
    ///   - RMSE (Root Mean Squared Error): Square root of MSE, in the same units as your data
    ///   - R2 (R-squared): How much of the variation in the data is explained by the model (0-1)
    /// 
    /// Lower values of MAE, MSE, and RMSE indicate better performance.
    /// Higher values of R2 (closer to 1) indicate better performance.
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
    /// Gets a dictionary of the estimated intervention effects.
    /// </summary>
    /// <returns>A dictionary mapping intervention identifiers to their estimated effects.</returns>
    /// <remarks>
    /// <para>
    /// This method returns a dictionary containing the estimated effects of each intervention considered by the model.
    /// The keys are identifiers for the interventions, and the values are the estimated magnitudes of their effects.
    /// </para>
    /// <para><b>For Beginners:</b> This shows how much each special event affected your data.
    /// 
    /// The method:
    /// - Returns a list of all interventions the model knows about
    /// - For each intervention, shows:
    ///   - When it started
    ///   - How long it lasted
    ///   - How much it affected the data (its effect size)
    /// 
    /// This helps you understand:
    /// - Which interventions had the biggest impact
    /// - Whether an intervention had a positive or negative effect
    /// - The relative importance of different interventions
    /// 
    /// For example, you might learn that a marketing campaign increased sales by 15%,
    /// while a website redesign increased sales by only 3%.
    /// </para>
    /// </remarks>
    public Dictionary<string, double> GetInterventionEffects()
    {
        return _interventionEffects.ToDictionary(
            effect => $"Intervention_{effect.StartIndex}_{effect.Duration}",
            effect => effect.Effect
        );
    }

    /// <summary>
    /// Serializes the model's core parameters to a binary writer.
    /// </summary>
    /// <param name="writer">The binary writer to write to.</param>
    /// <remarks>
    /// <para>
    /// This method writes the model's essential parameters to a binary stream, allowing the model to be saved
    /// to a file or database. The serialized parameters include the AR and MA parameters, intervention effects,
    /// and model options.
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
        // Write model parameters
        SerializationHelper<T>.SerializeVector(writer, _arParameters);
        SerializationHelper<T>.SerializeVector(writer, _maParameters);
            
        // Write intervention effects
        writer.Write(_interventionEffects.Count);
        foreach (var effect in _interventionEffects)
        {
            writer.Write(effect.StartIndex);
            writer.Write(effect.Duration);
            writer.Write(Convert.ToDouble(effect.Effect));
        }

        // Write options
        writer.Write(_iaOptions.AROrder);
        writer.Write(_iaOptions.MAOrder);
    }

    /// <summary>
    /// Deserializes the model's core parameters from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader to read from.</param>
    /// <remarks>
    /// <para>
    /// This method reads the model's essential parameters from a binary stream, allowing a previously saved model
    /// to be loaded from a file or database. The deserialized parameters include the AR and MA parameters,
    /// intervention effects, and model options.
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
        // Read model parameters
        _arParameters = SerializationHelper<T>.DeserializeVector(reader);
        _maParameters = SerializationHelper<T>.DeserializeVector(reader);

        // Read intervention effects
        int effectCount = reader.ReadInt32();
        _interventionEffects = new List<InterventionEffect<T>>();
        for (int i = 0; i < effectCount; i++)
        {
            _interventionEffects.Add(new InterventionEffect<T>
            {
                StartIndex = reader.ReadInt32(),
                Duration = reader.ReadInt32(),
                Effect = reader.ReadDouble()
            });
        }

        // Read options
        _iaOptions.AROrder = reader.ReadInt32();
        _iaOptions.MAOrder = reader.ReadInt32();
    }
}