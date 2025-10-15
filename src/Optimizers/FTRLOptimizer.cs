namespace AiDotNet.Optimizers;

/// <summary>
/// Represents a Follow The Regularized Leader (FTRL) optimizer for machine learning models.
/// </summary>
/// <remarks>
/// <para>
/// The FTRL optimizer is an online learning algorithm that adapts regularization in a per-coordinate fashion.
/// It's particularly effective for sparse datasets and is widely used in click-through rate (CTR) prediction
/// and other online learning scenarios.
/// </para>
/// <para><b>For Beginners:</b> FTRL is an advanced optimization technique that's good at handling large-scale,
/// sparse data. It's often used in online advertising and recommendation systems.
/// 
/// Think of FTRL like a smart learning system that:
/// - Adjusts its learning speed for each feature independently
/// - Can handle situations where most features are zero (sparse data)
/// - Is good at balancing between finding a good solution and not overfitting
/// 
/// For example, in an online advertising system, FTRL might:
/// - Quickly learn which ad categories are important for a user
/// - Ignore or learn slowly from features that rarely appear
/// - Automatically adjust how much it learns from each new piece of data
/// 
/// This makes FTRL particularly good for systems that need to learn and predict in real-time with lots of data.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class FTRLOptimizer<T, TInput, TOutput> : GradientBasedOptimizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// The options specific to the FTRL algorithm.
    /// </summary>
    private FTRLOptimizerOptions<T, TInput, TOutput> _options = default!;

    /// <summary>
    /// Auxiliary vector used in the FTRL update rule.
    /// </summary>
    private Vector<T>? _z;

    /// <summary>
    /// Vector<double> of accumulated squared gradients.
    /// </summary>
    private Vector<T>? _n;

    /// <summary>
    /// The current time step or iteration count.
    /// </summary>
    private int _t;

    /// <summary>
    /// Initializes a new instance of the FTRLOptimizer class.
    /// </summary>
    /// <param name="model">The model to be optimized.</param>
    /// <param name="options">The options for configuring the FTRL algorithm.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This constructor sets up the FTRL optimizer with its initial configuration.
    /// You provide the model to optimize and can customize various aspects of how the optimizer works,
    /// or use default settings if you're unsure.
    /// </para>
    /// </remarks>
    public FTRLOptimizer(
        IFullModel<T, TInput, TOutput> model,
        FTRLOptimizerOptions<T, TInput, TOutput>? options = null)
        : base(model, options ?? new())
    {
        _options = options ?? new FTRLOptimizerOptions<T, TInput, TOutput>();
        _z = null;
        _n = null;
        _t = 0;

        InitializeAdaptiveParameters();
    }

    /// <summary>
    /// Initializes the adaptive parameters used in the FTRL algorithm.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method sets up the initial learning rate and resets the time step.
    /// The learning rate determines how big the steps are when the algorithm is searching for the best solution.
    /// </para>
    /// </remarks>
    protected override void InitializeAdaptiveParameters()
    {
        base.InitializeAdaptiveParameters();

        CurrentLearningRate = NumOps.FromDouble(_options.Alpha);
        _t = 0;

        // Initialize _z and _n if Model is available
        if (Model != null)
        {
            var parameterLength = Model.GetParameters().Length;
            _z = new Vector<T>(parameterLength);
            _n = new Vector<T>(parameterLength);
            for (int i = 0; i < parameterLength; i++)
            {
                _n[i] = NumOps.Zero;
                _z[i] = NumOps.Zero;
            }
        }
    }

    /// <summary>
    /// Performs the main optimization process using the FTRL algorithm.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the heart of the FTRL algorithm. It starts with a random solution
    /// and iteratively improves it. In each iteration, it:
    /// 1. Calculates how to improve the current solution (the gradient)
    /// 2. Updates the solution based on this calculation
    /// 3. Checks if this new solution is the best one found so far
    /// 4. Decides whether to stop or continue improving
    /// 
    /// It's like a climber trying to find the highest peak, taking steps based on the slope they feel under their feet.
    /// </para>
    /// </remarks>
    /// <param name="inputData">The input data for the optimization process.</param>
    /// <returns>The result of the optimization process.</returns>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        ValidationHelper<T>.ValidateInputData(inputData);

        var currentSolution = Model.DeepCopy();
        var bestStepData = new OptimizationStepData<T, TInput, TOutput>
        {
            Solution = Model.DeepCopy(),
            FitnessScore = FitnessCalculator.IsHigherScoreBetter ? NumOps.MinValue : NumOps.MaxValue
        };
        var previousStepData = new OptimizationStepData<T, TInput, TOutput>();

        // _z and _n are now initialized in InitializeAdaptiveParameters
        // Make sure adaptive parameters are initialized
        InitializeAdaptiveParameters();

        for (int iteration = 0; iteration < _options.MaxIterations; iteration++)
        {
            _t++;
            var gradient = CalculateGradient(currentSolution, inputData.XTrain, inputData.YTrain);
            var newSolution = UpdateSolution(currentSolution, gradient);

            var currentStepData = EvaluateSolution(newSolution, inputData);
            UpdateBestSolution(currentStepData, ref bestStepData);

            UpdateAdaptiveParameters(currentStepData, previousStepData);

            if (UpdateIterationHistoryAndCheckEarlyStopping(iteration, bestStepData))
            {
                break;
            }

            if (NumOps.LessThan(NumOps.Abs(NumOps.Subtract(bestStepData.FitnessScore, currentStepData.FitnessScore)), NumOps.FromDouble(_options.Tolerance)))
            {
                break;
            }

            currentSolution = newSolution;
            previousStepData = currentStepData;
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    /// <summary>
    /// Updates the current solution using the FTRL update rule.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method applies the FTRL algorithm's specific way of updating the solution.
    /// It's like adjusting the recipe based on how well the last batch of cookies turned out, but in a very
    /// sophisticated way that considers the history of all previous batches.
    /// </para>
    /// </remarks>
    /// <param name="currentSolution">The current solution.</param>
    /// <param name="gradient">The calculated gradient.</param>
    /// <returns>The updated solution.</returns>
    protected override IFullModel<T, TInput, TOutput> UpdateSolution(IFullModel<T, TInput, TOutput> currentSolution, Vector<T> gradient)
    {
        var parameters = currentSolution.GetParameters();
        var newCoefficients = new Vector<T>(parameters.Length);
        var alpha = NumOps.FromDouble(_options.Alpha);
        var beta = NumOps.FromDouble(_options.Beta);
        var lambda1 = NumOps.FromDouble(_options.Lambda1);
        var lambda2 = NumOps.FromDouble(_options.Lambda2);

        for (int i = 0; i < parameters.Length; i++)
        {
            var sigma = NumOps.Divide(
                NumOps.Subtract(NumOps.Sqrt(NumOps.Add(_n![i], NumOps.Multiply(gradient[i], gradient[i]))), NumOps.Sqrt(_n[i])),
                alpha
            );
            _z![i] = NumOps.Add(_z[i], NumOps.Subtract(gradient[i], NumOps.Multiply(sigma, parameters[i])));
            _n![i] = NumOps.Add(_n[i], NumOps.Multiply(gradient[i], gradient[i]));

            var sign = NumOps.SignOrZero(_z[i]);
            if (NumOps.GreaterThan(NumOps.Abs(_z[i]), lambda1))
            {
                newCoefficients[i] = NumOps.Divide(
                    NumOps.Multiply(
                        NumOps.Subtract(lambda1, _z[i]),
                        sign
                    ),
                    NumOps.Add(
                        NumOps.Multiply(lambda2, NumOps.FromDouble(1 + _options.Beta)),
                        NumOps.Divide(
                            NumOps.Sqrt(_n[i]),
                            alpha
                        )
                    )
                );
            }
            else
            {
                newCoefficients[i] = NumOps.FromDouble(0);
            }
        }

        return currentSolution.WithParameters(newCoefficients);
    }

    /// <summary>
    /// Updates the adaptive parameters based on the optimization progress.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method adjusts how fast the algorithm learns based on its recent performance.
    /// If it's improving, it might speed up. If it's not improving, it might slow down to be more careful.
    /// It's like adjusting your study pace based on how well you're understanding the material.
    /// </para>
    /// </remarks>
    /// <param name="currentStepData">Data from the current optimization step.</param>
    /// <param name="previousStepData">Data from the previous optimization step.</param>
    protected override void UpdateAdaptiveParameters(OptimizationStepData<T, TInput, TOutput> currentStepData, OptimizationStepData<T, TInput, TOutput> previousStepData)
    {
        base.UpdateAdaptiveParameters(currentStepData, previousStepData);

        // Skip if previous step data is null (first iteration)
        if (previousStepData.Solution == null)
            return;

        if (_options.UseAdaptiveLearningRate)
        {
            if (NumOps.GreaterThan(currentStepData.FitnessScore, previousStepData.FitnessScore))
            {
                CurrentLearningRate = NumOps.Multiply(CurrentLearningRate, NumOps.FromDouble(_options.LearningRateIncreaseFactor));
            }
            else
            {
                CurrentLearningRate = NumOps.Multiply(CurrentLearningRate, NumOps.FromDouble(_options.LearningRateDecreaseFactor));
            }

            CurrentLearningRate = MathHelper.Clamp(CurrentLearningRate,
                NumOps.FromDouble(_options.MinLearningRate),
                NumOps.FromDouble(_options.MaxLearningRate));
        }
    }

    /// <summary>
    /// Updates the options for the FTRL optimizer.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method allows you to change the settings of the optimizer while it's running.
    /// It's like adjusting the controls on a machine while it's operating. This can be useful if you want to
    /// fine-tune the optimizer's behavior based on its performance or other factors.
    /// </para>
    /// </remarks>
    /// <param name="options">The new options to be set.</param>
    /// <exception cref="ArgumentException">Thrown when the provided options are not of type FTRLOptimizerOptions.</exception>
    protected override void UpdateOptions(OptimizationAlgorithmOptions<T, TInput, TOutput> options)
    {
        if (options is FTRLOptimizerOptions<T, TInput, TOutput> ftrlOptions)
        {
            _options = ftrlOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected FTRLOptimizerOptions.");
        }
    }

    /// <summary>
    /// Retrieves the current options of the FTRL optimizer.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method lets you see what settings the optimizer is currently using.
    /// It's like checking the current settings on a machine to understand how it's configured.
    /// </para>
    /// </remarks>
    /// <returns>The current optimization algorithm options.</returns>
    public override OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions()
    {
        return _options;
    }

    /// <summary>
    /// Serializes the FTRL optimizer to a byte array.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method converts the current state of the optimizer into a format
    /// that can be easily stored or transmitted. It's like taking a snapshot of the optimizer's memory,
    /// including all its settings and learned information, so you can save it or send it somewhere else.
    /// </para>
    /// </remarks>
    /// <returns>A byte array representing the serialized state of the optimizer.</returns>
    public override byte[] Serialize()
    {
        using (MemoryStream ms = new MemoryStream())
        using (BinaryWriter writer = new BinaryWriter(ms))
        {
            byte[] baseData = base.Serialize();
            writer.Write(baseData.Length);
            writer.Write(baseData);

            string optionsJson = JsonConvert.SerializeObject(_options);
            writer.Write(optionsJson);

            writer.Write(_t);

            // Serialize _z if it exists
            if (_z != null)
            {
                byte[] zData = _z.Serialize();
                writer.Write(zData.Length);
                writer.Write(zData);
            }
            else
            {
                writer.Write(0);
            }

            // Serialize _n if it exists
            if (_n != null)
            {
                byte[] nData = _n.Serialize();
                writer.Write(nData.Length);
                writer.Write(nData);
            }
            else
            {
                writer.Write(0);
            }

            return ms.ToArray();
        }
    }

    /// <summary>
    /// Deserializes the FTRL optimizer from a byte array.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method takes a previously serialized optimizer state and
    /// reconstructs the optimizer from it. It's like restoring the optimizer's memory from a saved snapshot,
    /// allowing you to continue from where you left off or use a pre-trained optimizer.
    /// </para>
    /// </remarks>
    /// <param name="data">The byte array containing the serialized optimizer state.</param>
    /// <exception cref="InvalidOperationException">Thrown when deserialization of optimizer options fails.</exception>
    public override void Deserialize(byte[] data)
    {
        using (MemoryStream ms = new MemoryStream(data))
        using (BinaryReader reader = new BinaryReader(ms))
        {
            int baseDataLength = reader.ReadInt32();
            byte[] baseData = reader.ReadBytes(baseDataLength);
            base.Deserialize(baseData);

            string optionsJson = reader.ReadString();
            _options = JsonConvert.DeserializeObject<FTRLOptimizerOptions<T, TInput, TOutput>>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");

            _t = reader.ReadInt32();

            // Deserialize _z if it exists
            int zLength = reader.ReadInt32();
            if (zLength > 0)
            {
                byte[] zData = reader.ReadBytes(zLength);
                _z = Vector<T>.Deserialize(zData);
            }

            // Deserialize _n if it exists
            int nLength = reader.ReadInt32();
            if (nLength > 0)
            {
                byte[] nData = reader.ReadBytes(nLength);
                _n = Vector<T>.Deserialize(nData);
            }
        }
    }

    /// <summary>
    /// Generates a unique key for caching gradients.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method creates a unique identifier for storing and retrieving
    /// calculated gradients. It's like creating a label for a file drawer where you store specific calculations.
    /// This helps the optimizer avoid redoing calculations it has already performed, making it more efficient.
    /// </para>
    /// </remarks>
    /// <param name="model">The current model.</param>
    /// <param name="X">The input data matrix.</param>
    /// <param name="y">The target values vector.</param>
    /// <returns>A string that uniquely identifies the gradient for the given inputs.</returns>
    protected override string GenerateGradientCacheKey(IFullModel<T, TInput, TOutput> model, TInput X, TOutput y)
    {
        var baseKey = base.GenerateGradientCacheKey(model, X, y);
        return $"{baseKey}_FTRL_{_options.Alpha}_{_options.Beta}_{_options.Lambda1}_{_options.Lambda2}_{_t}";
    }
}