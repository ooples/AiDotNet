namespace AiDotNet.Optimizers;

/// <summary>
/// Implements the Broyden-Fletcher-Goldfarb-Shanno (BFGS) optimization algorithm.
/// </summary>
/// <remarks>
/// <para>
/// BFGS is a quasi-Newton method for solving unconstrained nonlinear optimization problems.
/// It approximates the Hessian matrix of second derivatives without explicitly computing it,
/// making it more efficient for high-dimensional problems.
/// </para>
/// <para><b>For Beginners:</b>
/// BFGS is like driving to a destination with a smart navigation system that:
/// 
/// 1. Not only tells you which direction to go (like standard gradient descent)
/// 2. But also how sharp to turn based on its understanding of the road's curvature
/// 3. And continuously updates this understanding as you travel
/// 
/// This makes BFGS especially good at:
/// - Finding optimal solutions faster than simpler methods
/// - Navigating difficult optimization landscapes with ravines and curves
/// - Adjusting its step size automatically based on the local terrain
/// 
/// While it's more complex than basic gradient descent, it often requires fewer iterations
/// to reach a good solution, making it especially useful for expensive-to-evaluate functions.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <typeparam name="TInput">The type of input data for the model.</typeparam>
/// <typeparam name="TOutput">The type of output data for the model.</typeparam>
public class BFGSOptimizer<T, TInput, TOutput> : GradientBasedOptimizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// The options specific to the BFGS optimization algorithm.
    /// </summary>
    private BFGSOptimizerOptions<T, TInput, TOutput> _bfgsOptions = default!;

    /// <summary>
    /// The approximation of the inverse Hessian matrix.
    /// </summary>
    private Matrix<T>? _inverseHessian;

    /// <summary>
    /// The gradient from the previous iteration.
    /// </summary>
    private new Vector<T>? _previousGradient;

    /// <summary>
    /// The parameters from the previous iteration.
    /// </summary>
    private Vector<T>? _previousParameters;

    /// <summary>
    /// The current iteration count.
    /// </summary>
    private int _iteration;

    /// <summary>
    /// Initializes a new instance of the BFGSOptimizer class.
    /// </summary>
    /// <param name="model">The machine learning model to optimize.</param>
    /// <param name="options">The options for configuring the BFGS algorithm.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This constructor sets up the BFGS optimizer with its initial configuration.
    /// You provide the model you want to optimize, and you can customize various aspects of how it works,
    /// or use default settings.
    /// </para>
    /// </remarks>
    public BFGSOptimizer(
        IFullModel<T, TInput, TOutput> model,
        BFGSOptimizerOptions<T, TInput, TOutput>? options = null)
        : base(model, options ?? new BFGSOptimizerOptions<T, TInput, TOutput>())
    {
        _bfgsOptions = options ?? new BFGSOptimizerOptions<T, TInput, TOutput>();
        InitializeAdaptiveParameters();
    }

    /// <summary>
    /// Initializes the adaptive parameters used in the BFGS algorithm.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method sets up the initial state for the optimizer,
    /// including the learning rate and iteration count.
    /// </para>
    /// </remarks>
    protected override void InitializeAdaptiveParameters()
    {
        base.InitializeAdaptiveParameters();
        CurrentLearningRate = NumOps.FromDouble(_bfgsOptions.InitialLearningRate);
        _iteration = 0;
    }

    /// <summary>
    /// Performs the main optimization process using the BFGS algorithm.
    /// </summary>
    /// <param name="inputData">The input data for the optimization process.</param>
    /// <returns>The result of the optimization process.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the heart of the BFGS algorithm. It iteratively improves the solution
    /// by updating the parameters based on the gradient and the approximated inverse Hessian matrix.
    /// The process continues until it reaches the maximum number of iterations or meets the convergence criteria.
    /// </para>
    /// </remarks>
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
        var parameters = currentSolution.GetParameters();

        _inverseHessian = Matrix<T>.CreateIdentity(parameters.Length);
        _previousGradient = null;
        _previousParameters = null;
        InitializeAdaptiveParameters();

        for (int iteration = 0; iteration < Options.MaxIterations; iteration++)
        {
            _iteration++;
            var gradient = CalculateGradient(currentSolution, inputData.XTrain, inputData.YTrain);
            var newSolution = UpdateSolution(currentSolution, gradient, inputData);

            var currentStepData = EvaluateSolution(newSolution, inputData);
            UpdateBestSolution(currentStepData, ref bestStepData);

            UpdateAdaptiveParameters(currentStepData, previousStepData);

            if (UpdateIterationHistoryAndCheckEarlyStopping(iteration, bestStepData))
            {
                break;
            }

            if (NumOps.LessThan(NumOps.Abs(NumOps.Subtract(bestStepData.FitnessScore, currentStepData.FitnessScore)), NumOps.FromDouble(_bfgsOptions.Tolerance)))
            {
                break;
            }

            _previousGradient = gradient;
            _previousParameters = parameters;
            parameters = newSolution.GetParameters();
            currentSolution = newSolution;
            previousStepData = currentStepData;
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    /// <summary>
    /// Updates the current solution using the BFGS update formula.
    /// </summary>
    /// <param name="currentSolution">The current solution.</param>
    /// <param name="gradient">The current gradient.</param>
    /// <param name="inputData">The input data for the optimization process.</param>
    /// <returns>The updated solution.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method calculates the next step in the optimization process.
    /// It uses the inverse Hessian approximation to determine the direction and magnitude of the update.
    /// </para>
    /// </remarks>
    private IFullModel<T, TInput, TOutput> UpdateSolution(IFullModel<T, TInput, TOutput> currentSolution, Vector<T> gradient,
        OptimizationInputData<T, TInput, TOutput> inputData)
    {
        var parameters = currentSolution.GetParameters();
        if (_previousGradient != null && _previousParameters != null)
        {
            UpdateInverseHessian(parameters, gradient);
        }

        var direction = _inverseHessian!.Multiply(gradient);
        direction = direction.Transform(x => NumOps.Negate(x));

        var step = LineSearch(currentSolution, direction, gradient, inputData);
        var scaledDirection = direction.Transform(x => NumOps.Multiply(x, step));
        var newCoefficients = parameters.Add(scaledDirection);

        return currentSolution.WithParameters(newCoefficients);
    }

    /// <summary>
    /// Updates the approximation of the inverse Hessian matrix.
    /// </summary>
    /// <param name="currentParameters">The current parameter values.</param>
    /// <param name="currentGradient">The current gradient.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method updates the BFGS algorithm's internal representation of the function's curvature.
    /// It helps the algorithm make more informed decisions about how to update the parameters in future iterations.
    /// </para>
    /// </remarks>
    private void UpdateInverseHessian(Vector<T> currentParameters, Vector<T> currentGradient)
    {
        var s = currentParameters.Subtract(_previousParameters!);
        var y = currentGradient.Subtract(_previousGradient!);

        var rho = NumOps.Divide(NumOps.FromDouble(1), y.DotProduct(s));
        var I = Matrix<T>.CreateIdentity(currentParameters.Length);

        var term1 = I.Subtract(s.OuterProduct(y).Multiply(rho));
        var term2 = I.Subtract(y.OuterProduct(s).Multiply(rho));
        var term3 = s.OuterProduct(s).Multiply(rho);

        _inverseHessian = term1.Multiply(_inverseHessian!).Multiply(term2).Add(term3);
    }

    /// <summary>
    /// Updates the adaptive parameters of the optimizer.
    /// </summary>
    /// <param name="currentStepData">The current step data.</param>
    /// <param name="previousStepData">The previous step data.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method adjusts the learning rate based on the performance of the current step
    /// compared to the previous step. If the current step improved the fitness score, the learning rate is increased;
    /// otherwise, it's decreased. This helps the optimizer adapt to the landscape of the problem.
    /// </para>
    /// </remarks>
    protected override void UpdateAdaptiveParameters(OptimizationStepData<T, TInput, TOutput> currentStepData, OptimizationStepData<T, TInput, TOutput> previousStepData)
    {
        // Skip if previous step data is null (first iteration)
        if (previousStepData.Solution == null)
            return;

        base.UpdateAdaptiveParameters(currentStepData, previousStepData);

        if (_bfgsOptions.UseAdaptiveLearningRate)
        {
            if (NumOps.GreaterThan(currentStepData.FitnessScore, previousStepData.FitnessScore))
            {
                CurrentLearningRate = NumOps.Multiply(CurrentLearningRate, NumOps.FromDouble(_bfgsOptions.LearningRateIncreaseFactor));
            }
            else
            {
                CurrentLearningRate = NumOps.Multiply(CurrentLearningRate, NumOps.FromDouble(_bfgsOptions.LearningRateDecreaseFactor));
            }

            CurrentLearningRate = MathHelper.Clamp(CurrentLearningRate,
                NumOps.FromDouble(_bfgsOptions.MinLearningRate),
                NumOps.FromDouble(_bfgsOptions.MaxLearningRate));
        }
    }

    /// <summary>
    /// Updates the options for the BFGS optimizer.
    /// </summary>
    /// <param name="options">The new options to be set.</param>
    /// <exception cref="ArgumentException">Thrown when the provided options are not of the correct type.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method allows you to change the settings of the BFGS optimizer during runtime.
    /// It checks to make sure you're providing the right kind of options specific to the BFGS algorithm.
    /// </para>
    /// </remarks>
    protected override void UpdateOptions(OptimizationAlgorithmOptions<T, TInput, TOutput> options)
    {
        if (options is BFGSOptimizerOptions<T, TInput, TOutput> bfgsOptions)
        {
            _bfgsOptions = bfgsOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected BFGSOptimizerOptions.");
        }
    }

    /// <summary>
    /// Gets the current options for the BFGS optimizer.
    /// </summary>
    /// <returns>The current optimization options.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method lets you see what settings the BFGS optimizer is currently using.
    /// </para>
    /// </remarks>
    public override OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions()
    {
        return _bfgsOptions;
    }

    /// <summary>
    /// Converts the current state of the BFGS optimizer into a byte array for storage or transmission.
    /// </summary>
    /// <returns>A byte array representing the serialized state of the optimizer.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method takes all the important information about the current state
    /// of the BFGS Optimizer and turns it into a format that can be easily saved or sent to another computer.
    /// It includes both the base optimizer data and BFGS-specific data.
    /// </para>
    /// </remarks>
    public override byte[] Serialize()
    {
        using (MemoryStream ms = new MemoryStream())
        using (BinaryWriter writer = new BinaryWriter(ms))
        {
            byte[] baseData = base.Serialize();
            writer.Write(baseData.Length);
            writer.Write(baseData);

            string optionsJson = JsonConvert.SerializeObject(_bfgsOptions);
            writer.Write(optionsJson);

            writer.Write(_iteration);

            // Serialize _inverseHessian if it exists
            if (_inverseHessian != null)
            {
                writer.Write(true); // Indicates _inverseHessian exists
                writer.Write(_inverseHessian.Rows);
                writer.Write(_inverseHessian.Columns);
                for (int i = 0; i < _inverseHessian.Rows; i++)
                {
                    for (int j = 0; j < _inverseHessian.Columns; j++)
                    {
                        writer.Write(Convert.ToDouble(_inverseHessian[i, j]));
                    }
                }
            }
            else
            {
                writer.Write(false); // Indicates _inverseHessian doesn't exist
            }

            // Serialize _previousGradient if it exists
            if (_previousGradient != null)
            {
                writer.Write(true); // Indicates _previousGradient exists
                writer.Write(_previousGradient.Length);
                for (int i = 0; i < _previousGradient.Length; i++)
                {
                    writer.Write(Convert.ToDouble(_previousGradient[i]));
                }
            }
            else
            {
                writer.Write(false); // Indicates _previousGradient doesn't exist
            }

            // Serialize _previousParameters if it exists
            if (_previousParameters != null)
            {
                writer.Write(true); // Indicates _previousParameters exists
                writer.Write(_previousParameters.Length);
                for (int i = 0; i < _previousParameters.Length; i++)
                {
                    writer.Write(Convert.ToDouble(_previousParameters[i]));
                }
            }
            else
            {
                writer.Write(false); // Indicates _previousParameters doesn't exist
            }

            return ms.ToArray();
        }
    }

    /// <summary>
    /// Restores the state of the BFGS optimizer from a byte array.
    /// </summary>
    /// <param name="data">The byte array containing the serialized state of the optimizer.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method takes a saved state of the BFGS Optimizer (in the form of a byte array)
    /// and uses it to restore the optimizer to that state. It's like loading a saved game, bringing back all the
    /// important settings and progress that were saved earlier.
    /// </para>
    /// </remarks>
    public override void Deserialize(byte[] data)
    {
        using (MemoryStream ms = new MemoryStream(data))
        using (BinaryReader reader = new BinaryReader(ms))
        {
            int baseDataLength = reader.ReadInt32();
            byte[] baseData = reader.ReadBytes(baseDataLength);
            base.Deserialize(baseData);

            string optionsJson = reader.ReadString();
            _bfgsOptions = JsonConvert.DeserializeObject<BFGSOptimizerOptions<T, TInput, TOutput>>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");

            _iteration = reader.ReadInt32();

            // Deserialize _inverseHessian if it exists
            bool inverseHessianExists = reader.ReadBoolean();
            if (inverseHessianExists)
            {
                int rows = reader.ReadInt32();
                int columns = reader.ReadInt32();
                _inverseHessian = new Matrix<T>(rows, columns);
                for (int i = 0; i < rows; i++)
                {
                    for (int j = 0; j < columns; j++)
                    {
                        _inverseHessian[i, j] = NumOps.FromDouble(reader.ReadDouble());
                    }
                }
            }
            else
            {
                _inverseHessian = null;
            }

            // Deserialize _previousGradient if it exists
            bool previousGradientExists = reader.ReadBoolean();
            if (previousGradientExists)
            {
                int length = reader.ReadInt32();
                _previousGradient = new Vector<T>(length);
                for (int i = 0; i < length; i++)
                {
                    _previousGradient[i] = NumOps.FromDouble(reader.ReadDouble());
                }
            }
            else
            {
                _previousGradient = null;
            }

            // Deserialize _previousParameters if it exists
            bool previousParametersExists = reader.ReadBoolean();
            if (previousParametersExists)
            {
                int length = reader.ReadInt32();
                _previousParameters = new Vector<T>(length);
                for (int i = 0; i < length; i++)
                {
                    _previousParameters[i] = NumOps.FromDouble(reader.ReadDouble());
                }
            }
            else
            {
                _previousParameters = null;
            }
        }
    }

    /// <summary>
    /// Generates a unique key for caching gradients in the BFGS optimization process.
    /// </summary>
    /// <param name="model">The current model.</param>
    /// <param name="X">The input data matrix.</param>
    /// <param name="y">The target values vector.</param>
    /// <returns>A string representing the unique cache key.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method creates a unique identifier for storing and retrieving gradients
    /// during the optimization process. It helps avoid recalculating gradients unnecessarily, which can save time.
    /// The key includes BFGS-specific information to ensure it's unique to this optimizer's current state.
    /// </para>
    /// </remarks>
    protected override string GenerateGradientCacheKey(IFullModel<T, TInput, TOutput> model, TInput X, TOutput y)
    {
        var baseKey = base.GenerateGradientCacheKey(model, X, y);
        return $"{baseKey}_BFGS_{_bfgsOptions.InitialLearningRate}_{_bfgsOptions.Tolerance}_{_iteration}";
    }
}