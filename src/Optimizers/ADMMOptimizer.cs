using Newtonsoft.Json;

namespace AiDotNet.Optimizers;

/// <summary>
/// Implements the Alternating Direction Method of Multipliers (ADMM) optimization algorithm.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// ADMM is an algorithm for solving convex optimization problems, particularly useful for large-scale and distributed optimization.
/// It combines the benefits of dual decomposition and augmented Lagrangian methods.
/// </para>
/// <para><b>For Beginners:</b> ADMM is like solving a complex puzzle by breaking it into smaller, manageable pieces.
/// It's particularly good at handling problems with constraints or when you want to distribute the computation across multiple processors.
/// </para>
/// </remarks>
public class ADMMOptimizer<T, TInput, TOutput> : GradientBasedOptimizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// The options specific to the ADMM optimizer.
    /// </summary>
    private ADMMOptimizerOptions<T, TInput, TOutput> _options;

    /// <summary>
    /// The current iteration count.
    /// </summary>
    private int _iteration;

    /// <summary>
    /// The regularization method used in the optimization.
    /// </summary>
    private IRegularization<T, TInput, TOutput> _regularization;

    /// <summary>
    /// The auxiliary variable in ADMM algorithm.
    /// </summary>
    private Vector<T> _z;

    /// <summary>
    /// The dual variable in ADMM algorithm.
    /// </summary>
    private Vector<T> _u;

    /// <summary>
    /// Initializes a new instance of the ADMMOptimizer class.
    /// </summary>
    /// <param name="model">The model to optimize.</param>
    /// <param name="options">The options for configuring the ADMM optimizer.</param>
    /// <param name="engine">The computation engine (CPU or GPU) for vectorized operations.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This sets up the ADMM optimizer with its initial configuration.
    /// You can customize various aspects of how it solves the optimization problem, or use default settings.
    /// </para>
    /// </remarks>
    public ADMMOptimizer(
        IFullModel<T, TInput, TOutput> model,
        ADMMOptimizerOptions<T, TInput, TOutput>? options = null,
        IEngine? engine = null)
        : base(model, options ?? new())
    {
        _options = options ?? new ADMMOptimizerOptions<T, TInput, TOutput>();
        _regularization = _options.Regularization;
        _z = Vector<T>.Empty();
        _u = Vector<T>.Empty();

        InitializeAdaptiveParameters();
    }

    /// <summary>
    /// Initializes the adaptive parameters used by the ADMM optimizer.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This resets the iteration count to zero, preparing the optimizer for a new optimization run.
    /// </para>
    /// </remarks>
    protected override void InitializeAdaptiveParameters()
    {
        base.InitializeAdaptiveParameters();
        _iteration = 0;
    }

    /// <summary>
    /// Performs the optimization process using the ADMM algorithm.
    /// </summary>
    /// <param name="inputData">The input data for optimization, including training data and targets.</param>
    /// <returns>The result of the optimization process, including the best solution found.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the main optimization process. It repeatedly updates the solution
    /// using the ADMM steps until it reaches the best possible solution or hits a stopping condition.
    /// </para>
    /// </remarks>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        ValidationHelper<T>.ValidateInputData(inputData);

        var currentSolution = InitializeRandomSolution(inputData.XTrain);
        var parameters = currentSolution.GetParameters();
        _z = new Vector<T>(parameters.Length);
        _u = new Vector<T>(parameters.Length);

        var bestStepData = new OptimizationStepData<T, TInput, TOutput>();
        var previousStepData = new OptimizationStepData<T, TInput, TOutput>();

        InitializeAdaptiveParameters();

        for (int iteration = 0; iteration < _options.MaxIterations; iteration++)
        {
            _iteration++;

            // ADMM steps
            currentSolution = UpdateX(currentSolution, inputData.XTrain, inputData.YTrain);
            parameters = currentSolution.GetParameters();
            UpdateZ(parameters);
            UpdateU(parameters);

            var currentStepData = EvaluateSolution(currentSolution, inputData);
            UpdateBestSolution(currentStepData, ref bestStepData);

            UpdateAdaptiveParameters(currentStepData, previousStepData);

            if (UpdateIterationHistoryAndCheckEarlyStopping(iteration, bestStepData))
            {
                return CreateOptimizationResult(bestStepData, inputData);
            }

            if (CheckConvergence(parameters))
            {
                return CreateOptimizationResult(bestStepData, inputData);
            }

            previousStepData = currentStepData;
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    /// <summary>
    /// Updates the primal variable x in the ADMM algorithm.
    /// </summary>
    /// <param name="currentSolution">The current solution being optimized.</param>
    /// <param name="X">The input matrix.</param>
    /// <param name="y">The target vector.</param>
    /// <returns>A new solution with updated coefficients.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This step solves a linear system to update the main variable (x) in the optimization problem.
    /// It's like finding the best compromise between fitting the data and satisfying the constraints.
    /// </para>
    /// </remarks>
    private IFullModel<T, TInput, TOutput> UpdateX(IFullModel<T, TInput, TOutput> currentSolution, TInput X, TOutput y)
    {
        // === Partially Vectorized X Update using IEngine (Phase B: US-GPU-015) ===
        // Solve (X^T X + rho I)x = X^T y + rho(z - u)

        var matrix = ConversionsHelper.ConvertToMatrix<T, TInput>(X);
        var XTranspose = matrix.Transpose();
        var XTX = XTranspose.Multiply(matrix);
        var rhoI = Matrix<T>.CreateIdentity(XTX.Rows).Multiply(NumOps.FromDouble(_options.Rho));
        var leftSide = XTX.Add(rhoI);

        var XTy = XTranspose.Multiply(ConversionsHelper.ConvertToVector<T, TOutput>(y));

        // Vectorized right-hand side computation
        var zMinusU = (Vector<T>)Engine.Subtract(_z, _u);
        var rho = NumOps.FromDouble(_options.Rho);
        var rhoZMinusU = (Vector<T>)Engine.Multiply(zMinusU, rho);
        var rightSide = XTy.Add(rhoZMinusU);

        var newCoefficients = MatrixSolutionHelper.SolveLinearSystem(leftSide, rightSide, _options.DecompositionType);

        return currentSolution.WithParameters(newCoefficients);
    }

    /// <summary>
    /// Updates the auxiliary variable z in the ADMM algorithm.
    /// </summary>
    /// <param name="x">The current primal variable.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This step applies the regularization to the solution.
    /// It's like smoothing out the solution to prevent overfitting.
    /// </para>
    /// </remarks>
    private void UpdateZ(Vector<T> x)
    {
        // === Vectorized Z Update using IEngine (Phase B: US-GPU-015) ===
        // z = regularize((x + u) / rho)

        var xPlusU = (Vector<T>)Engine.Add(x, _u);
        var invRho = NumOps.FromDouble(1.0 / _options.Rho);
        var scaledXPlusU = (Vector<T>)Engine.Multiply(xPlusU, invRho);
        _z = _regularization.Regularize(scaledXPlusU);
    }

    /// <summary>
    /// Updates the dual variable u in the ADMM algorithm.
    /// </summary>
    /// <param name="x">The current primal variable.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This step adjusts the dual variable, which helps enforce the constraints.
    /// It's like fine-tuning the balance between the main solution and the regularized solution.
    /// </para>
    /// </remarks>
    private void UpdateU(Vector<T> x)
    {
        // === Vectorized U Update using IEngine (Phase B: US-GPU-015) ===
        // u = u + (x - z)

        var xMinusZ = (Vector<T>)Engine.Subtract(x, _z);
        _u = (Vector<T>)Engine.Add(_u, xMinusZ);
    }

    /// <summary>
    /// Checks if the optimization has converged based on primal and dual residuals.
    /// </summary>
    /// <param name="x">The current primal variable.</param>
    /// <returns>True if the optimization has converged, false otherwise.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This checks if the solution is good enough to stop the optimization.
    /// It's like checking if you're close enough to the finish line in a race.
    /// </para>
    /// </remarks>
    private bool CheckConvergence(Vector<T> x)
    {
        var primalResidual = x.Subtract(_z);
        var dualResidual = _z.Subtract(_z.Subtract(_u));

        var primalNorm = primalResidual.Norm();
        var dualNorm = dualResidual.Norm();

        return NumOps.LessThan(primalNorm, NumOps.FromDouble(_options.AbsoluteTolerance)) &&
               NumOps.LessThan(dualNorm, NumOps.FromDouble(_options.AbsoluteTolerance));
    }

    /// <summary>
    /// Updates the adaptive parameters of the optimizer based on the current and previous optimization steps.
    /// </summary>
    /// <param name="currentStepData">Data from the current optimization step.</param>
    /// <param name="previousStepData">Data from the previous optimization step.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method adjusts how the optimizer behaves based on its recent performance.
    /// It can change certain parameters to help the optimizer find a better solution more quickly.
    /// </para>
    /// </remarks>
    protected override void UpdateAdaptiveParameters(OptimizationStepData<T, TInput, TOutput> currentStepData, OptimizationStepData<T, TInput, TOutput> previousStepData)
    {
        base.UpdateAdaptiveParameters(currentStepData, previousStepData);

        if (_options.UseAdaptiveRho)
        {
            var primalResidual = currentStepData.Solution.GetParameters().Subtract(_z);
            var dualResidual = _z.Subtract(_z.Subtract(_u));

            var primalNorm = primalResidual.Norm();
            var dualNorm = dualResidual.Norm();

            if (NumOps.GreaterThan(primalNorm, NumOps.Multiply(NumOps.FromDouble(_options.AdaptiveRhoFactor), dualNorm)))
            {
                _options.Rho *= _options.AdaptiveRhoIncrease;
            }
            else if (NumOps.GreaterThan(dualNorm, NumOps.Multiply(NumOps.FromDouble(_options.AdaptiveRhoFactor), primalNorm)))
            {
                _options.Rho /= _options.AdaptiveRhoDecrease;
            }
        }
    }

    /// <summary>
    /// Updates the optimizer's options with new settings.
    /// </summary>
    /// <param name="options">The new options to be applied to the optimizer.</param>
    /// <exception cref="ArgumentException">Thrown when the provided options are not of the correct type.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method allows you to change the settings of the optimizer while it's running.
    /// It's like adjusting the controls on a machine that's already operating.
    /// </para>
    /// </remarks>
    protected override void UpdateOptions(OptimizationAlgorithmOptions<T, TInput, TOutput> options)
    {
        if (options is ADMMOptimizerOptions<T, TInput, TOutput> admmOptions)
        {
            _options = admmOptions;
            _regularization = GetRegularizationFromOptions(admmOptions);
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected ADMMOptimizerOptions.");
        }
    }

    /// <summary>
    /// Creates a regularization object based on the provided options.
    /// </summary>
    /// <param name="options">The ADMM optimizer options containing regularization settings.</param>
    /// <returns>An instance of IRegularization<T> based on the specified regularization type.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method chooses the right kind of regularization based on your settings.
    /// Regularization helps prevent overfitting, which is when a model performs well on training data but poorly on new data.
    /// </para>
    /// </remarks>
    private IRegularization<T, TInput, TOutput> GetRegularizationFromOptions(ADMMOptimizerOptions<T, TInput, TOutput> options)
    {
        return options.RegularizationType switch
        {
            RegularizationType.L1 => new L1Regularization<T, TInput, TOutput>(new RegularizationOptions { Strength = options.RegularizationStrength }),
            RegularizationType.L2 => new L2Regularization<T, TInput, TOutput>(new RegularizationOptions { Strength = options.RegularizationStrength }),
            RegularizationType.ElasticNet => new ElasticNetRegularization<T, TInput, TOutput>(new RegularizationOptions { Strength = options.RegularizationStrength, L1Ratio = options.ElasticNetMixing }),
            _ => new NoRegularization<T, TInput, TOutput>()
        };
    }

    /// <summary>
    /// Retrieves the current options of the optimizer.
    /// </summary>
    /// <returns>The current optimization algorithm options.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method lets you check what settings the optimizer is currently using.
    /// It's like looking at the current settings on a machine.
    /// </para>
    /// </remarks>
    public override OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions()
    {
        return _options;
    }

    /// <summary>
    /// Converts the current state of the optimizer into a byte array for storage or transmission.
    /// </summary>
    /// <returns>A byte array representing the serialized state of the optimizer.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method saves all the important information about the optimizer's current state.
    /// It's like taking a snapshot of the optimizer that can be used to recreate its exact state later.
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

            string optionsJson = JsonConvert.SerializeObject(_options);
            writer.Write(optionsJson);

            writer.Write(_iteration);
            writer.Write(_z.Serialize());
            writer.Write(_u.Serialize());

            return ms.ToArray();
        }
    }

    /// <summary>
    /// Restores the optimizer's state from a byte array previously created by the Serialize method.
    /// </summary>
    /// <param name="data">The byte array containing the serialized optimizer state.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method rebuilds the optimizer's state from a saved snapshot.
    /// It's like restoring a machine to a previous configuration using a backup.
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
            _options = JsonConvert.DeserializeObject<ADMMOptimizerOptions<T, TInput, TOutput>>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");

            _iteration = reader.ReadInt32();
            _z = Vector<T>.Deserialize(reader.ReadBytes(reader.ReadInt32()));
            _u = Vector<T>.Deserialize(reader.ReadBytes(reader.ReadInt32()));

            _regularization = GetRegularizationFromOptions(_options);
        }
    }

    /// <summary>
    /// Generates a unique key for caching gradients based on the current state of the optimizer and input data.
    /// </summary>
    /// <param name="model">The symbolic model being optimized.</param>
    /// <param name="X">The input matrix.</param>
    /// <param name="y">The target vector.</param>
    /// <returns>A string that uniquely identifies the current optimization state for gradient caching.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method creates a unique label for the current state of the optimization.
    /// It's used to efficiently store and retrieve calculated gradients, which helps speed up the optimization process.
    /// </para>
    /// </remarks>
    protected override string GenerateGradientCacheKey(IFullModel<T, TInput, TOutput> model, TInput X, TOutput y)
    {
        var baseKey = base.GenerateGradientCacheKey(model, X, y);
        return $"{baseKey}_ADMM_{_options.Rho}_{_regularization?.GetType().Name}_{_options.AbsoluteTolerance}_{_iteration}";
    }
}
