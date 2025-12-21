using Newtonsoft.Json;

namespace AiDotNet.Optimizers;

/// <summary>
/// Implements the Levenberg-Marquardt optimization algorithm for non-linear least squares problems.
/// </summary>
/// <remarks>
/// <para>
/// The Levenberg-Marquardt algorithm is a popular method for solving non-linear least squares problems. It combines 
/// the Gauss-Newton algorithm and the method of gradient descent, providing a robust solution that works well in 
/// many situations.
/// </para>
/// <para><b>For Beginners:</b>
/// This optimizer is like a smart problem-solver that's really good at fitting curves to data points. It's especially 
/// useful when the relationship between your inputs and outputs isn't a straight line. It works by making small 
/// adjustments to its guess, getting closer to the best solution with each step.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class LevenbergMarquardtOptimizer<T, TInput, TOutput> : GradientBasedOptimizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// The options specific to the Levenberg-Marquardt algorithm.
    /// </summary>
    private LevenbergMarquardtOptimizerOptions<T, TInput, TOutput> _options;

    /// <summary>
    /// The current iteration count of the optimization process.
    /// </summary>
    private int _iteration;

    /// <summary>
    /// The damping factor used in the Levenberg-Marquardt algorithm to balance between gradient descent and Gauss-Newton steps.
    /// </summary>
    private T _dampingFactor;

    /// <summary>
    /// Initializes a new instance of the LevenbergMarquardtOptimizer class.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This constructor sets up the optimizer with the provided options and dependencies. If no options are provided, 
    /// it uses default settings.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is where we set up our smart problem-solver. We can give it special instructions (options) on how to work, 
    /// or let it use its default settings. We also give it tools (like modelEvaluator and fitDetector) that it needs 
    /// to do its job well.
    /// </para>
    /// </remarks>
    /// <param name="model">The model to optimize.</param>
    /// <param name="options">Custom options for the Levenberg-Marquardt algorithm.</param>
    /// <param name="engine">The computation engine (CPU or GPU) for vectorized operations.</param>
    public LevenbergMarquardtOptimizer(
        IFullModel<T, TInput, TOutput> model,
        LevenbergMarquardtOptimizerOptions<T, TInput, TOutput>? options = null,
        IEngine? engine = null)
        : base(model, options ?? new())
    {
        _options = options ?? new LevenbergMarquardtOptimizerOptions<T, TInput, TOutput>();
        _dampingFactor = NumOps.Zero;

        InitializeAdaptiveParameters();
    }

    /// <summary>
    /// Initializes the adaptive parameters used by the optimizer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method sets up the initial values for the damping factor and iteration count. The damping factor is a key 
    /// component of the Levenberg-Marquardt algorithm, controlling the balance between the Gauss-Newton and gradient 
    /// descent approaches.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like setting up the starting point for our problem-solver. We give it an initial "cautiousness" level 
    /// (damping factor) and reset its step counter. The cautiousness helps it decide whether to take big steps or 
    /// small steps when trying to find the best solution.
    /// </para>
    /// </remarks>
    protected override void InitializeAdaptiveParameters()
    {
        base.InitializeAdaptiveParameters();
        _dampingFactor = NumOps.FromDouble(_options.InitialDampingFactor);
        _iteration = 0;
    }

    /// <summary>
    /// Performs the optimization process using the Levenberg-Marquardt algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method implements the main optimization loop. It iteratively improves the solution by calculating the 
    /// Jacobian matrix, residuals, and updating the solution until a stopping criterion is met.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is where the magic happens! Our problem-solver repeatedly tries to improve its guess. In each round:
    /// 1. It calculates how sensitive the output is to small changes (Jacobian).
    /// 2. It checks how far off its current guess is (residuals).
    /// 3. It uses this information to make a better guess.
    /// 4. It keeps doing this until it's happy with the result or runs out of attempts.
    /// </para>
    /// </remarks>
    /// <param name="inputData">The input data for the optimization process.</param>
    /// <returns>The result of the optimization process.</returns>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        ValidationHelper<T>.ValidateInputData(inputData);

        var currentSolution = InitializeRandomSolution(inputData.XTrain);
        var bestStepData = new OptimizationStepData<T, TInput, TOutput>();
        var previousStepData = new OptimizationStepData<T, TInput, TOutput>();

        InitializeAdaptiveParameters();

        for (int iteration = 0; iteration < _options.MaxIterations; iteration++)
        {
            _iteration++;
            var jacobian = CalculateJacobian(currentSolution, inputData.XTrain);
            var residuals = CalculateResiduals(currentSolution, inputData.XTrain, inputData.YTrain);
            var newSolution = UpdateSolution(currentSolution, jacobian, residuals);

            var currentStepData = EvaluateSolution(newSolution, inputData);
            UpdateBestSolution(currentStepData, ref bestStepData);

            UpdateAdaptiveParameters(currentStepData, previousStepData);

            if (UpdateIterationHistoryAndCheckEarlyStopping(iteration, bestStepData))
            {
                return CreateOptimizationResult(bestStepData, inputData);
            }

            if (NumOps.LessThan(NumOps.Abs(NumOps.Subtract(bestStepData.FitnessScore, currentStepData.FitnessScore)), NumOps.FromDouble(_options.Tolerance)))
            {
                return CreateOptimizationResult(bestStepData, inputData);
            }

            currentSolution = newSolution;
            previousStepData = currentStepData;
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    /// <summary>
    /// Calculates the Jacobian matrix for the current model and input data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The Jacobian matrix represents the local sensitivity of the model's output to changes in its parameters.
    /// Each element (i,j) in the matrix is the partial derivative of the i-th output with respect to the j-th parameter.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like creating a sensitivity map. It shows how much each part of our guess affects the final answer.
    /// It helps the optimizer understand which parts of the guess to change to get better results.
    /// </para>
    /// </remarks>
    /// <param name="model">The current model being optimized.</param>
    /// <param name="X">The input data matrix.</param>
    /// <returns>The Jacobian matrix.</returns>
    private Matrix<T> CalculateJacobian(IFullModel<T, TInput, TOutput> model, TInput X)
    {
        // Get batch size (number of examples)
        int m = InputHelper<T, TInput>.GetBatchSize(X);

        // Get number of parameters
        int n = model.GetParameters().Length;

        // Create the Jacobian matrix
        var jacobian = new Matrix<T>(m, n);

        // For each example and each parameter, calculate the partial derivative
        for (int i = 0; i < m; i++)
        {
            // Get the i-th example from the input batch
            var exampleX = InputHelper<T, TInput>.GetItem(X, i);

            for (int j = 0; j < n; j++)
            {
                // Calculate the partial derivative of the model output with respect to parameter j
                // for the current example
                jacobian[i, j] = CalculatePartialDerivative(model, exampleX, j);
            }
        }

        return jacobian;
    }

    /// <summary>
    /// Calculates the partial derivative of the model output with respect to a specific parameter.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method uses numerical differentiation to estimate the partial derivative. It slightly perturbs the 
    /// parameter and observes the change in the model's output.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like gently poking each part of our guess and seeing how much it changes the answer. 
    /// It helps us understand which parts of our guess are most important.
    /// </para>
    /// </remarks>
    /// <param name="model">The current model being optimized.</param>
    /// <param name="x">A single input data point.</param>
    /// <param name="paramIndex">The index of the parameter for which to calculate the derivative.</param>
    /// <returns>The estimated partial derivative.</returns>
    private T CalculatePartialDerivative(IFullModel<T, TInput, TOutput> model, TInput x, int paramIndex)
    {
        var epsilon = NumOps.FromDouble(1e-8);
        var parameters = model.GetParameters();
        var originalParam = parameters[paramIndex];

        // Create a batch with just one example
        TInput singleItemBatch = InputHelper<T, TInput>.CreateSingleItemBatch(x);

        // Modify the parameter and get prediction with increased value
        parameters[paramIndex] = NumOps.Add(originalParam, epsilon);
        var modifiedModel = model.WithParameters(parameters);
        TOutput yPlusBatch = modifiedModel.Predict(singleItemBatch);
        T yPlus = ConversionsHelper.ConvertToScalar<T, TOutput>(yPlusBatch);

        // Modify the parameter and get prediction with decreased value
        parameters[paramIndex] = NumOps.Subtract(originalParam, epsilon);
        modifiedModel = model.WithParameters(parameters);
        TOutput yMinusBatch = modifiedModel.Predict(singleItemBatch);
        T yMinus = ConversionsHelper.ConvertToScalar<T, TOutput>(yMinusBatch);

        // Restore the original parameter
        parameters[paramIndex] = originalParam;
        model.WithParameters(parameters);

        // Calculate the central difference approximation of the derivative
        return NumOps.Divide(NumOps.Subtract(yPlus, yMinus), NumOps.FromDouble(2 * 1e-8));
    }

    /// <summary>
    /// Calculates the residuals (differences between predicted and actual values) for the current model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Residuals represent the errors in the model's predictions. They are used to guide the optimization process
    /// towards minimizing these errors.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This checks how far off our current guess is from the real answers. It's like measuring the gap between 
    /// what our model predicts and what actually happened.
    /// </para>
    /// </remarks>
    /// <param name="model">The current model being optimized.</param>
    /// <param name="X">The input data matrix.</param>
    /// <param name="y">The actual output values.</param>
    /// <returns>A vector of residuals.</returns>
    private Vector<T> CalculateResiduals(IFullModel<T, TInput, TOutput> model, TInput X, TOutput y)
    {
        // === Vectorized Residuals Calculation using IEngine (Phase B: US-GPU-015) ===
        // residuals = y - predictions

        var predictions = ConversionsHelper.ConvertToVector<T, TOutput>(model.Predict(X));
        var targets = ConversionsHelper.ConvertToVector<T, TOutput>(y);
        return (Vector<T>)Engine.Subtract(targets, predictions);
    }

    /// <summary>
    /// Updates the current solution based on the Levenberg-Marquardt algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method implements the core of the Levenberg-Marquardt algorithm. It calculates the update step
    /// by solving a modified normal equation system, which balances between the Gauss-Newton method and gradient descent.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is where the optimizer decides how to improve its current guess. It's like a smart navigator
    /// that looks at a map (the Jacobian) and the current errors (residuals) to figure out which direction
    /// to move and how big a step to take.
    /// </para>
    /// </remarks>
    /// <param name="currentSolution">The current model solution.</param>
    /// <param name="jacobian">The Jacobian matrix of the current solution.</param>
    /// <param name="residuals">The residuals (errors) of the current solution.</param>
    /// <returns>An updated symbolic model with improved coefficients.</returns>
    private IFullModel<T, TInput, TOutput> UpdateSolution(IFullModel<T, TInput, TOutput> currentSolution, Matrix<T> jacobian, Vector<T> residuals)
    {
        // === Vectorized Solution Update using IEngine (Phase B: US-GPU-015) ===
        // Levenberg-Marquardt: (J^T J + λ diag(J^T J)) δ = J^T r
        // where λ is the damping factor

        var jTj = jacobian.Transpose().Multiply(jacobian);

        // Vectorized diagonal scaling
        var diagonalValues = (Vector<T>)Engine.Multiply(jTj.Diagonal(), _dampingFactor);
        var dampedDiagonal = Matrix<T>.CreateDiagonal(diagonalValues);

        var jTr = jacobian.Transpose().Multiply(residuals);
        var lhs = jTj.Add(dampedDiagonal);
        var delta = SolveLinearSystem(lhs, jTr);

        // Vectorized parameter update
        var newCoefficients = (Vector<T>)Engine.Add(currentSolution.GetParameters(), delta);
        return currentSolution.WithParameters(newCoefficients);
    }

    /// <summary>
    /// Solves the linear system of equations in the Levenberg-Marquardt algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method solves the linear system Ax = b, where A is the left-hand side matrix and b is the right-hand side vector.
    /// It uses either a custom decomposition method provided in the options, or falls back to default methods (LU decomposition
    /// with SVD as a fallback).
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like solving a complex puzzle to find out how to adjust our guess. It uses advanced math techniques
    /// to figure out the best way to change our model's parameters. If we've provided a special method to solve
    /// this puzzle, it uses that. Otherwise, it tries a standard method, and if that doesn't work, it uses a more
    /// robust (but slower) method as a backup.
    /// </para>
    /// </remarks>
    /// <param name="lhs">The left-hand side matrix of the linear system.</param>
    /// <param name="jTr">The right-hand side vector of the linear system.</param>
    /// <returns>The solution vector of the linear system.</returns>
    private Vector<T> SolveLinearSystem(Matrix<T> lhs, Vector<T> jTr)
    {
        if (_options.CustomDecomposition != null)
        {
            // Use the custom decomposition if provided
            return _options.CustomDecomposition.Solve(jTr);
        }
        else
        {
            // Use the default method if no custom decomposition is provided
            try
            {
                return MatrixSolutionHelper.SolveLinearSystem(lhs, jTr, MatrixDecompositionType.Lu);
            }
            catch (InvalidOperationException)
            {
                return MatrixSolutionHelper.SolveLinearSystem(lhs, jTr, MatrixDecompositionType.Svd);
            }
        }
    }

    /// <summary>
    /// Updates the adaptive parameters of the optimizer based on the current and previous optimization steps.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method adjusts the damping factor of the Levenberg-Marquardt algorithm based on the performance
    /// of the current step compared to the previous step. If the current step improved the solution, the damping
    /// factor is decreased to allow larger steps. If not, it's increased to take more conservative steps.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like adjusting how cautious our optimizer is. If the last guess was better, it becomes more
    /// confident and takes bigger steps. If the last guess wasn't so good, it becomes more careful and takes
    /// smaller steps. This helps the optimizer adapt to the shape of the problem it's solving.
    /// </para>
    /// </remarks>
    /// <param name="currentStepData">Data from the current optimization step.</param>
    /// <param name="previousStepData">Data from the previous optimization step.</param>
    protected override void UpdateAdaptiveParameters(OptimizationStepData<T, TInput, TOutput> currentStepData, OptimizationStepData<T, TInput, TOutput> previousStepData)
    {
        base.UpdateAdaptiveParameters(currentStepData, previousStepData);

        if (_options.UseAdaptiveDampingFactor)
        {
            if (NumOps.GreaterThan(currentStepData.FitnessScore, previousStepData.FitnessScore))
            {
                _dampingFactor = NumOps.Multiply(_dampingFactor, NumOps.FromDouble(_options.DampingFactorDecreaseFactor));
            }
            else
            {
                _dampingFactor = NumOps.Multiply(_dampingFactor, NumOps.FromDouble(_options.DampingFactorIncreaseFactor));
            }

            _dampingFactor = MathHelper.Clamp(_dampingFactor,
                NumOps.FromDouble(_options.MinDampingFactor),
                NumOps.FromDouble(_options.MaxDampingFactor));
        }
    }

    /// <summary>
    /// Updates the optimizer's options with new settings.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method allows updating the optimizer's settings during runtime. It ensures that only compatible
    /// option types are used with this optimizer.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like changing the settings on a machine while it's running. It makes sure that we're only
    /// using settings that work for this specific type of optimizer. If someone tries to use the wrong
    /// type of settings, it lets them know there's a problem.
    /// </para>
    /// </remarks>
    /// <param name="options">The new options to be applied to the optimizer.</param>
    /// <exception cref="ArgumentException">Thrown when the provided options are not of the correct type.</exception>
    protected override void UpdateOptions(OptimizationAlgorithmOptions<T, TInput, TOutput> options)
    {
        if (options is LevenbergMarquardtOptimizerOptions<T, TInput, TOutput> lmOptions)
        {
            _options = lmOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected LevenbergMarquardtOptimizerOptions.");
        }
    }

    /// <summary>
    /// Retrieves the current options of the optimizer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method provides access to the current settings of the Levenberg-Marquardt optimizer.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like checking the current settings on a machine. It lets you see how the optimizer
    /// is currently configured without changing anything.
    /// </para>
    /// </remarks>
    /// <returns>The current options of the Levenberg-Marquardt optimizer.</returns>
    public override OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions()
    {
        return _options;
    }

    /// <summary>
    /// Serializes the optimizer's state into a byte array.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method converts the current state of the optimizer, including its options and internal parameters,
    /// into a byte array. This is useful for saving the optimizer's state or transferring it between systems.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like taking a snapshot of the optimizer's current state and packing it into a compact form.
    /// It's useful if you want to save the optimizer's progress and continue from this point later, or if
    /// you want to move the optimizer to a different computer.
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

            writer.Write(_iteration);
            writer.Write(Convert.ToDouble(_dampingFactor));

            return ms.ToArray();
        }
    }

    /// <summary>
    /// Deserializes a byte array to restore the optimizer's state.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method takes a byte array (previously created by the Serialize method) and uses it to restore
    /// the optimizer's state, including its options and internal parameters.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like unpacking that snapshot we took earlier and setting up the optimizer exactly as it was
    /// when we saved it. It's the reverse process of serialization, allowing us to continue optimization
    /// from a previously saved state.
    /// </para>
    /// </remarks>
    /// <param name="data">The byte array containing the serialized optimizer state.</param>
    /// <exception cref="InvalidOperationException">Thrown when the optimizer options cannot be deserialized.</exception>
    public override void Deserialize(byte[] data)
    {
        using (MemoryStream ms = new MemoryStream(data))
        using (BinaryReader reader = new BinaryReader(ms))
        {
            int baseDataLength = reader.ReadInt32();
            byte[] baseData = reader.ReadBytes(baseDataLength);
            base.Deserialize(baseData);

            string optionsJson = reader.ReadString();
            _options = JsonConvert.DeserializeObject<LevenbergMarquardtOptimizerOptions<T, TInput, TOutput>>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");

            _iteration = reader.ReadInt32();
            _dampingFactor = NumOps.FromDouble(reader.ReadDouble());
        }
    }
}
