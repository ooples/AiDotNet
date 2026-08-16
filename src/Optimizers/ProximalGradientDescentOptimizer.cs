using AiDotNet.Helpers;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using Newtonsoft.Json;

using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.Optimizers;

/// <summary>
/// Implements a Proximal Gradient Descent optimization algorithm which combines gradient descent with regularization.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// Proximal Gradient Descent (PGD) is an extension of standard gradient descent that handles regularization more
/// efficiently. The algorithm alternates between performing a gradient descent step to minimize the loss function
/// and applying a proximal operator to enforce regularization. This approach is particularly effective for
/// problems where regularization is important to prevent overfitting or to enforce specific properties in the solution.
/// </para>
/// <para><b>For Beginners:</b> Proximal Gradient Descent is like walking downhill while staying within certain boundaries.
/// 
/// Imagine you're hiking down a mountain to find the lowest point:
/// - Standard gradient descent is like always walking directly downhill
/// - Proximal gradient descent adds boundaries or "guardrails" to your path
/// - These guardrails keep you from wandering into areas that might look good but are actually not helpful
/// - For example, the guardrails might prevent solutions that are too complex and would overfit the data
/// 
/// This approach helps find solutions that not only fit the data well but also have desirable properties
/// like simplicity or stability.
/// </para>
/// </remarks>
[ComponentType(ComponentType.Optimizer)]
[PipelineStage(PipelineStage.Training)]
public class ProximalGradientDescentOptimizer<T, TInput, TOutput> : GradientBasedOptimizerBase<T, TInput, TOutput>, Fused.IFusedOptimizerSpec
{
    /// <summary>
    /// Describes this optimizer for the compiled fused-training kernel.
    /// </summary>
    /// <remarks>
    /// <para>
    /// What can be fused depends entirely on the proximal operator, because the proximal operator IS the
    /// algorithm — the gradient step in front of it is plain SGD. Two regularizers have kernels:
    /// </para>
    /// <list type="bullet">
    /// <item><description>
    /// <see cref="NoRegularization{T, TInput, TOutput}"/>, or an L1/L2 regularizer whose strength is zero —
    /// the prox is the identity, so the tape step reduces to <c>param -= lr*grad</c> and
    /// <see cref="Tensors.Engines.Compilation.OptimizerType.SGD"/> reproduces it exactly.
    /// </description></item>
    /// <item><description>
    /// <see cref="L1Regularization{T, TInput, TOutput}"/> — the canonical proximal case (ISTA), fused by
    /// <see cref="Tensors.Engines.Compilation.OptimizerType.ProximalL1"/>.
    /// </description></item>
    /// </list>
    /// <para>
    /// The L1 case needs one conversion. The kernel writes the textbook ISTA step,
    /// <c>z = param - lr*grad; param = sign(z)*max(|z| - lr*l1, 0)</c>, thresholding at <c>lr*l1</c>
    /// because that is the prox of <c>l1*||x||_1</c> for a step of size <c>lr</c>. This library's
    /// <c>IRegularization.Regularize(Vector)</c> is defined the other way round — it soft-thresholds at the
    /// raw <c>Strength</c>, with no step size in it — and that is what the eager path applies. So the spec
    /// passes <c>L1 = Strength/lr</c>, which makes the kernel's <c>lr*l1</c> collapse back to <c>Strength</c>
    /// and the two paths agree element for element. Fidelity here is to what the eager optimizer does, not to
    /// the textbook scaling; changing <c>Regularize</c> would change every model that regularizes.
    /// </para>
    /// <para>
    /// That conversion is only stable while <c>lr</c> is the same number the kernel later multiplies by, so
    /// the L1 case declines whenever the learning rate can move underneath it — an attached schedule or
    /// adaptive learning rates — and whenever <c>lr</c> is not a positive finite number to divide by. The
    /// no-regularization case has no such coupling and honours schedules normally.
    /// </para>
    /// <para>
    /// Every other regularizer (L2, ElasticNet, and any custom one) declines. Their prox is a different
    /// function and no kernel implements it; folding an L2 penalty into the gradient instead would only
    /// shrink coordinates toward zero rather than setting them there, which is a different algorithm.
    /// </para>
    /// </remarks>
    bool Fused.IFusedOptimizerSpec.TryGetFusedOptimizerConfig(out Fused.FusedOptimizerConfig config)
    {
        config = default;
        if (_options.UseAdaptiveLearningRate) return false;
        if (!TryGetFusedLrSchedule(out var schedule)) return false;

        double lr = GetCurrentLearningRate();

        // A zero-strength L1 or L2 prox is the identity function, exactly like no regularizer at all:
        // sign(x)*max(|x| - 0, 0) == x and x*(1 - 0) == x.
        bool identityProx = _regularization is NoRegularization<T, TInput, TOutput>
            || ((_regularization is L1Regularization<T, TInput, TOutput>
                    || _regularization is L2Regularization<T, TInput, TOutput>)
                && _regularization.GetOptions().Strength == 0.0);

        if (identityProx)
        {
            config = new Fused.FusedOptimizerConfig(
                Tensors.Engines.Compilation.OptimizerType.SGD,
                (float)lr,
                0f, 0f, 0f, 0f, schedule);
            return true;
        }

        if (_regularization is L1Regularization<T, TInput, TOutput>)
        {
            // The threshold conversion below only holds for the lr the kernel is configured with.
            if (schedule is not null) return false;
            if (!(lr > 0.0) || double.IsInfinity(lr)) return false;

            double strength = _regularization.GetOptions().Strength;
            config = new Fused.FusedOptimizerConfig(
                Tensors.Engines.Compilation.OptimizerType.ProximalL1,
                (float)lr,
                0f, 0f, 0f, 0f, schedule)
            {
                Extras = new Tensors.Engines.Compilation.FusedOptimizerExtras
                {
                    L1 = (float)(strength / lr),
                },
            };
            return true;
        }

        return false;
    }

    /// <summary>
    /// Configuration options specific to Proximal Gradient Descent optimization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the configuration parameters for the PGD algorithm, such as learning rate settings,
    /// tolerance for convergence, and maximum iterations. These parameters control the behavior of the
    /// optimizer and affect its performance and convergence properties.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the settings panel for the optimizer.
    /// 
    /// The options control:
    /// - How quickly the algorithm moves downhill (learning rate)
    /// - When to consider the optimization complete (tolerance)
    /// - How many attempts the algorithm makes before giving up (max iterations)
    /// - Whether and how to adjust the step size automatically (adaptive learning rate)
    /// 
    /// Adjusting these settings can help the algorithm work better for different types of problems.
    /// </para>
    /// </remarks>
    private ProximalGradientDescentOptimizerOptions<T, TInput, TOutput> _options;

    /// <summary>
    /// The current iteration count of the optimization process.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field keeps track of the number of completed iterations in the optimization process.
    /// It is used to enforce the maximum iteration limit and can also be useful for monitoring
    /// the progress of the optimization.
    /// </para>
    /// <para><b>For Beginners:</b> This is like a step counter for the optimization journey.
    /// 
    /// The iteration counter:
    /// - Keeps track of how many rounds of optimization have been completed
    /// - Helps decide when to stop if the maximum number of steps is reached
    /// - Can be used to monitor how efficiently the algorithm is working
    /// 
    /// This is important because we don't want the algorithm to run forever if it's not finding a good solution.
    /// </para>
    /// </remarks>
    private int _iteration;

    /// <summary>
    /// The regularization strategy applied to the optimization process.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the regularization component that is applied after each gradient step.
    /// Regularization helps prevent overfitting by penalizing certain properties of the solution,
    /// such as large coefficient values or complexity. Different regularization strategies can
    /// enforce different properties in the optimized solution.
    /// </para>
    /// <para><b>For Beginners:</b> This is like a set of rules that keeps the solution well-behaved.
    /// 
    /// The regularization:
    /// - Acts as a constraint on what kinds of solutions are acceptable
    /// - Prevents solutions that are too complex or extreme
    /// - For example, it might penalize solutions with very large numbers
    /// - Helps the model generalize better to new data rather than just memorizing the training data
    /// 
    /// Think of it as adding a preference for simpler, more stable solutions that are less likely to overfit.
    /// </para>
    /// </remarks>
    private IRegularization<T, TInput, TOutput> _regularization;

    /// <summary>
    /// Stores the pre-update parameters for approximate reverse updates.
    /// </summary>
    private Vector<T>? _previousParameters;

    /// <summary>
    /// Initializes a new instance of the <see cref="ProximalGradientDescentOptimizer{T}"/> class with the specified options and components.
    /// </summary>
    /// <param name="model">The model to optimize.</param>
    /// <param name="options">The proximal gradient descent optimization options, or null to use default options.</param>
    /// <param name="engine">The computation engine (CPU or GPU) for vectorized operations.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new proximal gradient descent optimizer with the specified options and components.
    /// If any parameter is null, a default implementation is used. The constructor initializes the options,
    /// regularization strategy, and adaptive parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This is the starting point for creating a new optimizer.
    ///
    /// Think of it like setting up equipment for a mountain hike:
    /// - You can provide custom settings (options) or use the default ones
    /// - You can provide specialized tools (evaluators, calculators) or use the basic ones
    /// - You can specify how to enforce boundaries (regularization) or use no boundaries
    /// - It gets everything ready so you can start the optimization process
    ///
    /// The options control things like how fast to move, when to stop, and how to adapt during the journey.
    /// </para>
    /// </remarks>
    public ProximalGradientDescentOptimizer(
        IFullModel<T, TInput, TOutput> model,
        ProximalGradientDescentOptimizerOptions<T, TInput, TOutput>? options = null,
        IEngine? engine = null)
        : base(model, options ?? new())
    {
        _options = options ?? new ProximalGradientDescentOptimizerOptions<T, TInput, TOutput>();
        _regularization = BuildProximalOperator(_options);

        InitializeAdaptiveParameters();
    }

    /// <summary>
    /// Resolves the proximal operator from the options, applying <c>RegularizationStrength</c> when the
    /// caller set one.
    /// </summary>
    /// <param name="options">The optimizer's options.</param>
    /// <returns>The regularizer whose <c>Regularize</c> is this optimizer's proximal operator.</returns>
    /// <remarks>
    /// <para>
    /// <c>RegularizationStrength</c> used to be read by nothing at all. A caller who set it — or who tuned
    /// the facade's <c>regularization</c> hyperparameter, which maps onto exactly this property — got the
    /// default L2 operator at strength 0.01 regardless, which is the failure mode where the knob turns and
    /// the model does not change.
    /// </para>
    /// <para>
    /// It is nullable so that the two ways of expressing a strength cannot contradict each other: the
    /// regularizer carries its own, and this overrides it only when explicitly set. A non-null value
    /// rebuilds the L1 or L2 operator at that strength; any other regularizer is used as configured,
    /// because its strength may not be its only parameter (ElasticNet's L1/L2 ratio, for instance) and
    /// silently reconstructing it would drop the rest.
    /// </para>
    /// </remarks>
    private static IRegularization<T, TInput, TOutput> BuildProximalOperator(
        ProximalGradientDescentOptimizerOptions<T, TInput, TOutput> options)
    {
        var configured = options.Regularization ?? new NoRegularization<T, TInput, TOutput>();
        double? strength = options.RegularizationStrength;
        if (strength is null)
        {
            return configured;
        }

        // Copy the configured options and override only Strength. Constructing a fresh
        // RegularizationOptions would reset Type to None and L1Ratio to its default, so anything reading
        // GetOptions() for metadata would see a regularizer that describes itself wrongly — harmless while
        // only Strength is read, and a trap the moment something else is.
        var rescaled = CloneWithStrength(configured.GetOptions(), strength.Value);
        return configured switch
        {
            L1Regularization<T, TInput, TOutput> => new L1Regularization<T, TInput, TOutput>(rescaled),
            L2Regularization<T, TInput, TOutput> => new L2Regularization<T, TInput, TOutput>(rescaled),
            _ => configured,
        };
    }

    /// <summary>
    /// Copies regularization options, replacing only the strength.
    /// </summary>
    /// <param name="source">The options to copy.</param>
    /// <param name="strength">The strength the copy should carry.</param>
    /// <returns>A copy identical to <paramref name="source"/> apart from its strength.</returns>
    internal static RegularizationOptions CloneWithStrength(RegularizationOptions source, double strength)
        => new RegularizationOptions
        {
            Type = source.Type,
            Strength = strength,
            L1Ratio = source.L1Ratio,
        };

    /// <summary>
    /// Initializes the adaptive parameters used by the Proximal Gradient Descent algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method overrides the base implementation to initialize PGD-specific adaptive parameters.
    /// It sets the initial learning rate from the options and resets the iteration counter to zero.
    /// The learning rate controls how large each step is during optimization.
    /// </para>
    /// <para><b>For Beginners:</b> This method prepares the optimizer for a fresh start.
    /// 
    /// It's like a hiker preparing for a new journey:
    /// - Setting their initial step size (learning rate) to a comfortable starting value
    /// - Resetting their step counter to zero
    /// - Getting ready to begin searching for the lowest point
    /// 
    /// These initial settings help the algorithm start with balanced movements that
    /// can be adjusted as it learns more about the landscape.
    /// </para>
    /// </remarks>
    protected override void InitializeAdaptiveParameters()
    {
        base.InitializeAdaptiveParameters();
        CurrentLearningRate = NumOps.FromDouble(_options.InitialLearningRate);
        _iteration = 0;
    }

    /// <summary>
    /// Performs the proximal gradient descent optimization to find the best solution for the given input data.
    /// </summary>
    /// <param name="inputData">The input data to optimize against.</param>
    /// <returns>An optimization result containing the best solution found and associated metrics.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the main PGD algorithm. It starts from a random solution and iteratively improves it
    /// by calculating the gradient, taking a step in the negative gradient direction, and then applying regularization.
    /// The process continues until either the maximum number of iterations is reached, early stopping criteria are met,
    /// or the improvement falls below the specified tolerance.
    /// </para>
    /// <para><b>For Beginners:</b> This is the main search process where the algorithm looks for the best solution.
    ///
    /// The process works like this:
    /// 1. Start at a random position on the "hill"
    /// 2. For each iteration:
    ///    - Figure out which direction is most downhill (calculate gradient)
    ///    - Take a step in that direction (update solution)
    ///    - Apply the guardrails to keep the solution well-behaved (apply regularization)
    ///    - Check if the new position is better than the best found so far
    ///    - Adjust the step size based on progress
    /// 3. Stop when enough iterations are done, when no more improvement is happening, or when the
    ///    improvement is very small
    ///
    /// This approach efficiently finds solutions that both fit the data well and satisfy the regularization constraints.
    /// </para>
    /// <para><b>DataLoader Integration:</b> This method uses the DataLoader API for efficient batch processing.
    /// It creates a batcher using <see cref="GradientBasedOptimizerBase{T,TInput,TOutput}.CreateBatcher"/>
    /// and notifies the sampler of epoch starts using
    /// <see cref="GradientBasedOptimizerBase{T,TInput,TOutput}.NotifyEpochStart"/>.
    /// </para>
    /// </remarks>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        ValidationHelper<T>.ValidateInputData(inputData);

        var currentSolution = InitializeWorkingSolution(inputData.XTrain);
        var bestStepData = new OptimizationStepData<T, TInput, TOutput>();
        var previousStepData = new OptimizationStepData<T, TInput, TOutput>();

        InitializeAdaptiveParameters();

        for (int epoch = 0; epoch < _options.MaxIterations; epoch++)
        {
            NotifyEpochStart(epoch);
            var batcher = CreateBatcher(inputData, _options.BatchSize, epoch);

            foreach (var (xBatch, yBatch, batchIndices) in batcher.GetBatches())
            {
                _iteration++;
                var gradient = CalculateGradient(currentSolution, xBatch, yBatch);
                currentSolution = UpdateSolution(currentSolution, gradient);
            }

            var currentStepData = EvaluateSolution(currentSolution, inputData);
            UpdateBestSolution(currentStepData, ref bestStepData);

            UpdateAdaptiveParameters(currentStepData, previousStepData);

            if (UpdateIterationHistoryAndCheckEarlyStopping(epoch, bestStepData))
            {
                return CreateOptimizationResult(bestStepData, inputData);
            }

            if (IsConvergedAgainstPreviousEpoch(epoch, currentStepData, previousStepData, _options.Tolerance))
            {
                // H6 convergence fix (PR #1364): compare CURRENT vs PREVIOUS
                // epoch (not bestStepData — UpdateBestSolution copies
                // currentStepData into bestStepData on epoch 0, so |best -
                // current| = 0 < tolerance would falsely converge). Skip
                // check on epoch 0 where previousStepData is the pre-training
                // baseline. Helper is on GradientBasedOptimizerBase.
                return CreateOptimizationResult(bestStepData, inputData);
            }

            previousStepData = currentStepData;
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    /// <summary>
    /// Updates the solution by applying a gradient step followed by regularization.
    /// </summary>
    /// <param name="currentSolution">The current solution to update.</param>
    /// <param name="gradient">The gradient vector indicating the direction of steepest ascent.</param>
    /// <returns>A new solution after applying the gradient step and regularization.</returns>
    /// <remarks>
    /// <para>
    /// This method performs a two-step update: first, it applies a gradient descent step by moving in the
    /// negative gradient direction; then, it applies regularization to enforce desired properties in the solution.
    /// The step size is determined by the current learning rate.
    /// </para>
    /// <para><b>For Beginners:</b> This method takes one step down the hill while respecting the guardrails.
    ///
    /// The process has two parts:
    /// 1. Take a step downhill:
    ///    - Look at the gradient to see which way is most downhill
    ///    - Move in that direction by an amount controlled by the learning rate
    ///
    /// 2. Apply the guardrails:
    ///    - The regularization takes the solution after the gradient step
    ///    - It adjusts the solution to make it satisfy the desired properties
    ///    - For example, it might reduce any extremely large values
    ///
    /// This combination of steps helps find solutions that both minimize the error and have good properties.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, TInput, TOutput> UpdateSolution(IFullModel<T, TInput, TOutput> currentSolution, Vector<T> gradient)
    {
        // #1413 CONSOLIDATION: NN solutions go through base.UpdateSolution
        // which synthesizes a TapeStepContext and delegates to Step
        // (one source of truth, matches PyTorch/TF/JAX). Non-NN solutions
        // (regression, clustering, classical models) keep the legacy
        // flat-vector path below for backward compatibility.
        if (currentSolution is AiDotNet.Interfaces.INeuralNetwork<T>)
        {
            return base.UpdateSolution(currentSolution, gradient);
        }
        // === Vectorized Proximal GD Update using IEngine (Phase B: US-GPU-015) ===
        // params_new = params - stepSize * gradient
        // Then apply proximal operator (regularization)

        var stepSize = CurrentLearningRate;
        var parameters = InterfaceGuard.Parameterizable(currentSolution).GetParameters();

        // Save pre-update parameters for reverse updates (vectorized copy)
        _previousParameters = new Vector<T>(parameters);

        // Vectorized gradient descent step
        var gradientStep = (Vector<T>)Engine.Multiply(gradient, stepSize);
        var newCoefficients = (Vector<T>)Engine.Subtract(parameters, gradientStep);

        // Apply proximal operator (regularization)
        newCoefficients = _regularization.Regularize(newCoefficients);

        return InterfaceGuard.Parameterizable(currentSolution).WithParameters(newCoefficients);
    }

    /// <summary>
    /// Updates parameters using GPU-accelerated proximal gradient descent.
    /// </summary>
    /// <remarks>
    /// Proximal gradient descent requires applying proximal operators which vary by regularizer type.
    /// GPU implementation is not yet available due to the variety of proximal operators
    /// (soft-thresholding for L1, projection for constraints, etc.).
    /// </remarks>
    public override void UpdateParametersGpu(IGpuBuffer parameters, IGpuBuffer gradients, int parameterCount, IDirectGpuBackend backend)
    {
        throw new NotSupportedException(
            "GPU-accelerated proximal gradient descent is not yet implemented. " +
            "Proximal operators vary by regularizer type and require specialized kernels. " +
            "Use CPU-based UpdateParameters or consider using Adam/AdamW with weight decay for GPU-resident training.");
    }

    /// <summary>
    /// Reverses a Proximal Gradient Descent update to recover original parameters.
    /// </summary>
    /// <param name="updatedParameters">Parameters after PGD update</param>
    /// <param name="appliedGradients">The gradients that were applied</param>
    /// <returns>Original parameters before the update</returns>
    /// <remarks>
    /// <para>
    /// PGD applies vanilla gradient descent followed by a proximal operator (regularization).
    /// The reverse update undoes the gradient step. Note: The regularization cannot be perfectly
    /// reversed since the proximal operator is generally not invertible.
    /// </para>
    /// <para><b>For Beginners:</b> This calculates where parameters were before a PGD update.
    /// PGD takes a gradient step then applies regularization. We can reverse the gradient step
    /// but the regularization effect remains, since regularization is one-way (like rounding numbers).
    /// </para>
    /// </remarks>
    public override Vector<T> ReverseUpdate(Vector<T> updatedParameters, Vector<T> appliedGradients)
    {
        if (updatedParameters == null)
            throw new ArgumentNullException(nameof(updatedParameters));
        if (appliedGradients == null)
            throw new ArgumentNullException(nameof(appliedGradients));

        if (updatedParameters.Length != appliedGradients.Length)
        {
            throw new ArgumentException(
                $"Updated parameters size ({updatedParameters.Length}) must match applied gradients size ({appliedGradients.Length})",
                nameof(appliedGradients));
        }

        if (_previousParameters == null || _previousParameters.Length != updatedParameters.Length)
        {
            throw new InvalidOperationException(
                "Proximal GD optimizer state is not initialized. ReverseUpdate must be called after UpdateSolution.");
        }

        // === Vectorized Reverse PGD Update (Phase B: US-GPU-015) ===
        // PGD's proximal operator (regularization) cannot be inverted.
        // Return the pre-update parameters that were saved in UpdateSolution.
        // This is the best we can do since the proximal operator is irreversible.
        return new Vector<T>(_previousParameters);
    }

    /// <summary>
    /// Updates adaptive parameters based on optimization progress.
    /// </summary>
    /// <param name="currentStepData">The data from the current optimization step.</param>
    /// <param name="previousStepData">The data from the previous optimization step.</param>
    /// <remarks>
    /// <para>
    /// This method overrides the base implementation to update PGD-specific adaptive parameters
    /// in addition to the base adaptive parameters. It adjusts the learning rate based on whether
    /// the algorithm is making progress. If there is improvement, the learning rate increases; otherwise,
    /// it decreases. The learning rate is kept within specified minimum and maximum limits.
    /// </para>
    /// <para><b>For Beginners:</b> This method adjusts how the algorithm searches based on its progress.
    /// 
    /// It's like a hiker changing their approach:
    /// - If they're finding better spots, they might take bigger steps to progress more quickly
    /// - If they're not finding improvements, they might take smaller steps to search more carefully
    /// - The step size always stays between minimum and maximum values to avoid extremes
    /// 
    /// These adaptive adjustments help the algorithm be more efficient by being bold when things
    /// are going well and cautious when progress is difficult.
    /// </para>
    /// </remarks>
    protected override void UpdateAdaptiveParameters(OptimizationStepData<T, TInput, TOutput> currentStepData, OptimizationStepData<T, TInput, TOutput> previousStepData)
    {
        base.UpdateAdaptiveParameters(currentStepData, previousStepData);

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
    /// Updates the optimizer's options with the provided options.
    /// </summary>
    /// <param name="options">The options to apply to this optimizer.</param>
    /// <exception cref="ArgumentException">Thrown when the options are not of the expected type.</exception>
    /// <remarks>
    /// <para>
    /// This method overrides the base implementation to update the PGD-specific options.
    /// It checks that the provided options are of the correct type (ProximalGradientDescentOptimizerOptions)
    /// and throws an exception if they are not.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates the settings that control how the optimizer works.
    /// 
    /// It's like changing the game settings:
    /// - You provide a set of options to use
    /// - The method checks that these are the right kind of options for a PGD optimizer
    /// - If they are, it applies these new settings
    /// - If not, it lets you know there's a problem
    /// 
    /// This ensures that only appropriate settings are used with this specific optimizer.
    /// </para>
    /// </remarks>
    protected override void UpdateOptions(OptimizationAlgorithmOptions<T, TInput, TOutput> options)
    {
        if (options is ProximalGradientDescentOptimizerOptions<T, TInput, TOutput> pgdOptions)
        {
            _options = pgdOptions;
            // The proximal operator is derived from the options, so it has to be re-derived with them —
            // otherwise a reconfigured optimizer keeps running the regularizer it was constructed with.
            _regularization = BuildProximalOperator(_options);
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected ProximalGradientDescentOptimizerOptions.");
        }
    }

    /// <summary>
    /// Gets the current options for this optimizer.
    /// </summary>
    /// <returns>The current proximal gradient descent optimization options.</returns>
    /// <remarks>
    /// <para>
    /// This method overrides the base implementation to return the PGD-specific options.
    /// </para>
    /// <para><b>For Beginners:</b> This method returns the current settings of the optimizer.
    /// 
    /// It's like checking what game settings are currently active:
    /// - You can see the current learning rate settings
    /// - You can see the current tolerance and iteration limits
    /// - You can see all the other parameters that control the optimizer
    /// 
    /// This is useful for understanding how the optimizer is currently configured
    /// or for making a copy of the settings to modify and apply later.
    /// </para>
    /// </remarks>
    public override OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions()
    {
        return _options;
    }

    /// <summary>
    /// Serializes the proximal gradient descent optimizer to a byte array for storage or transmission.
    /// </summary>
    /// <returns>A byte array containing the serialized optimizer.</returns>
    /// <remarks>
    /// <para>
    /// This method overrides the base implementation to include PGD-specific information in the serialization.
    /// It first serializes the base class data, then adds the PGD options and iteration count.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves the current state of the optimizer so it can be restored later.
    /// 
    /// It's like taking a snapshot of the optimizer:
    /// - First, it saves all the general optimizer information
    /// - Then, it saves the PGD-specific settings and state
    /// - It packages everything into a format that can be saved to a file or sent over a network
    /// 
    /// This allows you to:
    /// - Save a trained optimizer to use later
    /// - Share an optimizer with others
    /// - Create a backup before making changes
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

            return ms.ToArray();
        }
    }

    /// <summary>
    /// Reconstructs the proximal gradient descent optimizer from a serialized byte array.
    /// </summary>
    /// <param name="data">The byte array containing the serialized optimizer.</param>
    /// <exception cref="InvalidOperationException">Thrown when the options cannot be deserialized.</exception>
    /// <remarks>
    /// <para>
    /// This method overrides the base implementation to handle PGD-specific information during deserialization.
    /// It first deserializes the base class data, then reconstructs the PGD options and iteration count.
    /// </para>
    /// <para><b>For Beginners:</b> This method restores the optimizer from a previously saved state.
    /// 
    /// It's like restoring from a snapshot:
    /// - First, it loads all the general optimizer information
    /// - Then, it loads the PGD-specific settings and state
    /// - It reconstructs the optimizer to the exact state it was in when saved
    /// 
    /// This allows you to:
    /// - Continue working with an optimizer you previously saved
    /// - Use an optimizer that someone else created and shared
    /// - Revert to a backup if needed
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
            _options = JsonConvert.DeserializeObject<ProximalGradientDescentOptimizerOptions<T, TInput, TOutput>>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");
            // Same reason as UpdateOptions: a restored optimizer that kept the constructor's operator would
            // resume training with a different algorithm from the one that was serialized.
            _regularization = BuildProximalOperator(_options);

            _iteration = reader.ReadInt32();
        }
    }

    /// <summary>
    /// Generates a unique key for caching gradients based on the model, input data, and optimizer state.
    /// </summary>
    /// <param name="model">The model for which the gradient is calculated.</param>
    /// <param name="X">The input features matrix.</param>
    /// <param name="y">The target values vector.</param>
    /// <returns>A string key that uniquely identifies this gradient calculation.</returns>
    /// <remarks>
    /// <para>
    /// This method overrides the base implementation to include PGD-specific information in the cache key.
    /// It extends the base key with information about the learning rate, regularization type, tolerance,
    /// and current iteration. This ensures that gradients are properly cached and retrieved even as the
    /// optimizer's state changes.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a unique identification tag for each gradient calculation.
    /// 
    /// Think of it like a file naming system:
    /// - It includes information about the model and data being used
    /// - It adds details specific to the PGD optimizer's current state
    /// - This unique tag helps the optimizer avoid redundant calculations
    /// - If the same gradient is needed again, it can be retrieved from cache instead of recalculated
    /// 
    /// This caching mechanism improves efficiency by avoiding duplicate work.
    /// </para>
    /// </remarks>
    protected override string GenerateGradientCacheKey(IFullModel<T, TInput, TOutput> model, TInput X, TOutput y)
    {
        var baseKey = base.GenerateGradientCacheKey(model, X, y);
        return $"{baseKey}_PGD_{_options.InitialLearningRate}_{_regularization.GetType().Name}_{_options.Tolerance}_{_iteration}";
    }

    /// <summary>
    /// Applies one proximal-gradient step to a flat parameter vector: a gradient step, then the proximal
    /// operator.
    /// </summary>
    /// <param name="parameters">The current parameters.</param>
    /// <param name="gradient">The gradient at those parameters.</param>
    /// <returns>The updated parameters.</returns>
    /// <remarks>
    /// <para>
    /// Without this override the vector path inherited the base class's plain <c>param -= lr*grad</c> and
    /// never applied the proximal operator at all — so a caller who asked for proximal gradient descent got
    /// gradient descent, silently, with the regularizer they configured having no effect on the result. The
    /// prox is not an add-on to PGD; it is the half of the algorithm that makes it PGD.
    /// </para>
    /// <para>
    /// This is the same update <see cref="Step"/> applies on the tape path, so both eager paths and the fused
    /// kernel now agree.
    /// </para>
    /// </remarks>
    public override Vector<T> UpdateParameters(Vector<T> parameters, Vector<T> gradient)
    {
        if (parameters.Length != gradient.Length)
        {
            throw new ArgumentException(
                $"Parameter vector length ({parameters.Length}) must match gradient vector length ({gradient.Length}).",
                nameof(gradient));
        }

        var gradientStep = (Vector<T>)Engine.Multiply(gradient, CurrentLearningRate);
        var stepped = (Vector<T>)Engine.Subtract(parameters, gradientStep);

        return _regularization.Regularize(stepped);
    }

    /// <inheritdoc />
    public override void Step(TapeStepContext<T> context)
    {
        // GPU-resident ISTA step: when the regularizer is L1 (the canonical
        // proximal-gradient case), one proximal_l1_update kernel does
        // tmp = param - lr*grad; param = sign(tmp)*max(|tmp| - strength, 0),
        // matching the CPU "subtract gradient, then L1 soft-threshold prox" path
        // exactly. Other regularizers (or non-GPU-resident tensors) fall back to CPU.
        bool gpuL1 = typeof(T) == typeof(float)
            && System.Environment.GetEnvironmentVariable("AIDOTNET_GPU_ADAM") == "1"
            && AiDotNet.Tensors.Engines.AiDotNetEngine.Current is AiDotNet.Tensors.Engines.DirectGpuTensorEngine
            && _regularization is AiDotNet.Regularization.L1Regularization<T, TInput, TOutput>;
        float l1Strength = gpuL1 ? (float)_regularization.GetOptions().Strength : 0f;

        foreach (var param in context.Parameters)
        {
            // True sparse scatter Proximal-L1: SGD step + L1 soft-threshold at touched
            // indices. Only used when the regularizer is L1 (otherwise the proximal
            // operator may need to inspect every element).
            bool useSparseL1 = !gpuL1
                && _regularization is AiDotNet.Regularization.L1Regularization<T, TInput, TOutput>
                && SparseEmbeddingOptimizerHelpers.HasSparseEmbeddingGrad(param);
            if (useSparseL1)
            {
                if (SparseEmbeddingOptimizerHelpers.TryApplyProximalL1Sparse(
                        param,
                        NumOps.ToDouble(CurrentLearningRate),
                        _regularization.GetOptions().Strength))
                {
                    continue;
                }
            }

            if (!SparseEmbeddingOptimizerHelpers.TryGetEffectiveGradient(context, param, Engine, out var grad))
                continue;

            if (gpuL1 && param.Length == grad.Length
                && AiDotNet.Tensors.Engines.Gpu.GpuOptimizer.TryProximalL1Step((Tensor<float>)(object)param, (Tensor<float>)(object)grad,
                    (float)NumOps.ToDouble(CurrentLearningRate), l1Strength))
                continue;

            // Gradient descent step: param -= lr * grad
            var update = Engine.TensorMultiplyScalar(grad, CurrentLearningRate);
            Engine.TensorSubtractInPlace(param, update);

            // Apply proximal operator (regularization) via Vector<T> interface
            var paramVec = param.ToVector();
            var regularized = _regularization.Regularize(paramVec);
            for (int i = 0; i < param.Length && i < regularized.Length; i++)
                param[i] = regularized[i];
        }
    }
}




