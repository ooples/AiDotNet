using AiDotNet.Engines;
using System.Collections.Concurrent;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu;
using Newtonsoft.Json;
using AiDotNet.Helpers;

using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.Optimizers;

/// <summary>
/// Implements the Root Mean Square Propagation (RMSProp) optimization algorithm, an adaptive learning rate method.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// RMSProp is an adaptive learning rate optimization algorithm designed to handle non-stationary
/// objectives and accelerate convergence. It maintains a moving average of the squared gradients
/// for each parameter and divides the learning rate by the square root of this average. This
/// approach allows the algorithm to use a larger learning rate for parameters with small gradients
/// and a smaller learning rate for parameters with large gradients, leading to more efficient optimization.
/// </para>
/// <para><b>For Beginners:</b> RMSProp is like a hiker who adjusts their step size differently for each direction.
/// 
/// Imagine a hiker exploring mountains with different terrains:
/// - On steep slopes (large gradients), the hiker takes small, careful steps
/// - On gentle slopes (small gradients), the hiker takes larger, confident steps
/// - The hiker remembers how steep each direction has been recently (using a moving average)
/// - This memory helps the hiker adjust their steps even as the terrain changes
/// 
/// This adaptive approach helps the algorithm find good solutions more quickly by:
/// - Preventing wild overshooting on steep slopes
/// - Making faster progress on gentle terrain
/// - Adjusting automatically to different parts of the solution space
/// </para>
/// </remarks>
[ComponentType(ComponentType.Optimizer)]
[PipelineStage(PipelineStage.Training)]
public partial class RootMeanSquarePropagationOptimizer<T, TInput, TOutput> : GradientBasedOptimizerBase<T, TInput, TOutput>, Fused.IFusedOptimizerSpec
{
    /// <summary>
    /// Describes the momentum-free, uncentered RMSProp update supported by the
    /// fused kernel. The opt-in centered Graves variant remains on the eager
    /// path because the fused kernel cannot represent its mean-gradient or velocity state.
    /// </summary>
    bool Fused.IFusedOptimizerSpec.TryGetFusedOptimizerConfig(out Fused.FusedOptimizerConfig config)
    {
        config = default;
        if (_options.UseAdaptiveLearningRate || _options.Centered)
        {
            return false;
        }
        if (!TryGetFusedLrSchedule(out var schedule)) return false;
        config = new Fused.FusedOptimizerConfig(
            Tensors.Engines.Compilation.OptimizerType.RMSprop,
            (float)GetCurrentLearningRate(),
            0f, (float)_options.Decay, (float)_options.Epsilon, 0f, schedule);
        return true;
    }

    /// <summary>
    /// Moving average of squared gradients for each parameter.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores a running average of the squared gradients for each parameter in the model.
    /// It is used to adapt the learning rate individually for each parameter, with larger accumulated
    /// squared gradients resulting in smaller effective learning rates.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the hiker's memory of how steep each direction has been.
    /// 
    /// This moving average:
    /// - Keeps track of the squared gradient (steepness) for each direction
    /// - Gives more weight to recent observations and gradually forgets older ones
    /// - Helps determine how cautious to be when stepping in each direction
    /// - A consistently steep direction will have a large value, signaling the need for smaller steps
    /// 
    /// This adaptive memory allows the algorithm to respond differently to different parameters based on their history.
    /// </para>
    /// </remarks>
    private Vector<T> _squaredGradient;

    /// <summary>Moving average of gradients used by centered RMSProp.</summary>
    private Vector<T> _meanGradient;

    /// <summary>The most recent RMSProp velocity/update for each flat parameter.</summary>
    private Vector<T> _velocity;


    /// <summary>
    /// The current iteration count of the optimization process.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field keeps track of the number of completed iterations in the optimization process.
    /// It is used for gradient caching and can be useful for monitoring the progress of the optimization.
    /// </para>
    /// <para><b>For Beginners:</b> This is like counting how many steps the hiker has taken.
    /// 
    /// The iteration counter:
    /// - Keeps track of how many rounds of optimization have been completed
    /// - Helps with creating unique cache keys for gradient calculations
    /// - Can be used to monitor how the algorithm is progressing
    /// 
    /// This simple counter plays an important role in the optimization process.
    /// </para>
    /// </remarks>
    private int _t;

    /// <summary>
    /// Configuration options specific to the RMSProp algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the configuration parameters for the RMSProp algorithm, such as
    /// the decay rate for the moving average, epsilon for numerical stability, and the
    /// maximum number of iterations. These parameters control the behavior of the optimizer
    /// and affect its performance and convergence properties.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the instruction manual for the optimizer.
    /// 
    /// The options control:
    /// - How quickly the algorithm forgets old gradient information (decay rate)
    /// - How to prevent division by very small numbers (epsilon)
    /// - When to stop the optimization process (maximum iterations, tolerance)
    /// 
    /// Adjusting these settings can help the algorithm work better for different types of problems.
    /// </para>
    /// </remarks>
    /// <summary>Read from the single instance OptimizerBase.Options holds, so there is
    /// no second copy that could disagree with it.</summary>
    private RootMeanSquarePropagationOptimizerOptions<T, TInput, TOutput> _options
        => (RootMeanSquarePropagationOptimizerOptions<T, TInput, TOutput>)Options;

    /// <summary>
    /// Initializes a new instance of the <see cref="RootMeanSquarePropagationOptimizer{T}"/> class with the specified options and components.
    /// </summary>
    /// <param name="options">The RMSProp optimization options, or null to use default options.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new RMSProp optimizer with the specified options and components.
    /// If any parameter is null, a default implementation is used. The constructor initializes
    /// the iteration counter, squared gradient vector, and options.
    /// </para>
    /// <para><b>For Beginners:</b> This is the starting point for creating a new optimizer.
    /// 
    /// Think of it like preparing for a hiking expedition:
    /// - You can provide custom settings (options) or use the default ones
    /// - You can provide specialized tools (evaluators, calculators) or use the basic ones
    /// - It initializes everything the optimizer needs to start working
    /// - The squared gradient starts empty because there's no history yet
    /// - The step counter starts at zero because no steps have been taken
    /// 
    /// This constructor gets everything ready so you can start the optimization process.
    /// </para>
    /// </remarks>
    public RootMeanSquarePropagationOptimizer(
        IFullModel<T, TInput, TOutput> model,
        RootMeanSquarePropagationOptimizerOptions<T, TInput, TOutput>? options = null,
        IEngine? engine = null)
        : base(model, options ?? new())
    {
        _t = 0;
        _squaredGradient = Vector<T>.Empty();
        _meanGradient = Vector<T>.Empty();
        _velocity = Vector<T>.Empty();
        CurrentMomentum = NumOps.FromDouble(_options.InitialMomentum);
    }


    /// <summary>
    /// Creates an RMSProp optimizer for minimizing a plain function, with no model attached.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Use this with <see cref="GradientBasedOptimizerBase{T, TInput, TOutput}.Minimize(Vector{T}, Func{Vector{T}, ValueTuple{T, Vector{T}}}, int, T)"/>
    /// when you want to minimize a mathematical function directly rather than train a model.
    /// <see cref="Optimize"/> requires a model and is not available on an instance created
    /// this way.
    /// </para>
    /// <para><b>For Beginners:</b> The constructor above asks for a model because it is set up
    /// to tune that model against training data. If all you have is a formula you want to make
    /// as small as possible, there is no model to hand over — use this factory instead.
    /// </para>
    /// </remarks>
    /// <param name="options">The optimizer-specific options. If null, defaults are used.</param>
    public static RootMeanSquarePropagationOptimizer<T, TInput, TOutput> CreateForFunction(
        RootMeanSquarePropagationOptimizerOptions<T, TInput, TOutput>? options = null)
        => new(options);

    /// <summary>
    /// Backs <see cref="CreateForFunction"/>: the same setup with no model.
    /// </summary>
    private RootMeanSquarePropagationOptimizer(RootMeanSquarePropagationOptimizerOptions<T, TInput, TOutput>? options)
        : base(null, options ?? new())
    {
        _t = 0;
        _squaredGradient = Vector<T>.Empty();
        _meanGradient = Vector<T>.Empty();
        _velocity = Vector<T>.Empty();
        CurrentMomentum = NumOps.FromDouble(_options.InitialMomentum);
    }


    /// <summary>
    /// Performs the RMSProp optimization to find the best solution for the given input data.
    /// </summary>
    /// <param name="inputData">The input data to optimize against.</param>
    /// <returns>An optimization result containing the best solution found and associated metrics.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the main RMSProp algorithm. It starts from a random solution and
    /// iteratively improves it by calculating the gradient, applying momentum, updating the solution
    /// based on the adaptive learning rates, and evaluating the new solution. The process continues
    /// until either the maximum number of iterations is reached, early stopping criteria are met,
    /// or the improvement falls below the specified tolerance.
    /// </para>
    /// <para><b>For Beginners:</b> This is the main search process where the algorithm looks for the best solution.
    /// 
    /// The process works like this:
    /// 1. Start at a random position on the "landscape"
    /// 2. Initialize the squared gradient history and step counter
    /// 3. For each iteration:
    ///    - Figure out which direction is most uphill (calculate gradient)
    ///    - Apply momentum to smooth the movement
    ///    - Take a step using adaptive step sizes for each direction
    ///    - Check if the new position is better than the best found so far
    ///    - Update the adaptive parameters based on progress
    /// 4. Stop when enough iterations are done, when no more improvement is happening, or when the
    ///    improvement is very small
    /// 
    /// This approach efficiently finds good solutions by adapting its behavior based on the shape
    /// of the optimization landscape.
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

        _squaredGradient = new Vector<T>(InterfaceGuard.Parameterizable(currentSolution).GetParameters().Length);
        _t = 0;
        InitializeAdaptiveParameters();

        for (int epoch = 0; epoch < _options.MaxIterations; epoch++)
        {
            NotifyEpochStart(epoch);
            var batcher = CreateBatcher(inputData, _options.BatchSize, epoch);

            foreach (var (xBatch, yBatch, batchIndices) in batcher.GetBatches())
            {
                _t++;
                var gradient = CalculateGradient(currentSolution, xBatch, yBatch);
                if (!_options.Centered)
                    gradient = ApplyMomentum(gradient);
                currentSolution = UpdateSolution(currentSolution, gradient);
            }

            var currentStepData = EvaluateSolution(currentSolution, inputData);
            UpdateBestSolution(currentStepData, ref bestStepData);

            UpdateAdaptiveParameters(currentStepData, previousStepData);

            if (UpdateIterationHistoryAndCheckEarlyStopping(epoch, bestStepData))
            {
                break;
            }

            // Check convergence against previousStepData (per-epoch progress),
            // not bestStepData. UpdateBestSolution above copies currentStepData
            // into bestStepData on the first iteration, so |best - current|
            // would always be 0 < tolerance and the optimiser would exit after
            // the first epoch. Issue #1340 / PR #1351 fix swept across the
            // optimizer suite.
            if (epoch > 0 && NumOps.LessThan(NumOps.Abs(NumOps.Subtract(previousStepData.FitnessScore, currentStepData.FitnessScore)), NumOps.FromDouble(_options.Tolerance)))
            {
                break;
            }

            previousStepData = currentStepData;
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    /// <summary>
    /// Updates a vector of parameters using the RMSProp algorithm.
    /// </summary>
    /// <param name="parameters">The parameters to update.</param>
    /// <param name="gradient">The gradient vector for the parameters.</param>
    /// <returns>The updated parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the core RMSProp update rule. For each parameter, it:
    /// 1. Updates the running average of squared gradients
    /// 2. Calculates an adaptive learning rate by dividing the base learning rate by the square root
    ///    of the running average (plus epsilon for numerical stability)
    /// 3. Updates the parameter by subtracting the product of the adaptive learning rate and the gradient
    /// </para>
    /// <para><b>For Beginners:</b> This method adjusts each parameter based on its gradient history.
    /// 
    /// For each parameter:
    /// - It updates the memory of how steep this direction has been (squared gradient)
    /// - It calculates a custom step size based on the steepness history
    /// - Parameters with consistently large gradients get smaller steps
    /// - Parameters with consistently small gradients get larger steps
    /// - It then updates the parameter value using this custom step size
    /// 
    /// This adaptive approach helps the algorithm converge faster by giving each parameter
    /// exactly the step size it needs.
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

        if (_squaredGradient.Length != parameters.Length
            || (_options.Centered
                && (_meanGradient.Length != parameters.Length || _velocity.Length != parameters.Length)))
        {
            _squaredGradient = new Vector<T>(parameters.Length);
            _meanGradient = _options.Centered ? new Vector<T>(parameters.Length) : Vector<T>.Empty();
            _velocity = _options.Centered ? new Vector<T>(parameters.Length) : Vector<T>.Empty();
        }

        T decay = NumOps.FromDouble(_options.Decay);
        T oneMinusDecay = NumOps.FromDouble(1 - _options.Decay);
        T epsilon = NumOps.FromDouble(_options.Epsilon);
        T momentum = CurrentMomentum;
        T learningRate = CurrentLearningRate;

        var updatedParams = new Vector<T>(parameters.Length, skipZeroInit: true);
        var pSpan = parameters.AsSpan();
        var gSpan = gradient.AsSpan();
        var sqGradSpan = _squaredGradient.AsWritableSpan();
        var meanGradSpan = _meanGradient.AsWritableSpan();
        var velocitySpan = _velocity.AsWritableSpan();
        var outSpan = updatedParams.AsWritableSpan();

        for (int i = 0; i < pSpan.Length; i++)
        {
            T g = gSpan[i];
            T squaredGradient = NumOps.Add(
                NumOps.Multiply(sqGradSpan[i], decay),
                NumOps.Multiply(NumOps.Multiply(g, g), oneMinusDecay));
            sqGradSpan[i] = squaredGradient;

            if (_options.Centered)
            {
                T meanGradient = NumOps.Add(
                    NumOps.Multiply(meanGradSpan[i], decay),
                    NumOps.Multiply(g, oneMinusDecay));
                meanGradSpan[i] = meanGradient;
                T variance = NumOps.Subtract(squaredGradient, NumOps.Multiply(meanGradient, meanGradient));
                variance = MathHelper.Max(variance, NumOps.Zero);
                T denominator = NumOps.Sqrt(NumOps.Add(variance, epsilon));
                T normalizedStep = NumOps.Divide(NumOps.Multiply(g, learningRate), denominator);
                T velocity = NumOps.Add(NumOps.Multiply(velocitySpan[i], momentum), normalizedStep);
                velocitySpan[i] = velocity;
                outSpan[i] = NumOps.Subtract(pSpan[i], velocity);
            }
            else
            {
                T denominator = NumOps.Add(NumOps.Sqrt(squaredGradient), epsilon);
                T update = NumOps.Divide(NumOps.Multiply(g, learningRate), denominator);
                outSpan[i] = NumOps.Subtract(pSpan[i], update);
            }
        }

        return updatedParams;
    }

    // Per-parameter state for tape-based training. These dictionaries are also
    // captured by the optimizer's tape-state serializer.
    private readonly ConcurrentDictionary<Tensor<T>, Tensor<T>> _tapeSqGrad = new(TensorReferenceComparer<Tensor<T>>.Instance);
    private readonly ConcurrentDictionary<Tensor<T>, Tensor<T>> _tapeMeanGrad = new(TensorReferenceComparer<Tensor<T>>.Instance);
    private readonly ConcurrentDictionary<Tensor<T>, Tensor<T>> _tapeVelocity = new(TensorReferenceComparer<Tensor<T>>.Instance);

    /// <inheritdoc />
    public override void Step(TapeStepContext<T> context)
    {
        PrepareTapeState(context);

        T decay = NumOps.FromDouble(_options.Decay);
        T oneMinusDecay = NumOps.FromDouble(1 - _options.Decay);
        T epsilon = NumOps.FromDouble(_options.Epsilon);
        T momentum = CurrentMomentum;
        bool requiresExtendedState = _options.Centered;

        // Existing GPU and sparse kernels implement only uncentered, momentum-free RMSProp.
        bool gpuRmsProp = SupportsGpuUpdate
            && typeof(T) == typeof(float)
            && System.Environment.GetEnvironmentVariable("AIDOTNET_GPU_ADAM") == "1"
            && AiDotNet.Tensors.Engines.AiDotNetEngine.Current is AiDotNet.Tensors.Engines.DirectGpuTensorEngine;

        foreach (var param in context.Parameters)
        {
            if (!requiresExtendedState
                && !gpuRmsProp
                && SparseEmbeddingOptimizerHelpers.HasSparseEmbeddingGrad(param))
            {
                if (!_tapeSqGrad.TryGetValue(param, out var sqGradSp))
                {
                    sqGradSp = new Tensor<T>(param._shape);
                    _tapeSqGrad[param] = sqGradSp;
                }

                if (SparseEmbeddingOptimizerHelpers.TryApplyRmspropSparse(
                        param, sqGradSp,
                        NumOps.ToDouble(CurrentLearningRate),
                        _options.Decay, _options.Epsilon, weightDecay: 0.0))
                {
                    continue;
                }
            }

            if (!SparseEmbeddingOptimizerHelpers.TryGetEffectiveGradient(context, param, Engine, out var grad))
                continue;

            if (!_tapeSqGrad.TryGetValue(param, out var sqGrad))
            {
                sqGrad = gpuRmsProp
                    ? AiDotNet.Tensors.Helpers.TensorAllocator.RentPinnedOnGpu<T>(param._shape)
                    : new Tensor<T>(param._shape);
                if (gpuRmsProp) sqGrad.AsWritableSpan().Clear();
                _tapeSqGrad[param] = sqGrad;
            }

            if (gpuRmsProp && param.Length == grad.Length
                && AiDotNet.Tensors.Engines.Gpu.GpuOptimizer.TryRmspropStep(
                    (Tensor<float>)(object)param,
                    (Tensor<float>)(object)grad,
                    (Tensor<float>)(object)sqGrad,
                    (float)NumOps.ToDouble(CurrentLearningRate),
                    (float)_options.Decay,
                    (float)_options.Epsilon,
                    0f))
            {
                continue;
            }

            if (requiresExtendedState)
            {
                if (!_tapeMeanGrad.TryGetValue(param, out var meanGrad))
                {
                    meanGrad = new Tensor<T>(param._shape);
                    _tapeMeanGrad[param] = meanGrad;
                }

                if (!_tapeVelocity.TryGetValue(param, out var velocity))
                {
                    velocity = new Tensor<T>(param._shape);
                    _tapeVelocity[param] = velocity;
                }

                var paramSpan = param.AsWritableSpan();
                var gradSpan = grad.Data.Span;
                var sqGradSpan = sqGrad.AsWritableSpan();
                var meanGradSpan = meanGrad.AsWritableSpan();
                var velocitySpan = velocity.AsWritableSpan();

                for (int i = 0; i < param.Length; i++)
                {
                    T g = gradSpan[i];
                    T squaredGradient = NumOps.Add(
                        NumOps.Multiply(sqGradSpan[i], decay),
                        NumOps.Multiply(NumOps.Multiply(g, g), oneMinusDecay));
                    sqGradSpan[i] = squaredGradient;

                    T denominator;
                    if (_options.Centered)
                    {
                        T meanGradient = NumOps.Add(
                            NumOps.Multiply(meanGradSpan[i], decay),
                            NumOps.Multiply(g, oneMinusDecay));
                        meanGradSpan[i] = meanGradient;
                        T variance = NumOps.Subtract(squaredGradient, NumOps.Multiply(meanGradient, meanGradient));
                        variance = MathHelper.Max(variance, NumOps.Zero);
                        denominator = NumOps.Sqrt(NumOps.Add(variance, epsilon));
                    }
                    else
                    {
                        denominator = NumOps.Add(NumOps.Sqrt(squaredGradient), epsilon);
                    }

                    T normalizedStep = NumOps.Divide(NumOps.Multiply(g, CurrentLearningRate), denominator);
                    T nextVelocity = NumOps.Add(NumOps.Multiply(velocitySpan[i], momentum), normalizedStep);
                    velocitySpan[i] = nextVelocity;
                    paramSpan[i] = NumOps.Subtract(paramSpan[i], nextVelocity);
                }

                continue;
            }

            // Momentum-free, uncentered path retained for fused/eager parity.
            var sqGradNew = Engine.TensorAdd(
                Engine.TensorMultiplyScalar(sqGrad, decay),
                Engine.TensorMultiplyScalar(Engine.TensorMultiply(grad, grad), oneMinusDecay));
            Engine.TensorCopy(sqGradNew, sqGrad);

            var denom = Engine.TensorAddScalar(Engine.TensorSqrt(sqGrad), epsilon);
            var update = Engine.TensorMultiplyScalar(Engine.TensorDivide(grad, denom), CurrentLearningRate);
            Engine.TensorSubtractInPlace(param, update);
        }
    }

    /// <summary>
    /// Reverses an RMSProp gradient update to recover original parameters.
    /// </summary>
    /// <param name="updatedParameters">Parameters after RMSProp update</param>
    /// <param name="appliedGradients">The gradients that were applied</param>
    /// <returns>Original parameters before the update</returns>
    /// <remarks>
    /// <para>
    /// Updates a solution model using the RMSProp algorithm.
    /// </summary>
    /// <param name="currentSolution">The current solution model to update.</param>
    /// <param name="gradient">The gradient vector for the solution.</param>
    /// <returns>The updated solution model.</returns>
    /// <remarks>
    /// <para>
    /// This method applies the RMSProp update rule to the coefficients of a solution model.
    /// It follows the same steps as UpdateVector, but operates directly on the solution model's
    /// coefficients. For each coefficient, it:
    /// 1. Updates the running average of squared gradients
    /// 2. Calculates an adaptive learning rate by dividing the base learning rate by the square root
    ///    of the running average (plus epsilon for numerical stability)
    /// 3. Updates the coefficient by subtracting the product of the adaptive learning rate and the gradient
    /// </para>
    /// <para><b>For Beginners:</b> This method moves the solution in the direction of improvement.
    /// 
    /// Think of it as the hiker taking one step:
    /// - For each direction, it updates the memory of how steep that direction has been
    /// - It calculates custom step sizes for each direction based on their history
    /// - Steeper directions get smaller, more careful steps
    /// - Gentler directions get larger, more confident steps
    /// - The solution then moves according to these personalized step sizes
    /// 
    /// This adaptive movement helps the algorithm navigate efficiently toward better solutions.
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
        var parameters = InterfaceGuard.Parameterizable(currentSolution).GetParameters();
        var updatedParams = UpdateParameters(parameters, gradient);
        return InterfaceGuard.Parameterizable(currentSolution).WithParameters(updatedParams);
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
    /// This method overrides the base implementation to include RMSProp-specific information in the cache key.
    /// It extends the base key with information about the current learning rate, decay rate, epsilon value,
    /// and iteration count. This ensures that gradients are properly cached and retrieved even as the
    /// optimizer's state changes.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a unique identification tag for each gradient calculation.
    /// 
    /// Think of it like a file naming system:
    /// - It includes information about the model and data being used
    /// - It adds details specific to the RMSProp optimizer's current state
    /// - This unique tag helps the optimizer avoid redundant calculations
    /// - If the same gradient is needed again, it can be retrieved from cache instead of recalculated
    /// 
    /// This caching mechanism improves efficiency by avoiding duplicate work.
    /// </para>
    /// </remarks>
    protected override string GenerateGradientCacheKey(IFullModel<T, TInput, TOutput> model, TInput X, TOutput y)
    {
        var baseKey = base.GenerateGradientCacheKey(model, X, y);
        return $"{baseKey}_RMSprop_{CurrentLearningRate}_{CurrentMomentum}_{_options.Decay}_{_options.Epsilon}_{_options.Centered}_{_t}";
    }

    /// <summary>
    /// Resets the optimizer to its initial state.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method overrides the base implementation to reset RMSProp-specific state variables
    /// in addition to the base state. It resets the iteration counter and clears the squared
    /// gradient history, preparing the optimizer for a fresh start.
    /// </para>
    /// <para><b>For Beginners:</b> This method prepares the optimizer to start fresh.
    /// 
    /// It's like a hiker:
    /// - Returning to the starting point
    /// - Resetting their step counter to zero
    /// - Clearing their memory of previous terrain steepness
    /// 
    /// This allows the optimizer to begin a new optimization process without being influenced
    /// by previous runs.
    /// </para>
    /// </remarks>
    public override void Reset()
    {
        base.Reset();
        _t = 0;
        _squaredGradient = Vector<T>.Empty();
        _meanGradient = Vector<T>.Empty();
        _velocity = Vector<T>.Empty();
        _tapeSqGrad.Clear();
        _tapeMeanGrad.Clear();
        _tapeVelocity.Clear();
        CurrentMomentum = NumOps.FromDouble(_options.InitialMomentum);
        DisposeGpuState();
    }

    /// <summary>
    /// Gets the current options for this optimizer.
    /// </summary>
    /// <returns>The current RMSProp optimization options.</returns>
    /// <remarks>
    /// <para>
    /// This method overrides the base implementation to return the RMSProp-specific options.
    /// </para>
    /// <para><b>For Beginners:</b> This method returns the current settings of the optimizer.
    /// 
    /// It's like checking what settings are currently active:
    /// - You can see the current decay rate
    /// - You can see the current epsilon value
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
    /// Reverses an RMSprop gradient update to recover original parameters.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For RMSprop, the forward update is:
    /// 1. _squaredGradient[i] = decay * _squaredGradient[i] + (1 - decay) * gradient[i]^2
    /// 2. update = learning_rate * gradient[i] / (sqrt(_squaredGradient[i]) + epsilon)
    /// 3. params_new = params_old - update
    ///
    /// To reverse: params_old = params_new + update
    ///
    /// This requires access to the current squared gradient state and the applied gradients
    /// to recalculate the adaptive update that was applied.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like retracing the hiker's steps. Given where the hiker ended up (updated parameters)
    /// and the terrain steepness history (squared gradients), we can calculate the exact step size
    /// that was used and determine where the hiker started from.
    /// </para>
    /// </remarks>
    /// <param name="updatedParameters">Parameters after gradient application</param>
    /// <param name="appliedGradients">The gradients that were applied</param>
    /// <returns>Original parameters before the gradient update</returns>
    /// <exception cref="ArgumentNullException">If parameters or gradients are null</exception>
    /// <exception cref="ArgumentException">If parameter and gradient sizes do not match</exception>
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

        if (_options.Centered)
        {
            // The velocity stores the exact centered update subtracted by the latest call.
            return _velocity.Length == updatedParameters.Length
                ? (Vector<T>)Engine.Add(updatedParameters, _velocity)
                : base.ReverseUpdate(updatedParameters, appliedGradients);
        }

        if (_squaredGradient.Length != updatedParameters.Length)
            return base.ReverseUpdate(updatedParameters, appliedGradients);

        var numerator = (Vector<T>)Engine.Multiply(
            Vector<T>.CreateDefault(appliedGradients.Length, CurrentLearningRate),
            appliedGradients);
        var denominator = (Vector<T>)Engine.Add(
            Engine.Sqrt(_squaredGradient),
            Vector<T>.CreateDefault(appliedGradients.Length, NumOps.FromDouble(_options.Epsilon)));
        var update = (Vector<T>)Engine.Divide(numerator, denominator);
        return (Vector<T>)Engine.Add(updatedParameters, update);
    }

    #region GPU Optimizer Support

    /// <summary>
    /// GPU buffer for squared gradient moving average.
    /// </summary>
    private IGpuBuffer? _gpuSquaredAvg;

    /// <summary>
    /// Gets whether this optimizer supports GPU-accelerated parameter updates.
    /// The available kernel is uncentered and has no velocity state.
    /// </summary>
    public override bool SupportsGpuUpdate => !_options.Centered;

    /// <summary>
    /// Initializes RMSprop optimizer state on the GPU.
    /// </summary>
    public override void InitializeGpuState(int parameterCount, IDirectGpuBackend backend)
    {
        if (_gpuStateInitialized && _gpuSquaredAvg != null)
            return;

        var zeros = new float[parameterCount];
        _gpuSquaredAvg = backend.AllocateBuffer(zeros);

        _gpuStateInitialized = true;
    }

    /// <summary>
    /// Updates parameters on the GPU using the RMSprop kernel.
    /// </summary>
    public override void UpdateParametersGpu(IGpuBuffer parameters, IGpuBuffer gradients, int parameterCount, IDirectGpuBackend backend)
    {
        if (!SupportsGpuUpdate)
            throw new NotSupportedException("The GPU RMSProp kernel does not support centered RMSProp.");


        if (!_gpuStateInitialized || _gpuSquaredAvg == null)
        {
            InitializeGpuState(parameterCount, backend);
        }

        backend.RmspropUpdate(
            parameters,
            gradients,
            _gpuSquaredAvg!,
            (float)NumOps.ToDouble(CurrentLearningRate),
            (float)_options.Decay,
            (float)_options.Epsilon,
            0.0f, // RMSprop doesn't have weight decay in these options
            parameterCount
        );
    }

    /// <summary>
    /// Disposes GPU-allocated optimizer state.
    /// </summary>
    public override void DisposeGpuState()
    {
        _gpuSquaredAvg?.Dispose();
        _gpuSquaredAvg = null;
        _gpuStateInitialized = false;
    }

    #endregion
}
