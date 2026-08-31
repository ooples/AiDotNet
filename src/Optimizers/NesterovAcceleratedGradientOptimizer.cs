using AiDotNet.Tensors.Engines.DirectGpu;
using System.Collections.Concurrent;
using AiDotNet.Tensors.Engines.Autodiff;
using Newtonsoft.Json;
using AiDotNet.Helpers;

using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.Optimizers;

/// <summary>
/// Implements the Nesterov Accelerated Gradient optimization algorithm.
/// </summary>
/// <remarks>
/// <para>
/// The Nesterov Accelerated Gradient (NAG) is an optimization algorithm that improves upon standard gradient descent.
/// It introduces a smart prediction of the next position of the parameters, which helps to dampen oscillations and
/// improve convergence, especially in scenarios with high curvature or small but consistent gradients.
/// </para>
/// <para><b>For Beginners:</b>
/// Imagine you're skiing down a hill. Regular gradient descent is like looking at your current position to decide where to go next.
/// NAG is like looking ahead to where you'll be after your next move, and then deciding how to adjust your path.
/// This "look-ahead" helps you navigate the slope more efficiently, especially around tricky turns.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[ComponentType(ComponentType.Optimizer)]
[PipelineStage(PipelineStage.Training)]
public partial class NesterovAcceleratedGradientOptimizer<T, TInput, TOutput> : GradientBasedOptimizerBase<T, TInput, TOutput>, Fused.IFusedOptimizerSpec
{
    /// <summary>
    /// Describes this optimizer for the compiled fused-training kernel.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Maps to <see cref="Tensors.Engines.Compilation.OptimizerType.SGDMomentum"/> with the Nesterov flag
    /// set, which runs exactly the three lines <see cref="Step"/> and <see cref="UpdateParameters"/> run:
    /// <c>v = mu*v + g; update = g + mu*v; p -= lr*update</c>.
    /// </para>
    /// <para>
    /// This spec could not be written correctly until two things were fixed. The kernel took a hardcoded
    /// <c>false</c> for its nesterov argument, so the flag had no way to reach it (Tensors #949). And this
    /// optimizer applied CLASSICAL momentum despite its name, so requesting Nesterov here would have run a
    /// different algorithm from the eager path — the divergence a comment in this file had already recorded,
    /// and the reason the CUDA path was left unwired. Both are now closed, so eager, fused and GPU agree.
    /// </para>
    /// <para>
    /// Unlike the CoordinateDescent mapping, no constant-schedule guard is needed. The learning rate is no
    /// longer folded into the velocity, so <c>v</c> means the same thing on both sides at every step and a
    /// moving lr rescales only the current update rather than the accumulated history.
    /// </para>
    /// </remarks>
    bool Fused.IFusedOptimizerSpec.TryGetFusedOptimizerConfig(out Fused.FusedOptimizerConfig config)
    {
        config = default;
        if (_options.UseAdaptiveLearningRate) return false;
        if (!TryGetFusedLrSchedule(out var schedule)) return false;

        config = new Fused.FusedOptimizerConfig(
            Tensors.Engines.Compilation.OptimizerType.SGDMomentum,
            (float)GetCurrentLearningRate(),
            (float)NumOps.ToDouble(CurrentMomentum),   // Beta1 carries the momentum coefficient
            0f, 0f, 0f, schedule)
        {
            Extras = new Tensors.Engines.Compilation.FusedOptimizerExtras { Nesterov = true },
        };
        return true;
    }

    /// <summary>
    /// The options specific to the Nesterov Accelerated Gradient optimizer.
    /// </summary>
    /// <summary>Read from the single instance OptimizerBase.Options holds, so there is
    /// no second copy that could disagree with it.</summary>
    private NesterovAcceleratedGradientOptimizerOptions<T, TInput, TOutput> _options => (NesterovAcceleratedGradientOptimizerOptions<T, TInput, TOutput>)Options;

    /// <summary>
    /// The velocity vector used in the NAG algorithm.
    /// </summary>
    private Vector<T>? _velocity;

    /// <summary>
    /// Initializes a new instance of the NesterovAcceleratedGradientOptimizer class.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This constructor sets up the NAG optimizer with the provided options and dependencies.
    /// If no options are provided, it uses default settings.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like preparing your skis and gear before you start your descent. You're setting up all the tools and rules you'll use during your optimization journey.
    /// </para>
    /// </remarks>
    /// <param name="model">The model to optimize.</param>
    /// <param name="options">The NAG-specific optimization options.</param>
    /// <param name="predictionOptions">Options for prediction statistics.</param>
    /// <param name="modelOptions">Options for model statistics.</param>
    /// <param name="modelEvaluator">The model evaluator to use.</param>
    /// <param name="fitDetector">The fit detector to use.</param>
    /// <param name="fitnessCalculator">The fitness calculator to use.</param>
    /// <param name="modelCache">The model cache to use.</param>
    /// <param name="gradientCache">The gradient cache to use.</param>
    public NesterovAcceleratedGradientOptimizer(
        IFullModel<T, TInput, TOutput> model,
        NesterovAcceleratedGradientOptimizerOptions<T, TInput, TOutput>? options = null,
        IEngine? engine = null)
        : base(model, options ?? new())
    {

        InitializeAdaptiveParameters();
    }

    /// <summary>
    /// Creates a Nesterov accelerated gradient optimizer for minimizing a plain function, with no model attached.
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
    public static NesterovAcceleratedGradientOptimizer<T, TInput, TOutput> CreateForFunction(
        NesterovAcceleratedGradientOptimizerOptions<T, TInput, TOutput>? options = null)
        => new(options);

    /// <summary>
    /// Backs <see cref="CreateForFunction"/>: the same setup with no model.
    /// </summary>
    private NesterovAcceleratedGradientOptimizer(NesterovAcceleratedGradientOptimizerOptions<T, TInput, TOutput>? options)
        : base(null, options ?? new())
    {

        InitializeAdaptiveParameters();
    }

    /// <summary>
    /// Initializes the adaptive parameters for the NAG optimizer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method sets up the initial values for the learning rate and momentum.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like setting your initial speed and direction before you start skiing. You're deciding how fast to move and how much to consider your previous direction.
    /// </para>
    /// </remarks>
    protected override void InitializeAdaptiveParameters()
    {
        base.InitializeAdaptiveParameters();
        CurrentLearningRate = NumOps.FromDouble(_options.InitialLearningRate);
        CurrentMomentum = NumOps.FromDouble(_options.InitialMomentum);
    }

    /// <summary>
    /// Performs the optimization process using the Nesterov Accelerated Gradient algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method implements the main optimization loop. It uses the NAG algorithm to update the solution iteratively,
    /// aiming to find the optimal set of parameters that minimize the loss function.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is your actual ski run. You start at the top of the hill (your initial solution) and then repeatedly:
    /// 1. Look ahead to where you might be after your next move.
    /// 2. Check the steepness (gradient) at that future position.
    /// 3. Adjust your speed and direction based on what you see.
    /// 4. Make your move.
    /// You keep doing this until you reach the bottom of the hill or decide you're close enough to the best spot.
    /// </para>
    /// <para><b>DataLoader Integration:</b> This method uses the DataLoader API for efficient batch processing.
    /// It creates a batcher using <see cref="GradientBasedOptimizerBase{T,TInput,TOutput}.CreateBatcher"/>
    /// and notifies the sampler of epoch starts using
    /// <see cref="GradientBasedOptimizerBase{T,TInput,TOutput}.NotifyEpochStart"/>.
    /// </para>
    /// </remarks>
    /// <param name="inputData">The input data for the optimization process.</param>
    /// <returns>The result of the optimization process.</returns>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        ValidationHelper<T>.ValidateInputData(inputData);

        var currentSolution = InitializeWorkingSolution(inputData.XTrain);
        var bestStepData = new OptimizationStepData<T, TInput, TOutput>();
        var previousStepData = new OptimizationStepData<T, TInput, TOutput>();

        _velocity = new Vector<T>(InterfaceGuard.Parameterizable(currentSolution).GetParameters().Length);
        InitializeAdaptiveParameters();

        for (int epoch = 0; epoch < _options.MaxIterations; epoch++)
        {
            NotifyEpochStart(epoch);
            var batcher = CreateBatcher(inputData, _options.BatchSize, epoch);

            foreach (var (xBatch, yBatch, batchIndices) in batcher.GetBatches())
            {
                var lookaheadSolution = GetLookaheadSolution(currentSolution);
                var gradient = CalculateGradient(lookaheadSolution, xBatch, yBatch);

                IFullModel<T, TInput, TOutput> newSolution;
                if (currentSolution is AiDotNet.Interfaces.INeuralNetwork<T>)
                {
                    // #1413 CONSOLIDATION: NN solutions route through the
                    // base tape path, which expects the RAW (lookahead)
                    // gradient — Step's SGD-with-momentum kernel applies
                    // momentum bookkeeping per-parameter. Forwarding the
                    // already-momentum-accumulated _velocity would double-
                    // apply momentum (the base path would treat velocity
                    // AS the gradient and then accumulate again on top),
                    // breaking NAG dynamics for neural-net training.
                    newSolution = base.UpdateSolution(currentSolution, gradient);
                }
                else
                {
                    // Legacy non-NN path: accumulate momentum into
                    // _velocity here and subtract from params in
                    // UpdateSolution.
                    _velocity = UpdateVelocity(gradient);
                    newSolution = UpdateSolution(currentSolution, _velocity);
                }
                currentSolution = newSolution;
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
    /// Calculates the lookahead solution based on the current solution and velocity.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method computes a predicted future position of the solution, which is a key aspect of the NAG algorithm.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like looking ahead to where you think you'll be after your next move, based on your current position and how fast you're moving (velocity).
    /// </para>
    /// </remarks>
    /// <param name="currentSolution">The current solution.</param>
    /// <returns>A predicted future solution.</returns>
    private IFullModel<T, TInput, TOutput> GetLookaheadSolution(IFullModel<T, TInput, TOutput> currentSolution)
    {
        // === Vectorized NAG Lookahead using IEngine (Phase B: US-GPU-015) ===
        // lookahead = params - momentum * velocity

        var parameters = InterfaceGuard.Parameterizable(currentSolution).GetParameters();
        var momentumVelocity = (Vector<T>)Engine.Multiply(_velocity!, CurrentMomentum);
        var lookaheadCoefficients = (Vector<T>)Engine.Subtract(parameters, momentumVelocity);

        return InterfaceGuard.Parameterizable(currentSolution).WithParameters(lookaheadCoefficients);
    }

    /// <summary>
    /// Updates the velocity vector based on the current gradient.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method adjusts the velocity using the momentum and learning rate, incorporating the new gradient information.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like adjusting your speed and direction. You consider how fast you were going before (momentum) and the new information about the slope (gradient),
    /// to decide how to change your movement.
    /// </para>
    /// </remarks>
    /// <param name="gradient">The current gradient.</param>
    /// <returns>The updated velocity vector.</returns>
    private Vector<T> UpdateVelocity(Vector<T> gradient)
    {
        // === Vectorized NAG Velocity Update using IEngine (Phase B: US-GPU-015) ===
        // velocity = momentum * velocity + learningRate * gradient

        var momentumVelocity = (Vector<T>)Engine.Multiply(_velocity!, CurrentMomentum);
        var scaledGradient = (Vector<T>)Engine.Multiply(gradient, CurrentLearningRate);
        _velocity = (Vector<T>)Engine.Add(momentumVelocity, scaledGradient);

        return _velocity;
    }

    /// <summary>
    /// Updates the current solution using the velocity vector.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method computes the new solution by applying the velocity to the current solution.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like actually making your move down the slope. You take your current position and adjust it based on your speed and direction (velocity).
    /// </para>
    /// </remarks>
    /// <param name="currentSolution">The current solution.</param>
    /// <param name="velocity">The current velocity vector.</param>
    /// <returns>The updated solution.</returns>
    protected override IFullModel<T, TInput, TOutput> UpdateSolution(IFullModel<T, TInput, TOutput> currentSolution, Vector<T> velocity)
    {
        // #1413 CONSOLIDATION: NN solutions go through base.UpdateSolution
        // which synthesizes a TapeStepContext and delegates to Step
        // (one source of truth, matches PyTorch/TF/JAX). Non-NN solutions
        // (regression, clustering, classical models) keep the legacy
        // flat-vector path below for backward compatibility.
        if (currentSolution is AiDotNet.Interfaces.INeuralNetwork<T>)
        {
            return base.UpdateSolution(currentSolution, velocity);
        }
        // === Vectorized NAG Update using IEngine (Phase B: US-GPU-015) ===
        // params = params - velocity

        var parameters = InterfaceGuard.Parameterizable(currentSolution).GetParameters();
        var newCoefficients = (Vector<T>)Engine.Subtract(parameters, velocity);

        return InterfaceGuard.Parameterizable(currentSolution).WithParameters(newCoefficients);
    }

    /// <summary>
    /// Updates a vector of parameters using the Nesterov Accelerated Gradient algorithm.
    /// </summary>
    /// <param name="parameters">The current parameter vector to be updated.</param>
    /// <param name="gradient">The gradient vector corresponding to the parameters.</param>
    /// <returns>The updated parameter vector.</returns>
    /// <remarks>
    /// <para>
    /// NAG uses a lookahead mechanism where it evaluates the gradient at a predicted future position,
    /// then uses that gradient to update velocity. This lookahead gives NAG better convergence properties
    /// than standard momentum.
    /// </para>
    /// <para><b>For Beginners:</b> NAG is like looking ahead while skiing - you peek at the slope
    /// ahead before making your move, which helps you make smarter adjustments to your speed and direction.
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

        if (_velocity == null || _velocity.Length != parameters.Length)
        {
            _velocity = new Vector<T>(parameters.Length);
        }

        // Nesterov look-ahead, in the Sutskever et al. (2013) reformulation that PyTorch's
        // nesterov=True uses:
        //
        //     v      = momentum*v + g
        //     update = g + momentum*v          <- the look-ahead term
        //     theta -= lr*update
        //
        // This class previously applied v = momentum*v + lr*g; theta -= v, which is CLASSICAL
        // momentum, not Nesterov — an optimizer named for an algorithm it did not implement. The
        // divergence was already recorded in a comment here: the CUDA nag_update kernel does the true
        // look-ahead, so the GPU path was left unwired rather than reconcile the two. Reconciling in
        // the other direction — making the CPU formula correct — removes the divergence AND lets this
        // optimizer reach the fused SGDMomentum kernel with nesterov set, which implements exactly
        // these three lines.
        //
        // Note lr is no longer folded into the velocity. That matters beyond tidiness: with lr inside
        // v, a moving learning rate rescales the whole accumulated history rather than just the
        // current step, so a schedule would silently change the meaning of the stored state.
        var updatedParameters = new Vector<T>(parameters.Length, skipZeroInit: true);
        var parameterSpan = parameters.AsSpan();
        var gradientSpan = gradient.AsSpan();
        var velocitySpan = _velocity.AsWritableSpan();
        var updatedSpan = updatedParameters.AsWritableSpan();

        for (int i = 0; i < updatedSpan.Length; i++)
        {
            T velocity = NumOps.Add(
                NumOps.Multiply(CurrentMomentum, velocitySpan[i]),
                gradientSpan[i]);
            velocitySpan[i] = velocity;

            T update = NumOps.Add(gradientSpan[i], NumOps.Multiply(CurrentMomentum, velocity));
            updatedSpan[i] = NumOps.Subtract(parameterSpan[i], NumOps.Multiply(CurrentLearningRate, update));
        }

        return updatedParameters;
    }

    // Per-parameter velocity for tape-based NAG training
    private readonly ConcurrentDictionary<Tensor<T>, Tensor<T>> _tapeVelocity = new(TensorReferenceComparer<Tensor<T>>.Instance);

    /// <inheritdoc />
    public override void Step(TapeStepContext<T> context)
    {
        PrepareTapeState(context);

        // Nesterov look-ahead (Sutskever et al. 2013 reformulation, as in PyTorch's nesterov=True).
        // This path previously applied v = momentum*v + lr*grad; param -= v, i.e. CLASSICAL momentum —
        // which is why the CUDA nag_update kernel, which does the true look-ahead, was left unwired to
        // avoid a silent behaviour change. The CPU formula is now the correct one, so the two agree and
        // the fused SGDMomentum kernel can be reached with nesterov set.
        foreach (var param in context.Parameters)
        {
            // No sparse fast path here, deliberately. TryApplySgdSparse applies CLASSICAL momentum
            // (velocity = mu*v + lr*g; param -= v) and has no look-ahead term, so routing embedding
            // parameters through it would run a different algorithm from the dense parameters in the
            // same step — the silent-substitution failure this optimizer just stopped committing.
            // Densifying costs throughput on sparse embeddings; running the wrong optimizer costs
            // correctness, so the dense path below handles every parameter until a Nesterov-capable
            // sparse helper exists.

            if (!SparseEmbeddingOptimizerHelpers.TryGetEffectiveGradient(context, param, Engine, out var grad))
                continue;

            if (!_tapeVelocity.TryGetValue(param, out var vel)) { vel = new Tensor<T>(param._shape); _tapeVelocity[param] = vel; }

            // Nesterov look-ahead, matching UpdateParameters and the fused SGDMomentum kernel's
            // nesterov branch exactly:
            //     v      = momentum*v + g
            //     update = g + momentum*v
            //     param -= lr*update
            var velNew = Engine.TensorAdd(Engine.TensorMultiplyScalar(vel, CurrentMomentum), grad);
            Engine.TensorCopy(velNew, vel);

            var update = Engine.TensorAdd(grad, Engine.TensorMultiplyScalar(vel, CurrentMomentum));
            Engine.TensorSubtractInPlace(param, Engine.TensorMultiplyScalar(update, CurrentLearningRate));
        }
    }

    /// <summary>
    /// Reverses a Nesterov Accelerated Gradient update to recover original parameters.
    /// </summary>
    /// <param name="updatedParameters">Parameters after NAG update</param>
    /// <param name="appliedGradients">The gradients that were applied</param>
    /// <returns>Original parameters before the update</returns>
    /// <remarks>
    /// <para>
    /// NAG's reverse update requires the optimizer's internal velocity state from the forward pass.
    /// This method must be called immediately after UpdateParameters while the velocity is fresh.
    /// NAG evaluates gradients at a lookahead position, but the reversal only needs the final velocity.
    /// </para>
    /// <para><b>For Beginners:</b> This calculates where parameters were before a NAG update.
    /// NAG uses velocity (built from lookahead gradients) to update parameters. To reverse,
    /// we just need to know what velocity was used to take the step.
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

        if (_velocity == null || _velocity.Length != updatedParameters.Length)
        {
            throw new InvalidOperationException(
                "NAG optimizer velocity is not initialized. ReverseUpdate must be called after UpdateParameters.");
        }

        // === Vectorized Reverse NAG Update (Phase B: US-GPU-015) ===
        // Reverse the update: original = updated + velocity
        return (Vector<T>)Engine.Add(updatedParameters, _velocity);
    }

    /// <summary>
    /// Updates the adaptive parameters of the NAG optimizer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method adjusts the learning rate and momentum based on the improvement in fitness.
    /// It's used to fine-tune the algorithm's behavior as the optimization progresses.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like adjusting your skiing technique as you go down the hill. If you're making good progress, you might decide to go a bit faster or trust your momentum more.
    /// If you're not improving, you might slow down or be more cautious about following your previous direction.
    /// </para>
    /// </remarks>
    /// <param name="currentStepData">The current optimization step data.</param>
    /// <param name="previousStepData">The previous optimization step data.</param>
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

            CurrentLearningRate = MathHelper.Max(NumOps.FromDouble(_options.MinLearningRate),
                MathHelper.Min(NumOps.FromDouble(_options.MaxLearningRate), CurrentLearningRate));
        }

        if (_options.UseAdaptiveMomentum)
        {
            if (NumOps.GreaterThan(currentStepData.FitnessScore, previousStepData.FitnessScore))
            {
                CurrentMomentum = NumOps.Multiply(CurrentMomentum, NumOps.FromDouble(_options.MomentumIncreaseFactor));
            }
            else
            {
                CurrentMomentum = NumOps.Multiply(CurrentMomentum, NumOps.FromDouble(_options.MomentumDecreaseFactor));
            }

            CurrentMomentum = MathHelper.Max(NumOps.FromDouble(_options.MinMomentum),
                MathHelper.Min(NumOps.FromDouble(_options.MaxMomentum), CurrentMomentum));
        }
    }


    /// <summary>
    /// Gets the current options of the Nesterov Accelerated Gradient optimizer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method returns the current configuration options of the optimizer.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like asking to see the current set of rules the skier is following on their descent.
    /// </para>
    /// </remarks>
    /// <returns>The current optimization algorithm options.</returns>
    public override OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions()
    {
        return _options;
    }

    /// <summary>
    /// Generates a unique key for caching gradients in the Nesterov Accelerated Gradient optimizer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method creates a unique identifier for caching gradients, incorporating the base key and NAG-specific parameters.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like creating a special label for each unique skiing situation, considering not just the slope (model and data) but also the specific NAG skiing technique being used (initial momentum and learning rate).
    /// </para>
    /// </remarks>
    /// <param name="model">The symbolic model for which the gradient is being calculated.</param>
    /// <param name="X">The input data matrix.</param>
    /// <param name="y">The target vector.</param>
    /// <returns>A string key uniquely identifying the gradient calculation scenario for caching purposes.</returns>
    protected override string GenerateGradientCacheKey(IFullModel<T, TInput, TOutput> model, TInput X, TOutput y)
    {
        var baseKey = base.GenerateGradientCacheKey(model, X, y);
        return $"{baseKey}_NAG_{_options.InitialMomentum}_{_options.InitialLearningRate}";
    }

    #region GPU Optimizer Support

    /// <summary>
    /// GPU buffer for velocity state.
    /// </summary>
    private IGpuBuffer? _gpuVelocity;

    /// <summary>
    /// Gets whether this optimizer supports GPU-accelerated parameter updates.
    /// </summary>
    public override bool SupportsGpuUpdate => true;

    /// <summary>
    /// Initializes NAG optimizer state on the GPU.
    /// </summary>
    public override void InitializeGpuState(int parameterCount, IDirectGpuBackend backend)
    {
        if (_gpuStateInitialized && _gpuVelocity != null)
            return;

        var zeros = new float[parameterCount];
        _gpuVelocity = backend.AllocateBuffer(zeros);

        _gpuStateInitialized = true;
    }

    /// <summary>
    /// Updates parameters on the GPU using the NAG kernel.
    /// </summary>
    public override void UpdateParametersGpu(IGpuBuffer parameters, IGpuBuffer gradients, int parameterCount, IDirectGpuBackend backend)
    {
        if (!_gpuStateInitialized || _gpuVelocity == null)
        {
            InitializeGpuState(parameterCount, backend);
        }

        backend.NagUpdate(
            parameters,
            gradients,
            _gpuVelocity!,
            (float)NumOps.ToDouble(CurrentLearningRate),
            (float)NumOps.ToDouble(CurrentMomentum),
            0.0f, // NAG doesn't have weight decay in these options
            parameterCount
        );
    }

    /// <summary>
    /// Disposes GPU-allocated optimizer state.
    /// </summary>
    public override void DisposeGpuState()
    {
        _gpuVelocity?.Dispose();
        _gpuVelocity = null;
        _gpuStateInitialized = false;
    }

    #endregion
}
