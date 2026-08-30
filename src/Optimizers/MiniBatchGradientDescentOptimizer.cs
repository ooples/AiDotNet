using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Autodiff;
using Newtonsoft.Json;

using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.Optimizers;

/// <summary>
/// Implements the Mini-Batch Gradient Descent optimization algorithm.
/// </summary>
/// <remarks>
/// <para>
/// Mini-Batch Gradient Descent is a variation of gradient descent that splits the training data into small batches
/// to calculate model error and update model coefficients. This approach strikes a balance between the efficiency
/// of stochastic gradient descent and the stability of batch gradient descent.
/// </para>
/// <para><b>For Beginners:</b>
/// Imagine you're trying to find the bottom of a valley while blindfolded. Mini-Batch Gradient Descent is like taking 
/// a few steps, checking your position, adjusting your direction, and repeating. It's faster than checking after every 
/// single step (Stochastic Gradient Descent) but more precise than taking a lot of steps before checking (Batch Gradient Descent).
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[ComponentType(ComponentType.Optimizer)]
[PipelineStage(PipelineStage.Training)]
public partial class MiniBatchGradientDescentOptimizer<T, TInput, TOutput> : GradientBasedOptimizerBase<T, TInput, TOutput>, Fused.IFusedOptimizerSpec
{
    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// Unlike <c>GradientDescentOptimizer</c> and <c>StochasticGradientDescentOptimizer</c>, this
    /// optimizer's loop does NOT call <c>ApplyMomentum</c> — its update is plain
    /// <c>param -= lr * grad</c> over a mini-batch, which is exactly the fused <c>SGD</c> kernel.
    /// </para>
    /// <para>
    /// Declines when the learning rate adapts during training, since the fused plan bakes it in when
    /// the plan is built.
    /// </para>
    /// </remarks>
    bool Fused.IFusedOptimizerSpec.TryGetFusedOptimizerConfig(out Fused.FusedOptimizerConfig config)
    {
        config = default;
        if (GradientOptions.UseAdaptiveLearningRate) return false;
        if (!TryGetFusedLrSchedule(out var schedule)) return false;
        config = new Fused.FusedOptimizerConfig(
            Tensors.Engines.Compilation.OptimizerType.SGD,
            (float)GetCurrentLearningRate(),
            0f, 0f, 0f, 0f, schedule);
        return true;
    }

    /// <summary>
    /// The options specific to the Mini-Batch Gradient Descent algorithm.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// This is like your hiking plan, containing details such as how many steps to take before checking your position,
    /// how many times to repeat the process, and how to adjust your step size.
    /// </para>
    /// </remarks>
    /// <summary>Read from the single instance OptimizerBase.Options holds, so there is
    /// no second copy that could disagree with it.</summary>
    private MiniBatchGradientDescentOptions<T, TInput, TOutput> _options => (MiniBatchGradientDescentOptions<T, TInput, TOutput>)Options;

    /// <summary>
    /// Initializes a new instance of the MiniBatchGradientDescentOptimizer class.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This constructor sets up the optimizer with the provided options and dependencies. If no options are provided,
    /// it uses default settings. It also initializes a random number generator for shuffling data.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like setting up your hiking gear before starting the journey to find the valley's bottom. You're 
    /// deciding on your strategy (options) and packing your tools (dependencies) that you'll use along the way.
    /// </para>
    /// </remarks>
    /// <param name="model">The model to optimize.</param>
    public MiniBatchGradientDescentOptimizer(
        IFullModel<T, TInput, TOutput> model,
        MiniBatchGradientDescentOptions<T, TInput, TOutput>? options = null,
        IEngine? engine = null)
        : base(model, options ?? new())
    {

        InitializeAdaptiveParameters();
    }

    /// <summary>
    /// Creates a mini-batch gradient descent optimizer for minimizing a plain function, with no model attached.
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
    public static MiniBatchGradientDescentOptimizer<T, TInput, TOutput> CreateForFunction(
        MiniBatchGradientDescentOptions<T, TInput, TOutput>? options = null)
        => new(options);

    /// <summary>
    /// Backs <see cref="CreateForFunction"/>: the same setup with no model.
    /// </summary>
    private MiniBatchGradientDescentOptimizer(MiniBatchGradientDescentOptions<T, TInput, TOutput>? options)
        : base(null, options ?? new())
    {

        InitializeAdaptiveParameters();
    }

    /// <summary>
    /// Initializes adaptive parameters for the optimization process.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method sets up the initial learning rate for the optimization process based on the options provided.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like deciding how big your steps will be when you start your journey. The learning rate determines 
    /// how much you adjust your position based on each batch of information you process.
    /// </para>
    /// </remarks>
    protected override void InitializeAdaptiveParameters()
    {
        base.InitializeAdaptiveParameters();
        CurrentLearningRate = NumOps.FromDouble(_options.InitialLearningRate);
    }

    /// <summary>
    /// Performs the optimization process using Mini-Batch Gradient Descent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method implements the main optimization loop. It iterates through the data in mini-batches,
    /// calculating gradients and updating the model parameters for each batch. The process continues for
    /// a specified number of epochs or until a stopping criterion is met.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is the actual journey to find the valley's bottom. You're taking steps (processing batches of data),
    /// checking your position (evaluating the model), and adjusting your direction (updating the model parameters).
    /// You do this repeatedly (for each epoch) until you're satisfied with your position or you've taken the
    /// maximum number of steps you allowed yourself.
    /// </para>
    /// <para><b>DataLoader Integration:</b>
    /// This optimizer now uses the DataLoader batching infrastructure which supports:
    /// - Custom samplers (weighted, stratified, curriculum, importance, active learning)
    /// - Reproducible shuffling via RandomSeed
    /// - Option to drop incomplete final batches
    /// Set these options via GradientBasedOptimizerOptions.DataSampler, ShuffleData, DropLastBatch, and RandomSeed.
    /// </para>
    /// </remarks>
    /// <param name="inputData">The input data for the optimization process.</param>
    /// <returns>The result of the optimization process.</returns>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        // Initialize with random solution
        var currentSolution = InitializeWorkingSolution(inputData.XTrain);
        var bestStepData = new OptimizationStepData<T, TInput, TOutput>();
        var previousStepData = PrepareAndEvaluateSolution(currentSolution, inputData);

        // Initialize parameters
        InitializeAdaptiveParameters();

        for (int epoch = 0; epoch < Options.MaxIterations; epoch++)
        {
            // Notify sampler of new epoch (for curriculum/self-paced learning)
            NotifyEpochStart(epoch);

            // Create batcher for the current epoch using DataLoader infrastructure
            // This handles shuffling, sampling strategies, and batch creation
            var batcher = CreateBatcher(inputData, _options.BatchSize, epoch);

            foreach (var (xBatch, yBatch, batchIndices) in batcher.GetBatches())
            {
                // Process batch and calculate gradient using the batch data directly
                var gradient = CalculateGradient(currentSolution, xBatch, yBatch);

                // Update solution
                var newSolution = UpdateSolution(currentSolution, gradient);

                // Evaluate the solution
                var currentStepData = EvaluateSolution(newSolution, inputData);
                UpdateBestSolution(currentStepData, ref bestStepData);
                UpdateAdaptiveParameters(currentStepData, previousStepData);

                // Check early stopping criteria
                if (UpdateIterationHistoryAndCheckEarlyStopping(epoch, bestStepData))
                {
                    return CreateOptimizationResult(bestStepData, inputData);
                }

                // H6 convergence fix (PR #1364): compare CURRENT vs PREVIOUS epoch
            // (not bestStepData — UpdateBestSolution would falsely report
            // converged on epoch 0 because best == current after the copy)
            // AND skip the check on epoch 0 where previousStepData is the
            // pre-training baseline. Lifted to GradientBasedOptimizerBase
            // helper so every gradient optimizer satisfies the same contract.
            if (IsConvergedAgainstPreviousEpoch(epoch, currentStepData, previousStepData, _options.Tolerance))
            {
                return CreateOptimizationResult(bestStepData, inputData);
            }

                currentSolution = newSolution;
                previousStepData = currentStepData;
            }
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    /// <summary>
    /// Updates the current solution based on the calculated gradient.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method applies the gradient to the current solution, adjusting each coefficient by the gradient 
    /// scaled by the learning rate.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like taking a step in the direction you think will lead you closer to the valley's bottom. 
    /// The size of your step is determined by the learning rate, and the direction is given by the gradient.
    /// </para>
    /// </remarks>
    /// <param name="currentSolution">The current model solution.</param>
    /// <param name="gradient">The calculated gradient.</param>
    /// <returns>An updated symbolic model with improved coefficients.</returns>
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
        // === Vectorized Mini-Batch GD Update using IEngine (Phase B: US-GPU-015) ===
        // params = params - learningRate * gradient

        var parameters = InterfaceGuard.Parameterizable(currentSolution).GetParameters();
        var scaledGradient = (Vector<T>)Engine.Multiply(gradient, CurrentLearningRate);
        var newCoefficients = (Vector<T>)Engine.Subtract(parameters, scaledGradient);

        return InterfaceGuard.Parameterizable(currentSolution).WithParameters(newCoefficients);
    }

    /// <summary>
    /// Reverses a Mini-Batch Gradient Descent update to recover original parameters.
    /// </summary>
    /// <param name="updatedParameters">Parameters after Mini-Batch GD update</param>
    /// <param name="appliedGradients">The gradients that were applied</param>
    /// <returns>Original parameters before the update</returns>
    /// <remarks>
    /// <para>
    /// Mini-Batch Gradient Descent uses vanilla SGD update rule: params_new = params_old - lr * gradient.
    /// The reverse is straightforward: params_old = params_new + lr * gradient.
    /// </para>
    /// <para><b>For Beginners:</b> This calculates where parameters were before a Mini-Batch GD update.
    /// Since Mini-Batch GD uses simple steps (parameter minus learning_rate times gradient), reversing
    /// just means adding back that step.
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

        // === Vectorized Reverse Mini-Batch GD Update (Phase B: US-GPU-015) ===
        // Reverse: original = updated + lr * gradient
        var currentLrVec = Vector<T>.CreateDefault(appliedGradients.Length, CurrentLearningRate);
        var gradientStep = (Vector<T>)Engine.Multiply(currentLrVec, appliedGradients);
        return (Vector<T>)Engine.Add(updatedParameters, gradientStep);
    }

    /// <summary>
    /// Updates the adaptive parameters of the optimizer based on the current and previous optimization steps.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method adjusts the learning rate based on the performance of the current step compared to the previous step.
    /// If improvement is seen, the learning rate may be increased, otherwise it may be decreased.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like adjusting your step size based on how well you're doing. If you're making good progress, 
    /// you might take slightly bigger steps. If you're not improving, you might take smaller, more careful steps.
    /// </para>
    /// </remarks>
    /// <param name="currentStepData">Data from the current optimization step.</param>
    /// <param name="previousStepData">Data from the previous optimization step.</param>
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
    }


    /// <summary>
    /// Gets the current optimization algorithm options.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method returns the current options used by the Mini-Batch Gradient Descent optimizer.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like checking your current hiking plan. It lets you see all the settings and strategies 
    /// you're currently using in your journey to find the valley's bottom.
    /// </para>
    /// </remarks>
    /// <returns>The current MiniBatchGradientDescentOptions object.</returns>
    public override OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions()
    {
        return _options;
    }

    /// <summary>
    /// Generates a unique key for caching gradients based on the model and input data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method creates a unique identifier for caching gradients. It combines the base gradient cache key 
    /// with specific parameters of the Mini-Batch Gradient Descent algorithm.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// Imagine you're leaving markers along your hiking path. This method creates a unique label for each marker, 
    /// combining information about where you are (the model and data) with specifics about how you're hiking 
    /// (batch size and number of rounds). This helps you quickly recognize and use information from similar 
    /// situations you've encountered before.
    /// </para>
    /// </remarks>
    /// <param name="model">The symbolic model being optimized.</param>
    /// <param name="X">The input data matrix.</param>
    /// <param name="y">The target output vector.</param>
    /// <returns>A string representing the unique gradient cache key.</returns>
    protected override string GenerateGradientCacheKey(IFullModel<T, TInput, TOutput> model, TInput X, TOutput y)
    {
        var baseKey = base.GenerateGradientCacheKey(model, X, y);
        return $"{baseKey}_MiniBatchGD_{_options.BatchSize}_{_options.MaxEpochs}";
    }

    /// <summary>
    /// Updates parameters on the GPU using vanilla SGD (same as SGD for parameter updates).
    /// </summary>
    public override void UpdateParametersGpu(IGpuBuffer parameters, IGpuBuffer gradients, int parameterCount, IDirectGpuBackend backend)
    {
        backend.SgdUpdate(
            parameters,
            gradients,
            (float)NumOps.ToDouble(CurrentLearningRate),
            0.0f, // No weight decay
            parameterCount);
    }

    /// <inheritdoc />
    public override void Step(TapeStepContext<T> context)
    {
        bool gpuAdam = typeof(T) == typeof(float)
            && System.Environment.GetEnvironmentVariable("AIDOTNET_GPU_ADAM") == "1"
            && AiDotNet.Tensors.Engines.AiDotNetEngine.Current is AiDotNet.Tensors.Engines.DirectGpuTensorEngine;

        foreach (var param in context.Parameters)
        {
            // True sparse scatter plain SGD.
            if (!gpuAdam && SparseEmbeddingOptimizerHelpers.HasSparseEmbeddingGrad(param))
            {
                if (SparseEmbeddingOptimizerHelpers.TryApplySgdSparse(
                        param, velocity: null,
                        NumOps.ToDouble(CurrentLearningRate),
                        momentum: 0.0, weightDecay: 0.0))
                {
                    continue;
                }
            }

            if (SparseEmbeddingOptimizerHelpers.TryGetEffectiveGradient(context, param, Engine, out var grad))
            {
                if (gpuAdam && param.Length == grad.Length
                    && AiDotNet.Tensors.Engines.Gpu.GpuOptimizer.TrySgdStep((Tensor<float>)(object)param, (Tensor<float>)(object)grad,
                        (float)NumOps.ToDouble(CurrentLearningRate)))
                    continue;

                var update = Engine.TensorMultiplyScalar(grad, CurrentLearningRate);
                Engine.TensorSubtractInPlace(param, update);
            }
        }
    }
}
