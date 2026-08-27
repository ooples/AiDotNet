using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Autodiff;
using Newtonsoft.Json;

using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.Optimizers;

/// <summary>
/// Represents a Gradient Descent optimizer for machine learning models.
/// </summary>
/// <remarks>
/// <para>
/// Gradient Descent is a first-order iterative optimization algorithm for finding a local minimum of a differentiable function.
/// It takes steps proportional to the negative of the gradient of the function at the current point.
/// </para>
/// <para><b>For Beginners:</b> Imagine you're trying to find the lowest point in a valley:
/// 
/// - You start at a random point (initial model parameters)
/// - You look around to see which way is steepest downhill (calculate the gradient)
/// - You take a step in that direction (update the parameters)
/// - You repeat this process until you reach the bottom of the valley (optimize the model)
/// 
/// This optimizer helps the model learn by gradually adjusting its parameters to minimize errors.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[ComponentType(ComponentType.Optimizer)]
[PipelineStage(PipelineStage.Training)]
public partial class GradientDescentOptimizer<T, TInput, TOutput> : GradientBasedOptimizerBase<T, TInput, TOutput>, Fused.IFusedOptimizerSpec
{
    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// Maps to plain <see cref="Tensors.Engines.Compilation.OptimizerType.SGD"/>, matching what
    /// <see cref="Step"/> does: <c>param -= lr * grad</c>, with the sparse path passing an explicit
    /// <c>momentum: 0.0</c>.
    /// </para>
    /// <para>
    /// <c>Optimize</c> calls <c>ApplyMomentum</c> and <c>InitialMomentum</c> defaults to 0.9, which suggests
    /// SGDMomentum — but that is the wrong loop to match. <c>Optimize</c>'s flat-vector path serves non-neural
    /// models and never reaches the compiled plan; <c>Step</c> is the tape path the fused kernel replaces.
    /// Mapping to SGDMomentum would add momentum on the fused path that the eager tape path does not apply.
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
    /// Options specific to the Gradient Descent optimizer.
    /// </summary>
    private GradientDescentOptimizerOptions<T, TInput, TOutput> _gdOptions;

    /// <summary>
    /// The regularization technique used to prevent overfitting.
    /// </summary>
    private readonly IRegularization<T, TInput, TOutput> _regularization;

    /// <summary>
    /// Initializes a new instance of the GradientDescentOptimizer class.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This sets up the Gradient Descent optimizer with its initial settings.
    /// It's like preparing for your hike by choosing your starting point, deciding how big your steps
    /// will be, and how you'll adjust your path to avoid getting stuck in small dips.
    /// </para>
    /// </remarks>
    /// <param name="model">The model to optimize.</param>
    /// <param name="options">Options for the Gradient Descent optimizer.</param>
    public GradientDescentOptimizer(
        IFullModel<T, TInput, TOutput> model,
        GradientDescentOptimizerOptions<T, TInput, TOutput>? options = null,
        IEngine? engine = null)
        : base(model, options ?? new GradientDescentOptimizerOptions<T, TInput, TOutput>())
    {
        _gdOptions = options ?? new GradientDescentOptimizerOptions<T, TInput, TOutput>();
        _regularization = _gdOptions.Regularization ?? CreateRegularization(_gdOptions);
    }

    /// <summary>
    /// Creates a gradient descent optimizer for minimizing a plain function, with no model attached.
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
    public static GradientDescentOptimizer<T, TInput, TOutput> CreateForFunction(
        GradientDescentOptimizerOptions<T, TInput, TOutput>? options = null)
        => new(options);

    /// <summary>
    /// Backs <see cref="CreateForFunction"/>: the same setup with no model.
    /// </summary>
    private GradientDescentOptimizer(GradientDescentOptimizerOptions<T, TInput, TOutput>? options)
        : base(null, options ?? new())
    {
        _gdOptions = options ?? new GradientDescentOptimizerOptions<T, TInput, TOutput>();
        _regularization = _gdOptions.Regularization ?? CreateRegularization(_gdOptions);
    }

    /// <summary>
    /// Performs the main optimization process using the Gradient Descent algorithm.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the heart of the Gradient Descent algorithm. It:
    /// 1. Starts with a random solution
    /// 2. Calculates how to improve the solution (the gradient)
    /// 3. Updates the solution by taking a step in the direction of improvement
    /// 4. Repeats this process many times
    /// 
    /// It's like repeatedly adjusting your path as you hike, always trying to move towards lower ground.
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
        var bestStepData = EvaluateSolution(currentSolution, inputData);
        var previousStepData = bestStepData;
        InitializeAdaptiveParameters();

        for (int epoch = 0; epoch < _gdOptions.MaxIterations; epoch++)
        {
            NotifyEpochStart(epoch);
            var batcher = CreateBatcher(inputData, _gdOptions.BatchSize, epoch);

            foreach (var (xBatch, yBatch, batchIndices) in batcher.GetBatches())
            {
                var gradient = CalculateGradient(currentSolution, xBatch, yBatch);
                gradient = ApplyMomentum(gradient);
                currentSolution = UpdateSolution(currentSolution, gradient);
            }

            var currentStepData = EvaluateSolution(currentSolution, inputData);

            UpdateAdaptiveParameters(currentStepData, previousStepData);

            UpdateBestSolution(currentStepData, ref bestStepData);

            if (UpdateIterationHistoryAndCheckEarlyStopping(epoch, bestStepData))
            {
                break;
            }

            previousStepData = currentStepData;
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    /// <summary>
    /// Updates the current solution based on the calculated gradient.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method adjusts the current solution to make it better.
    /// It's like taking a step in the direction you've determined will lead you downhill.
    /// </para>
    /// </remarks>
    /// <param name="currentSolution">The current solution.</param>
    /// <param name="gradient">The calculated gradient.</param>
    /// <returns>The updated solution.</returns>
    protected override IFullModel<T, TInput, TOutput> UpdateSolution(
        IFullModel<T, TInput, TOutput> currentSolution,
        Vector<T> gradient)
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
        // === Vectorized Gradient Descent Update using IEngine (Phase B: US-GPU-015) ===
        // params = params - learningRate * gradient

        Vector<T> currentParams = InterfaceGuard.Parameterizable(currentSolution).GetParameters();
        var scaledGradient = (Vector<T>)Engine.Multiply(gradient, CurrentLearningRate);
        var updatedParams = (Vector<T>)Engine.Subtract(currentParams, scaledGradient);

        return InterfaceGuard.Parameterizable(currentSolution).WithParameters(updatedParams);
    }

    /// <summary>
    /// Reverses a Gradient Descent update to recover original parameters.
    /// </summary>
    /// <param name="updatedParameters">Parameters after GD update</param>
    /// <param name="appliedGradients">The gradients that were applied</param>
    /// <returns>Original parameters before the update</returns>
    /// <remarks>
    /// <para>
    /// Gradient Descent uses vanilla SGD update rule: params_new = params_old - lr * gradient.
    /// The reverse is straightforward: params_old = params_new + lr * gradient.
    /// </para>
    /// <para><b>For Beginners:</b> This calculates where parameters were before a Gradient Descent update.
    /// Since GD uses simple steps (parameter minus learning_rate times gradient), reversing
    /// just means adding back that step.
    /// </para>
    /// </remarks>

    /// <inheritdoc />
    public override void Step(TapeStepContext<T> context)
    {
        bool gpuAdam = typeof(T) == typeof(float)
            && System.Environment.GetEnvironmentVariable("AIDOTNET_GPU_ADAM") == "1"
            && AiDotNet.Tensors.Engines.AiDotNetEngine.Current is AiDotNet.Tensors.Engines.DirectGpuTensorEngine;

        foreach (var param in context.Parameters)
        {
            // True sparse scatter plain SGD: param -= lr·g at touched indices.
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

        // === Vectorized Reverse Gradient Descent Update using IEngine (Phase B: US-GPU-015) ===
        // Reverse: original = updated + lr * gradient
        var currentLrVec = Vector<T>.CreateDefault(appliedGradients.Length, CurrentLearningRate);
        var gradientStep = (Vector<T>)Engine.Multiply(currentLrVec, appliedGradients);
        return (Vector<T>)Engine.Add(updatedParameters, gradientStep);
    }

    /// <summary>
    /// Calculates the loss for a given solution and input data.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method measures how well the current solution performs.
    /// It's like checking your altitude to see how close you are to the bottom of the valley.
    /// The method also includes a regularization term to prevent overfitting.
    /// </para>
    /// </remarks>
    /// <param name="solution">The current solution to evaluate.</param>
    /// <param name="X">The input features.</param>
    /// <param name="y">The target values.</param>
    /// <returns>The calculated loss value.</returns>
    private T CalculateLoss(IFullModel<T, TInput, TOutput> solution, TInput X, TOutput y)
    {
        TOutput predictions = solution.Predict(X);
        var parameters = InterfaceGuard.Parameterizable(solution).GetParameters();
        T loss;

        if (predictions is Tensor<T> tensorPredictions && y is Tensor<T> tensorY)
        {
            loss = LossFunction.CalculateLoss(tensorPredictions.ToVector(), tensorY.ToVector());
        }
        else if (predictions is Vector<T> vectorPredictions && y is Vector<T> vectorY)
        {
            loss = LossFunction.CalculateLoss(vectorPredictions, vectorY);
        }
        else
        {
            throw new ArgumentException("Unsupported prediction or target type");
        }

        Vector<T> regularizedCoefficients = _regularization.Regularize(parameters);
        T regularizationTerm = regularizedCoefficients.Subtract(parameters).Transform(NumOps.Abs).Sum();

        return NumOps.Add(loss, regularizationTerm);
    }

    /// <summary>
    /// Updates the options for the Gradient Descent optimizer.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method allows you to change the settings of the optimizer
    /// while it's running. It's like adjusting your hiking strategy mid-journey based on the terrain you encounter.
    /// </para>
    /// </remarks>
    /// <param name="options">The new options to apply to the optimizer.</param>
    /// <exception cref="ArgumentException">Thrown when the provided options are not of the correct type.</exception>
    protected override void UpdateOptions(OptimizationAlgorithmOptions<T, TInput, TOutput> options)
    {
        if (options is GradientDescentOptimizerOptions<T, TInput, TOutput> gdOptions)
        {
            _gdOptions = gdOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected GradientDescentOptions.");
        }
    }

    /// <summary>
    /// Retrieves the current options for the Gradient Descent optimizer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method returns the current configuration options for the Gradient Descent optimizer.
    /// These options control various aspects of the optimization process, such as learning rate,
    /// maximum iterations, and regularization settings.
    /// </para>
    /// <para><b>For Beginners:</b> Think of this method as checking your current hiking plan:
    /// 
    /// - It tells you things like how big your steps are (learning rate)
    /// - How long you plan to hike (maximum iterations)
    /// - What rules you're following to avoid getting lost (regularization settings)
    /// 
    /// This information is useful if you want to understand or adjust how the optimizer is currently set up.
    /// </para>
    /// </remarks>
    /// <returns>The current Gradient Descent optimizer options.</returns>
    public override OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions()
    {
        return _gdOptions;
    }

    /// <summary>
    /// Generates a unique key for caching gradients specific to the Gradient Descent optimizer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method extends the base class's gradient cache key generation by adding Gradient Descent-specific
    /// parameters. The resulting key is unique to the current state of the optimizer and the input data,
    /// allowing for efficient caching and retrieval of previously calculated gradients.
    /// </para>
    /// <para><b>For Beginners:</b> Think of this method as creating a unique label for each gradient calculation:
    /// 
    /// - It starts with a basic label (from the base class) that describes the model and data
    /// - Then it adds specific details about the Gradient Descent optimizer, like how big steps it's taking (learning rate)
    ///   and how many times it plans to adjust the model (max iterations)
    /// - This unique label helps the optimizer remember and quickly find previous calculations,
    ///   making the whole process faster and more efficient
    /// 
    /// It's like keeping a well-organized hiking journal where you can quickly look up information
    /// about specific points in your journey.
    /// </para>
    /// </remarks>
    /// <param name="model">The current model being optimized.</param>
    /// <param name="X">The input features used for gradient calculation.</param>
    /// <param name="y">The target values used for gradient calculation.</param>
    /// <returns>A string that uniquely identifies the current gradient calculation scenario.</returns>
    protected override string GenerateGradientCacheKey(IFullModel<T, TInput, TOutput> model, TInput X, TOutput y)
    {
        var baseKey = base.GenerateGradientCacheKey(model, X, y);
        return $"{baseKey}_GD_{CurrentLearningRate}_{_gdOptions.MaxIterations}";
    }

    /// <summary>
    /// Updates parameters on the GPU using vanilla SGD.
    /// </summary>
    public override void UpdateParametersGpu(IGpuBuffer parameters, IGpuBuffer gradients, int parameterCount, IDirectGpuBackend backend)
    {
        backend.SgdUpdate(
            parameters,
            gradients,
            (float)NumOps.ToDouble(CurrentLearningRate),
            0.0f, // No weight decay for basic GD
            parameterCount);
    }
}
