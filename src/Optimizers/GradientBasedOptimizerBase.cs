using AiDotNet.Engines;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.MixedPrecision;
using AiDotNet.Models.Options;

namespace AiDotNet.Optimizers;

/// <summary>
/// Represents a base class for gradient-based optimization algorithms.
/// </summary>
/// <remarks>
/// <para>
/// Gradient-based optimizers use the gradient of the loss function to update the model parameters
/// in a direction that minimizes the loss. This base class provides common functionality for
/// various gradient-based optimization techniques.
/// </para>
/// <para><b>For Beginners:</b> Think of gradient-based optimization like finding the bottom of a valley:
/// 
/// - You start at a random point on a hilly landscape (your initial model parameters)
/// - You look around to see which way is steepest downhill (calculate the gradient)
/// - You take a step in that direction (update the parameters)
/// - You repeat this process until you reach the bottom of the valley (optimize the model)
/// 
/// This approach helps the model learn by gradually adjusting its parameters to minimize errors.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public abstract class GradientBasedOptimizerBase<T, TInput, TOutput> : OptimizerBase<T, TInput, TOutput>, IGradientBasedOptimizer<T, TInput, TOutput>
{
    /// <summary>
    /// Options specific to gradient-based optimization algorithms.
    /// </summary>
    protected GradientBasedOptimizerOptions<T, TInput, TOutput> GradientOptions;

    /// <summary>
    /// The current learning rate used in the optimization process.
    /// </summary>
    private double _currentLearningRate;

    /// <summary>
    /// The current momentum factor used in the optimization process.
    /// </summary>
    private double _currentMomentum;

    /// <summary>
    /// The gradient from the previous optimization step, used for momentum calculations.
    /// </summary>
    protected Vector<T> _previousGradient;

    /// <summary>
    /// The gradients computed during the last optimization step.
    /// </summary>
    /// <remarks>
    /// This field stores the gradients calculated in the most recent call to CalculateGradient().
    /// It enables external access to gradients for features like gradient clipping, distributed
    /// training (true DDP), debugging, and visualization.
    /// Returns Vector&lt;T&gt;.Empty() if no gradients have been computed yet.
    /// </remarks>
    protected Vector<T> _lastComputedGradients;

    /// <summary>
    /// A cache for storing and retrieving gradients to improve performance.
    /// </summary>
    protected IGradientCache<T> GradientCache;

    /// <summary>
    /// A method used to compare the predicted values vs the actual values.
    /// </summary>
    protected ILossFunction<T> LossFunction;

    /// <summary>
    /// A method used to regularize the parameters so they don't get out of control.
    /// </summary>
    protected IRegularization<T, TInput, TOutput> Regularization;

    /// <summary>
    /// Mixed-precision training context (null if mixed-precision is disabled).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Mixed-precision training uses both 16-bit (FP16) and 32-bit (FP32) floating-point
    /// numbers during optimization. This context manages the conversion between precisions and handles
    /// loss scaling to prevent numerical issues. When enabled, this can provide:
    /// - 2-3x faster training on modern GPUs (V100, A100, RTX 3000+)
    /// - ~50% memory reduction
    /// - Maintained accuracy through careful precision management
    /// </para>
    /// </remarks>
    protected MixedPrecisionContext? _mixedPrecisionContext;

    /// <summary>
    /// The learning rate scheduler to use for adjusting learning rate during training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A learning rate scheduler automatically adjusts how fast your model
    /// learns during training. Common strategies include starting high and decreasing over time,
    /// or using warmup to slowly increase the learning rate at the beginning.
    /// </para>
    /// </remarks>
    protected ILearningRateScheduler? _learningRateScheduler;

    /// <summary>
    /// Specifies when to step the learning rate scheduler.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Controls whether the scheduler updates after each batch, each epoch, or uses warmup
    /// followed by per-epoch stepping.
    /// </para>
    /// </remarks>
    protected SchedulerStepMode _schedulerStepMode;

    /// <summary>
    /// The current step (batch) number for scheduler tracking.
    /// </summary>
    protected int _currentStep = 0;

    /// <summary>
    /// The current epoch number for scheduler tracking.
    /// </summary>
    protected int _currentEpoch = 0;

    /// <summary>
    /// Gets whether mixed-precision training is enabled for this optimizer.
    /// </summary>
    public bool IsMixedPrecisionEnabled => _mixedPrecisionContext != null;

    /// <summary>
    /// Gets the current learning rate scheduler, if one is configured.
    /// </summary>
    public ILearningRateScheduler? LearningRateScheduler => _learningRateScheduler;

    /// <summary>
    /// Gets the current scheduler step mode.
    /// </summary>
    public SchedulerStepMode SchedulerStepMode => _schedulerStepMode;

    /// <summary>
    /// Initializes a new instance of the GradientBasedOptimizerBase class.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This sets up the gradient-based optimizer with its initial settings.
    /// It's like preparing for your hike by choosing your starting point, deciding how big your steps
    /// will be, and how much you'll consider your previous direction when choosing your next step.
    /// </para>
    /// </remarks>
    /// <param name="model">The model to optimize (can be null if set later).</param>
    /// <param name="options">Options for the gradient-based optimizer.</param>
    protected GradientBasedOptimizerBase(
        IFullModel<T, TInput, TOutput>? model,
        GradientBasedOptimizerOptions<T, TInput, TOutput> options) :
        base(model, options)
    {
        GradientOptions = options;
        _currentLearningRate = GradientOptions.InitialLearningRate;
        _currentMomentum = GradientOptions.InitialMomentum;
        _previousGradient = Vector<T>.Empty();
        _lastComputedGradients = Vector<T>.Empty();
        LossFunction = options.LossFunction;
        GradientCache = options.GradientCache;
        Regularization = options.Regularization;

        // Initialize learning rate scheduler from options
        _learningRateScheduler = options.LearningRateScheduler;
        _schedulerStepMode = options.SchedulerStepMode;

        // If a scheduler is provided, sync the learning rate
        if (_learningRateScheduler != null)
        {
            _currentLearningRate = _learningRateScheduler.CurrentLearningRate;
        }
    }

    /// <inheritdoc/>
    public virtual Vector<T> LastComputedGradients => _lastComputedGradients;

    /// <inheritdoc/>
    /// <remarks>
    /// <para><b>Note:</b> This overload extracts parameters from the model. For distributed training,
    /// use the safer 3-parameter overload that accepts originalParameters explicitly to prevent
    /// double-stepping bugs.
    /// </para>
    /// </remarks>
    public virtual IFullModel<T, TInput, TOutput> ApplyGradients(Vector<T> gradients, IFullModel<T, TInput, TOutput> model)
    {
        if (model == null)
            throw new ArgumentNullException(nameof(model));

        // Delegate to the safe 3-parameter overload
        var parameters = model.GetParameters();
        return ApplyGradients(parameters, gradients, model);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para><b>Production-Ready for Distributed Training:</b>
    /// This overload prevents double-stepping by accepting originalParameters explicitly.
    /// The model parameter is used only as a template for structure.
    /// </para>
    /// <para>
    /// Correct implementation: applies gradients to originalParameters (not model.GetParameters()),
    /// ensuring single-step behavior even if model contains post-update parameters.
    /// </para>
    /// </remarks>
    public virtual IFullModel<T, TInput, TOutput> ApplyGradients(Vector<T> originalParameters, Vector<T> gradients, IFullModel<T, TInput, TOutput> model)
    {
        if (originalParameters == null)
            throw new ArgumentNullException(nameof(originalParameters));
        if (gradients == null)
            throw new ArgumentNullException(nameof(gradients));
        if (model == null)
            throw new ArgumentNullException(nameof(model));

        if (gradients.Length != originalParameters.Length)
        {
            throw new ArgumentException(
                $"Gradient size ({gradients.Length}) must match original parameter count ({originalParameters.Length})",
                nameof(gradients));
        }

        // CRITICAL: Apply gradients to originalParameters (explicitly passed in),
        // NOT to model.GetParameters(). This prevents double-stepping.
        //
        // UpdateParameters applies optimizer-specific logic: params_new = params_old - optimizer_update(gradients)
        var updatedParameters = UpdateParameters(originalParameters, gradients);

        // Create new model instance with updated parameters
        return model.WithParameters(updatedParameters);
    }

    /// <summary>
    /// Reverses a gradient update to recover original parameters.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This base implementation uses the vanilla SGD reversal formula:
    /// params_old = params_new + learning_rate * gradients
    /// </para>
    /// <para>
    /// <b>For Adaptive Optimizers (Adam, RMSprop, etc.):</b>
    /// This method should be overridden to account for optimizer-specific state.
    /// The base implementation is only accurate for vanilla SGD.
    /// </para>
    /// <para><b>For Beginners:</b> This calculates where the parameters were before
    /// a gradient update was applied. Think of it like rewinding a step you took.
    /// </para>
    /// </remarks>
    /// <param name="updatedParameters">Parameters after gradient application</param>
    /// <param name="appliedGradients">The gradients that were applied</param>
    /// <returns>Estimated original parameters</returns>
    public virtual Vector<T> ReverseUpdate(Vector<T> updatedParameters, Vector<T> appliedGradients)
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

        // Use current learning rate (may differ from initial due to decay/scheduling)
        var lr = NumOps.FromDouble(_currentLearningRate);

        // Reverse the SGD update using vectorized operations: params_old = params_new + lr * gradients
        var lrTimesGradients = (Vector<T>)Engine.Multiply(appliedGradients, lr);
        return (Vector<T>)Engine.Add(updatedParameters, lrTimesGradients);
    }

    /// <summary>
    /// Enables mixed-precision training for this optimizer.
    /// </summary>
    /// <param name="config">Configuration for mixed-precision training (optional, uses defaults if null).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Mixed-precision training uses a mix of 16-bit (FP16) and 32-bit (FP32) floating-point
    /// numbers during optimization to achieve faster training while maintaining accuracy.
    ///
    /// Benefits:
    /// - **2-3x faster** on modern GPUs with Tensor Cores (V100, A100, RTX 3000+)
    /// - **~50% memory reduction** allows larger batches or models
    /// - **Maintained accuracy** through FP32 master weights and loss scaling
    ///
    /// When to use:
    /// - ✅ Training large models with gradient-based optimizers
    /// - ✅ Using modern GPUs with Tensor Core support
    /// - ✅ Memory-constrained scenarios
    /// - ❌ CPU-only training (minimal benefit)
    /// - ❌ Non-gradient optimizers (genetic algorithms, etc.)
    ///
    /// Note: Only works with float (FP32) as the base type T.
    /// </para>
    /// </remarks>
    /// <example>
    /// <code>
    /// var optimizer = new AdamOptimizer&lt;float, Matrix&lt;float&gt;, Vector&lt;float&gt;&gt;(model, options);
    /// optimizer.EnableMixedPrecision();
    ///
    /// // Or with custom configuration
    /// optimizer.EnableMixedPrecision(MixedPrecisionConfig.Conservative());
    /// </code>
    /// </example>
    /// <exception cref="NotSupportedException">Thrown when T is not float.</exception>
    /// <exception cref="InvalidOperationException">Thrown when mixed-precision is already enabled.</exception>
    internal virtual void EnableMixedPrecision(MixedPrecisionConfig? config = null)
    {
        // Check that T is float
        if (typeof(T) != typeof(float))
        {
            throw new NotSupportedException(
                $"Mixed-precision training is only supported for optimizers with type parameter float. " +
                $"Current type: {typeof(T).Name}. " +
                $"Use Optimizer<float, ...> to enable mixed-precision training.");
        }

        if (_mixedPrecisionContext != null)
        {
            throw new InvalidOperationException(
                "Mixed-precision training is already enabled. Call DisableMixedPrecision() first if you want to change the configuration.");
        }

        _mixedPrecisionContext = new MixedPrecisionContext(config);
    }

    /// <summary>
    /// Disables mixed-precision training and releases associated resources.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This turns off mixed-precision training and returns the optimizer to
    /// standard FP32 operation. Useful for debugging or comparing performance.
    /// </para>
    /// </remarks>
    internal virtual void DisableMixedPrecision()
    {
        if (_mixedPrecisionContext != null)
        {
            _mixedPrecisionContext.Dispose();
            _mixedPrecisionContext = null;
        }
    }

    /// <summary>
    /// Gets the mixed-precision training context (if enabled).
    /// </summary>
    /// <returns>The mixed-precision context, or null if mixed-precision is disabled.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This provides access to the mixed-precision training internals,
    /// such as the current loss scale and overflow statistics. Useful for monitoring and debugging.
    /// </para>
    /// </remarks>
    internal virtual MixedPrecisionContext? GetMixedPrecisionContext()
    {
        return _mixedPrecisionContext;
    }

    /// <summary>
    /// Applies gradients with mixed-precision support (if enabled).
    /// </summary>
    /// <param name="originalParameters">The original parameters in FP32.</param>
    /// <param name="gradients">The gradients (may be in FP16 if mixed-precision is enabled).</param>
    /// <param name="model">The model to update.</param>
    /// <param name="scaledLoss">Optional scaled loss value (if mixed-precision is enabled).</param>
    /// <returns>The updated model with new parameters.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method handles gradient application with mixed-precision support.
    /// If mixed-precision is enabled, it:
    /// 1. Unscales the gradients
    /// 2. Checks for overflow/underflow
    /// 3. Updates parameters in FP32 (master weights)
    /// 4. Skips the update if overflow is detected
    /// </para>
    /// </remarks>
    internal virtual IFullModel<T, TInput, TOutput> ApplyGradientsWithMixedPrecision(
        Vector<T> originalParameters,
        Vector<T> gradients,
        IFullModel<T, TInput, TOutput> model)
    {
        // If mixed-precision is not enabled, use standard application
        if (_mixedPrecisionContext == null)
        {
            return ApplyGradients(originalParameters, gradients, model);
        }

        // Cast to float (required for mixed-precision context)
        var gradientsFloat = gradients as Vector<float>
            ?? throw new InvalidOperationException("Gradients must be Vector<float> for mixed-precision training.");

        // Unscale gradients and check for overflow
        bool isValid = _mixedPrecisionContext.LossScaler.UnscaleGradientsAndCheck(gradientsFloat);

        if (!isValid)
        {
            // Overflow detected - return model unchanged
            return model;
        }

        // Apply gradients normally (now unscaled)
        return ApplyGradients(originalParameters, gradients, model);
    }

    /// <summary>
    /// Creates a regularization technique based on the provided options.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method sets up a way to prevent the model from becoming too complex.
    /// It's like adding rules to your hiking strategy to avoid taking unnecessarily complicated paths.
    /// </para>
    /// </remarks>
    /// <param name="options">The options specifying the regularization technique to use.</param>
    /// <returns>An instance of the specified regularization technique.</returns>
    protected IRegularization<T, TInput, TOutput> CreateRegularization(GradientDescentOptimizerOptions<T, TInput, TOutput> options)
    {
        return RegularizationFactory.CreateRegularization<T, TInput, TOutput>(options.RegularizationOptions);
    }

    /// <summary>
    /// Calculates the gradient for the given model and input data.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method calculates how steep the hill is and in which direction.
    /// It helps determine which way the optimizer should step to improve the model.
    /// </para>
    /// </remarks>
    /// <param name="model">The current model.</param>
    /// <param name="X">The input features.</param>
    /// <param name="y">The target values.</param>
    /// <returns>The calculated gradient.</returns>
    /// <summary>
    /// Calculates the gradient for the given solution and input data.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method calculates how steep the hill is and in which direction.
    /// It helps determine which way the optimizer should step to improve the model.
    /// This implementation uses the loss function's derivative for efficient gradient calculation.
    /// </para>
    /// <para><b>Production Enhancement:</b>
    /// If the model implements IGradientComputable, this method automatically uses efficient
    /// backpropagation-based gradient computation. Otherwise, it falls back to the traditional
    /// approach using loss function derivatives.
    /// </para>
    /// </remarks>
    /// <param name="solution">The current solution.</param>
    /// <param name="X">The input features.</param>
    /// <param name="y">The target values.</param>
    /// <returns>The calculated gradient.</returns>
    protected virtual Vector<T> CalculateGradient(
        IFullModel<T, TInput, TOutput> solution,
        TInput X,
        TOutput y)
    {
        string cacheKey = GenerateGradientCacheKey(solution, X, y);
        var cachedGradient = GradientCache.GetCachedGradient(cacheKey);
        if (cachedGradient != null)
        {
            // CRITICAL: Clone the cached gradient to prevent external modifications from corrupting the cache.
            // If we return the cached vector directly, callers could modify it (e.g., during AllReduce operations),
            // which would corrupt the cache for future calls with the same key.
            var clonedGradient = new Vector<T>(cachedGradient.Parameters.ToArray());
            _lastComputedGradients = clonedGradient;
            return clonedGradient;
        }

        Vector<T> gradient;

        // Try to use explicit gradient computation if available (more efficient and accurate)
        if (solution is IGradientComputable<T, TInput, TOutput> gradientComputable)
        {
            gradient = gradientComputable.ComputeGradients(X, y, LossFunction);
        }
        else
        {
            // Fallback to traditional gradient computation via loss function derivative
            TOutput predictions = solution.Predict(X);

            if (predictions is Tensor<T> tensorPredictions && y is Tensor<T> tensorY)
            {
                gradient = LossFunction.CalculateDerivative(tensorPredictions.ToVector(), tensorY.ToVector());
            }
            else if (predictions is Vector<T> vectorPredictions && y is Vector<T> vectorY)
            {
                gradient = LossFunction.CalculateDerivative(vectorPredictions, vectorY);
            }
            else
            {
                throw new ArgumentException("Unsupported prediction or target type");
            }
        }

        // Apply regularization to the gradient
        var parameters = solution.GetParameters();
        var regularizationGradient = Regularization.Regularize(parameters);
        gradient = gradient.Add(regularizationGradient);

        // Scale the gradient by the batch size
        int batchSize = InputHelper<T, TInput>.GetBatchSize(X);
        gradient = gradient.Divide(NumOps.FromDouble(batchSize));

        // Apply gradient clipping if enabled
        gradient = ApplyGradientClipping(gradient);

        var gradientModel = new GradientModel<T>(gradient);
        GradientCache.CacheGradient(cacheKey, gradientModel);

        // Store for external access (enables gradient clipping, true DDP, debugging, etc.)
        _lastComputedGradients = gradient;

        return gradient;
    }

    /// <summary>
    /// Applies gradient clipping based on the configured options.
    /// </summary>
    /// <param name="gradient">The gradient to clip.</param>
    /// <returns>The clipped gradient.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Gradient clipping prevents training instability by limiting
    /// how large gradients can become. This is especially important for deep networks and RNNs
    /// where gradients can "explode" (become extremely large) during backpropagation.
    /// </para>
    /// </remarks>
    protected virtual Vector<T> ApplyGradientClipping(Vector<T> gradient)
    {
        if (!GradientOptions.EnableGradientClipping)
        {
            return gradient;
        }

        return GradientOptions.GradientClippingMethod switch
        {
            GradientClippingMethod.ByNorm => GradientClippingHelper.ClipByNorm(gradient, GradientOptions.MaxGradientNorm) ?? gradient,
            GradientClippingMethod.ByValue => GradientClippingHelper.ClipByValue(gradient, GradientOptions.MaxGradientValue) ?? gradient,
            _ => gradient
        };
    }

    /// <summary>
    /// Checks if the current gradients are exhibiting exploding gradient behavior.
    /// </summary>
    /// <param name="threshold">The threshold above which gradients are considered exploding. Default is 1000.</param>
    /// <returns>True if gradients are exploding, false otherwise.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method helps detect when training is becoming unstable.
    /// If gradients become too large, it usually indicates a problem with the learning rate
    /// or model architecture that needs to be addressed.
    /// </para>
    /// </remarks>
    public bool AreGradientsExploding(double threshold = 1000.0)
    {
        if (_lastComputedGradients == null || _lastComputedGradients.Length == 0)
        {
            return false;
        }

        return GradientClippingHelper.AreGradientsExploding(_lastComputedGradients, threshold);
    }

    /// <summary>
    /// Checks if the current gradients are exhibiting vanishing gradient behavior.
    /// </summary>
    /// <param name="threshold">The threshold below which gradients are considered vanishing. Default is 1e-7.</param>
    /// <returns>True if gradients are vanishing, false otherwise.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Vanishing gradients occur when gradients become so small that
    /// learning effectively stops. This is common in deep networks and can indicate the need
    /// for techniques like residual connections, batch normalization, or different activation functions.
    /// </para>
    /// </remarks>
    public bool AreGradientsVanishing(double threshold = 1e-7)
    {
        if (_lastComputedGradients == null || _lastComputedGradients.Length == 0)
        {
            return false;
        }

        return GradientClippingHelper.AreGradientsVanishing(_lastComputedGradients, threshold);
    }

    /// <summary>
    /// Gets the L2 norm of the last computed gradients.
    /// </summary>
    /// <returns>The gradient norm, or 0 if no gradients have been computed.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The gradient norm is a measure of how "strong" the overall
    /// gradient is. Monitoring this value during training can help diagnose issues with
    /// exploding or vanishing gradients.
    /// </para>
    /// </remarks>
    public T GetGradientNorm()
    {
        if (_lastComputedGradients == null || _lastComputedGradients.Length == 0)
        {
            return NumOps.Zero;
        }

        return GradientClippingHelper.ComputeNorm(_lastComputedGradients);
    }

    /// <summary>
    /// Computes the Hessian matrix (second derivatives) more efficiently when the model supports explicit gradient computation.
    /// </summary>
    /// <param name="model">The model to compute Hessian for.</param>
    /// <param name="inputData">The input data for optimization.</param>
    /// <returns>The Hessian matrix.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The Hessian tells us how the gradient changes - it's the "curvature" of the loss landscape.
    /// This is crucial for second-order optimization methods like Newton's method.
    /// </para>
    /// <para><b>Production Enhancement:</b>
    /// If the model implements IGradientComputable, this method computes the Hessian by taking gradients
    /// of the gradient (using finite differences on the gradient function), which is much more efficient
    /// than the traditional double finite differences approach. This is O(n) gradient evaluations instead
    /// of O(n²) loss evaluations.
    /// </para>
    /// <para><b>Note:</b>
    /// For models implementing IGradientComputable with ComputeSecondOrderGradients support,
    /// true Hessian-vector products could be computed even more efficiently. This is currently
    /// a middle ground that works with any model implementing ComputeGradients.
    /// </para>
    /// </remarks>
    protected virtual Matrix<T> ComputeHessianEfficiently(
        IFullModel<T, TInput, TOutput> model,
        OptimizationInputData<T, TInput, TOutput> inputData)
    {
        var parameters = model.GetParameters();
        int n = parameters.Length;
        var hessian = new Matrix<T>(n, n);
        var epsilon = NumOps.FromDouble(1e-5);

        // Check if model supports explicit gradient computation
        if (model is IGradientComputable<T, TInput, TOutput> gradientComputable)
        {
            // Efficient approach: Compute gradients at perturbed points
            // This is O(n) gradient computations instead of O(n²) loss computations
            var baseGradient = gradientComputable.ComputeGradients(
                inputData.XTrain,
                inputData.YTrain,
                LossFunction);

            for (int i = 0; i < n; i++)
            {
                // Perturb parameter i
                var perturbedParams = parameters.Clone();
                perturbedParams[i] = NumOps.Add(perturbedParams[i], epsilon);

                var perturbedModel = model.WithParameters(perturbedParams);

                if (perturbedModel is not IGradientComputable<T, TInput, TOutput> perturbedGradientModel)
                {
                    // Fallback to finite differences when perturbed model loses IGradientComputable
                    return ComputeHessianFiniteDifferences(model, inputData);
                }

                var perturbedGradient = perturbedGradientModel.ComputeGradients(
                    inputData.XTrain,
                    inputData.YTrain,
                    LossFunction);

                // Hessian column i = (∇f(x + εe_i) - ∇f(x)) / ε
                for (int j = 0; j < n; j++)
                {
                    var diff = NumOps.Subtract(perturbedGradient[j], baseGradient[j]);
                    hessian[j, i] = NumOps.Divide(diff, epsilon);
                }
            }
        }
        else
        {
            // Fallback: Traditional finite differences (slower but works for all models)
            hessian = ComputeHessianFiniteDifferences(model, inputData);
        }

        return hessian;
    }

    /// <summary>
    /// Computes the Hessian matrix using traditional finite differences (fallback method).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the slower but more universally applicable method.
    /// It approximates the curvature by testing small changes in parameters.
    /// </para>
    /// </remarks>
    protected virtual Matrix<T> ComputeHessianFiniteDifferences(
        IFullModel<T, TInput, TOutput> model,
        OptimizationInputData<T, TInput, TOutput> inputData)
    {
        var parameters = model.GetParameters();
        int n = parameters.Length;
        var hessian = new Matrix<T>(n, n);
        var epsilon = NumOps.FromDouble(1e-5);

        for (int i = 0; i < n; i++)
        {
            for (int j = i; j < n; j++)  // Symmetric matrix, only compute upper triangle
            {
                // Compute second partial derivative ∂²f/∂xi∂xj using finite differences
                // f''(x,y) ≈ [f(x+h,y+h) - f(x+h,y-h) - f(x-h,y+h) + f(x-h,y-h)] / (4h²)

                var params_pp = parameters.Clone();
                params_pp[i] = NumOps.Add(params_pp[i], epsilon);
                params_pp[j] = NumOps.Add(params_pp[j], epsilon);
                var model_pp = model.WithParameters(params_pp);
                var loss_pp = CalculateLoss(model_pp, inputData);

                var params_pm = parameters.Clone();
                params_pm[i] = NumOps.Add(params_pm[i], epsilon);
                params_pm[j] = NumOps.Subtract(params_pm[j], epsilon);
                var model_pm = model.WithParameters(params_pm);
                var loss_pm = CalculateLoss(model_pm, inputData);

                var params_mp = parameters.Clone();
                params_mp[i] = NumOps.Subtract(params_mp[i], epsilon);
                params_mp[j] = NumOps.Add(params_mp[j], epsilon);
                var model_mp = model.WithParameters(params_mp);
                var loss_mp = CalculateLoss(model_mp, inputData);

                var params_mm = parameters.Clone();
                params_mm[i] = NumOps.Subtract(params_mm[i], epsilon);
                params_mm[j] = NumOps.Subtract(params_mm[j], epsilon);
                var model_mm = model.WithParameters(params_mm);
                var loss_mm = CalculateLoss(model_mm, inputData);

                // Compute second derivative
                var numerator = NumOps.Add(
                    NumOps.Subtract(loss_pp, loss_pm),
                    NumOps.Subtract(loss_mm, loss_mp));
                var denominator = NumOps.Multiply(
                    NumOps.Multiply(NumOps.FromDouble(4.0), epsilon),
                    epsilon);
                var secondDerivative = NumOps.Divide(numerator, denominator);

                hessian[i, j] = secondDerivative;
                hessian[j, i] = secondDerivative;  // Symmetric
            }
        }

        return hessian;
    }

    /// <summary>
    /// Performs a line search to find an appropriate step size.
    /// </summary>
    /// <param name="currentSolution">The current solution.</param>
    /// <param name="direction">The search direction.</param>
    /// <param name="gradient">The current gradient.</param>
    /// <param name="inputData">The input data for the optimization process.</param>
    /// <returns>The step size to use.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method determines how big of a step to take in the chosen direction.
    /// It tries to find a step size that sufficiently decreases the function value while not being too small.
    /// </para>
    /// </remarks>
    protected T LineSearch(IFullModel<T, TInput, TOutput> currentSolution, Vector<T> direction, Vector<T> gradient, OptimizationInputData<T, TInput, TOutput> inputData)
    {
        var alpha = NumOps.FromDouble(1.0);
        var c1 = NumOps.FromDouble(1e-4);
        var c2 = NumOps.FromDouble(0.9);
        var xTrain = inputData.XTrain;
        var yTrain = inputData.YTrain;

        var initialValue = CalculateLoss(currentSolution, inputData);
        var initialSlope = gradient.DotProduct(direction);

        while (true)
        {
            var newCoefficients = currentSolution.GetParameters().Add(direction.Multiply(alpha));
            var newSolution = currentSolution.WithParameters(newCoefficients);
            var newValue = CalculateLoss(newSolution, inputData);

            if (NumOps.LessThanOrEquals(newValue, NumOps.Add(initialValue, NumOps.Multiply(NumOps.Multiply(c1, alpha), initialSlope))))
            {
                var newGradient = CalculateGradient(newSolution, xTrain, yTrain);
                var newSlope = newGradient.DotProduct(direction);

                if (NumOps.GreaterThanOrEquals(NumOps.Abs(newSlope), NumOps.Multiply(c2, NumOps.Abs(initialSlope))))
                {
                    return alpha;
                }
            }

            alpha = NumOps.Multiply(alpha, NumOps.FromDouble(0.5));

            if (NumOps.LessThan(alpha, NumOps.FromDouble(1e-10)))
            {
                return NumOps.FromDouble(1e-10);
            }
        }
    }


    /// <summary>
    /// Calculates the gradient for a given solution using a batch of training data.
    /// </summary>
    /// <param name="solution">The current solution (model).</param>
    /// <param name="xTrain">The training input data.</param>
    /// <param name="yTrain">The training target data.</param>
    /// <param name="batchIndices">The indices to use for the current batch.</param>
    /// <returns>A vector representing the gradient of the loss function with respect to the model parameters.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The gradient tells us which direction to adjust our model's
    /// parameters to improve performance. It's like a compass showing the way to a better solution.
    /// </para>
    /// </remarks>
    protected virtual Vector<T> CalculateGradient(
        IFullModel<T, TInput, TOutput> solution,
        TInput xTrain,
        TOutput yTrain,
        int[] batchIndices)
    {
        // Extract batch data using your InputHelper
        var xBatch = InputHelper<T, TInput>.GetBatch(xTrain, batchIndices);
        var yBatch = InputHelper<T, TOutput>.GetBatch(yTrain, batchIndices);

        // Get the current parameters
        var parameters = solution.GetParameters();
        var gradient = new Vector<T>(parameters.Length);

        // Initialize gradient vector with zeros
        for (int i = 0; i < gradient.Length; i++)
        {
            gradient[i] = NumOps.Zero;
        }

        // For each sample in the batch
        for (int i = 0; i < batchIndices.Length; i++)
        {
            // Get the current input and output for this sample
            var input = InputHelper<T, TInput>.GetItem(xBatch, i);
            var target = InputHelper<T, TOutput>.GetItem(yBatch, i);

            // Predict the output
            var prediction = solution.Predict(input);

            // Calculate the error
            var error = LossFunction.CalculateLoss(ConversionsHelper.ConvertToVector<T, TOutput>(prediction), ConversionsHelper.ConvertToVector<T, TOutput>(target));

            // Update gradient based on the error
            for (int j = 0; j < parameters.Length; j++)
            {
                var featureValue = InputHelper<T, TInput>.GetFeatureValue(input, j);
                var contribution = NumOps.Multiply(error, featureValue);
                gradient[j] = NumOps.Add(gradient[j], contribution);
            }
        }

        // Average the gradient using vectorized division
        var batchSizeScalar = NumOps.FromDouble(batchIndices.Length);
        gradient = (Vector<T>)Engine.Divide(gradient, batchSizeScalar);

        // Store for external access (enables gradient clipping, true DDP, debugging, etc.)
        _lastComputedGradients = gradient;

        return gradient;
    }

    /// <summary>
    /// Updates the current solution based on the calculated gradient.
    /// </summary>
    /// <param name="currentSolution">The current solution being optimized.</param>
    /// <param name="gradient">The calculated gradient.</param>
    /// <returns>A new solution with updated parameters.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method moves the model's parameters in the direction
    /// indicated by the gradient, hopefully improving the model's performance.
    /// </para>
    /// </remarks>
    protected virtual IFullModel<T, TInput, TOutput> UpdateSolution(
        IFullModel<T, TInput, TOutput> currentSolution,
        Vector<T> gradient)
    {
        var parameters = currentSolution.GetParameters();
        var newParameters = UpdateParameters(parameters, gradient);

        return currentSolution.WithParameters(newParameters);
    }

    /// <summary>
    /// Generates a unique key for caching gradients.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method creates a unique identifier for each gradient calculation.
    /// It's like labeling each spot on the hill so you can remember what the gradient was there.
    /// </para>
    /// </remarks>
    /// <param name="model">The current model.</param>
    /// <param name="X">The input features.</param>
    /// <param name="y">The target values.</param>
    /// <returns>A string key for caching the gradient.</returns>
    protected virtual string GenerateGradientCacheKey(IFullModel<T, TInput, TOutput> model, TInput X, TOutput y)
    {
        return $"{model.GetType().Name}_{InputHelper<T, TInput>.GetBatchSize(X)}_{InputHelper<T, TInput>.GetInputSize(X)}_{GradientOptions.GetType().Name}";
    }

    /// <summary>
    /// Resets the optimizer to its initial state.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method clears all the remembered information and starts fresh.
    /// It's like wiping your map clean and starting your hike from the beginning.
    /// </para>
    /// </remarks>
    public override void Reset()
    {
        base.Reset();
        GradientCache.ClearCache();

        // Reset learning rate scheduler state
        _learningRateScheduler?.Reset();
        _currentStep = 0;
        _currentEpoch = 0;

        // Restore initial learning rate
        if (_learningRateScheduler != null)
        {
            _currentLearningRate = _learningRateScheduler.CurrentLearningRate;
        }
        else
        {
            _currentLearningRate = GradientOptions.InitialLearningRate;
        }
    }

    /// <summary>
    /// Steps the learning rate scheduler and updates the current learning rate.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method advances the scheduler by one step and synchronizes the optimizer's
    /// learning rate with the scheduler's current value.
    /// </para>
    /// <para><b>For Beginners:</b> Call this method to update the learning rate according
    /// to the scheduler's policy. The scheduler will automatically adjust the learning rate
    /// based on how many steps have been taken.
    /// </para>
    /// </remarks>
    /// <returns>The new learning rate after stepping.</returns>
    public double StepScheduler()
    {
        if (_learningRateScheduler != null)
        {
            _currentLearningRate = _learningRateScheduler.Step();
        }
        return _currentLearningRate;
    }

    /// <summary>
    /// Called at the end of each training epoch to update scheduler state if applicable.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When using <see cref="SchedulerStepMode.StepPerEpoch"/> or <see cref="SchedulerStepMode.WarmupThenEpoch"/>,
    /// this method should be called at the end of each epoch to advance the scheduler.
    /// </para>
    /// <para><b>For Beginners:</b> An epoch is one complete pass through all your training data.
    /// Many learning rate schedules (like step decay or cosine annealing) work on an epoch basis,
    /// reducing the learning rate after each complete pass through the data.
    /// </para>
    /// </remarks>
    public virtual void OnEpochEnd()
    {
        _currentEpoch++;

        if (_learningRateScheduler != null)
        {
            bool shouldStep = _schedulerStepMode switch
            {
                SchedulerStepMode.StepPerEpoch => true,
                SchedulerStepMode.WarmupThenEpoch => !IsInWarmupPhase(),
                _ => false
            };

            if (shouldStep)
            {
                StepScheduler();
            }
        }
    }

    /// <summary>
    /// Called at the end of each training batch to update scheduler state if applicable.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When using <see cref="SchedulerStepMode.StepPerBatch"/> or during the warmup phase of
    /// <see cref="SchedulerStepMode.WarmupThenEpoch"/>, this method should be called after each
    /// batch to advance the scheduler.
    /// </para>
    /// <para><b>For Beginners:</b> A batch is a small subset of your training data processed at once.
    /// Some schedulers (like warmup or cyclical learning rates) need to update after every batch
    /// for smooth, fine-grained control of the learning rate.
    /// </para>
    /// </remarks>
    public virtual void OnBatchEnd()
    {
        _currentStep++;

        if (_learningRateScheduler != null)
        {
            bool shouldStep = _schedulerStepMode switch
            {
                SchedulerStepMode.StepPerBatch => true,
                SchedulerStepMode.WarmupThenEpoch => IsInWarmupPhase(),
                _ => false
            };

            if (shouldStep)
            {
                StepScheduler();
            }
        }
    }

    /// <summary>
    /// Determines whether the scheduler is currently in the warmup phase.
    /// </summary>
    /// <returns>True if in warmup phase, false otherwise.</returns>
    /// <remarks>
    /// <para>
    /// Warmup is a technique where the learning rate starts very low and gradually increases
    /// to the base learning rate over a specified number of steps. This helps stabilize
    /// training in the early phases.
    /// </para>
    /// </remarks>
    protected virtual bool IsInWarmupPhase()
    {
        if (_learningRateScheduler == null)
        {
            return false;
        }

        // Check if the scheduler supports warmup detection
        // A scheduler is in warmup if current LR is below base LR and still increasing
        var currentLr = _learningRateScheduler.CurrentLearningRate;
        var baseLr = _learningRateScheduler.BaseLearningRate;

        // If current LR is less than base LR, we're likely still in warmup
        // This is a heuristic - specific schedulers may override this behavior
        return currentLr < baseLr;
    }

    /// <summary>
    /// Gets the current learning rate being used by this optimizer.
    /// </summary>
    /// <returns>The current learning rate.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The learning rate controls how big each update step is.
    /// This value may change during training if a learning rate scheduler is configured.
    /// </para>
    /// </remarks>
    public double GetCurrentLearningRate()
    {
        return _currentLearningRate;
    }

    /// <summary>
    /// Gets the current training step (batch count).
    /// </summary>
    public int CurrentStep => _currentStep;

    /// <summary>
    /// Gets the current training epoch.
    /// </summary>
    public int CurrentEpoch => _currentEpoch;

    /// <summary>
    /// Applies momentum to the gradient calculation.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method considers the direction you were moving in previously
    /// when deciding which way to go next. It's like considering your momentum when hiking -
    /// you might keep going in roughly the same direction rather than abruptly changing course.
    /// </para>
    /// </remarks>
    /// <param name="gradient">The current gradient.</param>
    /// <returns>The gradient adjusted for momentum.</returns>
    protected virtual Vector<T> ApplyMomentum(Vector<T> gradient)
    {
        if (_previousGradient == null || _previousGradient.Length == 0 || _previousGradient.Length != gradient.Length)
        {
            _previousGradient = gradient;
            return gradient;
        }

        var momentumGradient = _previousGradient.Add(gradient.Multiply(NumOps.FromDouble(_currentMomentum)));
        _previousGradient = momentumGradient;
        return momentumGradient;
    }

    /// <summary>
    /// Updates the parameters of the model based on the calculated gradients.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method adjusts the model's parameters to improve its performance.
    /// It's like taking steps in the direction that will lead to better results, based on what we've learned
    /// from the data.</para>
    /// </remarks>
    /// <param name="layers">The layers of the neural network containing the parameters to update.</param>
    public virtual void UpdateParameters(List<ILayer<T>> layers)
    {
        foreach (var layer in layers)
        {
            if (layer.SupportsTraining)
            {
                Vector<T> parameters = layer.GetParameters();
                Vector<T> gradients = layer.GetParameterGradients();

                // Apply simple gradient descent update using vectorized operations
                var lr = NumOps.FromDouble(_currentLearningRate);
                var scaledGradients = (Vector<T>)Engine.Multiply(gradients, lr);
                var newParameters = (Vector<T>)Engine.Subtract(parameters, scaledGradients);

                layer.SetParameters(newParameters);
                layer.ClearGradients();
            }
        }
    }

    /// <summary>
    /// Updates a matrix of parameters based on the calculated gradient.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method adjusts the model's parameters to improve its performance.
    /// It's like taking a step in the direction you've determined will lead you downhill.
    /// </para>
    /// </remarks>
    /// <param name="parameters">The current parameters.</param>
    /// <param name="gradient">The calculated gradient.</param>
    /// <returns>The updated parameters.</returns>
    public virtual Matrix<T> UpdateParameters(Matrix<T> parameters, Matrix<T> gradient)
    {
        var learningRate = NumOps.FromDouble(_currentLearningRate);
        return parameters.Subtract(gradient.Multiply(learningRate));
    }

    /// <summary>
    /// Updates a tensor of parameters based on the calculated gradient.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method adjusts the model's parameters stored in tensor format to improve its performance.
    /// It's like taking a step in the direction you've determined will lead you downhill, but for more complex
    /// multi-dimensional data structures. Tensors are useful for representing parameters in deep neural networks
    /// where data has multiple dimensions (like images with width, height, and channels).
    /// </para>
    /// </remarks>
    /// <param name="parameters">The current tensor parameters.</param>
    /// <param name="gradient">The calculated gradient tensor.</param>
    /// <returns>The updated tensor parameters.</returns>
    public virtual Tensor<T> UpdateParameters(Tensor<T> parameters, Tensor<T> gradient)
    {
        var learningRate = NumOps.FromDouble(_currentLearningRate);

        // Scale the gradient by the learning rate
        var scaledGradient = gradient.Multiply(learningRate);

        // Subtract the scaled gradient from the parameters
        return parameters.Subtract(scaledGradient);
    }

    /// <summary>
    /// Updates a vector of parameters based on the calculated gradient.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method is similar to UpdateMatrix, but for when the parameters
    /// are in a vector format instead of a matrix. It's another way of taking a step to improve the model.
    /// </para>
    /// </remarks>
    /// <param name="parameters">The current parameters.</param>
    /// <param name="gradient">The calculated gradient.</param>
    /// <returns>The updated parameters.</returns>
    public virtual Vector<T> UpdateParameters(Vector<T> parameters, Vector<T> gradient)
    {
        var learningRate = NumOps.FromDouble(_currentLearningRate);
        return parameters.Subtract(gradient.Multiply(learningRate));
    }

    /// <summary>
    /// Updates the options for the gradient-based optimizer.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method allows you to change the settings of the optimizer
    /// while it's running. It's like adjusting your hiking strategy mid-journey based on the terrain you encounter.
    /// </para>
    /// </remarks>
    /// <param name="options">The new options to apply to the optimizer.</param>
    protected override void UpdateOptions(OptimizationAlgorithmOptions<T, TInput, TOutput> options)
    {
        if (options is GradientBasedOptimizerOptions<T, TInput, TOutput> gradientOptions)
        {
            GradientOptions = gradientOptions;
        }
    }
}
