using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Autodiff;
using Newtonsoft.Json;

using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.Optimizers;

/// <summary>
/// Implements the Broyden-Fletcher-Goldfarb-Shanno (BFGS) optimization algorithm.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// BFGS is a quasi-Newton method for solving unconstrained nonlinear optimization problems.
/// It approximates the Hessian matrix of second derivatives of the function to be minimized.
/// </para>
/// <para><b>For Beginners:</b> BFGS is an advanced optimization algorithm that tries to find the best solution
/// by making smart steps based on the function's behavior. It's particularly good at handling complex problems
/// where the function being optimized is smooth but potentially has many variables.
/// </para>
/// </remarks>
[ComponentType(ComponentType.Optimizer)]
[PipelineStage(PipelineStage.Training)]
public partial class BFGSOptimizer<T, TInput, TOutput> : GradientBasedOptimizerBase<T, TInput, TOutput>, Fused.IFusedOptimizerSpec
{
    /// <summary>
    /// Declines to fuse, always.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Stated explicitly rather than by omission, because an absent spec reads as "nobody got to it yet"
    /// and that has twice been the wrong reading in this codebase. This one is a property of the method:
    /// BFGS carries a dense n×n inverse-Hessian approximation and updates it with two rank-one terms per
    /// step. That state is quadratic in the parameter count and every element of the update touches all of
    /// it, so there is nothing for a fused optimizer — which owns per-parameter buffers and a flat step —
    /// to hold or apply. At the sizes fused training exists for, the n×n matrix does not fit in memory at
    /// all.
    /// </para>
    /// <para>
    /// The field's own answer to that is the compact representation of Byrd, Nocedal &amp; Schnabel (1994),
    /// which stores m curvature PAIRS instead of the matrix — and that is L-BFGS, which does fuse here.
    /// PyTorch ships only <c>LBFGS</c> from this family and documents it as supporting neither
    /// <c>foreach</c> nor <c>fused</c>. So the way to fuse BFGS is to use <see cref="LBFGSOptimizer{T,
    /// TInput, TOutput}"/> with a large enough memory, not to write a BFGS kernel.
    /// </para>
    /// </remarks>
    bool Fused.IFusedOptimizerSpec.TryGetFusedOptimizerConfig(out Fused.FusedOptimizerConfig config)
    {
        config = default;
        return false;
    }

    /// <summary>
    /// The options specific to the BFGS optimization algorithm.
    /// </summary>
    /// <summary>Read from the single instance OptimizerBase.Options holds, so there is
    /// no second copy that could disagree with it.</summary>
    private BFGSOptimizerOptions<T, TInput, TOutput> _options => (BFGSOptimizerOptions<T, TInput, TOutput>)Options;

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
    /// <param name="model">The model to optimize.</param>
    /// <param name="options">The options for configuring the BFGS algorithm.</param>
    /// <param name="engine">The computation engine (CPU or GPU) for vectorized operations.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This constructor sets up the BFGS optimizer with its initial configuration.
    /// You can customize various aspects of how it works, or use default settings.
    /// </para>
    /// </remarks>
    public BFGSOptimizer(
        IFullModel<T, TInput, TOutput> model,
        BFGSOptimizerOptions<T, TInput, TOutput>? options = null,
        IEngine? engine = null)
        : base(model, options ?? new())
    {
        InitializeAdaptiveParameters();
    }

    /// <summary>
    /// Creates a BFGS optimizer for minimizing a plain function, with no model attached.
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
    public static BFGSOptimizer<T, TInput, TOutput> CreateForFunction(
        BFGSOptimizerOptions<T, TInput, TOutput>? options = null)
        => new(options);

    /// <summary>
    /// Backs <see cref="CreateForFunction"/>: the same setup with no model.
    /// </summary>
    private BFGSOptimizer(BFGSOptimizerOptions<T, TInput, TOutput>? options)
        : base(null, options ?? new())
    {
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
        CurrentLearningRate = NumOps.FromDouble(_options.InitialLearningRate);
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
    /// <para><b>DataLoader Integration:</b> This method uses the DataLoader API for epoch management.
    /// BFGS typically operates on the full dataset because it builds an approximation of the inverse
    /// Hessian matrix that requires consistent gradients between iterations. The method notifies the
    /// sampler of epoch starts using <see cref="GradientBasedOptimizerBase{T,TInput,TOutput}.NotifyEpochStart"/>
    /// for compatibility with curriculum learning and sampling strategies.
    /// </para>
    /// </remarks>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        ValidationHelper<T>.ValidateInputData(inputData);

        var currentSolution = InitializeWorkingSolution(inputData.XTrain);
        var bestStepData = new OptimizationStepData<T, TInput, TOutput>();
        var previousStepData = new OptimizationStepData<T, TInput, TOutput>();
        var parameters = InterfaceGuard.Parameterizable(currentSolution).GetParameters();

        _inverseHessian = Matrix<T>.CreateIdentity(parameters.Length);
        _previousGradient = null;
        _previousParameters = null;
        InitializeAdaptiveParameters();

        for (int epoch = 0; epoch < _options.MaxIterations; epoch++)
        {
            NotifyEpochStart(epoch);
            _iteration++;

            parameters = InterfaceGuard.Parameterizable(currentSolution).GetParameters();
            var gradient = CalculateGradient(currentSolution, inputData.XTrain, inputData.YTrain);
            var newSolution = UpdateSolution(currentSolution, gradient, inputData);

            var currentStepData = EvaluateSolution(newSolution, inputData);
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

            _previousGradient = gradient;
            _previousParameters = parameters;
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
        // === Vectorized BFGS Update using IEngine (Phase B: US-GPU-015) ===

        var parameters = InterfaceGuard.Parameterizable(currentSolution).GetParameters();

        if (_inverseHessian is null)
        {
            throw new InvalidOperationException(
                "Inverse Hessian matrix has not been initialized. Ensure Initialize() is called before UpdateSolution().");
        }

        if (_previousGradient != null && _previousParameters != null)
        {
            UpdateInverseHessian(parameters, gradient);
        }

        var direction = _inverseHessian.Multiply(gradient);
        // Vectorized negation
        direction = (Vector<T>)Engine.Multiply(direction, NumOps.Negate(NumOps.One));

        var step = LineSearch(currentSolution, direction, gradient, inputData);
        // Vectorized scaling
        var scaledDirection = (Vector<T>)Engine.Multiply(direction, step);
        var newCoefficients = parameters.Add(scaledDirection);

        return InterfaceGuard.Parameterizable(currentSolution).WithParameters(newCoefficients);
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
        // === Partially Vectorized Hessian Update using IEngine (Phase B: US-GPU-015) ===
        // s = current_params - previous_params
        // y = current_grad - previous_grad

        var previousParams = _previousParameters ?? throw new InvalidOperationException("Previous parameters have not been initialized.");
        var previousGrad = _previousGradient ?? throw new InvalidOperationException("Previous gradient has not been initialized.");
        var inverseHessian = _inverseHessian ?? throw new InvalidOperationException("Inverse Hessian has not been initialized.");

        var s = (Vector<T>)Engine.Subtract(currentParameters, previousParams);
        var y = (Vector<T>)Engine.Subtract(currentGradient, previousGrad);

        // Curvature condition: only update when y·s > 0 (positive definiteness)
        var ys = y.DotProduct(s);
        double ysDouble = Convert.ToDouble(ys);
        if (ysDouble <= 1e-10)
        {
            // Skip update — curvature condition not satisfied
            return;
        }

        var rho = NumOps.Divide(NumOps.FromDouble(1), ys);
        var I = Matrix<T>.CreateIdentity(currentParameters.Length);

        var term1 = I.Subtract(s.OuterProduct(y).Multiply(rho));
        var term2 = I.Subtract(y.OuterProduct(s).Multiply(rho));
        var term3 = s.OuterProduct(s).Multiply(rho);

        _inverseHessian = term1.Multiply(inverseHessian).Multiply(term2).Add(term3);
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
    /// Updates parameters using the BFGS algorithm with inverse Hessian approximation.
    /// </summary>
    /// <param name="parameters">The current parameter values.</param>
    /// <param name="gradient">The gradient at the current parameters.</param>
    /// <returns>The updated parameters.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method implements the core BFGS update formula.
    /// It uses the inverse Hessian approximation to determine a search direction that
    /// typically converges faster than standard gradient descent.
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

        _iteration++;

        // Initialize inverse Hessian as identity on first call
        if (_inverseHessian is null || _inverseHessian.Rows != parameters.Length)
        {
            _inverseHessian = Matrix<T>.CreateIdentity(parameters.Length);
            _previousGradient = null;
            _previousParameters = null;
            _iteration = 0;
        }

        // Compute gradient norm for adaptive scaling
        var gradientNorm = gradient.Norm();

        // Skip update if gradient is near zero
        if (NumOps.LessThanOrEquals(gradientNorm, NumOps.FromDouble(1e-10)))
        {
            return parameters;
        }

        // Gradient clipping to prevent overshooting on ill-conditioned problems
        // Clip gradients with norm > 10 to prevent explosive updates
        var maxGradientNorm = NumOps.FromDouble(10.0);
        Vector<T> clippedGradient;
        if (NumOps.GreaterThan(gradientNorm, maxGradientNorm))
        {
            var scale = NumOps.Divide(maxGradientNorm, gradientNorm);
            clippedGradient = (Vector<T>)Engine.Multiply(gradient, scale);
        }
        else
        {
            clippedGradient = gradient;
        }

        // Update inverse Hessian if we have previous state
        if (_previousGradient is not null && _previousParameters is not null)
        {
            // s = x_k - x_{k-1}
            var s = (Vector<T>)Engine.Subtract(parameters, _previousParameters);
            // y = g_k - g_{k-1}
            var y = (Vector<T>)Engine.Subtract(clippedGradient, _previousGradient);

            var sDotY = s.DotProduct(y);

            // Only update if curvature condition is satisfied
            if (NumOps.GreaterThan(sDotY, NumOps.FromDouble(1e-10)))
            {
                var rho = NumOps.Divide(NumOps.One, sDotY);
                var I = Matrix<T>.CreateIdentity(parameters.Length);

                // BFGS update formula:
                // H_{k+1} = (I - rho * s * y^T) * H_k * (I - rho * y * s^T) + rho * s * s^T
                var term1 = I.Subtract(s.OuterProduct(y).Multiply(rho));
                var term2 = I.Subtract(y.OuterProduct(s).Multiply(rho));
                var term3 = s.OuterProduct(s).Multiply(rho);

                _inverseHessian = term1.Multiply(_inverseHessian).Multiply(term2).Add(term3);
            }
        }

        // Compute search direction: d = -H * g
        var direction = _inverseHessian.Multiply(clippedGradient);
        var directionSpan = direction.AsWritableSpan();
        for (int i = 0; i < directionSpan.Length; i++)
        {
            directionSpan[i] = NumOps.Negate(directionSpan[i]);
        }

        // Limit step size to prevent overshooting
        // The maximum step should be proportional to parameter magnitudes
        var directionNorm = direction.Norm();
        var parameterNorm = parameters.Norm();
        var maxStepNorm = NumOps.GreaterThan(parameterNorm, NumOps.FromDouble(1.0))
            ? NumOps.Multiply(parameterNorm, NumOps.FromDouble(0.5))
            : NumOps.FromDouble(1.0);

        // Compute the proposed step
        for (int i = 0; i < directionSpan.Length; i++)
        {
            directionSpan[i] = NumOps.Multiply(directionSpan[i], CurrentLearningRate);
        }
        var scaledNorm = direction.Norm();

        // If step is too large, scale it down
        if (NumOps.GreaterThan(scaledNorm, maxStepNorm))
        {
            var stepScale = NumOps.Divide(maxStepNorm, scaledNorm);
            for (int i = 0; i < directionSpan.Length; i++)
            {
                directionSpan[i] = NumOps.Multiply(directionSpan[i], stepScale);
            }
        }

        var parameterSpan = parameters.AsSpan();
        for (int i = 0; i < directionSpan.Length; i++)
        {
            directionSpan[i] = NumOps.Add(parameterSpan[i], directionSpan[i]);
        }

        // Store state for next iteration
        if (_previousParameters is null || _previousParameters.Length != parameters.Length)
        {
            _previousParameters = new Vector<T>(parameters.Length, skipZeroInit: true);
            _previousGradient = new Vector<T>(parameters.Length, skipZeroInit: true);
        }
        parameterSpan.CopyTo(_previousParameters.AsWritableSpan());
        clippedGradient.AsSpan().CopyTo(_previousGradient!.AsWritableSpan());

        return direction;
    }

    /// <summary>
    /// Updates parameters using GPU-accelerated BFGS.
    /// </summary>
    /// <remarks>
    /// BFGS is a second-order quasi-Newton method that requires Hessian approximation.
    /// GPU implementation is not yet available due to the complexity of maintaining
    /// the inverse Hessian approximation across GPU memory.
    /// </remarks>
    public override void UpdateParametersGpu(IGpuBuffer parameters, IGpuBuffer gradients, int parameterCount, IDirectGpuBackend backend)
    {
        throw new NotSupportedException(
            "GPU-accelerated BFGS is not yet implemented. BFGS requires maintaining an inverse Hessian " +
            "approximation which is complex to implement efficiently on GPU. Use CPU-based UpdateParameters " +
            "or consider using Adam/AdamW for GPU-resident training.");
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
        return _options;
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
        return $"{baseKey}_BFGS_{_options.InitialLearningRate}_{_options.Tolerance}_{_iteration}";
    }

    /// <inheritdoc />
    public override void Step(TapeStepContext<T> context)
    {
        // Sparse-by-default: walk any embedding params whose gradient lives
        // only in the sparse list (Tensors stopped seeding dense alongside) and
        // materialise via ToDense into context.Gradients before GetFlatGradients
        // / Hessian assembly reads from the dict. No-op for non-embedding params
        // and for embedding params that already have a dense entry.
        SparseEmbeddingOptimizerHelpers.MaterializeSparseIntoGradientsDict(context, Engine);

        var pv = context.GetFlatParameters();
        var gv = context.GetFlatGradients();
        var updated = UpdateParameters(pv, gv);
        context.SetFlatParameters(updated);

        // BFGS gives a direction; the step length along it is the line search's job (Nocedal & Wright,
        // Algorithm 3.1). This used to claim to be a backtracking line search while doing the opposite —
        // taking a SECOND full step from the point it had just measured as worse.
        if (_options.UseLineSearch && context.SupportsReevaluation)
        {
            ApplyBacktrackingLineSearch(context, pv, updated, gv, _options.MaxLineSearchIterations);
        }
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// BFGS maintains a dense estimate of the INVERSE Hessian, so a step is a matrix-vector
    /// product rather than a linear solve — which is the difference between O(n^2) and O(n^3) per
    /// iteration, and the reason nobody implements it the other way round.
    /// </para>
    /// <para>
    /// The estimate starts at the identity and is corrected by the rank-two update of Broyden,
    /// Fletcher, Goldfarb and Shanno (Nocedal and Wright, "Numerical Optimization", Algorithm 6.1)
    /// after each accepted step. A step satisfying the strong Wolfe conditions guarantees
    /// <c>s'y &gt; 0</c>, which is what keeps the estimate positive definite; a pair that fails it
    /// is discarded rather than used.
    /// </para>
    /// <para><b>For Beginners:</b> This is Newton's method without the Hessian. Rather than
    /// computing the curvature at every step, it works the curvature out from how the slope
    /// changed over the steps it has already taken — which costs nothing extra, because it
    /// measured those gradients anyway.
    /// </para>
    /// </remarks>
    public override Vector<T> Minimize(
        Vector<T> initialParameters,
        Func<Vector<T>, (T objective, Vector<T> gradient)> objectiveAndGradient,
        int maxIterations,
        T tolerance,
        Func<Vector<T>, Vector<T>>? projection)
        => MinimizeWithLineSearch(
            initialParameters, objectiveAndGradient, maxIterations, tolerance, projection,
            new BfgsDirection(this), "BFGS");

    /// <summary>The BFGS update of the inverse Hessian estimate, as a direction rule.</summary>
    private sealed class BfgsDirection : ISearchDirectionRule
    {
        private readonly BFGSOptimizer<T, TInput, TOutput> _owner;
        private T[,] _inverse = new T[0, 0];
        private int _size;
        private bool _scaled;

        public BfgsDirection(BFGSOptimizer<T, TInput, TOutput> owner) => _owner = owner;

        public void Reset(int parameterCount)
        {
            _size = parameterCount;
            _inverse = new T[parameterCount, parameterCount];
            _scaled = false;

            for (int i = 0; i < parameterCount; i++)
            {
                for (int j = 0; j < parameterCount; j++)
                {
                    _inverse[i, j] = i == j ? _owner.NumOps.One : _owner.NumOps.Zero;
                }
            }
        }

        public Vector<T> ComputeDirection(Vector<T> gradient)
        {
            var numOps = _owner.NumOps;
            var direction = new Vector<T>(_size);

            for (int i = 0; i < _size; i++)
            {
                T total = numOps.Zero;
                for (int j = 0; j < _size; j++)
                {
                    total = numOps.Add(total, numOps.Multiply(_inverse[i, j], gradient[j]));
                }

                direction[i] = numOps.Negate(total);
            }

            return direction;
        }

        public void Observe(
            Vector<T> step, Vector<T> gradientChange, Vector<T> gradient, bool satisfiedWolfe)
        {
            var numOps = _owner.NumOps;

            T curvature = gradientChange.DotProduct(step);
            if (!numOps.GreaterThan(curvature, numOps.FromDouble(1e-12)))
            {
                // Using a pair with non-positive curvature makes the estimate indefinite, so the
                // next direction could point uphill. Skipping loses information; using it loses
                // the property that makes the estimate usable at all.
                return;
            }

                        if (!_scaled)
            {
                // Nocedal and Wright equation 6.20. The identity has no idea of the problem's
                // scale, so the first step is wrong by whatever factor the units happen to be;
                // this rescales it using the only measurement available.
                T scale = numOps.Divide(curvature, gradientChange.DotProduct(gradientChange));

                for (int i = 0; i < _size; i++)
                {
                    for (int j = 0; j < _size; j++)
                    {
                        _inverse[i, j] = i == j ? scale : numOps.Zero;
                    }
                }

                _scaled = true;
            }

T rho = numOps.Divide(numOps.One, curvature);

            // (I - rho s y') H (I - rho y s') + rho s s', written out so no matrix is allocated
            // beyond the one being replaced.
            var updated = new T[_size, _size];

            for (int i = 0; i < _size; i++)
            {
                for (int j = 0; j < _size; j++)
                {
                    T total = numOps.Zero;

                    for (int k = 0; k < _size; k++)
                    {
                        T leftEntry = numOps.Subtract(
                            i == k ? numOps.One : numOps.Zero,
                            numOps.Multiply(rho, numOps.Multiply(step[i], gradientChange[k])));

                        T inner = numOps.Zero;
                        for (int m = 0; m < _size; m++)
                        {
                            T rightEntry = numOps.Subtract(
                                m == j ? numOps.One : numOps.Zero,
                                numOps.Multiply(rho, numOps.Multiply(gradientChange[m], step[j])));

                            inner = numOps.Add(inner, numOps.Multiply(_inverse[k, m], rightEntry));
                        }

                        total = numOps.Add(total, numOps.Multiply(leftEntry, inner));
                    }

                    updated[i, j] = numOps.Add(
                        total, numOps.Multiply(rho, numOps.Multiply(step[i], step[j])));
                }
            }

            _inverse = updated;
        }
    }

}
