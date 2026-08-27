using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Autodiff;
using Newtonsoft.Json;

using AiDotNet.Attributes;
using AiDotNet.Enums;

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
[ComponentType(ComponentType.Optimizer)]
[PipelineStage(PipelineStage.Training)]
public partial class ADMMOptimizer<T, TInput, TOutput> : GradientBasedOptimizerBase<T, TInput, TOutput>
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
    [AiDotNet.Attributes.Buffer]
    private Vector<T> _z;

    /// <summary>
    /// The dual variable in ADMM algorithm.
    /// </summary>
    [AiDotNet.Attributes.Buffer]
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
    /// Creates an ADMM optimizer for minimizing a plain function, with no model attached.
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
    public static ADMMOptimizer<T, TInput, TOutput> CreateForFunction(
        ADMMOptimizerOptions<T, TInput, TOutput>? options = null)
        => new(options);

    /// <summary>
    /// Backs <see cref="CreateForFunction"/>: the same setup with no model.
    /// </summary>
    private ADMMOptimizer(ADMMOptimizerOptions<T, TInput, TOutput>? options)
        : base(null, options ?? new())
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
    /// <para><b>DataLoader Integration:</b> This method uses the DataLoader API for epoch management.
    /// ADMM typically operates on the full dataset for the X update (which involves solving linear systems),
    /// but notifies the sampler of epoch starts using
    /// <see cref="GradientBasedOptimizerBase{T,TInput,TOutput}.NotifyEpochStart"/> for compatibility with
    /// curriculum learning and sampling strategies.
    /// </para>
    /// </remarks>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        ValidationHelper<T>.ValidateInputData(inputData);

        var currentSolution = InitializeWorkingSolution(inputData.XTrain);
        var parameters = InterfaceGuard.Parameterizable(currentSolution).GetParameters();
        _z = new Vector<T>(parameters.Length);
        _u = new Vector<T>(parameters.Length);

        var bestStepData = new OptimizationStepData<T, TInput, TOutput>();
        var previousStepData = new OptimizationStepData<T, TInput, TOutput>();

        InitializeAdaptiveParameters();

        for (int epoch = 0; epoch < _options.MaxIterations; epoch++)
        {
            NotifyEpochStart(epoch);
            _iteration++;

            // ADMM steps - operates on full dataset for linear system solving
            currentSolution = UpdateX(currentSolution, inputData.XTrain, inputData.YTrain);
            parameters = InterfaceGuard.Parameterizable(currentSolution).GetParameters();
            UpdateZ(parameters);
            UpdateU(parameters);

            var currentStepData = EvaluateSolution(currentSolution, inputData);
            UpdateBestSolution(currentStepData, ref bestStepData);

            UpdateAdaptiveParameters(currentStepData, previousStepData);

            if (UpdateIterationHistoryAndCheckEarlyStopping(epoch, bestStepData))
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

        return InterfaceGuard.Parameterizable(currentSolution).WithParameters(newCoefficients);
    }

    /// <summary>
    /// Updates the auxiliary variable z in the ADMM algorithm.
    /// </summary>
    /// <param name="x">The current primal variable.</param>
    /// <remarks>
    /// <para>
    /// Boyd et al. (2011): <c>z = argmin_z ( g(z) + (rho/2)·‖x - z + u‖² ) = prox_{g/rho}(x + u)</c>.
    /// The penalty parameter scales the STRENGTH of the proximal operator, not its argument. For an L1
    /// split that means soft-thresholding <c>x + u</c> at <c>Strength/rho</c>.
    /// </para>
    /// <para>
    /// This used to compute <c>Regularize((x + u)/rho)</c> — scaling the argument and leaving the
    /// threshold alone, which is a different function: for L1 it equals
    /// <c>(1/rho)·soft_threshold(x + u, Strength·rho)</c>, so both the threshold and the magnitude of z
    /// are wrong by factors of rho. The two coincide exactly at rho = 1, which is the default, which is
    /// why it went unnoticed.
    /// </para>
    /// <para>
    /// The scaling is applied by rebuilding the regularizer at <c>Strength/rho</c> rather than by adding a
    /// rho-aware method to every regularizer. A custom regularizer cannot be rebuilt that way — its
    /// strength may not be its only parameter — so it is applied as configured, and a caller using one
    /// with rho != 1 is choosing their own proximal scaling.
    /// </para>
    /// <para><b>For Beginners:</b> This step applies the regularization to the solution.
    /// It's like smoothing out the solution to prevent overfitting.
    /// </para>
    /// </remarks>
    private void UpdateZ(Vector<T> x)
    {
        var xPlusU = (Vector<T>)Engine.Add(x, _u);

        // L2 needs its proximal operator computed here rather than borrowed from the regularizer.
        //
        // The penalty is fixed by this library's own L2 gradient, which is grad + Strength*w
        // (L2Regularization.Regularize(gradient, coefficients)). That is the gradient of
        // g(w) = (Strength/2)*||w||^2 -- the same convention as PyTorch's weight_decay. ADMM's z-step is
        // prox_{g/rho}(v) = argmin_z g(z) + (rho/2)*||z-v||^2, so Strength*z + rho*(z-v) = 0, giving
        //
        //     z = v / (1 + Strength/rho)
        //
        // L2Regularization.Regularize(v) instead returns v*(1-Strength), which is only the first-order
        // expansion of that: 1/(1+s) = 1 - s + s^2 - ... So the two agree for small s and diverge as s
        // grows, and once s >= 1 the shrinkage crosses zero and FLIPS THE SIGN of every coordinate while
        // the prox stays positive and bounded. With Strength rescaled to Strength/rho that happens
        // whenever Strength >= Rho, so the z-step was wrong for any run with rho != 1.
        //
        // Fixed at this call site on purpose. Regularize(Vector) is the shrinkage API roughly twenty
        // regression models call, and ProximalGradientDescentOptimizer documents why redefining it is
        // not an option: "changing Regularize would change every model that regularizes." L1's
        // Regularize IS its own prox (soft-thresholding), so that path already agrees and is untouched;
        // ElasticNet and custom regularizers keep their existing behaviour.
        if (_regularization is L2Regularization<T, TInput, TOutput>)
        {
            double rho = _options.Rho;
            double strength = _regularization.GetOptions().Strength;
            double scaled = (rho > 0.0 && !double.IsInfinity(rho)) ? strength / rho : strength;
            _z = (Vector<T>)Engine.Multiply(xPlusU, NumOps.FromDouble(1.0 / (1.0 + scaled)));
            return;
        }

        _z = ProximalOperator.Regularize(xPlusU);
    }

    /// <summary>
    /// The configured regularizer rescaled to <c>Strength/rho</c>, which is the proximal operator ADMM's
    /// z-step actually calls for.
    /// </summary>
    /// <remarks>
    /// Rebuilt on each access rather than cached, because <c>Rho</c> and <c>Regularization</c> both come
    /// from the options object and can be replaced through <c>UpdateOptions</c> or a deserialize; a cached
    /// operator would keep applying the previous configuration's scaling.
    /// </remarks>
    private IRegularization<T, TInput, TOutput> ProximalOperator
    {
        get
        {
            double rho = _options.Rho;
            if (!(rho > 0.0) || double.IsInfinity(rho))
            {
                return _regularization;
            }

            // Copy the configured options and override only Strength, so Type and L1Ratio survive — a
            // fresh RegularizationOptions would reset them and make GetOptions() describe a different
            // regularizer than the one being applied.
            var rescaled = ProximalGradientDescentOptimizer<T, TInput, TOutput>.CloneWithStrength(
                _regularization.GetOptions(), _regularization.GetOptions().Strength / rho);
            return _regularization switch
            {
                L1Regularization<T, TInput, TOutput> => new L1Regularization<T, TInput, TOutput>(rescaled),
                L2Regularization<T, TInput, TOutput> => new L2Regularization<T, TInput, TOutput>(rescaled),
                _ => _regularization,
            };
        }
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
            var primalResidual = InterfaceGuard.Parameterizable(currentStepData.Solution).GetParameters().Subtract(_z);
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
    /// Updates parameters using GPU-accelerated ADMM.
    /// </summary>
    /// <remarks>
    /// ADMM (Alternating Direction Method of Multipliers) requires maintaining dual variables
    /// and performing alternating minimization steps.
    /// GPU implementation is not yet available due to the multi-step iterative nature
    /// of the algorithm.
    /// </remarks>
    public override void UpdateParametersGpu(IGpuBuffer parameters, IGpuBuffer gradients, int parameterCount, IDirectGpuBackend backend)
    {
        throw new NotSupportedException(
            "GPU-accelerated ADMM is not yet implemented. " +
            "ADMM requires alternating optimization steps and dual variable updates " +
            "which don't map well to single-pass GPU kernels. " +
            "Use CPU-based UpdateParameters or consider using Adam/AdamW for GPU-resident training.");
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

    /// <inheritdoc />
    public override void Step(TapeStepContext<T> context)
    {
        // Sparse-by-default: walk any embedding params whose gradient lives
        // only in the sparse list (Tensors stopped seeding dense alongside) and
        // materialise via ToDense into context.Gradients before GetFlatGradients
        // / Hessian assembly reads from the dict. No-op for non-embedding params
        // and for embedding params that already have a dense entry.
        SparseEmbeddingOptimizerHelpers.MaterializeSparseIntoGradientsDict(context, Engine);

        var updated = UpdateParameters(context.GetFlatParameters(), context.GetFlatGradients());
        context.SetFlatParameters(updated);
    }

    /// <summary>
    /// Applies one linearized-ADMM iteration to a flat parameter vector.
    /// </summary>
    /// <param name="parameters">The current parameters (the x block).</param>
    /// <param name="gradient">The gradient of the loss at those parameters.</param>
    /// <returns>The updated parameters.</returns>
    /// <remarks>
    /// <para>
    /// Without this override, <see cref="Step"/> resolved to
    /// <see cref="GradientBasedOptimizerBase{T, TInput, TOutput}"/>'s default <c>theta -= lr * g</c>. The
    /// splitting was never performed at all on the tape path: no z block, no dual variable, and the
    /// regularizer that is the whole reason to run ADMM was simply not applied. Training a neural network
    /// with ADMMOptimizer silently produced plain gradient descent.
    /// </para>
    /// <para>
    /// The three ADMM blocks, with the augmented-Lagrangian coupling that makes it ADMM rather than a
    /// gradient step next to an unrelated projection:
    /// </para>
    /// <code>
    /// x &lt;- x - lr * ( grad L(x) + rho * (x - z + u) )   // linearized x-update
    /// z &lt;- regularize( (x + u) / rho )                   // prox of the regularizer
    /// u &lt;- u + (x - z)                                   // scaled dual ascent
    /// </code>
    /// <para>
    /// <b>Deviation, stated explicitly.</b> <see cref="Optimize"/> solves the x-block in closed form,
    /// <c>(X^T X + rho I) x = X^T y + rho(z - u)</c>, which needs the design matrix X. The tape has no design
    /// matrix — it produces one gradient per step — so the x-block is instead LINEARIZED: a gradient step on
    /// the augmented Lagrangian. That is the standard variant for problems where the exact prox of the smooth
    /// term is unavailable (linearized / proximal-gradient ADMM, as in Parikh and Boyd's proximal-algorithms
    /// treatment), not an invention for this file. It converges to the same solution under the usual step-size
    /// condition, just not in one x-solve per iteration.
    /// </para>
    /// <para>
    /// The z and u blocks are exactly the ones <see cref="Optimize"/> uses — the same
    /// <c>UpdateZ</c>/<c>UpdateU</c> methods, not reimplementations — so the regularizer and dual update
    /// cannot drift between the two paths.
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

        // Both are non-nullable and start as Vector<T>.Empty(), so the length test alone covers the
        // first call as well as a parameter-count change.
        int n = parameters.Length;
        if (_z.Length != n) _z = Vector<T>.CreateDefault(n, NumOps.Zero);
        if (_u.Length != n) _u = Vector<T>.CreateDefault(n, NumOps.Zero);

        var rho = NumOps.FromDouble(_options.Rho);

        // Linearized x-update on the augmented Lagrangian:
        //   x <- x - lr * ( grad L(x) + rho * (x - z + u) )
        // The rho term is what couples x to the split variable; drop it and this degenerates into the
        // gradient step this override exists to replace.
        var coupling = (Vector<T>)Engine.Add(Engine.Subtract(parameters, _z), _u);
        var totalGradient = (Vector<T>)Engine.Add(gradient, Engine.Multiply(coupling, rho));
        var x = (Vector<T>)Engine.Subtract(parameters, Engine.Multiply(totalGradient, CurrentLearningRate));

        // Reuse Optimize()'s own z and u blocks so the two paths cannot disagree about them.
        UpdateZ(x);
        UpdateU(x);

        _iteration++;
        return x;
    }
}
