using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Autodiff;
using Newtonsoft.Json;

using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.Optimizers;

/// <summary>
/// Implements the Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) optimization algorithm.
/// </summary>
/// <remarks>
/// <para>
/// L-BFGS is a quasi-Newton method for solving unconstrained nonlinear optimization problems. It approximates the
/// Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm using a limited amount of computer memory, making it suitable 
/// for optimization problems with many variables.
/// </para>
/// <para><b>For Beginners:</b> 
/// L-BFGS is an advanced optimization algorithm that efficiently finds the minimum of a function, especially useful 
/// for problems with many variables. It uses information from previous iterations to make intelligent decisions 
/// about where to search next, while keeping memory usage low.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[ComponentType(ComponentType.Optimizer)]
[PipelineStage(PipelineStage.Training)]
public class LBFGSOptimizer<T, TInput, TOutput> : GradientBasedOptimizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// Options specific to the L-BFGS optimizer.
    /// </summary>
    private LBFGSOptimizerOptions<T, TInput, TOutput> _options;

    /// <summary>
    /// List of position (solution) differences used in the L-BFGS update.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This list stores the differences between consecutive solutions, which are used to approximate the inverse Hessian matrix.
    /// </para>
    /// <para><b>For Beginners:</b> 
    /// Think of this as the optimizer's memory of how the solution has changed over recent iterations.
    /// </para>
    /// </remarks>
    private List<Vector<T>> _s;

    /// <summary>
    /// List of gradient differences used in the L-BFGS update.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This list stores the differences between consecutive gradients, which are used along with the solution differences 
    /// to approximate the inverse Hessian matrix.
    /// </para>
    /// <para><b>For Beginners:</b> 
    /// This represents how the direction of steepest descent has changed over recent iterations.
    /// </para>
    /// </remarks>
    private List<Vector<T>> _y;

    /// <summary>
    /// The current iteration count of the optimization process.
    /// </summary>
    private int _iteration;

    /// <summary>
    /// Stores the previous parameters for computing position differences in UpdateParameters.
    /// </summary>
    private Vector<T>? _lbfgsPreviousParameters;

    /// <summary>
    /// Stores the previous gradient for computing gradient differences in UpdateParameters.
    /// </summary>
    private Vector<T>? _lbfgsPreviousGradient;

    /// <summary>
    /// The scaling γ = sᵀy / yᵀy of the most recently accepted correction pair. It defines the
    /// scaled-identity Hessian model B ≈ (1/γ)·I that Powell damping measures curvature against,
    /// and matches the initial scaling the two-loop recursion applies. Starts at 1, which is the
    /// conventional choice before any curvature information exists.
    /// </summary>
    private T _lbfgsInverseHessianScale;

    /// <summary>
    /// Reused scalar coefficients for the two-loop recursion.
    /// </summary>
    private T[] _twoLoopAlphas = Array.Empty<T>();

    /// <summary>
    /// Initializes a new instance of the LBFGSOptimizer class.
    /// </summary>
    /// <param name="model">The model to optimize.</param>
    /// <param name="options">Options for the L-BFGS optimizer. If null, default options are used.</param>
    /// <param name="engine">The computation engine (CPU or GPU) for vectorized operations.</param>
    public LBFGSOptimizer(
        IFullModel<T, TInput, TOutput> model,
        LBFGSOptimizerOptions<T, TInput, TOutput>? options = null,
        IEngine? engine = null)
        : base(model, options ?? new())
    {
        _options = options ?? new LBFGSOptimizerOptions<T, TInput, TOutput>();
        _s = new List<Vector<T>>();
        _y = new List<Vector<T>>();
        _lbfgsInverseHessianScale = NumOps.One;

        InitializeAdaptiveParameters();
    }

    /// <summary>
    /// Creates an L-BFGS optimizer for minimizing a plain function, with no model attached.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Use this overload with <see cref="Minimize(Vector{T}, Func{Vector{T}, ValueTuple{T, Vector{T}}}, int, T)"/>
    /// when you want to minimize a mathematical function directly rather than train a model.
    /// <see cref="Optimize"/> requires a model and is not available on an instance created this way.
    /// </para>
    /// <para><b>For Beginners:</b> The other constructor asks for a model because it is set up to
    /// tune that model against training data. If all you have is a formula you want to make as
    /// small as possible, there is no model to hand over — use this factory instead.
    /// </para>
    /// <para><b>Breaking change:</b> The former public options-only constructor was replaced by
    /// this named factory because passing <c>null</c> to <c>new LBFGSOptimizer(...)</c> was ambiguous with the
    /// model-based constructor.</para>
    /// </remarks>
    /// <param name="options">The L-BFGS-specific optimization options.</param>
    public static LBFGSOptimizer<T, TInput, TOutput> CreateForFunction(
        LBFGSOptimizerOptions<T, TInput, TOutput>? options = null)
        => new(options);

    private LBFGSOptimizer(LBFGSOptimizerOptions<T, TInput, TOutput>? options)
        : base(null, options ?? new())
    {
        _options = options ?? new LBFGSOptimizerOptions<T, TInput, TOutput>();
        _s = new List<Vector<T>>();
        _y = new List<Vector<T>>();
        _lbfgsInverseHessianScale = NumOps.One;

        InitializeAdaptiveParameters();
    }

    /// <summary>
    /// Initializes or resets the adaptive parameters used in the optimization process.
    /// </summary>
    protected override void InitializeAdaptiveParameters()
    {
        base.InitializeAdaptiveParameters();

        CurrentLearningRate = NumOps.FromDouble(_options.InitialLearningRate);
        _iteration = 0;
        _lbfgsPreviousParameters = null;
        _lbfgsPreviousGradient = null;

        // The curvature memory describes a specific objective surface at specific parameter values.
        // Carrying it into a fresh run would build search directions from correction pairs that
        // belong to the previous problem. Optimize() used to clear these itself, which left every
        // other reset path (Reset, a second Minimize call) reusing stale pairs.
        _s.Clear();
        _y.Clear();
        _lbfgsInverseHessianScale = NumOps.One;
    }

    /// <summary>
    /// Performs the main optimization process using the L-BFGS algorithm.
    /// </summary>
    /// <param name="inputData">The input data for the optimization process.</param>
    /// <returns>The result of the optimization process.</returns>
    /// <remarks>
    /// <para><b>DataLoader Integration:</b> This method uses the DataLoader API for epoch management.
    /// L-BFGS typically operates on the full dataset because it maintains a history of gradient and
    /// position differences that require consistent gradients between iterations. The method notifies
    /// the sampler of epoch starts using <see cref="GradientBasedOptimizerBase{T,TInput,TOutput}.NotifyEpochStart"/>
    /// for compatibility with curriculum learning and sampling strategies.
    /// </para>
    /// </remarks>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        ValidationHelper<T>.ValidateInputData(inputData);

        var currentSolution = InitializeWorkingSolution(inputData.XTrain);
        var bestStepData = new OptimizationStepData<T, TInput, TOutput>();
        var previousStepData = new OptimizationStepData<T, TInput, TOutput>();

        // InitializeAdaptiveParameters clears the curvature memory.
        InitializeAdaptiveParameters();

        Vector<T> previousGradient = Vector<T>.Empty();

        for (int epoch = 0; epoch < _options.MaxIterations; epoch++)
        {
            NotifyEpochStart(epoch);
            _iteration++;

            var gradient = CalculateGradient(currentSolution, inputData.XTrain, inputData.YTrain);
            var direction = CalculateDirection(gradient);
            var newSolution = UpdateSolution(currentSolution, direction, gradient, inputData);

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

            UpdateLBFGSMemory(InterfaceGuard.Parameterizable(currentSolution).GetParameters(), InterfaceGuard.Parameterizable(newSolution).GetParameters(), gradient, previousGradient);

            previousGradient = gradient;
            currentSolution = newSolution;
            previousStepData = currentStepData;
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    /// <summary>
    /// Calculates the search direction using the L-BFGS algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method computes the search direction using the L-BFGS two-loop recursion algorithm. It uses the stored 
    /// solution and gradient differences to approximate the inverse Hessian matrix.
    /// </para>
    /// <para><b>For Beginners:</b> 
    /// This method determines the best direction to move in the solution space, using information from previous iterations 
    /// to make a more informed decision than just following the steepest descent.
    /// </para>
    /// </remarks>
    /// <param name="gradient">The current gradient.</param>
    /// <returns>The calculated search direction.</returns>
    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// L-BFGS overrides the base implementation because it is not a fixed-step-rule optimizer: it
    /// proposes a search direction from its curvature memory and then performs a <b>line search</b>
    /// to decide how far along that direction to move. The base implementation drives
    /// <see cref="Step(TapeStepContext{T})"/>, which can only line-search when the context supports
    /// re-evaluation — and a context built from a plain <c>(f, ∇f)</c> closure cannot, because
    /// re-evaluation is defined in terms of a tensor forward/loss pair. This override does the line
    /// search directly against the caller's objective, while reusing the same two-loop recursion
    /// (<see cref="CalculateDirection"/>) and curvature memory
    /// (<see cref="UpdateLBFGSMemory"/>) as the model-training path.
    /// </para>
    /// <para>
    /// Steps are accepted on the Armijo sufficient-decrease condition, backtracking by
    /// <see cref="LBFGSOptimizerOptions{T, TInput, TOutput}.LineSearchContractionFactor"/> up to
    /// <see cref="LBFGSOptimizerOptions{T, TInput, TOutput}.LineSearchMaxSteps"/> times
    /// (Nocedal and Wright, "Numerical Optimization", Algorithms 3.1 and 7.4).
    /// </para>
    /// <para><b>For Beginners:</b> L-BFGS is usually the best choice on this list for a smooth
    /// function of a few dozen variables. It remembers how the slope changed over its last several
    /// steps and uses that to guess the shape of the surface, which lets it aim much better than
    /// plain gradient descent — and then it checks its guess by trying the step before committing
    /// to it.
    /// </para>
    /// </remarks>
    public override Vector<T> Minimize(
        Vector<T> initialParameters,
        Func<Vector<T>, (T objective, Vector<T> gradient)> objectiveAndGradient,
        int maxIterations,
        T tolerance,
        Func<Vector<T>, Vector<T>>? projection)
    {
        Guard.NotNull(initialParameters);
        Guard.NotNull(objectiveAndGradient);

        int parameterCount = initialParameters.Length;
        if (parameterCount == 0)
        {
            throw new ArgumentException(
                "Initial parameters must contain at least one element.",
                nameof(initialParameters));
        }

        if (maxIterations <= 0)
        {
            throw new ArgumentException(
                $"Maximum iterations must be positive, got {maxIterations}.",
                nameof(maxIterations));
        }

        // A fresh minimization must not inherit curvature pairs from a previous run.
        Reset();

        var current = ApplyProjection(projection, initialParameters.Clone(), parameterCount);
        var bestPoint = current.Clone();
        T bestObjective = NumOps.Zero;
        bool hasBestObjective = false;

        void Observe(Vector<T> point, T value)
        {
            double asDouble = NumOps.ToDouble(value);
            if (double.IsNaN(asDouble) || double.IsInfinity(asDouble))
            {
                return;
            }

            if (!hasBestObjective || NumOps.LessThan(value, bestObjective))
            {
                bestObjective = value;
                bestPoint = point.Clone();
                hasBestObjective = true;
            }
        }

        Vector<T>? previousPoint = null;
        Vector<T>? previousGradient = null;
        bool previousStepSatisfiedWolfe = false;

        // The accepted point's evaluation IS the next iteration's evaluation, so carrying it
        // forward makes the line search's final trial free rather than an extra call.
        T carriedObjective = NumOps.Zero;
        Vector<T>? carriedGradient = null;

        for (int iteration = 0; iteration < maxIterations; iteration++)
        {
            T objective;
            Vector<T> gradient;

            if (carriedGradient is not null)
            {
                objective = carriedObjective;
                gradient = carriedGradient;
            }
            else
            {
                (objective, gradient) = objectiveAndGradient(current);
            }

            Guard.NotNull(gradient);
            if (gradient.Length != parameterCount)
            {
                throw new ArgumentException(
                    $"Gradient length ({gradient.Length}) must match parameter length ({parameterCount}).",
                    nameof(objectiveAndGradient));
            }

            EnsureFiniteEvaluation(objective, gradient, iteration);
            Observe(current, objective);

            if (!NumOps.GreaterThan(InfinityNorm(gradient), tolerance))
            {
                break;
            }

            // Record the correction pair for the step just completed. Both endpoints come from
            // consecutive iterations, which is what the two-loop recursion assumes. A step that met
            // the strong Wolfe curvature condition already guarantees positive curvature, so
            // damping it would distort information already known to be sound.
            if (previousPoint is not null && previousGradient is not null)
            {
                UpdateLBFGSMemory(
                    previousPoint, current, gradient, previousGradient,
                    skipDamping: previousStepSatisfiedWolfe);
            }

            var direction = CalculateDirection(gradient);
            var directionalDerivative = gradient.DotProduct(direction);

            // The curvature guard in UpdateLBFGSMemory makes an ascent direction unlikely, but a
            // memory built from an ill-conditioned surface can still produce one. Fall back to
            // steepest descent rather than line-searching uphill, which would fail every trial and
            // waste the whole iteration.
            if (!NumOps.LessThan(directionalDerivative, NumOps.Zero))
            {
                direction = (Vector<T>)Engine.Multiply(gradient, NumOps.Negate(NumOps.One));
                directionalDerivative = NumOps.Negate(gradient.DotProduct(gradient));
            }

            var problem = new LineSearchProblem(
                current, direction, objective,
                objectiveAndGradient, projection, parameterCount,
                _options.ArmijoConstant, _options.WolfeCurvatureConstant,
                NumOps.ToDouble(objective), NumOps.ToDouble(directionalDerivative));

            LineSearchOutcome? outcome = _options.UseStrongWolfeLineSearch
                ? StrongWolfeLineSearch(problem, Observe)
                : null;

            outcome ??= BacktrackingLineSearch(problem, Observe);

            if (outcome is null)
            {
                // Neither search found a usable step. Take a deliberately tiny one along the descent
                // direction rather than stalling, so the curvature memory can refresh.
                var fallbackPoint = ApplyProjection(
                    projection,
                    (Vector<T>)Engine.Add(
                        current,
                        (Vector<T>)Engine.Multiply(
                            direction, NumOps.FromDouble(_options.LineSearchFallbackStep))),
                    parameterCount);

                var (fallbackObjective, fallbackGradient) = objectiveAndGradient(fallbackPoint);
                EnsureFiniteEvaluation(fallbackObjective, fallbackGradient, iteration);
                outcome = new LineSearchOutcome(
                    fallbackPoint, fallbackObjective, fallbackGradient, satisfiedWolfe: false);
            }

            Observe(outcome.Value.Point, outcome.Value.Objective);

            previousPoint = current;
            previousGradient = gradient;
            previousStepSatisfiedWolfe = outcome.Value.SatisfiedWolfe;

            current = outcome.Value.Point;
            carriedObjective = outcome.Value.Objective;
            carriedGradient = outcome.Value.Gradient;
        }

        return bestPoint;
    }

    /// <summary>
    /// Everything a line search along one direction needs, gathered so both phases of the strong
    /// Wolfe search can share it without a dozen parameters each.
    /// </summary>
    private sealed class LineSearchProblem
    {
        public LineSearchProblem(
            Vector<T> current,
            Vector<T> direction,
            T objectiveAtCurrent,
            Func<Vector<T>, (T objective, Vector<T> gradient)> objectiveAndGradient,
            Func<Vector<T>, Vector<T>>? projection,
            int parameterCount,
            double sufficientDecreaseConstant,
            double curvatureConstant,
            double valueAtZero,
            double slopeAtZero)
        {
            Current = current;
            Direction = direction;
            ObjectiveAtCurrent = objectiveAtCurrent;
            ObjectiveAndGradient = objectiveAndGradient;
            Projection = projection;
            ParameterCount = parameterCount;
            SufficientDecreaseConstant = sufficientDecreaseConstant;
            CurvatureConstant = curvatureConstant;
            ValueAtZero = valueAtZero;
            SlopeAtZero = slopeAtZero;
        }

        /// <summary>The point the search starts from.</summary>
        public Vector<T> Current { get; }

        /// <summary>The search direction, which must point downhill.</summary>
        public Vector<T> Direction { get; }

        /// <summary>The objective at <see cref="Current"/>.</summary>
        public T ObjectiveAtCurrent { get; }

        /// <summary>The caller's objective.</summary>
        public Func<Vector<T>, (T objective, Vector<T> gradient)> ObjectiveAndGradient { get; }

        /// <summary>Maps a trial point back into the feasible set, when there is one.</summary>
        public Func<Vector<T>, Vector<T>>? Projection { get; }

        /// <summary>How many variables the problem has.</summary>
        public int ParameterCount { get; }

        /// <summary>The constant c1 in the sufficient-decrease condition.</summary>
        public double SufficientDecreaseConstant { get; }

        /// <summary>The constant c2 in the Wolfe curvature condition.</summary>
        public double CurvatureConstant { get; }

        /// <summary>The objective where the search starts, as a plain double.</summary>
        public double ValueAtZero { get; }

        /// <summary>The slope along the direction at the start. Negative for a descent direction.</summary>
        public double SlopeAtZero { get; }

        /// <summary>Whether a trial step meets the sufficient-decrease condition.</summary>
        public bool HasSufficientDecrease(double step, double value)
            => value <= ValueAtZero + SufficientDecreaseConstant * step * SlopeAtZero;

        /// <summary>Whether a trial step meets the strong Wolfe curvature condition.</summary>
        public bool HasFlatEnoughSlope(double slope)
            => Math.Abs(slope) <= -CurvatureConstant * SlopeAtZero;
    }

    /// <summary>One point along the search direction, with everything measured there.</summary>
    private readonly struct LineSearchTrial
    {
        public LineSearchTrial(Vector<T> point, T objective, Vector<T> gradient, double value, double slope)
        {
            Point = point;
            Objective = objective;
            Gradient = gradient;
            Value = value;
            Slope = slope;
        }

        /// <summary>The trial point.</summary>
        public Vector<T> Point { get; }

        /// <summary>The objective there.</summary>
        public T Objective { get; }

        /// <summary>The gradient there.</summary>
        public Vector<T> Gradient { get; }

        /// <summary>The objective as a plain double.</summary>
        public double Value { get; }

        /// <summary>The slope along the search direction here.</summary>
        public double Slope { get; }

        /// <summary>Whether the objective and slope here are real numbers.</summary>
        public bool IsUsable
            => !double.IsNaN(Value) && !double.IsInfinity(Value) && !double.IsNaN(Slope);
    }

    /// <summary>The step a line search settled on.</summary>
    private readonly struct LineSearchOutcome
    {
        public LineSearchOutcome(Vector<T> point, T objective, Vector<T>? gradient, bool satisfiedWolfe)
        {
            Point = point;
            Objective = objective;
            Gradient = gradient;
            SatisfiedWolfe = satisfiedWolfe;
        }

        /// <summary>Where the step landed.</summary>
        public Vector<T> Point { get; }

        /// <summary>The objective there.</summary>
        public T Objective { get; }

        /// <summary>The gradient there, when the search happened to compute it.</summary>
        public Vector<T>? Gradient { get; }

        /// <summary>
        /// Whether the step met the strong Wolfe conditions, which is what makes the resulting
        /// correction pair sound without repair.
        /// </summary>
        public bool SatisfiedWolfe { get; }
    }

    /// <summary>Evaluates the objective a given distance along the search direction.</summary>
    /// <param name="problem">The line search being run.</param>
    /// <param name="step">How far along the direction to go.</param>
    /// <returns>The trial point and everything measured there.</returns>
    private LineSearchTrial EvaluateAlong(LineSearchProblem problem, double step)
    {
        var point = ApplyProjection(
            problem.Projection,
            (Vector<T>)Engine.Add(
                problem.Current,
                (Vector<T>)Engine.Multiply(problem.Direction, NumOps.FromDouble(step))),
            problem.ParameterCount);

        var (objective, gradient) = problem.ObjectiveAndGradient(point);
        Guard.NotNull(gradient);

        double value = NumOps.ToDouble(objective);
        double slope = gradient.Length == problem.Direction.Length
            ? NumOps.ToDouble(gradient.DotProduct(problem.Direction))
            : double.NaN;

        return new LineSearchTrial(point, objective, gradient, value, slope);
    }

    /// <summary>
    /// Finds a step length satisfying the strong Wolfe conditions, or reports that it could not.
    /// </summary>
    /// <param name="problem">The line search being run.</param>
    /// <param name="observe">Called with every trial point, so the caller can keep the best seen.</param>
    /// <returns>The accepted step, or <c>null</c> when none was found within the budget.</returns>
    /// <remarks>
    /// <para>
    /// This is the bracketing phase of Nocedal and Wright's Algorithm 3.5: start at a unit step —
    /// the right first guess for a quasi-Newton direction — and double until a step is found to be
    /// too long, at which point <see cref="ZoomLineSearch"/> refines the bracket.
    /// </para>
    /// <para>
    /// The curvature half of the conditions is the part that matters to L-BFGS. It guarantees the
    /// resulting correction pair has positive curvature, which is the property the two-loop
    /// recursion depends on and which sufficient decrease alone does not provide.
    /// </para>
    /// </remarks>
    private LineSearchOutcome? StrongWolfeLineSearch(
        LineSearchProblem problem, Action<Vector<T>, T> observe)
    {
        // A direction that does not point downhill has no acceptable step along it.
        if (!(problem.SlopeAtZero < 0.0)) return null;

        // The two conditions are only satisfiable together when c1 < c2 < 1.
        if (!(problem.SufficientDecreaseConstant < problem.CurvatureConstant)) return null;

        double maximumStep = _options.LineSearchMaxStep;
        if (!(maximumStep > 0.0)) return null;

        double previousStep = 0.0;
        double previousValue = problem.ValueAtZero;
        LineSearchTrial? previousTrial = null;

        double step = Math.Min(1.0, maximumStep);

        for (int attempt = 0; attempt < _options.LineSearchMaxSteps; attempt++)
        {
            var trial = EvaluateAlong(problem, step);
            observe(trial.Point, trial.Objective);

            if (!trial.IsUsable)
            {
                // Overflowed or left the domain, so any acceptable step is shorter than this one.
                step = 0.5 * (previousStep + step);
                if (!(step > previousStep)) return null;
                continue;
            }

            bool overshot = !problem.HasSufficientDecrease(step, trial.Value)
                || (previousTrial is not null && trial.Value >= previousValue);

            if (overshot)
            {
                return ZoomLineSearch(problem, previousStep, previousValue, previousTrial, step, observe);
            }

            if (problem.HasFlatEnoughSlope(trial.Slope))
            {
                return new LineSearchOutcome(trial.Point, trial.Objective, trial.Gradient, true);
            }

            if (trial.Slope >= 0.0)
            {
                // The slope has turned uphill, so the acceptable step is back towards the last one.
                return ZoomLineSearch(problem, step, trial.Value, trial, previousStep, observe);
            }

            if (step >= maximumStep)
            {
                // Still descending at the largest step allowed. Take it, and report that the
                // curvature condition was never established so the pair still gets damped.
                return new LineSearchOutcome(trial.Point, trial.Objective, trial.Gradient, false);
            }

            previousStep = step;
            previousValue = trial.Value;
            previousTrial = trial;
            step = Math.Min(2.0 * step, maximumStep);
        }

        return null;
    }

    /// <summary>
    /// Narrows a bracket known to contain a step satisfying the strong Wolfe conditions.
    /// </summary>
    /// <param name="problem">The line search being run.</param>
    /// <param name="low">The bracket end with the lower objective.</param>
    /// <param name="valueAtLow">The objective at <paramref name="low"/>.</param>
    /// <param name="trialAtLow">The measurement at <paramref name="low"/>, when there is one.</param>
    /// <param name="high">The other bracket end. It may be the smaller step of the two.</param>
    /// <param name="observe">Called with every trial point, so the caller can keep the best seen.</param>
    /// <returns>The accepted step, or <c>null</c> when the bracket produced nothing usable.</returns>
    /// <remarks>
    /// Nocedal and Wright's Algorithm 3.6, by bisection. Twenty bisections narrow the bracket by a
    /// factor of a million, which is far more than a quasi-Newton step normally needs.
    /// </remarks>
    private LineSearchOutcome? ZoomLineSearch(
        LineSearchProblem problem,
        double low,
        double valueAtLow,
        LineSearchTrial? trialAtLow,
        double high,
        Action<Vector<T>, T> observe)
    {
        for (int attempt = 0; attempt < _options.LineSearchMaxZoomSteps; attempt++)
        {
            double step = 0.5 * (low + high);
            if (!(step > 0.0) || step == low || step == high)
            {
                break;
            }

            var trial = EvaluateAlong(problem, step);
            observe(trial.Point, trial.Objective);

            if (!trial.IsUsable)
            {
                high = step;
                continue;
            }

            if (!problem.HasSufficientDecrease(step, trial.Value) || trial.Value >= valueAtLow)
            {
                high = step;
                continue;
            }

            if (problem.HasFlatEnoughSlope(trial.Slope))
            {
                return new LineSearchOutcome(trial.Point, trial.Objective, trial.Gradient, true);
            }

            if (trial.Slope * (high - low) >= 0.0)
            {
                high = low;
            }

            low = step;
            valueAtLow = trial.Value;
            trialAtLow = trial;
        }

        // The budget ran out. The best bracketed point is still a genuine improvement, so take it,
        // reporting that the curvature condition was never established.
        return trialAtLow is not null && trialAtLow.Value.Value < problem.ValueAtZero
            ? new LineSearchOutcome(
                trialAtLow.Value.Point, trialAtLow.Value.Objective, trialAtLow.Value.Gradient, false)
            : null;
    }

    /// <summary>
    /// Backtracking search on the sufficient-decrease condition alone.
    /// </summary>
    /// <param name="problem">The line search being run.</param>
    /// <param name="observe">Called with every trial point, so the caller can keep the best seen.</param>
    /// <returns>The accepted step, or <c>null</c> when every trial was rejected.</returns>
    /// <remarks>
    /// Nocedal and Wright's Algorithm 3.1. This is what runs when
    /// <see cref="LBFGSOptimizerOptions{T, TInput, TOutput}.UseStrongWolfeLineSearch"/> is turned
    /// off, and what catches the cases where the Wolfe search cannot bracket an acceptable step.
    /// </remarks>
    private LineSearchOutcome? BacktrackingLineSearch(
        LineSearchProblem problem, Action<Vector<T>, T> observe)
    {
        if (!(problem.SlopeAtZero < 0.0)) return null;

        double step = 1.0;
        double contraction = _options.LineSearchContractionFactor;

        for (int attempt = 0; attempt < _options.LineSearchMaxSteps; attempt++)
        {
            var trial = EvaluateAlong(problem, step);
            observe(trial.Point, trial.Objective);

            if (!trial.IsUsable)
            {
                step *= contraction;
                continue;
            }

            if (problem.HasSufficientDecrease(step, trial.Value))
            {
                return new LineSearchOutcome(
                    trial.Point, trial.Objective, trial.Gradient,
                    satisfiedWolfe: problem.HasFlatEnoughSlope(trial.Slope));
            }

            step *= contraction;
        }

        return null;
    }

    private void EnsureFiniteEvaluation(T objective, Vector<T> gradient, int iteration)
    {
        double objectiveValue = NumOps.ToDouble(objective);
        if (double.IsNaN(objectiveValue) || double.IsInfinity(objectiveValue))
        {
            throw new ArithmeticException($"Objective became non-finite at L-BFGS iteration {iteration}.");
        }

        for (int i = 0; i < gradient.Length; i++)
        {
            double value = NumOps.ToDouble(gradient[i]);
            if (double.IsNaN(value) || double.IsInfinity(value))
            {
                throw new ArithmeticException(
                    $"Gradient component {i} became non-finite at L-BFGS iteration {iteration}.");
            }
        }
    }

    /// <summary>
    /// Returns the infinity norm (largest absolute component) of a vector.
    /// </summary>
    private T InfinityNorm(Vector<T> vector)
    {
        var absolute = Engine.TensorAbs(Tensor<T>.FromVector(vector));
        return Engine.ReduceMax(absolute, [0], keepDims: false)[0];
    }

    /// <summary>
    /// Applies an optional feasible-set projection, validating that it preserves the vector length.
    /// </summary>
    private static Vector<T> ApplyProjection(
        Func<Vector<T>, Vector<T>>? projection,
        Vector<T> point,
        int parameterCount)
    {
        if (projection is null)
        {
            return point;
        }

        var projected = projection(point);
        Guard.NotNull(projected);
        if (projected.Length != parameterCount)
        {
            throw new ArgumentException(
                $"Projection returned a vector of length {projected.Length}, " +
                $"but the parameter vector has length {parameterCount}.",
                nameof(projection));
        }

        return projected;
    }

    private Vector<T> CalculateDirection(Vector<T> gradient)
    {
        // === Partially Vectorized L-BFGS Two-Loop Recursion using IEngine (Phase B: US-GPU-015) ===

        if (_s.Count == 0 || _y.Count == 0)
        {
            // First iteration: direction = -gradient
            return CreateNegativeGradient(gradient);
        }

        var q = new Vector<T>(gradient);
        if (_twoLoopAlphas.Length < _s.Count)
        {
            _twoLoopAlphas = new T[_s.Count];
        }

        // First loop (backward)
        for (int i = _s.Count - 1; i >= 0; i--)
        {
            _twoLoopAlphas[i] = NumOps.Divide(_s[i].DotProduct(q), _y[i].DotProduct(_s[i]));
            var qSpan = q.AsWritableSpan();
            var ySpan = _y[i].AsSpan();
            T alpha = _twoLoopAlphas[i];
            for (int j = 0; j < qSpan.Length; j++)
            {
                qSpan[j] = NumOps.Subtract(qSpan[j], NumOps.Multiply(alpha, ySpan[j]));
            }
        }

        var gamma = NumOps.Divide(_s[_s.Count - 1].DotProduct(_y[_s.Count - 1]), _y[_s.Count - 1].DotProduct(_y[_s.Count - 1]));
        // Vectorized: z = gamma * q
        var z = q;
        var zSpan = z.AsWritableSpan();
        for (int j = 0; j < zSpan.Length; j++)
        {
            zSpan[j] = NumOps.Multiply(zSpan[j], gamma);
        }

        // Second loop (forward)
        for (int i = 0; i < _s.Count; i++)
        {
            var beta = NumOps.Divide(_y[i].DotProduct(z), _y[i].DotProduct(_s[i]));
            var alphaMinusBeta = NumOps.Subtract(_twoLoopAlphas[i], beta);
            var sSpan = _s[i].AsSpan();
            for (int j = 0; j < zSpan.Length; j++)
            {
                zSpan[j] = NumOps.Add(zSpan[j], NumOps.Multiply(alphaMinusBeta, sSpan[j]));
            }
        }

        // Vectorized negation
        for (int j = 0; j < zSpan.Length; j++)
        {
            zSpan[j] = NumOps.Negate(zSpan[j]);
        }
        return z;
    }

    private Vector<T> CreateNegativeGradient(Vector<T> gradient)
    {
        var direction = new Vector<T>(gradient.Length, skipZeroInit: true);
        var gradientSpan = gradient.AsSpan();
        var directionSpan = direction.AsWritableSpan();
        for (int i = 0; i < directionSpan.Length; i++)
        {
            directionSpan[i] = NumOps.Negate(gradientSpan[i]);
        }
        return direction;
    }

    /// <summary>
    /// Applies Powell's damped update so the correction pair satisfies the curvature condition.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Given the step <c>s</c> and gradient change <c>y</c>, this returns
    /// <c>r = θ·y + (1 − θ)·B·s</c> where <c>θ</c> is the largest value in <c>(0, 1]</c> for which
    /// <c>sᵀr ≥ factor · sᵀBs</c>. When the pair already satisfies the condition, <c>θ = 1</c> and
    /// <c>y</c> is returned unchanged, so damping costs nothing on well-behaved problems.
    /// </para>
    /// <para>
    /// L-BFGS never forms the Hessian approximation <c>B</c> explicitly, so this uses the same
    /// scaled-identity model the two-loop recursion itself starts from,
    /// <c>B ≈ (1/γ)·I</c> with <c>γ = sᵀy / yᵀy</c> taken from the most recently accepted pair
    /// (1 before any pair exists). That is the standard approximation for damping a limited-memory
    /// method, and it is exact for the first update.
    /// </para>
    /// </remarks>
    /// <param name="s">The step taken, <c>x_{k+1} − x_k</c>.</param>
    /// <param name="y">The gradient change, <c>∇f(x_{k+1}) − ∇f(x_k)</c>.</param>
    /// <returns>The damped gradient change to store, or <paramref name="y"/> when no damping is needed.</returns>
    private Vector<T> ApplyPowellDamping(Vector<T> s, Vector<T> y)
    {
        var dampingFactor = NumOps.FromDouble(_options.PowellDampingFactor);
        if (!NumOps.GreaterThan(dampingFactor, NumOps.Zero))
        {
            return y;
        }

        // sᵀBs under the B ≈ (1/γ)·I model.
        T sDotS = s.DotProduct(s);
        if (!NumOps.GreaterThan(sDotS, NumOps.Zero))
        {
            // A zero step carries no curvature information; leave it to the caller's threshold.
            return y;
        }

        T sBs = NumOps.Divide(sDotS, _lbfgsInverseHessianScale);
        T sDotY = s.DotProduct(y);
        T required = NumOps.Multiply(dampingFactor, sBs);

        if (!NumOps.LessThan(sDotY, required))
        {
            return y;
        }

        // θ = (1 − factor)·sᵀBs / (sᵀBs − sᵀy). The denominator is positive here because
        // sᵀy < factor·sᵀBs ≤ sᵀBs for factor in (0, 1].
        T denominator = NumOps.Subtract(sBs, sDotY);
        if (!NumOps.GreaterThan(denominator, NumOps.Zero))
        {
            return y;
        }

        T theta = NumOps.Divide(
            NumOps.Multiply(NumOps.Subtract(NumOps.One, dampingFactor), sBs), denominator);

        // r = θ·y + (1 − θ)·B·s, with B·s = s / γ.
        var scaledStep = (Vector<T>)Engine.Multiply(
            s, NumOps.Divide(NumOps.One, _lbfgsInverseHessianScale));

        return (Vector<T>)Engine.Add(
            (Vector<T>)Engine.Multiply(y, theta),
            (Vector<T>)Engine.Multiply(scaledStep, NumOps.Subtract(NumOps.One, theta)));
    }

    /// <summary>
    /// Updates the L-BFGS memory with the latest step information.
    /// </summary>
    /// <param name="oldSolution">The previous solution vector.</param>
    /// <param name="newSolution">The current solution vector.</param>
    /// <param name="gradient">The current gradient.</param>
    /// <param name="previousGradient">The previous gradient.</param>
    /// <param name="skipDamping">
    /// When true the pair is stored exactly as measured. Set this only when the step is known to
    /// have satisfied the Wolfe curvature condition, which already guarantees positive curvature.
    /// </param>
    private void UpdateLBFGSMemory(Vector<T> oldSolution, Vector<T> newSolution, Vector<T> gradient, Vector<T> previousGradient, bool skipDamping = false)
    {
        // === Vectorized Memory Update using IEngine (Phase B: US-GPU-015) ===
        // s = new_solution - old_solution
        // y = current_gradient - previous_gradient

        // Skip first iteration when previousGradient is empty
        if (previousGradient.Length == 0)
        {
            return;
        }

        var s = (Vector<T>)Engine.Subtract(newSolution, oldSolution);
        var y = (Vector<T>)Engine.Subtract(gradient, previousGradient);

        // Curvature condition: only pairs with s·y > 0 keep the implied inverse-Hessian positive
        // definite. Storing a pair that violates it lets CalculateDirection return an ascent
        // direction, and a near-zero s·y makes its divisions (alpha, gamma, beta) blow up — the
        // division by s·y at the top of the two-loop recursion is unguarded.
        // See Nocedal and Wright, "Numerical Optimization", section 6.1.
        //
        // Simply DISCARDING violating pairs is wrong for a nonconvex objective, where the condition
        // fails routinely: the memory starves and L-BFGS silently degrades into steepest descent.
        // Measured on this repo's NOTEARS suites (nonconvex augmented-Lagrangian subproblems),
        // discard-only raised failures across the Linear/Nonlinear/LowRank families from 11 to 17.
        // Powell's damped update repairs the pair instead of dropping it, blending y toward Bs until
        // the curvature condition holds by construction (Powell, 1978; Nocedal and Wright,
        // Procedure 18.2).
        // Damping repairs a pair that would otherwise be unusable. A Wolfe step needs no repair,
        // and repairing it anyway costs the method its superlinear convergence.
        if (!skipDamping)
        {
            y = ApplyPowellDamping(s, y);
        }

        if (!NumOps.GreaterThan(s.DotProduct(y), NumOps.FromDouble(_options.MinimumCurvature)))
        {
            return;
        }

        // Remember the scaling implied by the accepted pair; it is the B0 = (1/gamma)·I model that
        // the next damping step measures curvature against.
        _lbfgsInverseHessianScale = NumOps.Divide(s.DotProduct(y), y.DotProduct(y));

        if (_s.Count >= _options.MemorySize)
        {
            _s.RemoveAt(0);
            _y.RemoveAt(0);
        }

        _s.Add(s);
        _y.Add(y);
    }

    /// <summary>
    /// Updates the current solution based on the calculated direction.
    /// </summary>
    /// <param name="currentSolution">The current solution.</param>
    /// <param name="direction">The search direction.</param>
    /// <param name="gradient">The current gradient.</param>
    /// <param name="inputData">The input data for the optimization process.</param>
    /// <returns>The updated solution.</returns>
    private IFullModel<T, TInput, TOutput> UpdateSolution(IFullModel<T, TInput, TOutput> currentSolution, Vector<T> direction, Vector<T> gradient, OptimizationInputData<T, TInput, TOutput> inputData)
    {
        var step = LineSearch(currentSolution, direction, gradient, inputData);
        var scaledDirection = direction.Transform(x => NumOps.Multiply(x, step));
        var newCoefficients = InterfaceGuard.Parameterizable(currentSolution).GetParameters().Add(scaledDirection);

        return InterfaceGuard.Parameterizable(currentSolution).WithParameters(newCoefficients);
    }

    /// <summary>
    /// Updates the adaptive parameters of the optimizer based on the current and previous optimization steps.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method adjusts the learning rate based on the performance of the current step compared to the previous step.
    /// If the adaptive learning rate option is enabled, it increases or decreases the learning rate accordingly.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This method helps the optimizer learn more efficiently by adjusting how big its steps are.
    /// If the current step improved the solution, it takes slightly bigger steps.
    /// If not, it takes smaller steps to be more careful.
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

            CurrentLearningRate = MathHelper.Clamp(CurrentLearningRate,
                NumOps.FromDouble(_options.MinLearningRate),
                NumOps.FromDouble(_options.MaxLearningRate));
        }
    }

    /// <summary>
    /// Updates parameters using the L-BFGS algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method implements the L-BFGS two-loop recursion algorithm for computing the search direction.
    /// It maintains internal state (previous parameters and gradients) to build up the L-BFGS memory
    /// across successive calls.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// Unlike simple gradient descent that just follows the steepest direction, L-BFGS uses information
    /// from previous steps to approximate the curvature of the function being optimized. This typically
    /// leads to faster convergence, especially for problems with many variables.
    /// </para>
    /// </remarks>
    /// <param name="parameters">The current parameter vector to update.</param>
    /// <param name="gradient">The gradient of the loss function with respect to the parameters.</param>
    /// <returns>The updated parameter vector.</returns>
    public override Vector<T> UpdateParameters(Vector<T> parameters, Vector<T> gradient)
    {
        if (parameters.Length != gradient.Length)
        {
            throw new ArgumentException(
                $"Parameter vector length ({parameters.Length}) must match gradient vector length ({gradient.Length}).",
                nameof(gradient));
        }

        if ((_lbfgsPreviousParameters is not null && _lbfgsPreviousParameters.Length != parameters.Length)
            || (_lbfgsPreviousGradient is not null && _lbfgsPreviousGradient.Length != parameters.Length)
            || (_s.Count > 0 && _s[0].Length != parameters.Length)
            || (_y.Count > 0 && _y[0].Length != parameters.Length))
        {
            _s.Clear();
            _y.Clear();
            _lbfgsPreviousParameters = null;
            _lbfgsPreviousGradient = null;
            _iteration = 0;
        }

        _iteration++;

        // Update L-BFGS memory with the difference between current and previous gradients/parameters
        if (_lbfgsPreviousParameters is not null && _lbfgsPreviousGradient is not null)
        {
            // Only add to memory if curvature condition is satisfied (s^T y > 0)
            // L-BFGS requires positive curvature to maintain positive-definite Hessian approximation
            T sDotY = NumOps.Zero;
            var currentParameterSpan = parameters.AsSpan();
            var currentGradientSpan = gradient.AsSpan();
            var previousParameterSpan = _lbfgsPreviousParameters.AsSpan();
            var previousGradientSpan = _lbfgsPreviousGradient.AsSpan();
            for (int i = 0; i < currentParameterSpan.Length; i++)
            {
                T sValue = NumOps.Subtract(currentParameterSpan[i], previousParameterSpan[i]);
                T yValue = NumOps.Subtract(currentGradientSpan[i], previousGradientSpan[i]);
                sDotY = NumOps.Add(sDotY, NumOps.Multiply(sValue, yValue));
            }

            if (NumOps.GreaterThan(sDotY, NumOps.FromDouble(1e-10)))
            {
                Vector<T> s;
                Vector<T> y;
                if (_s.Count >= _options.MemorySize)
                {
                    s = _s[0];
                    y = _y[0];
                    _s.RemoveAt(0);
                    _y.RemoveAt(0);
                }
                else
                {
                    s = new Vector<T>(parameters.Length, skipZeroInit: true);
                    y = new Vector<T>(parameters.Length, skipZeroInit: true);
                }

                var sSpan = s.AsWritableSpan();
                var ySpan = y.AsWritableSpan();
                for (int i = 0; i < sSpan.Length; i++)
                {
                    sSpan[i] = NumOps.Subtract(currentParameterSpan[i], previousParameterSpan[i]);
                    ySpan[i] = NumOps.Subtract(currentGradientSpan[i], previousGradientSpan[i]);
                }

                _s.Add(s);
                _y.Add(y);
            }
        }

        // Calculate the L-BFGS search direction using two-loop recursion
        var direction = CalculateDirection(gradient);

        // Transform the direction allocation into the returned parameter vector.
        var directionSpan = direction.AsWritableSpan();
        var parameterSpan = parameters.AsSpan();
        for (int i = 0; i < directionSpan.Length; i++)
        {
            directionSpan[i] = NumOps.Add(
                parameterSpan[i],
                NumOps.Multiply(directionSpan[i], CurrentLearningRate));
        }

        // Store current parameters and gradient for next iteration
        if (_lbfgsPreviousParameters is null || _lbfgsPreviousParameters.Length != parameters.Length)
        {
            _lbfgsPreviousParameters = new Vector<T>(parameters.Length, skipZeroInit: true);
            _lbfgsPreviousGradient = new Vector<T>(parameters.Length, skipZeroInit: true);
        }
        parameterSpan.CopyTo(_lbfgsPreviousParameters.AsWritableSpan());
        gradient.AsSpan().CopyTo(_lbfgsPreviousGradient!.AsWritableSpan());

        return direction;
    }

    /// <summary>
    /// Updates the optimizer's options with new settings.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method updates the optimizer's configuration with new options. It ensures that only valid
    /// LBFGSOptimizerOptions are applied to this optimizer.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like changing the settings on the optimizer. It makes sure you're using the right kind of settings
    /// for this specific type of optimizer.
    /// </para>
    /// </remarks>
    /// <param name="options">The new options to apply to the optimizer.</param>
    /// <exception cref="ArgumentException">Thrown when the provided options are not of type LBFGSOptimizerOptions.</exception>
    protected override void UpdateOptions(OptimizationAlgorithmOptions<T, TInput, TOutput> options)
    {
        if (options is LBFGSOptimizerOptions<T, TInput, TOutput> lbfgsOptions)
        {
            _options = lbfgsOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected LBFGSOptimizerOptions.");
        }
    }

    /// <summary>
    /// Retrieves the current options of the optimizer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method returns the current configuration options of the L-BFGS optimizer.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This lets you see what settings the optimizer is currently using.
    /// </para>
    /// </remarks>
    /// <returns>The current options of the optimizer.</returns>
    public override OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions()
    {
        return _options;
    }

    /// <summary>
    /// Updates parameters using GPU-accelerated L-BFGS.
    /// </summary>
    /// <remarks>
    /// L-BFGS is a limited-memory quasi-Newton method that maintains history of past gradients.
    /// GPU implementation is not yet available due to the complexity of two-loop recursion
    /// and history management across GPU memory.
    /// </remarks>
    public override void UpdateParametersGpu(IGpuBuffer parameters, IGpuBuffer gradients, int parameterCount, IDirectGpuBackend backend)
    {
        throw new NotSupportedException(
            "GPU-accelerated L-BFGS is not yet implemented. L-BFGS requires maintaining gradient history " +
            "and performing two-loop recursion which is complex to implement efficiently on GPU. " +
            "Use CPU-based UpdateParameters or consider using Adam/AdamW for GPU-resident training.");
    }

    /// <summary>
    /// Serializes the optimizer's state into a byte array.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method converts the current state of the optimizer, including its options and internal memory,
    /// into a byte array. This allows the optimizer's state to be saved or transmitted.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like taking a snapshot of the optimizer's current state so it can be saved or sent somewhere else.
    /// It includes all the important information about what the optimizer has learned so far.
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
            writer.Write(NumOps.ToDouble(_lbfgsInverseHessianScale));
            writer.Write(_s.Count);
            foreach (var vector in _s)
            {
                byte[] vectorData = vector.Serialize();
                writer.Write(vectorData.Length);
                writer.Write(vectorData);
            }
            writer.Write(_y.Count);
            foreach (var vector in _y)
            {
                byte[] vectorData = vector.Serialize();
                writer.Write(vectorData.Length);
                writer.Write(vectorData);
            }

            return ms.ToArray();
        }
    }

    /// <summary>
    /// Deserializes a byte array to restore the optimizer's state.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method takes a byte array (previously created by the Serialize method) and uses it to restore
    /// the optimizer's state, including its options and internal memory.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like loading a saved snapshot of the optimizer's state. It rebuilds the optimizer's memory
    /// and settings from the saved data, allowing it to continue from where it left off.
    /// </para>
    /// </remarks>
    /// <param name="data">The byte array containing the serialized optimizer state.</param>
    /// <exception cref="InvalidOperationException">Thrown when deserialization of optimizer options fails.</exception>
    public override void Deserialize(byte[] data)
    {
        using (MemoryStream ms = new MemoryStream(data))
        using (BinaryReader reader = new BinaryReader(ms))
        {
            int baseDataLength = reader.ReadInt32();
            byte[] baseData = reader.ReadBytes(baseDataLength);
            base.Deserialize(baseData);

            string optionsJson = reader.ReadString();
            _options = JsonConvert.DeserializeObject<LBFGSOptimizerOptions<T, TInput, TOutput>>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");

            _iteration = reader.ReadInt32();
            _lbfgsInverseHessianScale = NumOps.FromDouble(reader.ReadDouble());

            int sCount = reader.ReadInt32();
            _s = new List<Vector<T>>(sCount);
            for (int i = 0; i < sCount; i++)
            {
                int vectorLength = reader.ReadInt32();
                byte[] vectorData = reader.ReadBytes(vectorLength);
                _s.Add(Vector<T>.Deserialize(vectorData));
            }

            int yCount = reader.ReadInt32();
            _y = new List<Vector<T>>(yCount);
            for (int i = 0; i < yCount; i++)
            {
                int vectorLength = reader.ReadInt32();
                byte[] vectorData = reader.ReadBytes(vectorLength);
                _y.Add(Vector<T>.Deserialize(vectorData));
            }
        }
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

        // L-BFGS benefits from re-evaluation for line search
        if (context.SupportsReevaluation)
        {
            T origLoss = context.Loss;
            T newLoss = context.Reevaluate();
            if (NumOps.GreaterThan(newLoss, origLoss))
            {
                // Re-materialize sparse-embedding contributions before reading the
                // retry's flat gradient — Reevaluate refreshes the dense dict but
                // leaves SparseEmbeddingGradient<T> entries from the autodiff
                // backward unconsumed, so the retry's GetFlatGradients() would
                // silently omit sparse-only embedding params without this call.
                SparseEmbeddingOptimizerHelpers.MaterializeSparseIntoGradientsDict(context, Engine);
                var retry = UpdateParameters(context.GetFlatParameters(), context.GetFlatGradients());
                context.SetFlatParameters(retry);
            }
        }
    }
}
