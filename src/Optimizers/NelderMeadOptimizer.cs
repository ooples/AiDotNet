using AiDotNet.Helpers;
using Newtonsoft.Json;

using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.Optimizers;

/// <summary>
/// Implements the Nelder-Mead optimization algorithm, also known as the downhill simplex method.
/// </summary>
/// <remarks>
/// <para>
/// The Nelder-Mead method is a heuristic search method that can optimize a problem with N variables.
/// It attempts to minimize a scalar-valued nonlinear function of n real variables using only function values,
/// without any derivative information.
/// </para>
/// <para><b>For Beginners:</b>
/// Imagine you're trying to find the lowest point in a hilly landscape. The Nelder-Mead method is like
/// having a group of explorers who work together, moving and reshaping their search pattern to find the lowest point.
/// They don't need to know which way is downhill; they just compare their positions and adjust accordingly.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[ComponentType(ComponentType.Optimizer)]
[PipelineStage(PipelineStage.Training)]
public partial class NelderMeadOptimizer<T, TInput, TOutput> : OptimizerBase<T, TInput, TOutput>, IDerivativeFreeFunctionOptimizer<T>
{
    /// <summary>
    /// The options specific to the Nelder-Mead optimizer.
    /// </summary>
    private NelderMeadOptimizerOptions<T, TInput, TOutput> _options;

    /// <summary>
    /// The current iteration count.
    /// </summary>
    private int _iteration;

    /// <summary>
    /// The reflection coefficient.
    /// </summary>
    private T _alpha;

    /// <summary>
    /// The contraction coefficient.
    /// </summary>
    private T _beta;

    /// <summary>
    /// The expansion coefficient.
    /// </summary>
    private T _gamma;

    /// <summary>
    /// The shrinkage coefficient.
    /// </summary>
    private T _delta;

    /// <summary>
    /// Initializes a new instance of the NelderMeadOptimizer class.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This constructor sets up the Nelder-Mead optimizer with the provided options and dependencies.
    /// If no options are provided, it uses default settings.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like preparing your team of explorers before they start searching the landscape.
    /// You're giving them their initial instructions and tools.
    /// </para>
    /// </remarks>
    /// <param name="model">The model to optimize.</param>
    /// <param name="options">The Nelder-Mead-specific optimization options.</param>
    public NelderMeadOptimizer(
        IFullModel<T, TInput, TOutput> model,
        NelderMeadOptimizerOptions<T, TInput, TOutput>? options = null)
        : base(model, options ?? new())
    {
        _options = options ?? new NelderMeadOptimizerOptions<T, TInput, TOutput>();
        _alpha = NumOps.Zero;
        _beta = NumOps.Zero;
        _gamma = NumOps.Zero;
        _delta = NumOps.Zero;

        InitializeAdaptiveParameters();
    }

    /// <summary>
    /// Initializes a new instance of the NelderMeadOptimizer class for minimizing a plain function,
    /// with no model attached.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Use this overload with <see cref="Minimize"/> when you want to minimize a mathematical
    /// function directly rather than train a model. <see cref="Optimize"/> requires a model and is
    /// not available on an instance created this way.
    /// </para>
    /// <para><b>For Beginners:</b> The other constructor asks for a model because it is set up to
    /// tune that model against training data. If all you have is a formula you want to make as
    /// small as possible, there is no model to hand over — use this constructor instead.
    /// </para>
    /// </remarks>
    /// <param name="options">The Nelder-Mead-specific optimization options.</param>
    public NelderMeadOptimizer(NelderMeadOptimizerOptions<T, TInput, TOutput>? options = null)
        : base(null, options ?? new())
    {
        _options = options ?? new NelderMeadOptimizerOptions<T, TInput, TOutput>();
        _alpha = NumOps.Zero;
        _beta = NumOps.Zero;
        _gamma = NumOps.Zero;
        _delta = NumOps.Zero;

        InitializeAdaptiveParameters();
    }

    /// <summary>
    /// Initializes the adaptive parameters for the Nelder-Mead optimizer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method sets up the initial values for the reflection, contraction, expansion, and shrinkage coefficients.
    /// It also resets the iteration counter.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like giving your explorers their initial strategies for how to move around the landscape.
    /// You're setting up how far they should reflect, contract, expand, or shrink their search pattern.
    /// </para>
    /// </remarks>
    protected override void InitializeAdaptiveParameters()
    {
        base.InitializeAdaptiveParameters();
        _alpha = NumOps.FromDouble(_options.InitialAlpha);
        _beta = NumOps.FromDouble(_options.InitialBeta);
        _gamma = NumOps.FromDouble(_options.InitialGamma);
        _delta = NumOps.FromDouble(_options.InitialDelta);
        _iteration = 0;
    }

    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// Minimizes an arbitrary scalar objective with no model, dataset, or training pipeline
    /// involved — the gradient-free counterpart of SciPy's
    /// <c>scipy.optimize.minimize(method='Nelder-Mead')</c>. This runs the same simplex search
    /// that <see cref="Optimize"/> uses, so the function path and the model-training path cannot
    /// drift apart.
    /// </para>
    /// <para>
    /// The initial simplex is built from <paramref name="initialParameters"/> using the standard
    /// construction (Lagarias et al. 1998, and MATLAB's <c>fminsearch</c>): the starting point
    /// plus one vertex per dimension, each perturbing a single coordinate by
    /// <see cref="NelderMeadOptimizerOptions{T, TInput, TOutput}.InitialSimplexStep"/> relative to
    /// its magnitude, or by
    /// <see cref="NelderMeadOptimizerOptions{T, TInput, TOutput}.ZeroCoordinateSimplexStep"/> when
    /// the coordinate is zero.
    /// </para>
    /// <para>
    /// The search stops when the spread of objective values across the simplex falls below
    /// <paramref name="tolerance"/>, or when <paramref name="maxIterations"/> is reached.
    /// </para>
    /// <para><b>For Beginners:</b> Give this a starting guess and a function that scores any point,
    /// and it finds the point with the lowest score. It never asks for a derivative — it just
    /// keeps a cluster of trial points, throws away the worst one each round, and moves the
    /// cluster downhill. That makes it usable on functions with kinks, noise, or no formula at
    /// all (a simulation, a backtest, a black box).
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException">
    /// Thrown when <paramref name="initialParameters"/> or <paramref name="objective"/> is null.
    /// </exception>
    /// <exception cref="ArgumentException">
    /// Thrown when <paramref name="initialParameters"/> is empty or
    /// <paramref name="maxIterations"/> is not positive.
    /// </exception>
    public Vector<T> Minimize(
        Vector<T> initialParameters,
        Func<Vector<T>, T> objective,
        int maxIterations,
        T tolerance)
    {
        Guard.NotNull(initialParameters);
        Guard.NotNull(objective);

        int n = initialParameters.Length;
        if (n == 0)
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

        InitializeAdaptiveParameters();

        return RunSimplexSearch<object?>(
            BuildInitialSimplex(initialParameters),
            point => (objective(point), null),
            maxIterations,
            tolerance);
    }

    /// <summary>
    /// Builds the standard Nelder-Mead initial simplex around a single starting point.
    /// </summary>
    /// <param name="start">The starting point, which becomes the first vertex.</param>
    /// <returns>A simplex of <c>n + 1</c> vertices, where <c>n</c> is the dimension.</returns>
    private List<Vector<T>> BuildInitialSimplex(Vector<T> start)
    {
        int n = start.Length;
        var simplex = new List<Vector<T>>(n + 1) { start.Clone() };

        var relativeStep = NumOps.FromDouble(_options.InitialSimplexStep);
        var zeroStep = NumOps.FromDouble(_options.ZeroCoordinateSimplexStep);
        var zeroThreshold = NumOps.FromDouble(_options.ZeroCoordinateThreshold);

        for (int i = 0; i < n; i++)
        {
            var vertex = start.Clone();
            bool coordinateIsEffectivelyZero =
                NumOps.LessThanOrEquals(NumOps.Abs(start[i]), zeroThreshold);
            vertex[i] = coordinateIsEffectivelyZero
                ? NumOps.Add(start[i], zeroStep)
                : NumOps.Add(start[i], NumOps.Multiply(relativeStep, start[i]));
            simplex.Add(vertex);
        }

        return simplex;
    }

    /// <summary>
    /// Runs the Nelder-Mead simplex search over a vector space.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is the single implementation of the algorithm, shared by <see cref="Minimize"/> and
    /// <see cref="Optimize"/>. It follows the standard formulation (Nelder and Mead 1965, with
    /// the convergence analysis and inside/outside contraction distinction of Lagarias, Reeds,
    /// Wright and Wright 1998): reflect the worst vertex through the centroid of the rest, expand
    /// when reflection produced a new best, contract when it did not beat the second-worst, and
    /// shrink the whole simplex toward the best vertex when contraction also fails.
    /// </para>
    /// <para>
    /// The objective is always a <b>minimization</b> objective here — lower is better. Callers
    /// working with a fitness score where higher is better must negate before calling.
    /// </para>
    /// </remarks>
    /// <typeparam name="TEval">
    /// Caller-defined payload carried alongside each objective value. <see cref="Optimize"/> uses
    /// it to keep the full <c>OptimizationStepData</c> for each evaluated vertex, so the model
    /// path never has to re-evaluate a point just to recover its bookkeeping.
    /// </typeparam>
    /// <param name="simplex">The initial simplex; mutated in place.</param>
    /// <param name="objective">
    /// The scalar objective to minimize, returning the value and an optional payload.
    /// </param>
    /// <param name="maxIterations">Maximum number of simplex iterations.</param>
    /// <param name="tolerance">
    /// Convergence threshold on the spread of objective values across the simplex.
    /// </param>
    /// <param name="onIteration">
    /// Optional per-iteration callback receiving the iteration index, the current best vertex, its
    /// objective value and its payload. Returning <c>true</c> stops the search (used by
    /// <see cref="Optimize"/> for history tracking and early stopping).
    /// </param>
    /// <returns>The best vertex found.</returns>
    private Vector<T> RunSimplexSearch<TEval>(
        List<Vector<T>> simplex,
        Func<Vector<T>, (T value, TEval evaluation)> objective,
        int maxIterations,
        T tolerance,
        Func<int, Vector<T>, T, TEval, bool>? onIteration = null)
    {
        if (simplex.Count < 2)
        {
            throw new ArgumentException(
                "A Nelder-Mead simplex must contain at least two vertices.", nameof(simplex));
        }

        int n = simplex.Count - 1;
        if (simplex[0].Length != n || simplex[0].Length == 0)
        {
            throw new ArgumentException(
                $"A simplex with {simplex.Count} vertices must use non-empty vectors of length {n}.",
                nameof(simplex));
        }
        for (int i = 1; i < simplex.Count; i++)
        {
            if (simplex[i].Length != n)
            {
                throw new ArgumentException(
                    $"Simplex vertex {i} has length {simplex[i].Length}; expected {n}.",
                    nameof(simplex));
            }
        }

        var values = new List<T>(simplex.Count);
        var evaluations = new List<TEval>(simplex.Count);
        for (int i = 0; i < simplex.Count; i++)
        {
            var (value, evaluation) = objective(simplex[i]);
            values.Add(value);
            evaluations.Add(evaluation);
        }

        SortSimplex(simplex, values, evaluations);
        var previousBest = values[0];

        // A non-positive tolerance disables the simplex-spread criterion entirely. Optimize() uses
        // that to stop on its own fitness-progress and early-stopping rules instead. Without this
        // guard a zero tolerance still terminates whenever the spread is exactly zero — which it is
        // when every initial vertex happens to score the same — so the search would return before
        // recording a single iteration.
        bool useSpreadTolerance = NumOps.GreaterThan(tolerance, NumOps.Zero);

        for (int iteration = 0; iteration < maxIterations; iteration++)
        {
            _iteration++;

            // Converged when every vertex agrees on the objective to within tolerance. The test is
            // relative to the current best value (with tolerance itself as the additive floor so a
            // best value of zero still terminates), which makes it scale-invariant: the same
            // tolerance behaves identically whether objective values are near 1e-6 or near 1e6.
            if (useSpreadTolerance)
            {
                var spread = NumOps.Abs(NumOps.Subtract(values[n], values[0]));
                var threshold = NumOps.Multiply(tolerance, NumOps.Add(NumOps.Abs(values[0]), tolerance));
                if (NumOps.LessThanOrEquals(spread, threshold))
                {
                    break;
                }
            }

            var centroid = CalculateCentroid(simplex, n);
            var worst = simplex[n];

            // Reflection: xr = centroid + alpha * (centroid - worst)
            var reflected = Combine(centroid, Subtract(centroid, worst), _alpha);
            var (reflectedValue, reflectedEvaluation) = objective(reflected);

            if (NumOps.LessThan(reflectedValue, values[0]))
            {
                // Reflection produced a new best — try going further in the same direction.
                // xe = centroid + gamma * (xr - centroid)
                var expanded = Combine(centroid, Subtract(reflected, centroid), _gamma);
                var (expandedValue, expandedEvaluation) = objective(expanded);

                bool takeExpanded = NumOps.LessThan(expandedValue, reflectedValue);
                simplex[n] = takeExpanded ? expanded : reflected;
                values[n] = takeExpanded ? expandedValue : reflectedValue;
                evaluations[n] = takeExpanded ? expandedEvaluation : reflectedEvaluation;
            }
            else if (NumOps.LessThan(reflectedValue, values[n - 1]))
            {
                // Better than the second-worst but not a new best: accept the reflection as-is.
                simplex[n] = reflected;
                values[n] = reflectedValue;
                evaluations[n] = reflectedEvaluation;
            }
            else if (NumOps.LessThan(reflectedValue, values[n]))
            {
                // Outside contraction: xoc = centroid + beta * (xr - centroid)
                var contracted = Combine(centroid, Subtract(reflected, centroid), _beta);
                var (contractedValue, contractedEvaluation) = objective(contracted);

                if (NumOps.LessThanOrEquals(contractedValue, reflectedValue))
                {
                    simplex[n] = contracted;
                    values[n] = contractedValue;
                    evaluations[n] = contractedEvaluation;
                }
                else
                {
                    ShrinkSimplex(simplex, values, evaluations, objective);
                }
            }
            else
            {
                // Inside contraction: xic = centroid + beta * (worst - centroid)
                var contracted = Combine(centroid, Subtract(worst, centroid), _beta);
                var (contractedValue, contractedEvaluation) = objective(contracted);

                if (NumOps.LessThan(contractedValue, values[n]))
                {
                    simplex[n] = contracted;
                    values[n] = contractedValue;
                    evaluations[n] = contractedEvaluation;
                }
                else
                {
                    ShrinkSimplex(simplex, values, evaluations, objective);
                }
            }

            SortSimplex(simplex, values, evaluations);

            AdaptCoefficients(NumOps.Subtract(previousBest, values[0]));
            previousBest = values[0];

            if (onIteration is not null && onIteration(iteration, simplex[0], values[0], evaluations[0]))
            {
                break;
            }
        }

        return simplex[0];
    }

    /// <summary>
    /// Sorts the simplex with its cached objective values and payloads together, best (lowest) first.
    /// </summary>
    private void SortSimplex<TEval>(List<Vector<T>> simplex, List<T> values, List<TEval> evaluations)
    {
        // Insertion sort: the simplex is small (n + 1 vertices) and is already nearly sorted on
        // every iteration after the first, so this beats allocating a new ordered sequence.
        for (int i = 1; i < simplex.Count; i++)
        {
            var vertex = simplex[i];
            var value = values[i];
            var evaluation = evaluations[i];
            int j = i - 1;
            while (j >= 0 && NumOps.GreaterThan(values[j], value))
            {
                simplex[j + 1] = simplex[j];
                values[j + 1] = values[j];
                evaluations[j + 1] = evaluations[j];
                j--;
            }
            simplex[j + 1] = vertex;
            values[j + 1] = value;
            evaluations[j + 1] = evaluation;
        }
    }

    /// <summary>
    /// Shrinks every vertex except the best toward the best vertex, re-evaluating each.
    /// </summary>
    private void ShrinkSimplex<TEval>(
        List<Vector<T>> simplex,
        List<T> values,
        List<TEval> evaluations,
        Func<Vector<T>, (T value, TEval evaluation)> objective)
    {
        var best = simplex[0];
        for (int i = 1; i < simplex.Count; i++)
        {
            // xi = best + delta * (xi - best)
            simplex[i] = Combine(best, Subtract(simplex[i], best), _delta);
            var (value, evaluation) = objective(simplex[i]);
            values[i] = value;
            evaluations[i] = evaluation;
        }
    }

    /// <summary>
    /// Computes the centroid of the simplex excluding its worst vertex.
    /// </summary>
    /// <param name="simplex">The current simplex, sorted best-first.</param>
    /// <param name="n">The dimension of the search space (the worst vertex is at index n).</param>
    private Vector<T> CalculateCentroid(List<Vector<T>> simplex, int n)
    {
        var sum = new Vector<T>(simplex[0].Length);
        for (int i = 0; i < n; i++)
        {
            sum = (Vector<T>)Engine.Add(sum, simplex[i]);
        }

        return (Vector<T>)Engine.Multiply(sum, NumOps.Divide(NumOps.One, NumOps.FromDouble(n)));
    }

    /// <summary>
    /// Returns <c>left - right</c> elementwise.
    /// </summary>
    private Vector<T> Subtract(Vector<T> left, Vector<T> right)
    {
        return (Vector<T>)Engine.Subtract(left, right);
    }

    /// <summary>
    /// Returns <c>origin + scale * direction</c> elementwise.
    /// </summary>
    private Vector<T> Combine(Vector<T> origin, Vector<T> direction, T scale)
    {
        return (Vector<T>)Engine.Add(origin, (Vector<T>)Engine.Multiply(direction, scale));
    }

    /// <summary>
    /// Applies the adaptive-coefficient update shared by both search entry points.
    /// </summary>
    /// <param name="improvement">
    /// How much the best objective value improved this iteration (positive means it got better).
    /// </param>
    private void AdaptCoefficients(T improvement)
    {
        if (!_options.UseAdaptiveParameters)
        {
            return;
        }

        var adaptationRate = NumOps.FromDouble(_options.AdaptationRate);

        _alpha = NumOps.Add(_alpha, NumOps.Multiply(adaptationRate, improvement));
        _beta = NumOps.Add(_beta, NumOps.Multiply(adaptationRate, improvement));
        _gamma = NumOps.Add(_gamma, NumOps.Multiply(adaptationRate, improvement));
        _delta = NumOps.Add(_delta, NumOps.Multiply(adaptationRate, improvement));

        _alpha = MathHelper.Clamp(_alpha, NumOps.FromDouble(_options.MinAlpha), NumOps.FromDouble(_options.MaxAlpha));
        _beta = MathHelper.Clamp(_beta, NumOps.FromDouble(_options.MinBeta), NumOps.FromDouble(_options.MaxBeta));
        _gamma = MathHelper.Clamp(_gamma, NumOps.FromDouble(_options.MinGamma), NumOps.FromDouble(_options.MaxGamma));
        _delta = MathHelper.Clamp(_delta, NumOps.FromDouble(_options.MinDelta), NumOps.FromDouble(_options.MaxDelta));
    }

    /// <summary>
    /// Performs the optimization process using the Nelder-Mead algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method implements the main optimization loop. It creates and manipulates a simplex
    /// (a geometric figure in N dimensions) to find the optimal solution.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is the actual search process. Your team of explorers starts at different points,
    /// then repeatedly adjusts their positions based on which points are higher or lower.
    /// They reflect away from high points, expand towards promising areas, contract if they overshoot,
    /// and shrink their search area if they get stuck.
    /// </para>
    /// </remarks>
    /// <param name="inputData">The input data for the optimization process.</param>
    /// <returns>The result of the optimization process.</returns>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        ValidationHelper<T>.ValidateInputData(inputData);
        InitializeAdaptiveParameters();

        // The simplex lives in parameter space. A template individual supplies the model shape so
        // any vertex vector can be turned back into a model for evaluation.
        var template = InterfaceGuard.Parameterizable(SpawnIndividual(inputData.XTrain));
        var templateParameters = template.GetParameters();
        int n = templateParameters.Length;
        if (n == 0)
        {
            throw new InvalidOperationException(
                "Nelder-Mead requires a model with at least one optimizable parameter.");
        }

        var simplex = new List<Vector<T>>(n + 1) { templateParameters };
        for (int i = 0; i < n; i++)
        {
            var vertex = InterfaceGuard.Parameterizable(SpawnIndividual(inputData.XTrain)).GetParameters();
            if (vertex.Length != n)
            {
                throw new InvalidOperationException(
                    $"Spawned Nelder-Mead vertex {i + 1} has {vertex.Length} parameters; expected {n}.");
            }
            simplex.Add(vertex);
        }

        // RunSimplexSearch always minimizes, so a higher-is-better fitness score is negated on the
        // way in. Previously this optimizer compared raw fitness with hardcoded GreaterThan /
        // LessThan, which assumed a direction the configured IFitnessCalculator does not have to
        // agree with, and inverted the expansion branch for either direction.
        bool higherIsBetter = FitnessCalculator.IsHigherScoreBetter;

        (T value, OptimizationStepData<T, TInput, TOutput> evaluation) Objective(Vector<T> point)
        {
            var stepData = EvaluateSolution(template.WithParameters(point), inputData);
            var value = higherIsBetter ? NumOps.Negate(stepData.FitnessScore) : stepData.FitnessScore;
            return (value, stepData);
        }

        var bestStepData = new OptimizationStepData<T, TInput, TOutput>();
        var previousStepData = new OptimizationStepData<T, TInput, TOutput>();
        var tolerance = NumOps.FromDouble(_options.Tolerance);

        bool OnIteration(int iteration, Vector<T> best, T bestValue, OptimizationStepData<T, TInput, TOutput> currentStepData)
        {
            UpdateBestSolution(currentStepData, ref bestStepData);
            UpdateAdaptiveParameters(currentStepData, previousStepData);

            if (UpdateIterationHistoryAndCheckEarlyStopping(iteration, bestStepData))
            {
                return true;
            }

            // Check convergence against previousStepData (per-iteration progress),
            // not bestStepData. UpdateBestSolution above copies currentStepData
            // into bestStepData on the first iteration, so |best - current|
            // would always be 0 < tolerance and the optimiser would exit after
            // the first iteration. Issue #1340 / PR #1351 fix swept across the
            // optimizer suite.
            if (iteration > 0 &&
                NumOps.LessThan(
                    NumOps.Abs(NumOps.Subtract(previousStepData.FitnessScore, currentStepData.FitnessScore)),
                    tolerance))
            {
                return true;
            }

            previousStepData = currentStepData;
            return false;
        }

        // The simplex-spread tolerance is disabled here (zero) because this path stops on the
        // fitness-progress and early-stopping criteria above, which is what the optimizer suite's
        // iteration-history contract expects.
        RunSimplexSearch(simplex, Objective, _options.MaxIterations, NumOps.Zero, OnIteration);

        return CreateOptimizationResult(bestStepData, inputData);
    }

    /// <summary>
    /// Updates the adaptive parameters of the Nelder-Mead algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method adjusts the reflection, expansion, contraction, and shrinkage coefficients based on the improvement in fitness.
    /// It's used to fine-tune the algorithm's behavior as the optimization progresses.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like adjusting how far the explorers move based on how well they're doing. If they're finding better spots, they might move more boldly.
    /// </para>
    /// </remarks>
    /// <param name="currentStepData">The current optimization step data.</param>
    /// <param name="previousStepData">The previous optimization step data.</param>
    protected override void UpdateAdaptiveParameters(OptimizationStepData<T, TInput, TOutput> currentStepData, OptimizationStepData<T, TInput, TOutput> previousStepData)
    {
        // The Nelder-Mead coefficients themselves are adapted once per simplex iteration inside
        // RunSimplexSearch (see AdaptCoefficients), which is the single implementation shared by
        // Optimize and Minimize. Adapting them again here would apply the update twice per
        // iteration on the model path. This override remains so the base class's own adaptive
        // state (learning rate, momentum schedule) still advances.
        base.UpdateAdaptiveParameters(currentStepData, previousStepData);
    }

    /// <summary>
    /// Updates the optimizer's options with new settings.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method ensures that only compatible option types are used with this optimizer.
    /// It updates the internal options if the provided options are of the correct type.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like changing the rules for how the explorers should search. It makes sure you're only using rules that work for this specific type of search (Nelder-Mead method).
    /// </para>
    /// </remarks>
    /// <param name="options">The new options to be applied to the optimizer.</param>
    /// <exception cref="ArgumentException">Thrown when the provided options are not of the correct type.</exception>
    protected override void UpdateOptions(OptimizationAlgorithmOptions<T, TInput, TOutput> options)
    {
        if (options is NelderMeadOptimizerOptions<T, TInput, TOutput> nmOptions)
        {
            _options = nmOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected NelderMeadOptimizerOptions.");
        }
    }

    /// <summary>
    /// Gets the current options of the Nelder-Mead optimizer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method returns the current configuration options of the optimizer.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like asking to see the current set of rules the explorers are following in their search.
    /// </para>
    /// </remarks>
    /// <returns>The current optimization algorithm options.</returns>
    public override OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions()
    {
        return _options;
    }
}
