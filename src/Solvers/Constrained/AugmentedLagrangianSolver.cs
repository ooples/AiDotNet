using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Solvers.LinearProgramming;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Solvers.Constrained;

/// <summary>
/// Solves general nonlinear constrained problems by the method of multipliers, minimizing a sequence
/// of unconstrained augmented Lagrangians.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Implements the augmented Lagrangian method of M. R. Hestenes, "Multiplier and Gradient Methods",
/// <i>Journal of Optimization Theory and Applications</i> 4(5), 1969, pp. 303-320, and M. J. D.
/// Powell, "A Method for Nonlinear Constraints in Minimization Problems" (1969), with the inequality
/// treatment of R. T. Rockafellar, "A Dual Approach to Solving Nonlinear Programming Problems by
/// Unconstrained Optimization", <i>Mathematical Programming</i> 5, 1973, pp. 354-373. The outer loop
/// follows Framework 17.3 and Algorithm 17.4 of J. Nocedal and S. J. Wright, <i>Numerical
/// Optimization</i> (2nd ed., Springer 2006).
/// </para>
/// <para>
/// <b>Why not just a penalty.</b> The obvious way to handle a constraint is to add a large multiple
/// of its violation to the objective. It works, but only in the limit: the penalty must go to
/// infinity for the answer to become exactly feasible, and long before it gets there the subproblem
/// is too ill-conditioned to solve. The augmented Lagrangian adds an explicit multiplier term
/// alongside the penalty and updates the multiplier after each subproblem. Those multipliers
/// converge to the true Lagrange multipliers, at which point the constraints are satisfied exactly
/// with the penalty still at a moderate value — the ill-conditioning never arrives.
/// </para>
/// <para>
/// <b>The inequality form.</b> Inequalities enter through
/// <c>(1/2ρ)·Σ max(0, μ_j + ρ·g_j(x))² − μ_j²</c>, which is Rockafellar's. The <c>max</c> is what
/// makes it correct: a constraint with room to spare contributes nothing and does not distort the
/// objective, while one that is binding or violated contributes exactly as an equality would. The
/// expression is continuously differentiable despite the <c>max</c>, because the two branches meet
/// where the argument is zero and so does the derivative — which is why an ordinary smooth
/// unconstrained solver can minimize it.
/// </para>
/// <para><b>For Beginners:</b> Two mechanisms push the answer into the allowed region. The penalty
/// is a fine for breaking a rule, and it gets steeper if you keep breaking it. The multiplier is a
/// price on that rule, learned from how much you have been breaking it. Fines alone would have to
/// become infinite to work; prices converge to a finite right answer, and then the fine can stay
/// small.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // minimize x² + y² subject to x + y = 1; the answer is (0.5, 0.5).
/// var problem = new ConstrainedProblem&lt;double&gt;(
///     objective: p =&gt; (p[0] * p[0] + p[1] * p[1],
///                      new Vector&lt;double&gt;(new[] { 2 * p[0], 2 * p[1] })),
///     equalityConstraints: p =&gt;
///     {
///         var jacobian = new Matrix&lt;double&gt;(1, 2);
///         jacobian[0, 0] = 1.0;
///         jacobian[0, 1] = 1.0;
///         return (new Vector&lt;double&gt;(new[] { p[0] + p[1] - 1.0 }), jacobian);
///     });
///
/// var solution = new AugmentedLagrangianSolver&lt;double&gt;()
///     .Solve(problem, new Vector&lt;double&gt;(new[] { 0.0, 0.0 }));
/// </code>
/// </example>
public sealed class AugmentedLagrangianSolver<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly AugmentedLagrangianSolverOptions _options;
    private readonly IFunctionOptimizer<T> _innerOptimizer;

    /// <summary>
    /// Creates an augmented Lagrangian solver with the default options and inner optimizer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The default inner optimizer is L-BFGS, which is the standard choice: each subproblem is a
    /// smooth unconstrained minimization whose curvature changes between outer iterations as the
    /// penalty grows, and L-BFGS rebuilds its curvature estimate cheaply rather than carrying a
    /// stale one.
    /// </para>
    /// </remarks>
    public AugmentedLagrangianSolver()
        : this(new AugmentedLagrangianSolverOptions(), CreateDefaultInnerOptimizer())
    {
    }

    /// <summary>
    /// Creates an augmented Lagrangian solver with the given options and the default inner optimizer.
    /// </summary>
    /// <param name="options">Solver configuration.</param>
    public AugmentedLagrangianSolver(AugmentedLagrangianSolverOptions options)
        : this(options, CreateDefaultInnerOptimizer())
    {
    }

    /// <summary>
    /// Creates an augmented Lagrangian solver.
    /// </summary>
    /// <param name="options">Solver configuration.</param>
    /// <param name="innerOptimizer">
    /// The unconstrained optimizer used on each subproblem. Any
    /// <see cref="IFunctionOptimizer{T}"/> will do, so a caller who knows their problem's structure
    /// can substitute one that suits it — conjugate gradient for a very large problem, or Newton's
    /// method where second derivatives are cheap.
    /// </param>
    /// <exception cref="ArgumentNullException">
    /// Thrown when <paramref name="options"/> or <paramref name="innerOptimizer"/> is null.
    /// </exception>
    /// <exception cref="ArgumentException">
    /// Thrown when an iteration limit, tolerance or penalty setting is out of range.
    /// </exception>
    public AugmentedLagrangianSolver(
        AugmentedLagrangianSolverOptions options, IFunctionOptimizer<T> innerOptimizer)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _innerOptimizer = innerOptimizer ?? throw new ArgumentNullException(nameof(innerOptimizer));

        if (options.MaxOuterIterations <= 0)
        {
            throw new ArgumentException("MaxOuterIterations must be positive.", nameof(options));
        }

        if (options.MaxInnerIterations <= 0)
        {
            throw new ArgumentException("MaxInnerIterations must be positive.", nameof(options));
        }

        if (options.FeasibilityTolerance <= 0.0 || options.StationarityTolerance <= 0.0)
        {
            throw new ArgumentException("Tolerances must be positive.", nameof(options));
        }

        if (options.InitialPenalty <= 0.0)
        {
            throw new ArgumentException("InitialPenalty must be positive.", nameof(options));
        }

        if (options.PenaltyGrowthFactor <= 1.0)
        {
            throw new ArgumentException(
                "PenaltyGrowthFactor must exceed 1 — a factor of 1 or less never tightens the " +
                "penalty, so an infeasible iterate would never be driven back.", nameof(options));
        }

        if (options.MaximumPenalty < options.InitialPenalty)
        {
            throw new ArgumentException(
                "MaximumPenalty cannot be below InitialPenalty.", nameof(options));
        }

        if (options.RequiredViolationReduction <= 0.0 || options.RequiredViolationReduction >= 1.0)
        {
            throw new ArgumentException(
                "RequiredViolationReduction must lie strictly between 0 and 1.", nameof(options));
        }
    }

    /// <summary>
    /// Solves a constrained problem.
    /// </summary>
    /// <param name="problem">The problem to solve.</param>
    /// <param name="initialPoint">Where to start. It need not be feasible.</param>
    /// <returns>The best point found, its multipliers, and how nearly it satisfies the constraints.</returns>
    /// <exception cref="ArgumentNullException">
    /// Thrown when <paramref name="problem"/> or <paramref name="initialPoint"/> is null.
    /// </exception>
    /// <exception cref="ArgumentException">Thrown when the starting point is empty.</exception>
    public ConstrainedSolution<T> Solve(ConstrainedProblem<T> problem, Vector<T> initialPoint)
    {
        if (problem is null) throw new ArgumentNullException(nameof(problem));
        if (initialPoint is null) throw new ArgumentNullException(nameof(initialPoint));
        if (initialPoint.Length == 0)
        {
            throw new ArgumentException(
                "The starting point must have at least one variable.", nameof(initialPoint));
        }

        var point = initialPoint;

        int equalityCount = CountConstraints(problem.EqualityConstraints, point);
        int inequalityCount = CountConstraints(problem.InequalityConstraints, point);

        var equalityMultipliers = new Vector<T>(equalityCount);
        var inequalityMultipliers = new Vector<T>(inequalityCount);

        double penalty = _options.InitialPenalty;
        double previousViolation = double.PositiveInfinity;

        // With no constraints at all this degenerates to a single unconstrained solve, which is the
        // right answer rather than a special case worth rejecting.
        if (equalityCount == 0 && inequalityCount == 0)
        {
            point = _innerOptimizer.Minimize(
                point,
                problem.Objective,
                _options.MaxInnerIterations,
                NumOps.FromDouble(_options.StationarityTolerance));

            return Report(problem, point, null, null, LinearProgramStatus.Optimal, 1);
        }

        for (int iteration = 1; iteration <= _options.MaxOuterIterations; iteration++)
        {
            T penaltyValue = NumOps.FromDouble(penalty);

            var subproblem = BuildSubproblem(
                problem, equalityMultipliers, inequalityMultipliers, penaltyValue);

            point = _innerOptimizer.Minimize(
                point,
                subproblem,
                _options.MaxInnerIterations,
                NumOps.FromDouble(_options.StationarityTolerance));

            var equality = Evaluate(problem.EqualityConstraints, point, equalityCount);
            var inequality = Evaluate(problem.InequalityConstraints, point, inequalityCount);

            double violation = Violation(equality.Values, inequality.Values);

            if (violation <= _options.FeasibilityTolerance)
            {
                // The multipliers still need their final update: the ones in hand were computed for
                // the previous iterate, and it is the updated pair that satisfies the KKT conditions
                // at the point being returned.
                UpdateMultipliers(
                    equalityMultipliers, inequalityMultipliers,
                    equality.Values, inequality.Values, penaltyValue);

                return Report(
                    problem, point, equalityMultipliers, inequalityMultipliers,
                    LinearProgramStatus.Optimal, iteration);
            }

            if (violation <= _options.RequiredViolationReduction * previousViolation)
            {
                // Good progress: the penalty is doing its job, so buy accuracy with a multiplier
                // update rather than with more ill-conditioning.
                UpdateMultipliers(
                    equalityMultipliers, inequalityMultipliers,
                    equality.Values, inequality.Values, penaltyValue);

                previousViolation = violation;
            }
            else
            {
                // Stalled: raise the penalty and leave the multipliers where they are. Doing both at
                // once is what makes naive implementations oscillate.
                penalty = Math.Min(penalty * _options.PenaltyGrowthFactor, _options.MaximumPenalty);
                previousViolation = Math.Min(previousViolation, violation);
            }
        }

        return Report(
            problem, point,
            equalityCount > 0 ? equalityMultipliers : null,
            inequalityCount > 0 ? inequalityMultipliers : null,
            LinearProgramStatus.IterationLimit,
            _options.MaxOuterIterations);
    }

    /// <summary>
    /// Builds the augmented Lagrangian and its gradient at the current multipliers and penalty.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The function returned is
    /// <c>L(x) = f(x) + λᵀh(x) + (ρ/2)‖h(x)‖² + (1/2ρ)·Σ_j (max(0, μ_j + ρ g_j(x))² − μ_j²)</c>,
    /// whose gradient is
    /// <c>∇f + Σ_i (λ_i + ρ h_i)∇h_i + Σ_j max(0, μ_j + ρ g_j) ∇g_j</c>. Both branches of the
    /// inequality term have the same value and the same derivative where the argument is zero, so
    /// the whole expression is continuously differentiable and an ordinary smooth solver handles it.
    /// </para>
    /// </remarks>
    private static Func<Vector<T>, (T Value, Vector<T> Gradient)> BuildSubproblem(
        ConstrainedProblem<T> problem,
        Vector<T> equalityMultipliers,
        Vector<T> inequalityMultipliers,
        T penalty)
    {
        T half = NumOps.FromDouble(0.5);

        return point =>
        {
            var (value, gradient) = problem.Objective(point);

            var result = new Vector<T>(gradient.Length);
            for (int i = 0; i < gradient.Length; i++) result[i] = gradient[i];

            if (problem.EqualityConstraints is not null && equalityMultipliers.Length > 0)
            {
                var (values, jacobian) = problem.EqualityConstraints(point);

                for (int i = 0; i < equalityMultipliers.Length; i++)
                {
                    T violation = values[i];

                    // lambda_i * h_i + (rho/2) * h_i^2
                    value = NumOps.Add(value, NumOps.Multiply(equalityMultipliers[i], violation));
                    value = NumOps.Add(value, NumOps.Multiply(
                        NumOps.Multiply(half, penalty), NumOps.Multiply(violation, violation)));

                    // (lambda_i + rho * h_i) * grad h_i
                    T weight = NumOps.Add(
                        equalityMultipliers[i], NumOps.Multiply(penalty, violation));

                    AccumulateRow(result, jacobian, i, weight);
                }
            }

            if (problem.InequalityConstraints is not null && inequalityMultipliers.Length > 0)
            {
                var (values, jacobian) = problem.InequalityConstraints(point);

                T twicePenalty = NumOps.Multiply(NumOps.FromDouble(2.0), penalty);

                for (int j = 0; j < inequalityMultipliers.Length; j++)
                {
                    T shifted = NumOps.Add(
                        inequalityMultipliers[j], NumOps.Multiply(penalty, values[j]));

                    // A constraint with room to spare drives the shift negative, where the max
                    // clamps it to zero: it then contributes nothing and does not bend the
                    // objective away from what it would otherwise do.
                    T active = NumOps.GreaterThan(shifted, NumOps.Zero) ? shifted : NumOps.Zero;

                    T contribution = NumOps.Divide(
                        NumOps.Subtract(
                            NumOps.Multiply(active, active),
                            NumOps.Multiply(inequalityMultipliers[j], inequalityMultipliers[j])),
                        twicePenalty);

                    value = NumOps.Add(value, contribution);

                    AccumulateRow(result, jacobian, j, active);
                }
            }

            return (value, result);
        };
    }

    /// <summary>
    /// Adds <c>weight</c> times row <paramref name="row"/> of a Jacobian into an accumulating
    /// gradient.
    /// </summary>
    private static void AccumulateRow(Vector<T> target, Matrix<T> jacobian, int row, T weight)
    {
        for (int c = 0; c < target.Length; c++)
        {
            target[c] = NumOps.Add(target[c], NumOps.Multiply(weight, jacobian[row, c]));
        }
    }

    /// <summary>
    /// Applies the first-order multiplier update, which is what makes this more than a penalty
    /// method.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The equality update <c>λ ← λ + ρ·h(x)</c> is a gradient ascent step on the dual, and the
    /// inequality update <c>μ ← max(0, μ + ρ·g(x))</c> is the same step projected onto the
    /// non-negative orthant, which is where inequality multipliers are required to live. Both
    /// converge to the true Lagrange multipliers, at which point the constraints hold exactly with
    /// the penalty still finite.
    /// </para>
    /// </remarks>
    private static void UpdateMultipliers(
        Vector<T> equalityMultipliers,
        Vector<T> inequalityMultipliers,
        Vector<T>? equalityValues,
        Vector<T>? inequalityValues,
        T penalty)
    {
        if (equalityValues is not null)
        {
            for (int i = 0; i < equalityMultipliers.Length; i++)
            {
                equalityMultipliers[i] = NumOps.Add(
                    equalityMultipliers[i], NumOps.Multiply(penalty, equalityValues[i]));
            }
        }

        if (inequalityValues is not null)
        {
            for (int j = 0; j < inequalityMultipliers.Length; j++)
            {
                T updated = NumOps.Add(
                    inequalityMultipliers[j], NumOps.Multiply(penalty, inequalityValues[j]));

                inequalityMultipliers[j] =
                    NumOps.GreaterThan(updated, NumOps.Zero) ? updated : NumOps.Zero;
            }
        }
    }

    /// <summary>
    /// Returns the largest amount by which any constraint is violated. Equalities count in either
    /// direction; an inequality with room to spare counts as no violation at all.
    /// </summary>
    private static double Violation(Vector<T>? equalityValues, Vector<T>? inequalityValues)
    {
        double worst = 0.0;

        if (equalityValues is not null)
        {
            for (int i = 0; i < equalityValues.Length; i++)
            {
                worst = Math.Max(worst, Math.Abs(NumOps.ToDouble(equalityValues[i])));
            }
        }

        if (inequalityValues is not null)
        {
            for (int j = 0; j < inequalityValues.Length; j++)
            {
                worst = Math.Max(worst, NumOps.ToDouble(inequalityValues[j]));
            }
        }

        return worst;
    }

    private static (Vector<T>? Values, Matrix<T>? Jacobian) Evaluate(
        Func<Vector<T>, (Vector<T> Values, Matrix<T> Jacobian)>? constraints,
        Vector<T> point,
        int count)
    {
        if (constraints is null || count == 0) return (null, null);

        var (values, jacobian) = constraints(point);
        return (values, jacobian);
    }

    private static int CountConstraints(
        Func<Vector<T>, (Vector<T> Values, Matrix<T> Jacobian)>? constraints, Vector<T> point)
    {
        if (constraints is null) return 0;

        var (values, _) = constraints(point);
        return values.Length;
    }

    private static ConstrainedSolution<T> Report(
        ConstrainedProblem<T> problem,
        Vector<T> point,
        Vector<T>? equalityMultipliers,
        Vector<T>? inequalityMultipliers,
        LinearProgramStatus status,
        int iterations)
    {
        var (value, _) = problem.Objective(point);

        var equality = problem.EqualityConstraints is null
            ? null
            : problem.EqualityConstraints(point).Values;
        var inequality = problem.InequalityConstraints is null
            ? null
            : problem.InequalityConstraints(point).Values;

        return new ConstrainedSolution<T>(
            status, point, value, NumOps.FromDouble(Violation(equality, inequality)), iterations,
            equalityMultipliers, inequalityMultipliers);
    }

    private static IFunctionOptimizer<T> CreateDefaultInnerOptimizer()
        => LBFGSOptimizer<T, Tensor<T>, Tensor<T>>.CreateForFunction();
}
