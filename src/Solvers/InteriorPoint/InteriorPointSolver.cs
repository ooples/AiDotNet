using AiDotNet.DecompositionMethods.MatrixDecomposition;
using AiDotNet.Exceptions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Solvers.LinearProgramming;
using AiDotNet.Solvers.QuadraticProgramming;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Solvers.InteriorPoint;

/// <summary>
/// Solves linear and convex quadratic programs with a primal-dual path-following interior-point
/// method using Mehrotra's predictor-corrector.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Implements the algorithm of S. Mehrotra, "On the Implementation of a Primal-Dual Interior Point
/// Method", <i>SIAM Journal on Optimization</i> 2(4), 1992, pp. 575-601, following the presentation
/// and starting-point heuristic in J. Nocedal and S. J. Wright, <i>Numerical Optimization</i>
/// (2nd ed., Springer 2006), Chapter 14 and Section 16.6.
/// </para>
/// <para>
/// <b>How it differs from the simplex method.</b> <see cref="SimplexSolver{T}"/> walks from corner to
/// corner of the feasible region along its edges, and the number of corners it visits can in the
/// worst case grow exponentially. An interior-point method instead stays strictly inside the region
/// and drives toward the optimum along the <i>central path</i>, cutting across the middle. Each step
/// costs a matrix factorization rather than a cheap pivot, but the number of steps grows only
/// logarithmically with the accuracy demanded and barely at all with problem size — so simplex tends
/// to win on small problems and interior point on large or dense ones. Both implement
/// <see cref="ILinearProgramSolver{T}"/>, so swapping them changes nothing else.
/// </para>
/// <para>
/// <b>Why one class solves both.</b> The Newton system for a convex quadratic program is the linear
/// program's system with the objective's Hessian <c>Q</c> added to one block. Setting <c>Q = 0</c>
/// recovers the linear case exactly, so the two share every line of the iteration; the only
/// specialization is that when <c>Q</c> is absent that block is diagonal and inverts by division.
/// </para>
/// <para>
/// <b>Infeasibility and unboundedness are proved, not guessed.</b> Slow progress is not evidence of
/// anything, so this solver reports <see cref="LinearProgramStatus.Infeasible"/> or
/// <see cref="LinearProgramStatus.Unbounded"/> only after checking an explicit certificate: a
/// direction <c>y</c> with <c>Aᵀy ≤ 0</c> and <c>bᵀy &gt; 0</c> proves by Farkas' lemma that no
/// feasible point exists, and a direction <c>d ≥ 0</c> with <c>Ad = 0</c>, <c>Qd = 0</c> and
/// <c>cᵀd &lt; 0</c> is a ray along which the objective falls forever. The iterates of an
/// infeasible-start method converge to exactly these directions, so the check costs one matrix
/// product per iteration.
/// </para>
/// <para><b>For Beginners:</b> Picture the allowed region as a room and the goal as its lowest
/// corner. The simplex method feels its way along the walls from corner to corner. This method
/// starts in the middle of the room and walks downhill toward the low corner without ever touching a
/// wall — stopping just short each time, because the arithmetic breaks down exactly at the wall. It
/// gets very close very fast, which is why large-scale solvers reach for it.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // minimize -x - y  subject to  x + 2y &lt;= 4,  3x + 2y &lt;= 6,  x, y &gt;= 0
/// var program = new LinearProgram&lt;double&gt;(
///     objective: new Vector&lt;double&gt;(new[] { -1.0, -1.0 }),
///     inequalityMatrix: new Matrix&lt;double&gt;(new[,] { { 1.0, 2.0 }, { 3.0, 2.0 } }),
///     inequalityBounds: new Vector&lt;double&gt;(new[] { 4.0, 6.0 }));
///
/// var solution = new InteriorPointSolver&lt;double&gt;().Solve(program);
/// // solution.Status == LinearProgramStatus.Optimal
/// // solution.Solution is approximately (1, 1.5), objective -2.5
/// </code>
/// </example>
public sealed class InteriorPointSolver<T> : ILinearProgramSolver<T>, IQuadraticProgramSolver<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly InteriorPointSolverOptions _options;

    /// <summary>
    /// Creates an interior-point solver with the default options.
    /// </summary>
    public InteriorPointSolver()
        : this(new InteriorPointSolverOptions())
    {
    }

    /// <summary>
    /// Creates an interior-point solver.
    /// </summary>
    /// <param name="options">Solver configuration.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="options"/> is null.</exception>
    /// <exception cref="ArgumentException">
    /// Thrown when the iteration limit or tolerance is not positive, when the fraction-to-boundary
    /// parameter does not lie strictly between 0 and 1, or when the regularization is negative.
    /// </exception>
    public InteriorPointSolver(InteriorPointSolverOptions options)
    {
        if (options is null) throw new ArgumentNullException(nameof(options));
        _options = new InteriorPointSolverOptions(options);

        if (options.MaxIterations <= 0)
        {
            throw new ArgumentException("MaxIterations must be positive.", nameof(options));
        }

        if (options.Tolerance <= 0.0)
        {
            throw new ArgumentException("Tolerance must be positive.", nameof(options));
        }

        if (options.FractionToBoundary <= 0.0 || options.FractionToBoundary >= 1.0)
        {
            throw new ArgumentException(
                "FractionToBoundary must lie strictly between 0 and 1 — a step covering the full " +
                "distance lands on the boundary, where the next iteration divides by zero.",
                nameof(options));
        }

        if (options.Regularization < 0.0)
        {
            throw new ArgumentException("Regularization cannot be negative.", nameof(options));
        }

        if (options.CertificateTolerance <= 0.0 ||
            double.IsNaN(options.CertificateTolerance) ||
            double.IsInfinity(options.CertificateTolerance))
        {
            throw new ArgumentException(
                "CertificateTolerance must be a positive finite number.", nameof(options));
        }
    }

    /// <inheritdoc />
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="program"/> is null.</exception>
    public LinearProgramSolution<T> Solve(LinearProgram<T> program)
    {
        if (program is null) throw new ArgumentNullException(nameof(program));

        var standard = LinearProgramStandardForm<T>.Build(program);
        var problem = BuildProblem(standard, quadratic: null);
        var outcome = Iterate(problem);

        Vector<T>? solution = null;
        T objectiveValue = NumOps.Zero;

        if (outcome.Point is not null)
        {
            solution = standard.RecoverOriginalVariables(outcome.Point);
            objectiveValue = program.Objective.DotProduct(solution);
        }

        var (inequalityDuals, equalityDuals) = SplitDuals(standard, outcome);

        return new LinearProgramSolution<T>(
            outcome.Status, solution, objectiveValue, outcome.Iterations,
            inequalityDuals, equalityDuals);
    }

    /// <inheritdoc />
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="program"/> is null.</exception>
    public QuadraticProgramSolution<T> Solve(QuadraticProgram<T> program)
    {
        if (program is null) throw new ArgumentNullException(nameof(program));

        // The standard-form rewrite only touches the constraints and the linear objective, so it is
        // built from the quadratic program's linear half and the Hessian is projected separately.
        var linearHalf = new LinearProgram<T>(
            program.Linear,
            program.InequalityMatrix, program.InequalityBounds,
            program.EqualityMatrix, program.EqualityBounds,
            program.LowerBounds, program.UpperBounds);

        var standard = LinearProgramStandardForm<T>.Build(
            linearHalf, treatMissingLowerBoundsAsUnbounded: true);
        var problem = BuildProblem(standard, program.Quadratic);
        var outcome = Iterate(problem);

        Vector<T>? solution = null;
        T objectiveValue = NumOps.Zero;

        if (outcome.Point is not null)
        {
            solution = standard.RecoverOriginalVariables(outcome.Point);

            // 0.5 xᵀQx + cᵀx, evaluated on the caller's own data rather than on the rewritten
            // problem, so none of the rewrite's constants can leak into the reported value.
            T quadraticTerm = solution.DotProduct(Multiply(program.Quadratic, solution));
            objectiveValue = NumOps.Add(
                NumOps.Multiply(NumOps.FromDouble(0.5), quadraticTerm),
                program.Linear.DotProduct(solution));
        }

        var (inequalityDuals, equalityDuals) = SplitDuals(standard, outcome);

        return new QuadraticProgramSolution<T>(
            outcome.Status, solution, objectiveValue, outcome.Iterations,
            inequalityDuals, equalityDuals);
    }

    /// <summary>
    /// The rewritten problem the iteration works on: minimize <c>½zᵀQz + cᵀz</c> subject to
    /// <c>Az = b</c> and <c>z ≥ 0</c>.
    /// </summary>
    private sealed class Problem
    {
        public Matrix<T> A { get; }
        public Vector<T> B { get; }
        public Vector<T> C { get; }

        /// <summary>The projected Hessian, or <c>null</c> for a linear program.</summary>
        public Matrix<T>? Q { get; }

        public Problem(Matrix<T> a, Vector<T> b, Vector<T> c, Matrix<T>? q)
        {
            A = a;
            B = b;
            C = c;
            Q = q;
        }

        public int RowCount => A.Rows;

        public int ColumnCount => A.Columns;
    }

    /// <summary>
    /// The raw result of the iteration, in rewritten coordinates.
    /// </summary>
    private sealed class Outcome
    {
        public LinearProgramStatus Status { get; }

        /// <summary>The primal point, or <c>null</c> when there is no point to report.</summary>
        public Vector<T>? Point { get; }

        /// <summary>The dual values of the rewritten rows, or <c>null</c>.</summary>
        public Vector<T>? RowDuals { get; }

        public int Iterations { get; }

        public Outcome(
            LinearProgramStatus status, Vector<T>? point, Vector<T>? rowDuals, int iterations)
        {
            Status = status;
            Point = point;
            RowDuals = rowDuals;
            Iterations = iterations;
        }
    }

    /// <summary>
    /// Turns the standard form into equality-only constraints by giving every inequality row its own
    /// slack variable, and projects the Hessian when one is present.
    /// </summary>
    private static Problem BuildProblem(
        LinearProgramStandardForm<T> standard, Matrix<T>? quadratic)
    {
        int rowCount = standard.Rows.Count;
        int structuralCount = standard.VariableCount;

        int slackCount = 0;
        for (int r = 0; r < rowCount; r++)
        {
            if (!standard.IsEquality[r]) slackCount++;
        }

        int columnCount = structuralCount + slackCount;

        var a = new Matrix<T>(rowCount, columnCount);
        var b = new Vector<T>(rowCount);

        int slackCursor = structuralCount;
        for (int r = 0; r < rowCount; r++)
        {
            var row = standard.Rows[r];
            for (int c = 0; c < structuralCount; c++) a[r, c] = row[c];

            b[r] = standard.RightHandSides[r];

            if (!standard.IsEquality[r])
            {
                // A row still reading "<=" absorbs its slack with +1; one the rewrite negated into
                // ">=" needs -1.
                a[r, slackCursor++] = standard.RowWasNegated[r]
                    ? NumOps.Negate(NumOps.One)
                    : NumOps.One;
            }
        }

        var objective = new Vector<T>(columnCount);
        for (int i = 0; i < structuralCount; i++) objective[i] = standard.Objective[i];

        Matrix<T>? projectedQuadratic = null;
        if (quadratic is not null)
        {
            var (projected, linearCorrection, _) = standard.ProjectQuadratic(quadratic);

            // The rewrite's constant term is dropped: it shifts the objective without moving the
            // minimizer, and the reported objective is evaluated on the caller's own data anyway.
            // The slack variables extend the Hessian with zero rows and columns — they appear in no
            // quadratic term.
            projectedQuadratic = new Matrix<T>(columnCount, columnCount);
            for (int i = 0; i < structuralCount; i++)
            {
                for (int j = 0; j < structuralCount; j++)
                {
                    projectedQuadratic[i, j] = projected[i, j];
                }

                objective[i] = NumOps.Add(objective[i], linearCorrection[i]);
            }
        }

        return new Problem(a, b, objective, projectedQuadratic);
    }

    /// <summary>
    /// Maps the rewritten rows' dual values back onto the caller's inequality and equality blocks.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The rewrite negated any row whose right-hand side was negative, which negates that row's dual
    /// along with it; flipping the sign back restores the caller's convention, under which the dual
    /// values satisfy strong duality against the caller's own right-hand sides. Rows the rewrite
    /// added to encode a finite upper bound have no counterpart to report and are dropped.
    /// </para>
    /// </remarks>
    private static (Vector<T>? Inequality, Vector<T>? Equality) SplitDuals(
        LinearProgramStandardForm<T> standard, Outcome outcome)
    {
        if (outcome.RowDuals is null || outcome.Status != LinearProgramStatus.Optimal)
        {
            return (null, null);
        }

        var duals = outcome.RowDuals;

        Vector<T>? inequality = null;
        if (standard.InequalityRowCount > 0)
        {
            inequality = new Vector<T>(standard.InequalityRowCount);
            for (int r = 0; r < standard.InequalityRowCount; r++)
            {
                inequality[r] = standard.RowWasNegated[r] ? NumOps.Negate(duals[r]) : duals[r];
            }
        }

        Vector<T>? equality = null;
        if (standard.EqualityRowCount > 0)
        {
            equality = new Vector<T>(standard.EqualityRowCount);
            for (int r = 0; r < standard.EqualityRowCount; r++)
            {
                int row = standard.InequalityRowCount + r;
                equality[r] = standard.RowWasNegated[row] ? NumOps.Negate(duals[row]) : duals[row];
            }
        }

        return (inequality, equality);
    }

    /// <summary>
    /// Runs the predictor-corrector iteration.
    /// </summary>
    private Outcome Iterate(Problem problem)
    {
        int m = problem.RowCount;
        int n = problem.ColumnCount;

        if (m == 0) return SolveWithoutConstraints(problem);

        var (x, y, s, rangeResidual) = ComputeStartingPoint(problem);

        // Rows that contradict each other as a plain linear system — b outside the range of A — make
        // the problem infeasible before non-negativity is even considered, and the least-squares
        // residual of Az = b is itself the Farkas certificate: it satisfies Aᵀr = 0 and bᵀr = ‖r‖².
        // Waiting for the iterates to diverge along that direction would take far longer and, on a
        // singular system the regularization has made solvable, may not happen at all.
        if (rangeResidual is not null && IsPrimalInfeasibilityCertificate(problem, rangeResidual))
        {
            return new Outcome(LinearProgramStatus.Infeasible, null, null, 0);
        }

        double tolerance = _options.Tolerance;
        double primalScale = 1.0 + Norm(problem.B);
        double dualScale = 1.0 + Norm(problem.C);

        for (int iteration = 1; iteration <= _options.MaxIterations; iteration++)
        {
            // rp = b - Az
            var primalResidual = Subtract(problem.B, Multiply(problem.A, x));

            // rd = c + Qz - Aᵀy - s
            var dualResidual = Subtract(problem.C, Add(TransposeMultiply(problem.A, y), s));
            if (problem.Q is not null)
            {
                dualResidual = Add(dualResidual, Multiply(problem.Q, x));
            }

            double complementarity = ToDouble(x.DotProduct(s)) / n;

            double primalObjective = ToDouble(problem.C.DotProduct(x));
            if (problem.Q is not null)
            {
                primalObjective += 0.5 * ToDouble(x.DotProduct(Multiply(problem.Q, x)));
            }

            bool converged =
                Norm(primalResidual) / primalScale <= tolerance &&
                Norm(dualResidual) / dualScale <= tolerance &&
                complementarity / (1.0 + Math.Abs(primalObjective)) <= tolerance;

            if (converged)
            {
                return new Outcome(LinearProgramStatus.Optimal, x, y, iteration);
            }

            var certificate = CheckCertificates(problem, x, y);
            if (certificate is not null)
            {
                return new Outcome(certificate.Value, null, null, iteration);
            }

            var newton = NewtonSystem.Build(problem, x, s, _options.Regularization);
            if (newton is null)
            {
                // A factorization that fails even with regularization will not start succeeding on
                // the next iteration, so the best available point is returned as-is.
                return new Outcome(LinearProgramStatus.IterationLimit, x, y, iteration);
            }

            // --- Predictor: aim straight at complementarity zero and see how far that would get ---
            var affineTarget = new Vector<T>(n);
            for (int i = 0; i < n; i++)
            {
                affineTarget[i] = NumOps.Negate(NumOps.Multiply(x[i], s[i]));
            }

            var affine = newton.SolveDirection(
                problem, x, s, primalResidual, dualResidual, affineTarget);

            double affinePrimalStep = MaxStep(x, affine.Dx);
            double affineDualStep = MaxStep(s, affine.Ds);

            double affineComplementarity = 0.0;
            for (int i = 0; i < n; i++)
            {
                double xi = ToDouble(x[i]) + affinePrimalStep * ToDouble(affine.Dx[i]);
                double si = ToDouble(s[i]) + affineDualStep * ToDouble(affine.Ds[i]);
                affineComplementarity += xi * si;
            }

            affineComplementarity /= n;

            // Mehrotra's adaptive centering: when the aggressive step would have worked well, trust
            // it and barely center at all; when it would have stalled, pull hard back onto the
            // central path.
            double sigma = complementarity > 0.0
                ? Math.Pow(Math.Max(affineComplementarity, 0.0) / complementarity, 3.0)
                : 0.0;
            sigma = Math.Min(1.0, Math.Max(0.0, sigma));

            // --- Corrector: re-aim at sigma*mu, compensating for the predictor's own curvature ---
            T centeringTarget = NumOps.FromDouble(sigma * complementarity);
            var correctorTarget = new Vector<T>(n);
            for (int i = 0; i < n; i++)
            {
                correctorTarget[i] = NumOps.Subtract(
                    NumOps.Subtract(centeringTarget, NumOps.Multiply(x[i], s[i])),
                    NumOps.Multiply(affine.Dx[i], affine.Ds[i]));
            }

            var step = newton.SolveDirection(
                problem, x, s, primalResidual, dualResidual, correctorTarget);

            double primalStep = _options.FractionToBoundary * MaxStep(x, step.Dx);
            double dualStep = _options.FractionToBoundary * MaxStep(s, step.Ds);

            if (problem.Q is not null)
            {
                // With a Hessian present the dual residual couples z and y, so stepping them by
                // different amounts leaves a residual the next iteration cannot account for. Linear
                // programs have no such coupling and keep the two step lengths independent, which is
                // a real speedup rather than a detail.
                double common = Math.Min(primalStep, dualStep);
                primalStep = common;
                dualStep = common;
            }

            if (primalStep <= 0.0 && dualStep <= 0.0)
            {
                return new Outcome(LinearProgramStatus.IterationLimit, x, y, iteration);
            }

            x = Advance(x, step.Dx, primalStep);
            y = Advance(y, step.Dy, dualStep);
            s = Advance(s, step.Ds, dualStep);
        }

        return new Outcome(LinearProgramStatus.IterationLimit, x, y, _options.MaxIterations);
    }

    /// <summary>
    /// Handles the degenerate case of a problem whose only constraints are <c>z ≥ 0</c>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Without rows there is no linear system to factor and no iteration to run. A linear objective
    /// with any negative coefficient runs to minus infinity along that axis; otherwise the origin is
    /// optimal. A quadratic objective is passed to the general path with a single all-zero row, which
    /// constrains nothing but gives the iteration the shape it is written for.
    /// </para>
    /// </remarks>
    private Outcome SolveWithoutConstraints(Problem problem)
    {
        int n = problem.ColumnCount;

        if (problem.Q is null)
        {
            for (int i = 0; i < n; i++)
            {
                if (NumOps.LessThan(problem.C[i], NumOps.Zero))
                {
                    return new Outcome(LinearProgramStatus.Unbounded, null, null, 0);
                }
            }

            return new Outcome(LinearProgramStatus.Optimal, new Vector<T>(n), null, 0);
        }

        var padded = new Problem(
            new Matrix<T>(1, n), new Vector<T>(1), problem.C, problem.Q);

        var outcome = Iterate(padded);
        return new Outcome(outcome.Status, outcome.Point, null, outcome.Iterations);
    }

    /// <summary>
    /// Tests the normalized iterates against the Farkas certificates for infeasibility and
    /// unboundedness.
    /// </summary>
    /// <returns>The proved status, or <c>null</c> when neither certificate holds.</returns>
    private LinearProgramStatus? CheckCertificates(Problem problem, Vector<T> x, Vector<T> y)
    {
        double certificateTolerance = _options.CertificateTolerance;

        if (IsPrimalInfeasibilityCertificate(problem, y))
        {
            return LinearProgramStatus.Infeasible;
        }

        // Unbounded: a ray d >= 0 with Ad = 0, Qd = 0 and cᵀd < 0 drives the objective to minus
        // infinity. The primal iterate diverges along exactly such a ray when one exists.
        double normX = Norm(x);
        if (normX > 0.0)
        {
            double descent = ToDouble(problem.C.DotProduct(x)) / normX;
            if (descent < -certificateTolerance)
            {
                double rayResidual = Norm(Multiply(problem.A, x)) / normX;

                // A convex quadratic only runs away along a direction of zero curvature; with any
                // curvature the quadratic term eventually dominates the linear descent.
                double curvature = problem.Q is null
                    ? 0.0
                    : Norm(Multiply(problem.Q, x)) / normX;

                if (rayResidual <= certificateTolerance && curvature <= certificateTolerance)
                {
                    return LinearProgramStatus.Unbounded;
                }
            }
        }

        return null;
    }

    /// <summary>
    /// Verifies that <paramref name="candidate"/> proves the constraints have no non-negative
    /// solution.
    /// </summary>
    /// <remarks>
    /// <para>
    /// By Farkas' lemma, exactly one of these holds: some <c>z ≥ 0</c> satisfies <c>Az = b</c>, or
    /// some <c>y</c> satisfies <c>Aᵀy ≤ 0</c> and <c>bᵀy &gt; 0</c>. A vector meeting the second pair
    /// of conditions therefore rules out the first outright — no iteration count or convergence
    /// argument is involved. Both conditions are checked against the candidate's own magnitude, so
    /// the test is invariant to how far the iterate has diverged.
    /// </para>
    /// </remarks>
    private bool IsPrimalInfeasibilityCertificate(Problem problem, Vector<T> candidate)
    {
        double magnitude = Norm(candidate);
        if (magnitude <= 0.0) return false;

        var transposed = TransposeMultiply(problem.A, candidate);

        double worstViolation = 0.0;
        for (int i = 0; i < problem.ColumnCount; i++)
        {
            worstViolation = Math.Max(worstViolation, ToDouble(transposed[i]) / magnitude);
        }

        double alignment = ToDouble(problem.B.DotProduct(candidate)) / magnitude;

        return worstViolation <= _options.CertificateTolerance
            && alignment > _options.CertificateTolerance;
    }

    /// <summary>
    /// The factored Newton system for one iteration, reused by the predictor and the corrector.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Eliminating <c>ds</c> and then <c>dz</c> from the Newton equations leaves
    /// <c>(A K⁻¹ Aᵀ) dy = rp − A K⁻¹ (Z⁻¹ rc − rd)</c> with <c>K = Q + Z⁻¹S</c>. Only the right-hand
    /// side depends on the complementarity target, so the predictor and the corrector reuse the same
    /// factorizations — which is exactly what makes Mehrotra's method cost barely more per iteration
    /// than a plain path-following step, despite solving the system twice.
    /// </para>
    /// </remarks>
    private sealed class NewtonSystem
    {
        /// <summary>Diagonal of <c>K⁻¹</c> for a linear program; <c>null</c> for a quadratic one.</summary>
        private readonly Vector<T>? _diagonalInverse;

        /// <summary>Factorization of the dense <c>K</c> for a quadratic program.</summary>
        private readonly LuDecomposition<T>? _kernel;

        /// <summary>Factorization of the normal-equations matrix <c>A K⁻¹ Aᵀ</c>.</summary>
        private readonly LuDecomposition<T> _normal;

        private NewtonSystem(
            Vector<T>? diagonalInverse, LuDecomposition<T>? kernel, LuDecomposition<T> normal)
        {
            _diagonalInverse = diagonalInverse;
            _kernel = kernel;
            _normal = normal;
        }

        /// <summary>
        /// Factors the system at the current iterate, or returns <c>null</c> when it cannot be
        /// factored.
        /// </summary>
        public static NewtonSystem? Build(
            Problem problem, Vector<T> x, Vector<T> s, double regularization)
        {
            int n = problem.ColumnCount;
            int m = problem.RowCount;

            Vector<T>? diagonalInverse = null;
            LuDecomposition<T>? kernel = null;

            // Row r of this matrix holds K⁻¹ applied to row r of A — that is, the columns of K⁻¹Aᵀ,
            // which is everything the normal-equations product needs.
            var kernelInverseTransposed = new Matrix<T>(m, n);

            if (problem.Q is null)
            {
                // K = Z⁻¹S is diagonal, so K⁻¹ has entries z_i / s_i.
                diagonalInverse = new Vector<T>(n);
                for (int i = 0; i < n; i++)
                {
                    diagonalInverse[i] = NumOps.Divide(x[i], s[i]);
                }

                for (int r = 0; r < m; r++)
                {
                    for (int i = 0; i < n; i++)
                    {
                        kernelInverseTransposed[r, i] =
                            NumOps.Multiply(problem.A[r, i], diagonalInverse[i]);
                    }
                }
            }
            else
            {
                var k = new Matrix<T>(n, n);
                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j < n; j++) k[i, j] = problem.Q[i, j];

                    k[i, i] = NumOps.Add(k[i, i], NumOps.Divide(s[i], x[i]));
                }

                try
                {
                    kernel = new LuDecomposition<T>(k);
                }
                catch (MatrixFactorizationException)
                {
                    return null;
                }

                for (int r = 0; r < m; r++)
                {
                    var row = new Vector<T>(n);
                    for (int i = 0; i < n; i++) row[i] = problem.A[r, i];

                    var solved = kernel.Solve(row);
                    for (int i = 0; i < n; i++) kernelInverseTransposed[r, i] = solved[i];
                }
            }

            var normalMatrix = new Matrix<T>(m, m);
            T delta = NumOps.FromDouble(regularization);

            for (int r = 0; r < m; r++)
            {
                for (int c = 0; c <= r; c++)
                {
                    T accumulator = NumOps.Zero;
                    for (int i = 0; i < n; i++)
                    {
                        accumulator = NumOps.Add(
                            accumulator,
                            NumOps.Multiply(kernelInverseTransposed[r, i], problem.A[c, i]));
                    }

                    // A K⁻¹ Aᵀ is symmetric because K is, so only half of it is computed.
                    normalMatrix[r, c] = accumulator;
                    normalMatrix[c, r] = accumulator;
                }

                // Redundant rows make this matrix singular, and a redundant row is not a user error
                // — the standard-form rewrite can introduce one on its own.
                normalMatrix[r, r] = NumOps.Add(normalMatrix[r, r], delta);
            }

            try
            {
                return new NewtonSystem(
                    diagonalInverse, kernel, new LuDecomposition<T>(normalMatrix));
            }
            catch (MatrixFactorizationException)
            {
                return null;
            }
        }

        /// <summary>
        /// Solves the Newton system for one complementarity target.
        /// </summary>
        /// <param name="problem">The rewritten problem.</param>
        /// <param name="x">The current primal iterate.</param>
        /// <param name="s">The current dual slack iterate.</param>
        /// <param name="primalResidual">The residual <c>b − Az</c>.</param>
        /// <param name="dualResidual">The residual <c>c + Qz − Aᵀy − s</c>.</param>
        /// <param name="complementarityTarget">
        /// The right-hand side <c>rc</c> of <c>S dz + Z ds = rc</c>.
        /// </param>
        public (Vector<T> Dx, Vector<T> Dy, Vector<T> Ds) SolveDirection(
            Problem problem,
            Vector<T> x,
            Vector<T> s,
            Vector<T> primalResidual,
            Vector<T> dualResidual,
            Vector<T> complementarityTarget)
        {
            int n = problem.ColumnCount;

            // inner = Z⁻¹ rc − rd
            var inner = new Vector<T>(n);
            for (int i = 0; i < n; i++)
            {
                inner[i] = NumOps.Subtract(
                    NumOps.Divide(complementarityTarget[i], x[i]), dualResidual[i]);
            }

            var w = ApplyKernelInverse(inner);

            // dy solves (A K⁻¹ Aᵀ) dy = rp − A w
            var rightHandSide = Subtract(primalResidual, Multiply(problem.A, w));
            var dy = _normal.Solve(rightHandSide);

            // dz = K⁻¹ Aᵀ dy + w
            var dx = Add(ApplyKernelInverse(TransposeMultiply(problem.A, dy)), w);

            // ds = Z⁻¹ (rc − S dz), the equation dz was eliminated from
            var ds = new Vector<T>(n);
            for (int i = 0; i < n; i++)
            {
                ds[i] = NumOps.Divide(
                    NumOps.Subtract(complementarityTarget[i], NumOps.Multiply(s[i], dx[i])),
                    x[i]);
            }

            return (dx, dy, ds);
        }

        private Vector<T> ApplyKernelInverse(Vector<T> vector)
        {
            if (_diagonalInverse is null) return _kernel!.Solve(vector);

            var result = new Vector<T>(vector.Length);
            for (int i = 0; i < vector.Length; i++)
            {
                result[i] = NumOps.Multiply(_diagonalInverse[i], vector[i]);
            }

            return result;
        }
    }

    /// <summary>
    /// Builds the starting point from Mehrotra's heuristic (Nocedal and Wright, Section 14.2).
    /// </summary>
    /// <remarks>
    /// <para>
    /// The heuristic first takes the least-squares point satisfying the equality constraints and the
    /// dual equation while ignoring non-negativity, then shifts both <c>z</c> and <c>s</c> far enough
    /// into the positive orthant that neither starts near the boundary. A starting point hugging the
    /// boundary is the single most common cause of an interior-point method stalling, which is why
    /// this is worth two extra factorizations before the first iteration.
    /// </para>
    /// </remarks>
    /// <returns>
    /// The starting iterate, plus the least-squares residual of <c>Az = b</c> — which is a candidate
    /// infeasibility certificate — or <c>null</c> when the Gram matrix could not be factored.
    /// </returns>
    private (Vector<T> X, Vector<T> Y, Vector<T> S, Vector<T>? RangeResidual) ComputeStartingPoint(
        Problem problem)
    {
        int m = problem.RowCount;
        int n = problem.ColumnCount;

        var gram = new Matrix<T>(m, m);
        T delta = NumOps.FromDouble(Math.Max(_options.Regularization, 1e-12));

        for (int r = 0; r < m; r++)
        {
            for (int c = 0; c <= r; c++)
            {
                T accumulator = NumOps.Zero;
                for (int i = 0; i < n; i++)
                {
                    accumulator = NumOps.Add(
                        accumulator, NumOps.Multiply(problem.A[r, i], problem.A[c, i]));
                }

                gram[r, c] = accumulator;
                gram[c, r] = accumulator;
            }

            gram[r, r] = NumOps.Add(gram[r, r], delta);
        }

        Vector<T> leastSquaresPrimal;
        Vector<T> y;
        Vector<T> leastSquaresSlack;
        Vector<T> rangeResidual;

        try
        {
            var factored = new LuDecomposition<T>(gram);
            leastSquaresPrimal = TransposeMultiply(problem.A, factored.Solve(problem.B));
            y = factored.Solve(Multiply(problem.A, problem.C));
            leastSquaresSlack = Subtract(problem.C, TransposeMultiply(problem.A, y));

            // What the equality constraints cannot reach even ignoring non-negativity.
            rangeResidual = Subtract(problem.B, Multiply(problem.A, leastSquaresPrimal));
        }
        catch (MatrixFactorizationException)
        {
            // A Gram matrix too ill-conditioned to factor still leaves a usable neutral start; the
            // iteration recovers from a poor start, just more slowly.
            var neutralPrimal = new Vector<T>(n);
            var neutralSlack = new Vector<T>(n);
            for (int i = 0; i < n; i++)
            {
                neutralPrimal[i] = NumOps.One;
                neutralSlack[i] = NumOps.One;
            }

            return (neutralPrimal, new Vector<T>(m), neutralSlack, null);
        }

        double primalShift = Math.Max(-1.5 * Minimum(leastSquaresPrimal), 0.0);
        double dualShift = Math.Max(-1.5 * Minimum(leastSquaresSlack), 0.0);

        double primalSum = 0.0;
        double dualSum = 0.0;
        double product = 0.0;

        for (int i = 0; i < n; i++)
        {
            double shiftedPrimal = ToDouble(leastSquaresPrimal[i]) + primalShift;
            double shiftedSlack = ToDouble(leastSquaresSlack[i]) + dualShift;
            primalSum += shiftedPrimal;
            dualSum += shiftedSlack;
            product += shiftedPrimal * shiftedSlack;
        }

        double primalCorrection = primalShift + (dualSum > 0.0 ? 0.5 * product / dualSum : 1.0);
        double dualCorrection = dualShift + (primalSum > 0.0 ? 0.5 * product / primalSum : 1.0);

        var startPrimal = new Vector<T>(n);
        var startSlack = new Vector<T>(n);
        for (int i = 0; i < n; i++)
        {
            // Both must end strictly positive: a problem whose least-squares point is already
            // balanced can otherwise produce a correction of zero, and zero is the one value the
            // iteration cannot divide by.
            startPrimal[i] = NumOps.FromDouble(
                Math.Max(ToDouble(leastSquaresPrimal[i]) + primalCorrection, 1e-6));
            startSlack[i] = NumOps.FromDouble(
                Math.Max(ToDouble(leastSquaresSlack[i]) + dualCorrection, 1e-6));
        }

        return (startPrimal, y, startSlack, rangeResidual);
    }

    /// <summary>
    /// Returns the largest step in <c>[0, 1]</c> that keeps every entry of
    /// <paramref name="current"/> non-negative when moved along <paramref name="direction"/>.
    /// </summary>
    private static double MaxStep(Vector<T> current, Vector<T> direction)
    {
        double step = 1.0;
        for (int i = 0; i < current.Length; i++)
        {
            double rate = ToDouble(direction[i]);
            if (rate >= 0.0) continue;

            double limit = -ToDouble(current[i]) / rate;
            if (limit < step) step = limit;
        }

        return Math.Max(step, 0.0);
    }

    private static Vector<T> Advance(Vector<T> current, Vector<T> direction, double step)
    {
        var result = new Vector<T>(current.Length);
        T scaled = NumOps.FromDouble(step);
        for (int i = 0; i < current.Length; i++)
        {
            result[i] = NumOps.Add(current[i], NumOps.Multiply(scaled, direction[i]));
        }

        return result;
    }

    private static Vector<T> Multiply(Matrix<T> matrix, Vector<T> vector)
    {
        var result = new Vector<T>(matrix.Rows);
        for (int r = 0; r < matrix.Rows; r++)
        {
            T accumulator = NumOps.Zero;
            for (int c = 0; c < matrix.Columns; c++)
            {
                accumulator = NumOps.Add(accumulator, NumOps.Multiply(matrix[r, c], vector[c]));
            }

            result[r] = accumulator;
        }

        return result;
    }

    private static Vector<T> TransposeMultiply(Matrix<T> matrix, Vector<T> vector)
    {
        var result = new Vector<T>(matrix.Columns);
        for (int c = 0; c < matrix.Columns; c++)
        {
            T accumulator = NumOps.Zero;
            for (int r = 0; r < matrix.Rows; r++)
            {
                accumulator = NumOps.Add(accumulator, NumOps.Multiply(matrix[r, c], vector[r]));
            }

            result[c] = accumulator;
        }

        return result;
    }

    private static Vector<T> Add(Vector<T> left, Vector<T> right)
    {
        var result = new Vector<T>(left.Length);
        for (int i = 0; i < left.Length; i++) result[i] = NumOps.Add(left[i], right[i]);
        return result;
    }

    private static Vector<T> Subtract(Vector<T> left, Vector<T> right)
    {
        var result = new Vector<T>(left.Length);
        for (int i = 0; i < left.Length; i++) result[i] = NumOps.Subtract(left[i], right[i]);
        return result;
    }

    private static double Minimum(Vector<T> vector)
    {
        double smallest = double.PositiveInfinity;
        for (int i = 0; i < vector.Length; i++)
        {
            smallest = Math.Min(smallest, ToDouble(vector[i]));
        }

        return double.IsPositiveInfinity(smallest) ? 0.0 : smallest;
    }

    private static double Norm(Vector<T> vector)
    {
        double total = 0.0;
        for (int i = 0; i < vector.Length; i++)
        {
            double value = ToDouble(vector[i]);
            total += value * value;
        }

        return Math.Sqrt(total);
    }

    private static double ToDouble(T value) => NumOps.ToDouble(value);
}
