using AiDotNet.DecompositionMethods.MatrixDecomposition;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Solvers.LinearProgramming;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Solvers.QuadraticProgramming;

/// <summary>
/// Solves convex quadratic programs with a primal active-set method.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Implements the primal active-set method for convex quadratic programming (Nocedal and Wright,
/// "Numerical Optimization", Algorithm 16.3). The insight it exploits is that if you already knew
/// which inequality constraints press against the optimum, the problem would collapse to an
/// equality-constrained quadratic program — and that has a closed-form answer, obtained by solving
/// one linear KKT system. The method therefore guesses that set, solves, and uses the answer to
/// correct the guess:
/// <list type="bullet">
/// <item>if the equality-constrained step is blocked by a constraint outside the working set, that
/// constraint is added and the step truncated at the blocking point;</item>
/// <item>if the step is zero, the point solves the current subproblem; the Lagrange multipliers
/// then say whether every working constraint deserves to be there. A negative multiplier means
/// the objective would improve by moving off that constraint, so it is dropped;</item>
/// <item>when the step is zero and no multiplier is negative, the KKT conditions hold and the point
/// is optimal.</item>
/// </list>
/// </para>
/// <para>
/// The method needs a feasible starting point, and finding one is itself a linear-programming
/// feasibility problem — so a <see cref="SimplexSolver{T}"/> supplies it before the quadratic phase
/// begins. That reuse is deliberate: the phase-one machinery already exists and is already tested.
/// </para>
/// <para><b>For Beginners:</b> Imagine rolling a ball into a bowl that has walls. Without walls the
/// ball settles at the bottom, which is a one-step calculation. With walls it rolls until it hits
/// one, then rolls along that wall, possibly into a corner. The method simulates exactly that: roll,
/// hit a wall, add it to the list of walls you are touching, roll along them, and — importantly —
/// check whether any wall you are pressed against is one you could leave to get lower still.
/// </para>
/// </remarks>
public sealed class ActiveSetQuadraticProgramSolver<T> : IQuadraticProgramSolver<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly ActiveSetQuadraticProgramSolverOptions _options;
    private readonly ILinearProgramSolver<T> _feasibilitySolver;

    /// <summary>
    /// Creates an active-set quadratic-programming solver.
    /// </summary>
    /// <param name="options">
    /// Solver configuration. When omitted, the documented defaults on
    /// <see cref="ActiveSetQuadraticProgramSolverOptions"/> are used.
    /// </param>
    /// <param name="feasibilitySolver">
    /// The linear-programming solver used to find an initial feasible point. When omitted, a
    /// <see cref="SimplexSolver{T}"/> configured from
    /// <see cref="ActiveSetQuadraticProgramSolverOptions.FeasibilityOptions"/> is used.
    /// </param>
    public ActiveSetQuadraticProgramSolver(
        ActiveSetQuadraticProgramSolverOptions? options = null,
        ILinearProgramSolver<T>? feasibilitySolver = null)
    {
        _options = options ?? new ActiveSetQuadraticProgramSolverOptions();
        _feasibilitySolver = feasibilitySolver ?? new SimplexSolver<T>(_options.FeasibilityOptions);
    }

    /// <inheritdoc />
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="program"/> is null.</exception>
    public QuadraticProgramSolution<T> Solve(QuadraticProgram<T> program)
    {
        if (program is null) throw new ArgumentNullException(nameof(program));

        var tolerance = NumOps.FromDouble(_options.Tolerance);
        int variableCount = program.VariableCount;

        // Bounds are folded into the inequality rows so the working-set bookkeeping has a single
        // uniform kind of constraint to reason about.
        var (inequalityRows, inequalityBounds, boundRowOrigin) = FlattenConstraints(program);
        int inequalityCount = inequalityBounds.Length;
        int equalityCount = program.EqualityBounds?.Length ?? 0;

        var (feasibilityStatus, start) = FindFeasiblePoint(program, inequalityRows, inequalityBounds);
        if (start is null)
        {
            return new QuadraticProgramSolution<T>(
                feasibilityStatus, null, NumOps.Zero, 0);
        }

        var x = start;

        // The working set always contains every equality constraint; inequalities join and leave.
        var workingSet = new List<int>();
        for (int i = 0; i < inequalityCount; i++)
        {
            if (IsActive(inequalityRows, inequalityBounds, i, x, tolerance)) workingSet.Add(i);
        }

        var inequalityMultipliers = new Vector<T>(inequalityCount);
        var equalityMultipliers = new Vector<T>(Math.Max(equalityCount, 1));
        int iterations = 0;

        for (; iterations < _options.MaxIterations; iterations++)
        {
            // Gradient of ½xᵀQx + cᵀx at the current point.
            var gradient = (Vector<T>)Engine.Add(MultiplyMatrixVector(program.Quadratic, x), program.Linear);

            var kkt = SolveEqualityConstrainedStep(
                program, inequalityRows, workingSet, gradient, equalityCount, variableCount);

            if (kkt is null)
            {
                throw new InvalidOperationException(
                    "The active-set KKT system remained singular after regularization; " +
                    "no numerically certified quadratic-program step is available.");
            }

            var (step, multipliers) = kkt.Value;

            if (IsEffectivelyZero(step, tolerance))
            {
                // The point solves the equality-constrained subproblem. Check whether any working
                // inequality wants to be released.
                int mostNegative = -1;
                T mostNegativeValue = NumOps.Negate(tolerance);

                for (int k = 0; k < workingSet.Count; k++)
                {
                    T multiplier = multipliers[equalityCount + k];
                    if (NumOps.LessThan(multiplier, mostNegativeValue))
                    {
                        mostNegativeValue = multiplier;
                        mostNegative = k;
                    }
                }

                if (mostNegative < 0)
                {
                    // Every multiplier is non-negative: the KKT conditions hold and x is optimal.
                    for (int i = 0; i < inequalityCount; i++) inequalityMultipliers[i] = NumOps.Zero;
                    for (int k = 0; k < workingSet.Count; k++)
                    {
                        inequalityMultipliers[workingSet[k]] = multipliers[equalityCount + k];
                    }

                    for (int e = 0; e < equalityCount; e++) equalityMultipliers[e] = multipliers[e];

                    return Success(
                        program, x, iterations, inequalityCount, equalityCount,
                        inequalityMultipliers, equalityMultipliers, boundRowOrigin);
                }

                // Dropping the constraint with the most negative multiplier gives the steepest
                // first-order improvement among the available releases.
                workingSet.RemoveAt(mostNegative);
                continue;
            }

            // Take the largest step along the direction that keeps every constraint satisfied.
            var (stepLength, blockingConstraint) = ComputeStepLength(
                inequalityRows, inequalityBounds, workingSet, x, step, tolerance);

            x = (Vector<T>)Engine.Add(x, (Vector<T>)Engine.Multiply(step, stepLength));

            if (blockingConstraint >= 0)
            {
                workingSet.Add(blockingConstraint);
            }
        }

        // The iteration budget ran out. The point is feasible but its optimality is not certified.
        return new QuadraticProgramSolution<T>(
            LinearProgramStatus.IterationLimit, x, EvaluateObjective(program, x), iterations);
    }

    private static IEngine Engine => AiDotNet.Tensors.Engines.AiDotNetEngine.Current;

    /// <summary>
    /// Builds the complete list of inequality rows, folding finite variable bounds into ordinary
    /// <c>aᵀx ≤ b</c> rows so the active-set bookkeeping handles one uniform constraint type.
    /// </summary>
    /// <returns>
    /// The rows, their right-hand sides, and for each row the index of the original inequality it
    /// came from (or -1 when it came from a bound, whose multiplier the caller does not report).
    /// </returns>
    private static (List<Vector<T>> Rows, Vector<T> Bounds, int[] Origin) FlattenConstraints(
        QuadraticProgram<T> program)
    {
        int variableCount = program.VariableCount;
        var rows = new List<Vector<T>>();
        var bounds = new List<T>();
        var origin = new List<int>();

        int originalCount = program.InequalityBounds?.Length ?? 0;
        for (int r = 0; r < originalCount; r++)
        {
            var row = new Vector<T>(variableCount);
            for (int c = 0; c < variableCount; c++) row[c] = program.InequalityMatrix![r, c];
            rows.Add(row);
            bounds.Add(program.InequalityBounds![r]);
            origin.Add(r);
        }

        for (int i = 0; i < variableCount; i++)
        {
            if (program.UpperBounds is not null && IsFinite(program.UpperBounds[i]))
            {
                var row = new Vector<T>(variableCount);
                row[i] = NumOps.One;                          // x_i <= upper_i
                rows.Add(row);
                bounds.Add(program.UpperBounds[i]);
                origin.Add(-1);
            }

            if (program.LowerBounds is not null && IsFinite(program.LowerBounds[i]))
            {
                var row = new Vector<T>(variableCount);
                row[i] = NumOps.Negate(NumOps.One);           // -x_i <= -lower_i
                rows.Add(row);
                bounds.Add(NumOps.Negate(program.LowerBounds[i]));
                origin.Add(-1);
            }
        }

        var boundsVector = new Vector<T>(bounds.Count);
        for (int i = 0; i < bounds.Count; i++) boundsVector[i] = bounds[i];

        return (rows, boundsVector, origin.ToArray());
    }

    /// <summary>
    /// Finds any point satisfying every constraint, by solving the feasibility problem as a linear
    /// program with a zero objective.
    /// </summary>
    private (LinearProgramStatus Status, Vector<T>? Point) FindFeasiblePoint(
        QuadraticProgram<T> program, List<Vector<T>> inequalityRows, Vector<T> inequalityBounds)
    {
        int variableCount = program.VariableCount;

        var zeroObjective = new Vector<T>(variableCount);
        Matrix<T>? inequalityMatrix = null;
        if (inequalityRows.Count > 0)
        {
            inequalityMatrix = new Matrix<T>(inequalityRows.Count, variableCount);
            for (int r = 0; r < inequalityRows.Count; r++)
            {
                for (int c = 0; c < variableCount; c++) inequalityMatrix[r, c] = inequalityRows[r][c];
            }
        }

        // The bounds were already folded into the inequality rows, so the linear program itself is
        // told the variables are free; otherwise the simplex default of x >= 0 would silently add a
        // constraint the caller never asked for.
        var freeLower = new Vector<T>(variableCount);
        var freeUpper = new Vector<T>(variableCount);
        var negativeInfinity = NumOps.FromDouble(double.NegativeInfinity);
        var positiveInfinity = NumOps.FromDouble(double.PositiveInfinity);
        for (int i = 0; i < variableCount; i++)
        {
            freeLower[i] = negativeInfinity;
            freeUpper[i] = positiveInfinity;
        }

        var feasibilityProgram = new LinearProgram<T>(
            zeroObjective,
            inequalityMatrix,
            inequalityRows.Count > 0 ? inequalityBounds : null,
            program.EqualityMatrix,
            program.EqualityBounds,
            freeLower,
            freeUpper);

        var solution = _feasibilitySolver.Solve(feasibilityProgram);
        return (solution.Status,
            solution.Status == LinearProgramStatus.Optimal ? solution.Solution : null);
    }

    /// <summary>
    /// Solves the KKT system for the equality-constrained subproblem defined by the current working
    /// set, returning the step and the Lagrange multipliers.
    /// </summary>
    /// <remarks>
    /// The system solved is
    /// <code>
    ///   [ Q   Aᵀ ] [ p ]   [ -g ]
    ///   [ A   0  ] [ λ ] = [  0 ]
    /// </code>
    /// where <c>A</c> stacks the equality rows and the working inequality rows, <c>g</c> is the
    /// gradient at the current point, <c>p</c> is the step and <c>λ</c> the multipliers. Its second
    /// block row says the step stays on every working constraint; the first says the gradient at the
    /// new point is a combination of their normals, which is stationarity.
    /// </remarks>
    private (Vector<T> Step, Vector<T> Multipliers)? SolveEqualityConstrainedStep(
        QuadraticProgram<T> program,
        List<Vector<T>> inequalityRows,
        List<int> workingSet,
        Vector<T> gradient,
        int equalityCount,
        int variableCount)
    {
        int constraintCount = equalityCount + workingSet.Count;
        int size = variableCount + constraintCount;

        var system = new Matrix<T>(size, size);
        var rightHandSide = new Vector<T>(size);

        var regularization = NumOps.FromDouble(_options.SingularityRegularization);
        for (int i = 0; i < variableCount; i++)
        {
            for (int j = 0; j < variableCount; j++) system[i, j] = program.Quadratic[i, j];
            system[i, i] = NumOps.Add(system[i, i], regularization);
            rightHandSide[i] = NumOps.Negate(gradient[i]);
        }

        for (int e = 0; e < equalityCount; e++)
        {
            int row = variableCount + e;
            for (int c = 0; c < variableCount; c++)
            {
                T coefficient = program.EqualityMatrix![e, c];
                system[row, c] = coefficient;
                system[c, row] = coefficient;
            }
        }

        for (int k = 0; k < workingSet.Count; k++)
        {
            int row = variableCount + equalityCount + k;
            var constraintRow = inequalityRows[workingSet[k]];
            for (int c = 0; c < variableCount; c++)
            {
                system[row, c] = constraintRow[c];
                system[c, row] = constraintRow[c];
            }
        }

        Vector<T> solution;
        try
        {
            solution = new LuDecomposition<T>(system).Solve(rightHandSide);
        }
        catch (Exception exception) when (exception is not OutOfMemoryException
            and not StackOverflowException)
        {
            // A singular KKT system means the working constraints are linearly dependent. Reporting
            // this as "no step available" lets the caller stop at the current feasible point rather
            // than propagating a NaN direction into the answer.
            return null;
        }

        var step = new Vector<T>(variableCount);
        for (int i = 0; i < variableCount; i++)
        {
            if (double.IsNaN(NumOps.ToDouble(solution[i])) || double.IsInfinity(NumOps.ToDouble(solution[i])))
            {
                return null;
            }

            step[i] = solution[i];
        }

        var multipliers = new Vector<T>(Math.Max(constraintCount, 1));
        for (int k = 0; k < constraintCount; k++) multipliers[k] = solution[variableCount + k];

        return (step, multipliers);
    }

    /// <summary>
    /// Computes how far along the step direction the point can move before a constraint outside the
    /// working set blocks it, and which constraint that is.
    /// </summary>
    /// <returns>
    /// The step length, capped at 1 (the full subproblem step), and the index of the blocking
    /// constraint, or -1 when nothing blocks.
    /// </returns>
    private static (T Length, int BlockingConstraint) ComputeStepLength(
        List<Vector<T>> inequalityRows,
        Vector<T> inequalityBounds,
        List<int> workingSet,
        Vector<T> x,
        Vector<T> step,
        T tolerance)
    {
        T best = NumOps.One;
        int blocking = -1;

        for (int i = 0; i < inequalityRows.Count; i++)
        {
            if (workingSet.Contains(i)) continue;

            // Only constraints the step moves TOWARD can block it.
            T directionalRate = inequalityRows[i].DotProduct(step);
            if (!NumOps.GreaterThan(directionalRate, tolerance)) continue;

            T slack = NumOps.Subtract(inequalityBounds[i], inequalityRows[i].DotProduct(x));
            T limit = NumOps.Divide(slack, directionalRate);

            if (NumOps.LessThan(limit, best))
            {
                best = limit;
                blocking = i;
            }
        }

        // Numerical drift can make the slack marginally negative; clamping keeps the iterate
        // feasible rather than stepping backwards.
        if (NumOps.LessThan(best, NumOps.Zero)) best = NumOps.Zero;

        return (best, blocking);
    }

    private static bool IsActive(
        List<Vector<T>> rows, Vector<T> bounds, int index, Vector<T> x, T tolerance)
    {
        T slack = NumOps.Subtract(bounds[index], rows[index].DotProduct(x));
        return NumOps.LessThanOrEquals(NumOps.Abs(slack), tolerance);
    }

    private static bool IsEffectivelyZero(Vector<T> vector, T tolerance)
    {
        for (int i = 0; i < vector.Length; i++)
        {
            if (NumOps.GreaterThan(NumOps.Abs(vector[i]), tolerance)) return false;
        }

        return true;
    }

    private QuadraticProgramSolution<T> Success(
        QuadraticProgram<T> program,
        Vector<T> x,
        int iterations,
        int flattenedInequalityCount,
        int equalityCount,
        Vector<T> flattenedMultipliers,
        Vector<T> equalityMultipliers,
        int[] boundRowOrigin)
    {
        // Report multipliers only for the caller's own inequality rows; the extra rows synthesized
        // from variable bounds are an implementation detail.
        int originalCount = program.InequalityBounds?.Length ?? 0;
        Vector<T>? inequalityMultipliers = null;
        if (originalCount > 0)
        {
            inequalityMultipliers = new Vector<T>(originalCount);
            for (int i = 0; i < flattenedInequalityCount; i++)
            {
                int origin = boundRowOrigin[i];
                if (origin >= 0) inequalityMultipliers[origin] = flattenedMultipliers[i];
            }
        }

        return new QuadraticProgramSolution<T>(
            LinearProgramStatus.Optimal,
            x,
            EvaluateObjective(program, x),
            iterations,
            inequalityMultipliers,
            equalityCount > 0 ? equalityMultipliers : null);
    }

    private static T EvaluateObjective(QuadraticProgram<T> program, Vector<T> x)
    {
        var quadraticTerm = MultiplyMatrixVector(program.Quadratic, x);
        T quadratic = NumOps.Multiply(NumOps.FromDouble(0.5), x.DotProduct(quadraticTerm));
        return NumOps.Add(quadratic, program.Linear.DotProduct(x));
    }

    private static Vector<T> MultiplyMatrixVector(Matrix<T> matrix, Vector<T> vector)
    {
        var result = new Vector<T>(matrix.Rows);
        for (int r = 0; r < matrix.Rows; r++)
        {
            T sum = NumOps.Zero;
            for (int c = 0; c < matrix.Columns; c++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(matrix[r, c], vector[c]));
            }

            result[r] = sum;
        }

        return result;
    }

    private static bool IsFinite(T value)
    {
        double asDouble = NumOps.ToDouble(value);
        return !double.IsInfinity(asDouble) && !double.IsNaN(asDouble);
    }
}
