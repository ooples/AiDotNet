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
        var inWorkingSet = new bool[inequalityCount];
        for (int i = 0; i < inequalityCount; i++)
        {
            if (IsActive(inequalityRows, inequalityBounds, i, x, tolerance))
            {
                workingSet.Add(i);
                inWorkingSet[i] = true;
            }
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
                // inequality wants to be released: the most negative multiplier below -tolerance, the
                // first on ties (as -multiplier > tolerance, strictly increasing).
                int mostNegative = -1;
                Tensor<T>? workingMultipliers = null;
                if (workingSet.Count > 0)
                {
                    workingMultipliers = Engine.TensorNarrow(
                        SolverTensorOps<T>.AsTensor(multipliers), 0, equalityCount, workingSet.Count).Contiguous();
                    mostNegative = SolverTensorOps<T>.FirstStrictMaxIndex(Engine.TensorNegate(workingMultipliers), tolerance);
                }

                if (mostNegative < 0)
                {
                    // Every multiplier is non-negative: the KKT conditions hold and x is optimal.
                    var released = Tensor<T>.CreateDefault(new[] { Math.Max(inequalityCount, 1) }, NumOps.Zero);
                    if (workingMultipliers is not null)
                    {
                        var ids = workingSet.ToArray();
                        released = Engine.TensorIndexCopy(
                            released, 0, new Tensor<int>(new[] { ids.Length }, new Vector<int>(ids)), workingMultipliers);
                    }

                    released.AsSpan().Slice(0, inequalityCount).CopyTo(inequalityMultipliers.AsWritableSpan());
                    multipliers.AsSpan().Slice(0, equalityCount).CopyTo(equalityMultipliers.AsWritableSpan());

                    return Success(
                        program, x, iterations, inequalityCount, equalityCount,
                        inequalityMultipliers, equalityMultipliers, boundRowOrigin);
                }

                // Dropping the constraint with the most negative multiplier gives the steepest
                // first-order improvement among the available releases.
                inWorkingSet[workingSet[mostNegative]] = false;
                workingSet.RemoveAt(mostNegative);
                continue;
            }

            // Take the largest step along the direction that keeps every constraint satisfied.
            var (stepLength, blockingConstraint) = ComputeStepLength(
                inequalityRows, inequalityBounds, inWorkingSet, x, step, tolerance);

            x = (Vector<T>)Engine.Add(x, (Vector<T>)Engine.Multiply(step, stepLength));

            if (blockingConstraint >= 0)
            {
                workingSet.Add(blockingConstraint);
                inWorkingSet[blockingConstraint] = true;
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
        if (originalCount > 0)
        {
            var matrix = SolverTensorOps<T>.AsTensor(program.InequalityMatrix ?? throw new InvalidOperationException("program.InequalityMatrix is null where it is required."));
            var originalBounds = (program.InequalityBounds ?? throw new InvalidOperationException("program.InequalityBounds is null where it is required.")).ToArray();
            for (int r = 0; r < originalCount; r++)
            {
                rows.Add(Engine.TensorSliceAxis(matrix, 0, r).ToVector());
                bounds.Add(originalBounds[r]);
                origin.Add(r);
            }
        }

        // x_i <= upper_i as the unit row e_i, and -x_i <= -lower_i as the row 0 - e_i.
        var upperFinite = FiniteFlags(program.UpperBounds, variableCount);
        var lowerFinite = FiniteFlags(program.LowerBounds, variableCount);
        var identity = Engine.TensorEye<T>(variableCount);
        var negatedIdentity = Engine.ScalarMinusTensor(NumOps.Zero, identity);
        T[] uppers = program.UpperBounds?.ToArray() ?? Array.Empty<T>();
        T[] negatedLowers = program.LowerBounds is null
            ? Array.Empty<T>()
            : Engine.TensorNegate(SolverTensorOps<T>.AsTensor(program.LowerBounds)).ToArray();
        for (int i = 0; i < variableCount; i++)
        {
            if (upperFinite[i])
            {
                rows.Add(Engine.TensorSliceAxis(identity, 0, i).ToVector());
                bounds.Add(uppers[i]);
                origin.Add(-1);
            }

            if (lowerFinite[i])
            {
                rows.Add(Engine.TensorSliceAxis(negatedIdentity, 0, i).ToVector());
                bounds.Add(negatedLowers[i]);
                origin.Add(-1);
            }
        }

        var boundsVector = new Vector<T>(bounds);

        return (rows, boundsVector, origin.ToArray());
    }

    // True where the bound vector is present and its entry is finite.
    private static bool[] FiniteFlags(Vector<T>? bounds, int count)
    {
        var flags = new bool[count];
        if (bounds is null || count == 0) return flags;
        for (int i = 0; i < count; i++) flags[i] = true;
        var nonFinite = SolverTensorOps<T>.NonFinite(SolverTensorOps<T>.AsTensor(bounds));
        foreach (int i in Engine.TensorNonzero(nonFinite).ToArray()) flags[i] = false;
        return flags;
    }

    private (LinearProgramStatus Status, Vector<T>? Point) FindFeasiblePoint(
        QuadraticProgram<T> program, List<Vector<T>> inequalityRows, Vector<T> inequalityBounds)
    {
        int variableCount = program.VariableCount;

        var zeroObjective = new Vector<T>(variableCount);
        Matrix<T>? inequalityMatrix = null;
        if (inequalityRows.Count > 0)
        {
            inequalityMatrix = StackRows(inequalityRows).ToMatrix();
        }

        // The bounds were already folded into the inequality rows, so the linear program itself is
        // told the variables are free; otherwise the simplex default of x >= 0 would silently add a
        // constraint the caller never asked for.
        var negativeInfinity = NumOps.FromDouble(double.NegativeInfinity);
        var positiveInfinity = NumOps.FromDouble(double.PositiveInfinity);
        var freeLower = Tensor<T>.CreateDefault(new[] { variableCount }, negativeInfinity).ToVector();
        var freeUpper = Tensor<T>.CreateDefault(new[] { variableCount }, positiveInfinity).ToVector();

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

    private static Tensor<T> StackRows(List<Vector<T>> rows)
        => Engine.TensorStack(rows.Select(row => SolverTensorOps<T>.AsTensor(row)).ToArray(), 0);

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

        // [[Q + rI, C^T], [C, 0]] with C the equality rows then the working inequality rows, and
        // right-hand side [-g; 0]. The regularization only touches the diagonal.
        var regularization = NumOps.FromDouble(_options.SingularityRegularization);
        var quadratic = SolverTensorOps<T>.AsTensor(program.Quadratic);
        var identity = Engine.TensorEye<T>(variableCount);
        var topLeft = Engine.TensorWhere(identity, Engine.TensorAddScalar(quadratic, regularization), quadratic);
        var negatedGradient = Engine.TensorNegate(SolverTensorOps<T>.AsTensor(gradient));

        Tensor<T> systemTensor;
        Tensor<T> rightHandSideTensor;
        if (constraintCount == 0)
        {
            systemTensor = topLeft;
            rightHandSideTensor = negatedGradient;
        }
        else
        {
            var blocks = new List<Tensor<T>>(2);
            if (equalityCount > 0)
            {
                blocks.Add(Engine.TensorNarrow(
                    SolverTensorOps<T>.AsTensor(program.EqualityMatrix ?? throw new InvalidOperationException("program.EqualityMatrix is null where it is required.")),
                    0, 0, equalityCount));
            }

            if (workingSet.Count > 0)
            {
                blocks.Add(StackRows(workingSet.Select(k => inequalityRows[k]).ToList()));
            }

            var constraints = blocks.Count == 1 ? blocks[0] : Engine.TensorConcatenate(blocks.ToArray(), 0);
            var top = Engine.TensorConcatenate(new[] { topLeft, Engine.TensorTranspose(constraints) }, 1);
            var bottom = Engine.TensorConcatenate(
                new[] { constraints, Tensor<T>.CreateDefault(new[] { constraintCount, constraintCount }, NumOps.Zero) }, 1);
            systemTensor = Engine.TensorConcatenate(new[] { top, bottom }, 0);
            rightHandSideTensor = Engine.TensorConcatenate(
                new[] { negatedGradient, Tensor<T>.CreateDefault(new[] { constraintCount }, NumOps.Zero) }, 0);
        }

        var system = systemTensor.Contiguous().ToMatrix();
        var rightHandSide = rightHandSideTensor.Contiguous().ToVector();

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

        var solved = SolverTensorOps<T>.AsTensor(solution);
        var step = Engine.TensorNarrow(solved, 0, 0, variableCount).Contiguous();
        if (SolverTensorOps<T>.First(SolverTensorOps<T>.NonFinite(step)) is not null)
        {
            return null;
        }

        var multipliers = constraintCount > 0
            ? Engine.TensorNarrow(solved, 0, variableCount, constraintCount).Contiguous().ToVector()
            : new Vector<T>(1);

        return (step.ToVector(), multipliers);
    }

    private static (T Length, int BlockingConstraint) ComputeStepLength(
        List<Vector<T>> inequalityRows,
        Vector<T> inequalityBounds,
        bool[] inWorkingSet,
        Vector<T> x,
        Vector<T> step,
        T tolerance)
    {
        T best = NumOps.One;
        int blocking = -1;

        if (inequalityRows.Count > 0)
        {
            // Only constraints outside the working set that the step moves TOWARD (rate > tolerance) can
            // block it; the first strictly smallest limit below one wins, as -limit > -1.
            var rows = StackRows(inequalityRows);
            var rates = SolverTensorOps<T>.MatVec(rows, SolverTensorOps<T>.AsTensor(step));
            var slack = Engine.TensorSubtract(SolverTensorOps<T>.AsTensor(inequalityBounds), SolverTensorOps<T>.MatVec(rows, SolverTensorOps<T>.AsTensor(x)));
            var outside = Engine.ScalarMinusTensor(NumOps.One, SolverTensorOps<T>.Mask(inWorkingSet));
            var eligible = Engine.TensorMultiply(outside, Engine.TensorGreaterThan(rates, tolerance));
            var ones = Tensor<T>.CreateDefault(new[] { inequalityRows.Count }, NumOps.One);
            var limits = Engine.TensorDivide(slack, Engine.TensorWhere(eligible, rates, ones));
            T floor = NumOps.Negate(NumOps.One);
            var candidates = Engine.TensorWhere(
                eligible, Engine.TensorNegate(limits), Tensor<T>.CreateDefault(new[] { inequalityRows.Count }, floor));
            blocking = SolverTensorOps<T>.FirstStrictMaxIndex(candidates, floor);
            if (blocking >= 0) best = limits.GetFlat(blocking);
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
        if (vector.Length == 0) return true;
        var magnitudes = Engine.TensorAbs(SolverTensorOps<T>.AsTensor(vector));
        return SolverTensorOps<T>.First(Engine.TensorGreaterThan(magnitudes, tolerance)) is null;
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
        // from variable bounds are an implementation detail. FlattenConstraints lists the caller's rows
        // first, in order (origin[i] == i), so they are the leading entries.
        int originalCount = program.InequalityBounds?.Length ?? 0;
        Vector<T>? inequalityMultipliers = null;
        if (originalCount > 0)
        {
            inequalityMultipliers = new Vector<T>(originalCount);
            int copied = Math.Min(originalCount, flattenedInequalityCount);
            flattenedMultipliers.AsSpan().Slice(0, copied).CopyTo(inequalityMultipliers.AsWritableSpan());
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
        => SolverTensorOps<T>.MatVec(SolverTensorOps<T>.AsTensor(matrix), SolverTensorOps<T>.AsTensor(vector)).ToVector();
}
