using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Solvers.LinearProgramming;

/// <summary>
/// Solves linear programs with the two-phase simplex method.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Implements Dantzig's simplex method in tableau form (Dantzig, 1947; see also Nocedal and
/// Wright, "Numerical Optimization", chapter 13), with:
/// <list type="bullet">
/// <item>a <b>two-phase</b> start, so no feasible point has to be supplied — phase one minimizes
/// the sum of artificial variables to find one, phase two optimizes the real objective;</item>
/// <item><b>Bland's anti-cycling rule</b> engaged automatically after a run of degenerate pivots,
/// which guarantees termination on problems where Dantzig's rule alone can cycle;</item>
/// <item><b>dual values</b> (shadow prices) recovered from the final tableau, so the solution
/// reports not just the answer but the marginal worth of every constraint.</item>
/// </list>
/// </para>
/// <para>
/// <b>The geometry.</b> The constraints of a linear program carve out a polytope, and because the
/// objective is linear, an optimum always sits at one of its corners (vertices). The simplex method
/// exploits exactly this: it starts at a corner and repeatedly walks along an edge to an adjacent
/// corner that improves the objective, stopping when no adjacent corner is better — at which point
/// the corner is provably optimal, because a linear objective over a convex region has no local
/// optima that are not global.
/// </para>
/// <para><b>For Beginners:</b> Picture the set of allowed choices as a many-sided crystal. Because
/// the thing you are maximizing is linear, the best point is never in the middle of a face — it is
/// always at a corner. So instead of searching everywhere, the method hops from corner to corner,
/// always moving to a better one, and stops when every neighbouring corner is worse. That is the
/// whole idea, and it is why linear programs of enormous size are routinely solved exactly.
/// </para>
/// <example>
/// <code>
/// // Maximise 3x + 2y subject to x + y &lt;= 4 and 2x + y &lt;= 6. The objective is the coefficient
/// // vector; every constraint block is optional.
/// var program = new LinearProgram&lt;double&gt;(
///     objective: new Vector&lt;double&gt;(new double[] { 3.0, 2.0 }),
///     inequalityMatrix: new Matrix&lt;double&gt;(new double[,] { { 1.0, 1.0 }, { 2.0, 1.0 } }),
///     inequalityBounds: new Vector&lt;double&gt;(new double[] { 4.0, 6.0 }));
///
/// var solver = new SimplexSolver&lt;double&gt;();
/// var solution = solver.Solve(program);
/// if (solution.Status == LinearProgramStatus.Optimal)
/// {
///     Console.WriteLine(solution.ObjectiveValue);
///     Console.WriteLine(solution.InequalityDualValues); // what each limit is worth
/// }
/// </code>
/// </example>
/// </remarks>
public sealed class SimplexSolver<T> : ILinearProgramSolver<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly SimplexSolverOptions _options;

    /// <summary>
    /// Creates a simplex solver.
    /// </summary>
    /// <param name="options">
    /// Solver configuration. When omitted, the documented defaults on
    /// <see cref="SimplexSolverOptions"/> are used.
    /// </param>
    public SimplexSolver(SimplexSolverOptions? options = null)
    {
        _options = options ?? new SimplexSolverOptions();
    }

    /// <inheritdoc />
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="program"/> is null.</exception>
    public LinearProgramSolution<T> Solve(LinearProgram<T> program)
    {
        if (program is null) throw new ArgumentNullException(nameof(program));

        return StandardForm.Build(program, _options).Solve();
    }

    /// <summary>
    /// A linear program rewritten as <c>minimize cᵀz subject to Az = b, z ≥ 0, b ≥ 0</c>, which is
    /// the only form the simplex tableau understands, together with the mapping needed to translate
    /// a solution back to the caller's original variables.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Three rewrites get any problem into this form:
    /// </para>
    /// <list type="number">
    /// <item><b>Bounds become shifts and rows.</b> A variable with a finite lower bound is shifted
    /// so its floor is zero. One bounded above only is reflected. One bounded on neither side is
    /// split into a difference of two non-negative parts. A finite upper bound then becomes an
    /// ordinary inequality row.</item>
    /// <item><b>Inequalities become equalities.</b> Each row gains a slack or surplus variable.</item>
    /// <item><b>Negative right-hand sides are negated,</b> which flips that row's sense; rows that
    /// end up as <c>≥</c> or <c>=</c> receive an artificial variable so phase one has a starting
    /// basis.</item>
    /// </list>
    /// </remarks>
    private sealed class StandardForm
    {
        private readonly SimplexSolverOptions _options;
        private readonly T _tolerance;

        // Rows 0..constraintCount-1 hold the constraints; the final column is the right-hand side.
        private Tensor<T> _tableau;
        private readonly int _constraintCount;
        private readonly int _columnCount;
        private readonly int[] _basis;

        private readonly Vector<T> _phaseTwoObjective;
        private readonly HashSet<int> _artificialColumns;
        private readonly LinearProgramStandardForm<T> _nonNegativeForm;

        // Auxiliary column carrying each original constraint's dual value; -1 when the row has none.
        private readonly int[] _inequalitySlackColumns;
        private readonly int[] _equalityArtificialColumns;
        private readonly bool[] _rowWasNegated;

        private readonly T _originalObjectiveOffset;

        // 1 for artificial (respectively real) columns, and the tolerance in every column.
        private readonly Tensor<T> _artificialColumnMask;
        private readonly Tensor<T> _realColumnMask;
        private readonly Tensor<T> _toleranceRow;

        private bool _phaseOneActive = true;
        private int _iterations;

        private StandardForm(
            SimplexSolverOptions options,
            Tensor<T> tableau,
            int constraintCount,
            int columnCount,
            int[] basis,
            Vector<T> phaseTwoObjective,
            HashSet<int> artificialColumns,
            LinearProgramStandardForm<T> nonNegativeForm,
            int[] inequalitySlackColumns,
            int[] equalityArtificialColumns,
            bool[] rowWasNegated,
            T originalObjectiveOffset)
        {
            _options = options;
            _tolerance = NumOps.FromDouble(options.Tolerance);
            _tableau = tableau;
            _constraintCount = constraintCount;
            _columnCount = columnCount;
            _basis = basis;
            _phaseTwoObjective = phaseTwoObjective;
            _artificialColumns = artificialColumns;
            _nonNegativeForm = nonNegativeForm;
            _inequalitySlackColumns = inequalitySlackColumns;
            _equalityArtificialColumns = equalityArtificialColumns;
            _rowWasNegated = rowWasNegated;
            _originalObjectiveOffset = originalObjectiveOffset;

            var isArtificial = new bool[columnCount];
            var isReal = new bool[columnCount];
            for (int c = 0; c < columnCount; c++)
            {
                isArtificial[c] = artificialColumns.Contains(c);
                isReal[c] = !isArtificial[c];
            }

            _artificialColumnMask = SolverTensorOps<T>.Mask(isArtificial);
            _realColumnMask = SolverTensorOps<T>.Mask(isReal);
            _toleranceRow = Tensor<T>.CreateDefault(new[] { columnCount }, _tolerance);
        }

        public static StandardForm Build(LinearProgram<T> program, SimplexSolverOptions options)
        {
            // --- Steps 1-3: non-negative variables, shifted rows, non-negative right-hand side ---
            // Shared with the interior-point solver, which needs the same rewrite but different
            // auxiliary columns afterwards.
            var nonNegativeForm = LinearProgramStandardForm<T>.Build(program);

            int structuralColumnCount = nonNegativeForm.VariableCount;
            int inequalityRowCount = nonNegativeForm.InequalityRowCount;
            int equalityRowCount = nonNegativeForm.EqualityRowCount;
            int constraintCount = nonNegativeForm.Rows.Count;

            var rows = nonNegativeForm.Rows;
            var rightHandSides = nonNegativeForm.RightHandSides;
            var isEquality = nonNegativeForm.IsEquality;
            var rowWasNegated = nonNegativeForm.RowWasNegated;

            // --- Step 4: assign auxiliary columns ---
            //   <= row  -> slack (+1), which also serves as the initial basic variable
            //   >= row  -> surplus (-1) plus an artificial to start the basis
            //   =  row  -> artificial only
            int auxiliaryColumnCount = 0;
            for (int r = 0; r < constraintCount; r++)
            {
                bool needsSlackOrSurplus = !isEquality[r];
                if (needsSlackOrSurplus) auxiliaryColumnCount++;
                if (isEquality[r] || (needsSlackOrSurplus && rowWasNegated[r])) auxiliaryColumnCount++;
            }

            int columnCount = structuralColumnCount + auxiliaryColumnCount;
            var engine = SolverTensorOps<T>.Engine;
            var basis = new int[constraintCount];
            var artificialColumns = new HashSet<int>();
            var slackColumnOfRow = new int[constraintCount];
            var artificialColumnOfRow = new int[constraintCount];

            // Auxiliary-block entries as flat positions with a sign id (0 -> -1, 1 -> +1): slack +1, surplus -1,
            // artificial +1.
            var auxiliaryPositions = new List<int>();
            var auxiliarySigns = new List<int>();
            int auxiliaryCursor = structuralColumnCount;
            for (int r = 0; r < constraintCount; r++)
            {
                slackColumnOfRow[r] = -1;
                artificialColumnOfRow[r] = -1;

                bool needsSlackOrSurplus = !isEquality[r];
                bool needsArtificial = isEquality[r] || (needsSlackOrSurplus && rowWasNegated[r]);

                if (needsSlackOrSurplus)
                {
                    int slackColumn = auxiliaryCursor++;
                    slackColumnOfRow[r] = slackColumn;
                    auxiliaryPositions.Add(r * auxiliaryColumnCount + slackColumn - structuralColumnCount);
                    auxiliarySigns.Add(rowWasNegated[r] ? 0 : 1);
                }

                if (needsArtificial)
                {
                    int artificialColumn = auxiliaryCursor++;
                    artificialColumnOfRow[r] = artificialColumn;
                    artificialColumns.Add(artificialColumn);
                    auxiliaryPositions.Add(r * auxiliaryColumnCount + artificialColumn - structuralColumnCount);
                    auxiliarySigns.Add(1);
                    basis[r] = artificialColumn;
                }
                else
                {
                    basis[r] = slackColumnOfRow[r];
                }
            }

            // [structural rows | auxiliary columns | right-hand side].
            Tensor<T> tableau;
            if (constraintCount == 0)
            {
                tableau = new Tensor<T>(new[] { 0, columnCount + 1 });
            }
            else
            {
                var structural = engine.TensorStack(rows.Select(row => SolverTensorOps<T>.AsTensor(row)).ToArray(), 0);
                var positions = auxiliaryPositions.ToArray();
                var signs = auxiliarySigns.ToArray();
                var auxiliary = engine.Reshape(
                    engine.TensorIndexAdd(
                        Tensor<T>.CreateDefault(new[] { constraintCount * auxiliaryColumnCount }, NumOps.Zero),
                        0,
                        new Tensor<int>(new[] { positions.Length }, new Vector<int>(positions)),
                        engine.TensorTake(engine.TensorArange<T>(-1, 2, 2), new Tensor<int>(new[] { signs.Length }, new Vector<int>(signs)))),
                    new[] { constraintCount, auxiliaryColumnCount });
                var rightHandSideColumn = new Tensor<T>(new[] { constraintCount, 1 }, new Vector<T>(rightHandSides.ToArray()));
                tableau = engine.TensorConcatenate(new[] { structural, auxiliary, rightHandSideColumn }, 1);
            }

            // --- Step 5: widen the projected objective to cover the auxiliary columns ---
            // The auxiliary columns carry no objective cost, so they stay zero.
            var structuralObjective = engine.TensorNarrow(SolverTensorOps<T>.AsTensor(nonNegativeForm.Objective), 0, 0, structuralColumnCount);
            var phaseTwoObjective = auxiliaryColumnCount == 0
                ? structuralObjective.Contiguous().ToVector()
                : engine.TensorConcatenate(
                    new[] { structuralObjective, Tensor<T>.CreateDefault(new[] { auxiliaryColumnCount }, NumOps.Zero) }, 0).ToVector();

            T objectiveOffset = nonNegativeForm.ObjectiveOffset;

            var inequalitySlackColumns = new int[inequalityRowCount];
            for (int r = 0; r < inequalityRowCount; r++) inequalitySlackColumns[r] = slackColumnOfRow[r];

            var equalityArtificialColumns = new int[equalityRowCount];
            for (int r = 0; r < equalityRowCount; r++)
            {
                equalityArtificialColumns[r] = artificialColumnOfRow[inequalityRowCount + r];
            }

            return new StandardForm(
                options, tableau, constraintCount, columnCount, basis, phaseTwoObjective,
                artificialColumns, nonNegativeForm, inequalitySlackColumns, equalityArtificialColumns,
                rowWasNegated, objectiveOffset);
        }

        public LinearProgramSolution<T> Solve()
        {
            // --- Phase one: minimize the sum of the artificial variables ---
            if (_artificialColumns.Count > 0)
            {
                var phaseOneObjective = _artificialColumnMask.ToVector();

                var phaseOneStatus = RunSimplex(phaseOneObjective, out T phaseOneOptimum);
                if (phaseOneStatus == LinearProgramStatus.IterationLimit)
                {
                    return NoPoint(LinearProgramStatus.IterationLimit);
                }

                if (phaseOneStatus == LinearProgramStatus.Unbounded)
                {
                    throw new InvalidOperationException(
                        "Phase one reported an unbounded objective, which is impossible for the " +
                        "sum of non-negative artificial variables. The tableau is numerically " +
                        "invalid; rescale the problem data.");
                }

                // A positive phase-one optimum means no assignment of the real variables can drive
                // every artificial to zero — the constraints contradict each other.
                if (NumOps.GreaterThan(phaseOneOptimum, _tolerance))
                {
                    return NoPoint(LinearProgramStatus.Infeasible);
                }

                if (!DriveArtificialsOutOfBasis())
                {
                    var limitedSolution = ExtractSolution();
                    T limitedObjective = NumOps.Add(
                        ComputeObjectiveValue(_phaseTwoObjective), _originalObjectiveOffset);
                    return new LinearProgramSolution<T>(
                        LinearProgramStatus.IterationLimit,
                        limitedSolution,
                        limitedObjective,
                        _iterations);
                }
            }

            _phaseOneActive = false;

            // --- Phase two: minimize the real objective ---
            var status = RunSimplex(_phaseTwoObjective, out T optimum);
            if (status == LinearProgramStatus.Unbounded)
            {
                return NoPoint(LinearProgramStatus.Unbounded);
            }

            var solution = ExtractSolution();
            T objectiveValue = NumOps.Add(optimum, _originalObjectiveOffset);
            Vector<T>? inequalityDuals = null;
            Vector<T>? equalityDuals = null;
            if (status == LinearProgramStatus.Optimal)
            {
                (inequalityDuals, equalityDuals) = ExtractDualValues(_phaseTwoObjective);
            }

            return new LinearProgramSolution<T>(
                status, solution, objectiveValue, _iterations, inequalityDuals, equalityDuals);
        }

        private LinearProgramSolution<T> NoPoint(LinearProgramStatus status)
        {
            return new LinearProgramSolution<T>(status, null, NumOps.Zero, _iterations);
        }

        /// <summary>
        /// Runs simplex pivots against the supplied objective until optimality, unboundedness or the
        /// iteration limit, and reports the objective value reached.
        /// </summary>
        private LinearProgramStatus RunSimplex(Vector<T> objective, out T optimum)
        {
            int degeneratePivots = 0;

            while (true)
            {
                // Reduced costs r_j = c_j - c_Bᵀ B⁻¹ A_j. The tableau is kept canonical with respect
                // to the basis, so B⁻¹A is exactly what is stored and this is a direct sum.
                var reducedCosts = ComputeReducedCosts(objective);
                bool useBlandsRule = degeneratePivots >= _options.DegeneratePivotsBeforeBlandsRule;
                int enteringColumn = SelectEnteringColumn(reducedCosts, useBlandsRule);

                if (enteringColumn < 0)
                {
                    optimum = ComputeObjectiveValue(objective);
                    return LinearProgramStatus.Optimal;
                }

                int leavingRow = SelectLeavingRow(enteringColumn, useBlandsRule);
                if (leavingRow < 0)
                {
                    // Every entry in the entering column is non-positive, so the entering variable
                    // can grow forever without any basic variable reaching zero.
                    optimum = ComputeObjectiveValue(objective);
                    return LinearProgramStatus.Unbounded;
                }

                // Pivoting on a row whose right-hand side is already zero moves to a different basis
                // describing the SAME vertex, leaving the objective unchanged. A long run of these
                // is the signature of cycling.
                var leavingRightHandSide = SolverTensorOps<T>.Engine.TensorNarrow(TableauColumn(_columnCount), 0, leavingRow, 1);
                bool isDegenerate = SolverTensorOps<T>.First(SolverTensorOps<T>.AtMost(
                    SolverTensorOps<T>.Engine.TensorAbs(leavingRightHandSide), _tolerance)) is not null;
                degeneratePivots = isDegenerate ? degeneratePivots + 1 : 0;

                Pivot(leavingRow, enteringColumn);
                _basis[leavingRow] = enteringColumn;
                _iterations++;

                if (_iterations >= _options.MaxIterations)
                {
                    optimum = ComputeObjectiveValue(objective);
                    return LinearProgramStatus.IterationLimit;
                }
            }
        }

        private int Width => _columnCount + 1;

        private Tensor<int> BasisIds() => new Tensor<int>(new[] { _constraintCount }, new Vector<int>(_basis));

        private Tensor<T> TableauRow(int row) => SolverTensorOps<T>.Engine.TensorSliceAxis(_tableau, 0, row);

        private Tensor<T> TableauColumn(int column) => SolverTensorOps<T>.Engine.TensorSliceAxis(_tableau, 1, column);

        private Tensor<T> ComputeReducedCosts(Vector<T> objective)
        {
            var engine = SolverTensorOps<T>.Engine;
            var reducedCosts = SolverTensorOps<T>.AsTensor(objective);
            if (_constraintCount == 0) return reducedCosts;

            // Rows are folded in one at a time, in row order, exactly as the per-entry update
            // r_c -= c_B[r] * A[r, c] did; rows whose basic cost is exactly zero are skipped.
            var basicCosts = engine.TensorTake(reducedCosts, BasisIds());
            var costs = basicCosts.ToArray();
            foreach (int r in engine.TensorNonzero(engine.TensorNotEquals(basicCosts, NumOps.Zero)).ToArray())
            {
                var row = engine.TensorNarrow(TableauRow(r), 0, 0, _columnCount);
                reducedCosts = engine.TensorSubtract(reducedCosts, engine.TensorMultiplyScalar(row, costs[r]));
            }

            return reducedCosts;
        }

        private T ComputeObjectiveValue(Vector<T> objective)
        {
            if (_constraintCount == 0) return NumOps.Zero;
            var engine = SolverTensorOps<T>.Engine;
            var basicCosts = engine.TensorTake(SolverTensorOps<T>.AsTensor(objective), BasisIds());
            return engine.TensorSum(engine.TensorMultiply(basicCosts, TableauColumn(_columnCount)));
        }

        /// <summary>
        /// Chooses the variable to bring into the basis: the most negative reduced cost (Dantzig),
        /// or the lowest-index negative one when Bland's rule is in force.
        /// </summary>
        private int SelectEnteringColumn(Tensor<T> reducedCosts, bool useBlandsRule)
        {
            var engine = SolverTensorOps<T>.Engine;

            // r_c < -tolerance is -r_c > tolerance. Artificial variables must never re-enter after phase one
            // drove them out (doing so could leave the final point infeasible for the real problem), so in phase
            // two they sit at the tolerance, which never qualifies.
            var negated = engine.TensorNegate(reducedCosts);
            var candidates = _phaseOneActive ? negated : engine.TensorWhere(_realColumnMask, negated, _toleranceRow);

            if (useBlandsRule)
            {
                return SolverTensorOps<T>.First(engine.TensorGreaterThan(candidates, _tolerance))?[0] ?? -1;
            }

            return SolverTensorOps<T>.FirstStrictMaxIndex(candidates, _tolerance);
        }

        /// <summary>
        /// Chooses the basic variable to leave by the minimum-ratio test, breaking ties by lowest
        /// basis index (Bland's rule) so the two rules together cannot cycle.
        /// </summary>
        /// <remarks>
        /// The scan keeps the first eligible row and moves on a strictly smaller ratio; under Bland's rule it
        /// also moves to a later row within tolerance of the best ratio with a smaller basis index. The best
        /// ratio only changes on a strict decrease, so after the first minimum (row k) it is fixed, and the
        /// result is the smallest basis index among k and the later rows tied with it. A NaN first ratio is
        /// never displaced.
        /// </remarks>
        private int SelectLeavingRow(int enteringColumn, bool useBlandsRule)
        {
            if (_constraintCount == 0) return -1;
            var engine = SolverTensorOps<T>.Engine;
            var pivotCandidates = TableauColumn(enteringColumn);
            var eligible = engine.TensorGreaterThan(pivotCandidates, _tolerance);
            var firstEligible = SolverTensorOps<T>.First(eligible);
            if (firstEligible is null) return -1;

            var safePivots = engine.TensorWhere(
                eligible, pivotCandidates, Tensor<T>.CreateDefault(new[] { _constraintCount }, NumOps.One));
            var ratios = engine.TensorDivide(TableauColumn(_columnCount), safePivots);
            int firstRow = firstEligible[0];
            if (NumOps.IsNaN(ratios.GetFlat(firstRow))) return firstRow;

            int best = SolverTensorOps<T>.FirstMinIndex(ratios, eligible);
            if (!useBlandsRule) return best;

            T bestRatio = ratios.GetFlat(best);
            var tied = engine.TensorMultiply(
                eligible,
                SolverTensorOps<T>.AtMost(engine.TensorAbs(engine.TensorSubtractScalar(ratios, bestRatio)), _tolerance));
            int chosen = best;
            foreach (int r in engine.TensorNonzero(tied).ToArray())
            {
                if (r > best && _basis[r] < _basis[chosen]) chosen = r;
            }

            return chosen;
        }

        /// <summary>
        /// Performs the Gauss-Jordan elimination that makes <paramref name="enteringColumn"/> a unit
        /// column with its 1 in <paramref name="pivotRow"/>.
        /// </summary>
        private void Pivot(int pivotRow, int enteringColumn)
        {
            var engine = SolverTensorOps<T>.Engine;
            int width = Width;
            var row = TableauRow(pivotRow);
            T pivotValue = row.GetFlat(enteringColumn);
            var scaledRow = engine.TensorDivideScalar(row, pivotValue);

            // Every other row with a non-zero factor loses factor * (scaled pivot row); rows with an exactly
            // zero factor are left untouched, and the pivot row becomes the scaled row.
            var factors = TableauColumn(enteringColumn);
            var eliminated = engine.TensorSubtract(
                _tableau,
                engine.TensorOuter(factors.Reshape(_constraintCount), scaledRow.Reshape(width)));
            var isPivotRow = engine.TensorSliceAxis(engine.TensorEye<T>(_constraintCount), 0, pivotRow);
            var changes = engine.TensorMultiply(
                engine.TensorNotEquals(factors, NumOps.Zero), engine.ScalarMinusTensor(NumOps.One, isPivotRow));
            var updated = engine.TensorWhere(RowMask(changes), eliminated, _tableau);
            _tableau = engine.TensorWhere(
                RowMask(isPivotRow),
                engine.TensorBroadcastTo(engine.Reshape(scaledRow, new[] { 1, width }), new[] { _constraintCount, width }),
                updated).Contiguous();
        }

        private Tensor<T> RowMask(Tensor<T> perRow)
            => SolverTensorOps<T>.Engine.TensorBroadcastTo(
                SolverTensorOps<T>.Engine.Reshape(perRow, new[] { _constraintCount, 1 }), new[] { _constraintCount, Width });

        /// <summary>
        /// Pivots any artificial variable still sitting in the basis at value zero out of it, so
        /// phase two never has to consider artificial columns.
        /// </summary>
        private bool DriveArtificialsOutOfBasis()
        {
            var engine = SolverTensorOps<T>.Engine;
            for (int r = 0; r < _constraintCount; r++)
            {
                if (!_artificialColumns.Contains(_basis[r])) continue;

                // The first real column of largest magnitude above the tolerance.
                var magnitudes = engine.TensorAbs(engine.TensorNarrow(TableauRow(r), 0, 0, _columnCount));
                int replacement = SolverTensorOps<T>.FirstStrictMaxIndex(
                    engine.TensorWhere(_realColumnMask, magnitudes, _toleranceRow), _tolerance);

                if (replacement < 0)
                {
                    // The row is all zeros across the real columns: the original constraint was
                    // linearly dependent on the others and carries no information. Leaving the
                    // artificial basic at value zero is harmless, because phase two excludes
                    // artificial columns from the entering-column search.
                    continue;
                }

                if (_iterations >= _options.MaxIterations) return false;

                Pivot(r, replacement);
                _basis[r] = replacement;
                _iterations++;
            }

            return true;
        }

        private Vector<T> ExtractSolution()
        {
            // Basic variables take their row's right-hand side; every other variable is zero.
            var standardValues = Tensor<T>.CreateDefault(new[] { _columnCount }, NumOps.Zero);
            if (_constraintCount > 0)
            {
                standardValues = SolverTensorOps<T>.Engine.TensorIndexCopy(
                    standardValues, 0, BasisIds(), TableauColumn(_columnCount));
            }

            // Undo the shift, reflection and splitting to recover the caller's variables.
            return _nonNegativeForm.RecoverOriginalVariables(standardValues.ToVector());
        }

        /// <summary>
        /// Recovers the dual value of each original constraint from the final tableau.
        /// </summary>
        /// <remarks>
        /// At optimality the negated reduced cost of a constraint's auxiliary column is that
        /// constraint's dual value, the tableau form of <c>y = c_B^T B^-1</c>. Rows that were negated
        /// to make their right-hand side non-negative have their sign flipped back.
        /// </remarks>
        private (Vector<T>? Inequality, Vector<T>? Equality) ExtractDualValues(Vector<T> objective)
        {
            var reducedCosts = ComputeReducedCosts(objective);

            Vector<T>? inequalityDuals = _inequalitySlackColumns.Length > 0
                ? DualsOf(reducedCosts, _inequalitySlackColumns, 0)
                : null;
            Vector<T>? equalityDuals = _equalityArtificialColumns.Length > 0
                ? DualsOf(reducedCosts, _equalityArtificialColumns, _inequalitySlackColumns.Length)
                : null;

            return (inequalityDuals, equalityDuals);
        }

        // -reducedCost[column] (sign restored for a negated row), or zero for a row without the column.
        private Vector<T> DualsOf(Tensor<T> reducedCosts, int[] columns, int rowOffset)
        {
            var engine = SolverTensorOps<T>.Engine;
            var safeColumns = new int[columns.Length];
            var hasColumn = new bool[columns.Length];
            var negated = new bool[columns.Length];
            for (int r = 0; r < columns.Length; r++)
            {
                hasColumn[r] = columns[r] >= 0;
                safeColumns[r] = hasColumn[r] ? columns[r] : 0;
                negated[r] = _rowWasNegated[rowOffset + r];
            }

            var picked = engine.TensorTake(reducedCosts, new Tensor<int>(new[] { safeColumns.Length }, new Vector<int>(safeColumns)));
            var duals = engine.TensorWhere(SolverTensorOps<T>.Mask(negated), picked, engine.TensorNegate(picked));
            return engine.TensorWhere(
                SolverTensorOps<T>.Mask(hasColumn),
                duals,
                Tensor<T>.CreateDefault(new[] { columns.Length }, NumOps.Zero)).ToVector();
        }
    }
}
