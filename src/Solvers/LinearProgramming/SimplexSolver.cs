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
        private readonly Matrix<T> _tableau;
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

        private bool _phaseOneActive = true;
        private int _iterations;

        private StandardForm(
            SimplexSolverOptions options,
            Matrix<T> tableau,
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
            var tableau = new Matrix<T>(constraintCount, columnCount + 1);
            var basis = new int[constraintCount];
            var artificialColumns = new HashSet<int>();
            var slackColumnOfRow = new int[constraintCount];
            var artificialColumnOfRow = new int[constraintCount];

            int auxiliaryCursor = structuralColumnCount;
            for (int r = 0; r < constraintCount; r++)
            {
                slackColumnOfRow[r] = -1;
                artificialColumnOfRow[r] = -1;

                var row = rows[r];
                for (int c = 0; c < structuralColumnCount; c++) tableau[r, c] = row[c];
                tableau[r, columnCount] = rightHandSides[r];

                bool needsSlackOrSurplus = !isEquality[r];
                bool needsArtificial = isEquality[r] || (needsSlackOrSurplus && rowWasNegated[r]);

                if (needsSlackOrSurplus)
                {
                    int slackColumn = auxiliaryCursor++;
                    slackColumnOfRow[r] = slackColumn;
                    tableau[r, slackColumn] = rowWasNegated[r] ? NumOps.Negate(NumOps.One) : NumOps.One;
                }

                if (needsArtificial)
                {
                    int artificialColumn = auxiliaryCursor++;
                    artificialColumnOfRow[r] = artificialColumn;
                    artificialColumns.Add(artificialColumn);
                    tableau[r, artificialColumn] = NumOps.One;
                    basis[r] = artificialColumn;
                }
                else
                {
                    basis[r] = slackColumnOfRow[r];
                }
            }

            // --- Step 5: widen the projected objective to cover the auxiliary columns ---
            // The auxiliary columns carry no objective cost, so they stay zero.
            var phaseTwoObjective = new Vector<T>(columnCount);
            for (int c = 0; c < structuralColumnCount; c++)
            {
                phaseTwoObjective[c] = nonNegativeForm.Objective[c];
            }

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
                var phaseOneObjective = new Vector<T>(_columnCount);
                foreach (int column in _artificialColumns) phaseOneObjective[column] = NumOps.One;

                var phaseOneStatus = RunSimplex(phaseOneObjective, out T phaseOneOptimum);
                if (phaseOneStatus == LinearProgramStatus.IterationLimit)
                {
                    return NoPoint(LinearProgramStatus.IterationLimit);
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
                bool isDegenerate = NumOps.LessThanOrEquals(
                    NumOps.Abs(_tableau[leavingRow, _columnCount]), _tolerance);
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

        private Vector<T> ComputeReducedCosts(Vector<T> objective)
        {
            var reducedCosts = new Vector<T>(_columnCount);
            for (int c = 0; c < _columnCount; c++) reducedCosts[c] = objective[c];

            for (int r = 0; r < _constraintCount; r++)
            {
                T basicCost = objective[_basis[r]];
                if (NumOps.Equals(basicCost, NumOps.Zero)) continue;

                for (int c = 0; c < _columnCount; c++)
                {
                    reducedCosts[c] = NumOps.Subtract(
                        reducedCosts[c], NumOps.Multiply(basicCost, _tableau[r, c]));
                }
            }

            return reducedCosts;
        }

        private T ComputeObjectiveValue(Vector<T> objective)
        {
            T value = NumOps.Zero;
            for (int r = 0; r < _constraintCount; r++)
            {
                value = NumOps.Add(
                    value, NumOps.Multiply(objective[_basis[r]], _tableau[r, _columnCount]));
            }

            return value;
        }

        /// <summary>
        /// Chooses the variable to bring into the basis: the most negative reduced cost (Dantzig),
        /// or the lowest-index negative one when Bland's rule is in force.
        /// </summary>
        private int SelectEnteringColumn(Vector<T> reducedCosts, bool useBlandsRule)
        {
            int best = -1;
            T bestValue = NumOps.Negate(_tolerance);
            T threshold = NumOps.Negate(_tolerance);

            for (int c = 0; c < _columnCount; c++)
            {
                // Artificial variables must never re-enter after phase one drove them out; doing so
                // could leave the final point infeasible for the real problem.
                if (!_phaseOneActive && _artificialColumns.Contains(c)) continue;

                if (!NumOps.LessThan(reducedCosts[c], threshold)) continue;

                if (useBlandsRule) return c;

                if (NumOps.LessThan(reducedCosts[c], bestValue))
                {
                    bestValue = reducedCosts[c];
                    best = c;
                }
            }

            return best;
        }

        /// <summary>
        /// Chooses the basic variable to leave by the minimum-ratio test, breaking ties by lowest
        /// basis index (Bland's rule) so the two rules together cannot cycle.
        /// </summary>
        private int SelectLeavingRow(int enteringColumn, bool useBlandsRule)
        {
            int best = -1;
            T bestRatio = NumOps.Zero;

            for (int r = 0; r < _constraintCount; r++)
            {
                T pivotCandidate = _tableau[r, enteringColumn];
                if (!NumOps.GreaterThan(pivotCandidate, _tolerance)) continue;

                T ratio = NumOps.Divide(_tableau[r, _columnCount], pivotCandidate);

                if (best < 0 || NumOps.LessThan(ratio, bestRatio))
                {
                    best = r;
                    bestRatio = ratio;
                }
                else if (useBlandsRule
                    && NumOps.LessThanOrEquals(NumOps.Abs(NumOps.Subtract(ratio, bestRatio)), _tolerance)
                    && _basis[r] < _basis[best])
                {
                    best = r;
                }
            }

            return best;
        }

        /// <summary>
        /// Performs the Gauss-Jordan elimination that makes <paramref name="enteringColumn"/> a unit
        /// column with its 1 in <paramref name="pivotRow"/>.
        /// </summary>
        private void Pivot(int pivotRow, int enteringColumn)
        {
            T pivotValue = _tableau[pivotRow, enteringColumn];
            int width = _columnCount + 1;

            for (int c = 0; c < width; c++)
            {
                _tableau[pivotRow, c] = NumOps.Divide(_tableau[pivotRow, c], pivotValue);
            }

            for (int r = 0; r < _constraintCount; r++)
            {
                if (r == pivotRow) continue;

                T factor = _tableau[r, enteringColumn];
                if (NumOps.Equals(factor, NumOps.Zero)) continue;

                for (int c = 0; c < width; c++)
                {
                    _tableau[r, c] = NumOps.Subtract(
                        _tableau[r, c], NumOps.Multiply(factor, _tableau[pivotRow, c]));
                }
            }
        }

        /// <summary>
        /// Pivots any artificial variable still sitting in the basis at value zero out of it, so
        /// phase two never has to consider artificial columns.
        /// </summary>
        private bool DriveArtificialsOutOfBasis()
        {
            for (int r = 0; r < _constraintCount; r++)
            {
                if (!_artificialColumns.Contains(_basis[r])) continue;

                int replacement = -1;
                T largestMagnitude = NumOps.Zero;
                for (int c = 0; c < _columnCount; c++)
                {
                    if (_artificialColumns.Contains(c)) continue;
                    T magnitude = NumOps.Abs(_tableau[r, c]);
                    if (NumOps.GreaterThan(magnitude, _tolerance) &&
                        (replacement < 0 || NumOps.GreaterThan(magnitude, largestMagnitude)))
                    {
                        replacement = c;
                        largestMagnitude = magnitude;
                    }
                }

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
            var standardValues = new Vector<T>(_columnCount);
            for (int r = 0; r < _constraintCount; r++)
            {
                standardValues[_basis[r]] = _tableau[r, _columnCount];
            }

            // Undo the shift, reflection and splitting to recover the caller's variables.
            return _nonNegativeForm.RecoverOriginalVariables(standardValues);
        }

        /// <summary>
        /// Recovers the dual value of each original constraint from the final tableau.
        /// </summary>
        /// <remarks>
        /// At optimality the negated reduced cost of a constraint's auxiliary column is that
        /// constraint's dual value — the tableau form of <c>y = c_Bᵀ B⁻¹</c>. Rows that were negated
        /// to make their right-hand side non-negative have their sign flipped back.
        /// </remarks>
        private (Vector<T>? Inequality, Vector<T>? Equality) ExtractDualValues(Vector<T> objective)
        {
            var reducedCosts = ComputeReducedCosts(objective);

            Vector<T>? inequalityDuals = null;
            if (_inequalitySlackColumns.Length > 0)
            {
                inequalityDuals = new Vector<T>(_inequalitySlackColumns.Length);
                for (int r = 0; r < _inequalitySlackColumns.Length; r++)
                {
                    int column = _inequalitySlackColumns[r];
                    if (column < 0)
                    {
                        inequalityDuals[r] = NumOps.Zero;
                        continue;
                    }

                    T dual = NumOps.Negate(reducedCosts[column]);
                    inequalityDuals[r] = _rowWasNegated[r] ? NumOps.Negate(dual) : dual;
                }
            }

            Vector<T>? equalityDuals = null;
            if (_equalityArtificialColumns.Length > 0)
            {
                int inequalityRowCount = _inequalitySlackColumns.Length;
                equalityDuals = new Vector<T>(_equalityArtificialColumns.Length);
                for (int r = 0; r < _equalityArtificialColumns.Length; r++)
                {
                    int column = _equalityArtificialColumns[r];
                    if (column < 0)
                    {
                        equalityDuals[r] = NumOps.Zero;
                        continue;
                    }

                    T dual = NumOps.Negate(reducedCosts[column]);
                    equalityDuals[r] = _rowWasNegated[inequalityRowCount + r]
                        ? NumOps.Negate(dual)
                        : dual;
                }
            }

            return (inequalityDuals, equalityDuals);
        }
    }
}
