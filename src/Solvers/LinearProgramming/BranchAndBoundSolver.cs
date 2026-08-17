using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Solvers.LinearProgramming;

/// <summary>
/// Solves integer and mixed-integer linear programs by branch and bound over linear relaxations.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Implements branch and bound (Land and Doig, 1960) with best-bound node selection and branching
/// on the most fractional variable. At each node the integrality requirement is dropped and the
/// resulting linear program is solved:
/// <list type="bullet">
/// <item>an infeasible relaxation means the node's whole subtree is infeasible — prune;</item>
/// <item>a relaxation bound no better than the best whole-number solution found so far means the
/// subtree cannot contain a better one — prune, because relaxing constraints can only improve the
/// objective, so the relaxation is always at least as good as anything below it;</item>
/// <item>an all-integral relaxation solution is a genuine candidate — record it;</item>
/// <item>otherwise pick a variable sitting at a fractional value <c>v</c> and split the node into
/// <c>x ≤ ⌊v⌋</c> and <c>x ≥ ⌈v⌉</c>, which excludes the fractional point without excluding any
/// whole-number one.</item>
/// </list>
/// </para>
/// <para>
/// Because relaxation bounds prune whole subtrees, the search usually visits a small fraction of
/// the exponentially many possibilities — this bounding step is the entire reason branch and bound
/// is practical.
/// </para>
/// <para><b>For Beginners:</b> Imagine choosing how many of each item to produce, where fractions
/// are meaningless. The solver first ignores the whole-number rule and solves the easy version. If
/// the answer happens to be whole numbers, it is done. If some quantity comes back as 3.4, it knows
/// the true answer has that quantity at 3 or less, or at 4 or more — so it splits into those two
/// worlds and explores each. The trick that makes this fast: the easy version's answer is always at
/// least as good as any whole-number answer beneath it, so if it is already worse than something
/// found earlier, that entire world can be discarded unexamined.
/// </para>
/// </remarks>
public sealed class BranchAndBoundSolver<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly BranchAndBoundSolverOptions _options;
    private readonly ILinearProgramSolver<T> _relaxationSolver;

    private sealed class NodeBounds
    {
        public Vector<T> LowerValues { get; }
        public Vector<T> UpperValues { get; }
        public bool[] HasLower { get; }
        public bool[] HasUpper { get; }

        public NodeBounds(int variableCount)
        {
            LowerValues = new Vector<T>(variableCount);
            UpperValues = new Vector<T>(variableCount);
            HasLower = new bool[variableCount];
            HasUpper = new bool[variableCount];
        }

        private NodeBounds(NodeBounds other)
        {
            LowerValues = other.LowerValues.Clone();
            UpperValues = other.UpperValues.Clone();
            HasLower = (bool[])other.HasLower.Clone();
            HasUpper = (bool[])other.HasUpper.Clone();
        }

        public NodeBounds Clone() => new(this);
    }

    /// <summary>
    /// Creates a branch-and-bound solver.
    /// </summary>
    /// <param name="options">
    /// Solver configuration. When omitted, the documented defaults on
    /// <see cref="BranchAndBoundSolverOptions"/> are used.
    /// </param>
    /// <param name="relaxationSolver">
    /// The solver used for the linear relaxation at each node. When omitted, a
    /// <see cref="SimplexSolver{T}"/> configured from
    /// <see cref="BranchAndBoundSolverOptions.RelaxationOptions"/> is used. Supply an interior-point
    /// solver here instead when the relaxations are large.
    /// </param>
    public BranchAndBoundSolver(
        BranchAndBoundSolverOptions? options = null,
        ILinearProgramSolver<T>? relaxationSolver = null)
    {
        _options = options ?? new BranchAndBoundSolverOptions();
        _relaxationSolver = relaxationSolver ?? new SimplexSolver<T>(_options.RelaxationOptions);
    }

    /// <summary>
    /// Solves an integer or mixed-integer linear program.
    /// </summary>
    /// <param name="program">The problem to solve.</param>
    /// <returns>
    /// The best whole-number solution found. The status is
    /// <see cref="LinearProgramStatus.Optimal"/> when the search completed and proved optimality,
    /// <see cref="LinearProgramStatus.IterationLimit"/> when the node budget ran out with a
    /// candidate in hand, <see cref="LinearProgramStatus.Infeasible"/> when no whole-number point
    /// satisfies the constraints, and <see cref="LinearProgramStatus.Unbounded"/> when the
    /// relaxation itself is unbounded.
    /// </returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="program"/> is null.</exception>
    public LinearProgramSolution<T> Solve(IntegerProgram<T> program)
    {
        if (program is null) throw new ArgumentNullException(nameof(program));

        var integralityTolerance = NumOps.FromDouble(_options.IntegralityTolerance);
        var minimumImprovement = NumOps.FromDouble(_options.MinimumImprovement);

        Vector<T>? incumbent = null;
        T incumbentObjective = NumOps.Zero;
        bool hasIncumbent = false;
        bool exhaustedNodeBudget = false;
        int nodesExplored = 0;

        // Each node is the original problem with tightened variable bounds. Explored best-bound
        // first: the most promising relaxation is expanded next, which finds good incumbents early
        // and therefore prunes more.
        var frontier = new PriorityQueue<NodeBounds, T>(
            Comparer<T>.Create((left, right) => NumOps.Compare(left, right)));
        frontier.Enqueue(new NodeBounds(program.Relaxation.VariableCount), NumOps.MinValue);

        while (frontier.Count > 0)
        {
            if (nodesExplored >= _options.MaxNodes)
            {
                exhaustedNodeBudget = true;
                break;
            }

            var node = frontier.Dequeue();
            nodesExplored++;

            var relaxation = WithBranchBounds(program.Relaxation, node);
            var relaxed = _relaxationSolver.Solve(relaxation);

            if (relaxed.Status == LinearProgramStatus.Infeasible)
            {
                continue;
            }

            if (relaxed.Status == LinearProgramStatus.Unbounded)
            {
                // An unbounded relaxation at the root means the integer problem is unbounded too.
                // Deeper nodes only ever tighten bounds, so this can only happen at the root.
                return new LinearProgramSolution<T>(
                    LinearProgramStatus.Unbounded, null, NumOps.Zero, nodesExplored);
            }

            if (relaxed.Solution is null)
            {
                continue;
            }

            // Bounding: the relaxation is a lower bound on everything in this subtree, so if it is
            // not better than the incumbent, nothing below can be either.
            if (hasIncumbent &&
                !NumOps.LessThan(
                    relaxed.ObjectiveValue, NumOps.Subtract(incumbentObjective, minimumImprovement)))
            {
                continue;
            }

            int branchVariable = SelectBranchVariable(
                relaxed.Solution, program.IntegralityMask, integralityTolerance);

            if (branchVariable < 0)
            {
                // Every integer-constrained variable landed on a whole number: a real candidate.
                if (!hasIncumbent ||
                    NumOps.LessThan(
                        relaxed.ObjectiveValue,
                        NumOps.Subtract(incumbentObjective, minimumImprovement)))
                {
                    incumbent = RoundIntegralComponents(
                        relaxed.Solution, program.IntegralityMask);
                    incumbentObjective = relaxed.ObjectiveValue;
                    hasIncumbent = true;
                }

                continue;
            }

            // Branch: x_j <= floor(v) and x_j >= ceil(v). The fractional point satisfies neither,
            // and every whole-number point satisfies exactly one, so nothing is lost.
            T value = relaxed.Solution[branchVariable];
            T floorValue = NumOps.FromDouble(Math.Floor(NumOps.ToDouble(value)));
            T ceilingValue = NumOps.FromDouble(Math.Ceiling(NumOps.ToDouble(value)));

            var lowerBranch = node.Clone();
            if (!TryGetUpperBound(program.Relaxation, node, branchVariable, out T currentUpper) ||
                NumOps.LessThan(floorValue, currentUpper))
            {
                lowerBranch.UpperValues[branchVariable] = floorValue;
                lowerBranch.HasUpper[branchVariable] = true;
            }
            else
            {
                lowerBranch.UpperValues[branchVariable] = currentUpper;
                lowerBranch.HasUpper[branchVariable] = true;
            }

            if (!TryGetLowerBound(program.Relaxation, node, branchVariable, out T effectiveLower) ||
                !NumOps.GreaterThan(effectiveLower, lowerBranch.UpperValues[branchVariable]))
            {
                frontier.Enqueue(lowerBranch, relaxed.ObjectiveValue);
            }

            var upperBranch = node.Clone();
            if (!TryGetLowerBound(program.Relaxation, node, branchVariable, out T currentLower) ||
                NumOps.GreaterThan(ceilingValue, currentLower))
            {
                upperBranch.LowerValues[branchVariable] = ceilingValue;
                upperBranch.HasLower[branchVariable] = true;
            }
            else
            {
                upperBranch.LowerValues[branchVariable] = currentLower;
                upperBranch.HasLower[branchVariable] = true;
            }

            if (!TryGetUpperBound(program.Relaxation, node, branchVariable, out T effectiveUpper) ||
                !NumOps.GreaterThan(upperBranch.LowerValues[branchVariable], effectiveUpper))
            {
                frontier.Enqueue(upperBranch, relaxed.ObjectiveValue);
            }
        }

        if (!hasIncumbent)
        {
            // Only a search that ran to completion can prove infeasibility. One that was cut short
            // by the node budget simply does not know, and saying "infeasible" there would assert
            // that no whole-number point exists when unexplored branches might still contain one.
            return new LinearProgramSolution<T>(
                exhaustedNodeBudget ? LinearProgramStatus.IterationLimit : LinearProgramStatus.Infeasible,
                null,
                NumOps.Zero,
                nodesExplored);
        }

        // Only claim optimality when the search actually closed; a truncated search may have left a
        // better whole-number solution unexplored.
        var status = exhaustedNodeBudget
            ? LinearProgramStatus.IterationLimit
            : LinearProgramStatus.Optimal;

        return new LinearProgramSolution<T>(status, incumbent, incumbentObjective, nodesExplored);
    }

    /// <summary>
    /// Picks the integer-constrained variable furthest from a whole number, which is the split most
    /// likely to tighten the bound on both sides.
    /// </summary>
    private static int SelectBranchVariable(
        Vector<T> solution, IReadOnlyList<bool> integralityMask, T tolerance)
    {
        int mostFractional = -1;
        double worstDistance = 0.0;

        for (int i = 0; i < solution.Length; i++)
        {
            if (!integralityMask[i]) continue;

            double value = NumOps.ToDouble(solution[i]);
            double distance = Math.Abs(value - Math.Round(value));
            if (distance <= NumOps.ToDouble(tolerance)) continue;

            if (distance > worstDistance)
            {
                worstDistance = distance;
                mostFractional = i;
            }
        }

        return mostFractional;
    }

    /// <summary>
    /// Snaps integer-constrained components onto exact whole numbers, removing the floating-point
    /// residue left by the relaxation so callers get 3 rather than 2.9999999997.
    /// </summary>
    private static Vector<T> RoundIntegralComponents(
        Vector<T> solution, IReadOnlyList<bool> integralityMask)
    {
        var rounded = solution.Clone();
        for (int i = 0; i < rounded.Length; i++)
        {
            if (!integralityMask[i]) continue;
            rounded[i] = NumOps.FromDouble(Math.Round(NumOps.ToDouble(rounded[i])));
        }

        return rounded;
    }

    private static bool TryGetLowerBound(
        LinearProgram<T> program, NodeBounds node, int variable, out T value)
    {
        bool hasProgramBound = program.LowerBounds is null || IsFinite(program.LowerBounds[variable]);
        T programBound = program.LowerBounds is null ? NumOps.Zero : program.LowerBounds[variable];

        if (node.HasLower[variable] && hasProgramBound)
        {
            value = Maximum(node.LowerValues[variable], programBound);
            return true;
        }

        if (node.HasLower[variable])
        {
            value = node.LowerValues[variable];
            return true;
        }

        value = programBound;
        return hasProgramBound;
    }

    private static bool TryGetUpperBound(
        LinearProgram<T> program, NodeBounds node, int variable, out T value)
    {
        bool hasProgramBound = program.UpperBounds is not null && IsFinite(program.UpperBounds[variable]);
        T programBound = program.UpperBounds is null ? NumOps.Zero : program.UpperBounds[variable];

        if (node.HasUpper[variable] && hasProgramBound)
        {
            value = Minimum(node.UpperValues[variable], programBound);
            return true;
        }

        if (node.HasUpper[variable])
        {
            value = node.UpperValues[variable];
            return true;
        }

        value = programBound;
        return hasProgramBound;
    }

    private static LinearProgram<T> WithBranchBounds(LinearProgram<T> program, NodeBounds node)
    {
        int existingRows = program.InequalityMatrix?.Rows ?? 0;
        int branchRows = node.HasLower.Count(has => has) + node.HasUpper.Count(has => has);
        if (branchRows == 0) return program;

        var matrix = new Matrix<T>(existingRows + branchRows, program.VariableCount);
        var bounds = new Vector<T>(existingRows + branchRows);

        for (int row = 0; row < existingRows; row++)
        {
            for (int column = 0; column < program.VariableCount; column++)
            {
                matrix[row, column] = program.InequalityMatrix![row, column];
            }
            bounds[row] = program.InequalityBounds![row];
        }

        int nextRow = existingRows;
        for (int variable = 0; variable < program.VariableCount; variable++)
        {
            if (node.HasUpper[variable])
            {
                matrix[nextRow, variable] = NumOps.One;
                bounds[nextRow++] = node.UpperValues[variable];
            }

            if (node.HasLower[variable])
            {
                matrix[nextRow, variable] = NumOps.Negate(NumOps.One);
                bounds[nextRow++] = NumOps.Negate(node.LowerValues[variable]);
            }
        }

        return new LinearProgram<T>(
            program.Objective,
            matrix,
            bounds,
            program.EqualityMatrix,
            program.EqualityBounds,
            program.LowerBounds,
            program.UpperBounds);
    }

    private static bool IsFinite(T value)
    {
        double converted = NumOps.ToDouble(value);
        return !double.IsNaN(converted) && !double.IsInfinity(converted);
    }

    private static T Minimum(T left, T right) => NumOps.LessThan(left, right) ? left : right;

    private static T Maximum(T left, T right) => NumOps.GreaterThan(left, right) ? left : right;
}
