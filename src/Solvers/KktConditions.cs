using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Solvers;

/// <summary>
/// How badly a candidate point violates each of the Karush-Kuhn-Tucker optimality conditions.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// All four residuals are zero exactly when the point, together with its multipliers, satisfies the
/// KKT conditions. For a convex problem that is both necessary and sufficient for optimality, so
/// these numbers constitute a certificate: a solver's answer can be checked independently of the
/// solver that produced it.
/// </para>
/// </remarks>
public sealed class KktResidual<T>
{
    /// <summary>
    /// Gets the largest absolute component of <c>∇f(x) + Aᵀ_eq·ν + Aᵀ_ub·λ</c>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Stationarity: at an optimum the objective's gradient is exactly balanced by the pull of the
    /// active constraints, so no direction improves the objective without breaking one.
    /// </para>
    /// <para><b>For Beginners:</b> "There is nowhere left to move that helps." If you could still
    /// improve without violating a rule, you were not at the best point.
    /// </para>
    /// </remarks>
    public T Stationarity { get; }

    /// <summary>
    /// Gets the worst violation of the constraints themselves — how far outside the feasible region
    /// the point sits.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> "The answer actually obeys the rules." An answer that breaks a
    /// constraint is not a solution at all, however good its objective looks.
    /// </para>
    /// </remarks>
    public T PrimalFeasibility { get; }

    /// <summary>
    /// Gets the worst negative inequality multiplier, as a non-negative magnitude.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Dual feasibility: inequality multipliers must be non-negative. A negative one would mean the
    /// objective improves by moving off that constraint, which contradicts optimality.
    /// </para>
    /// <para><b>For Beginners:</b> "Every wall you are pressed against is actually holding you
    /// back, not helping you." A wall you could step away from to do better should not be counted
    /// as binding.
    /// </para>
    /// </remarks>
    public T DualFeasibility { get; }

    /// <summary>
    /// Gets the largest absolute value of <c>λ_i · (a_iᵀx − b_i)</c> over the inequality
    /// constraints.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Complementary slackness: for each inequality, either the constraint is tight or its
    /// multiplier is zero. A constraint with room to spare cannot have a price.
    /// </para>
    /// <para><b>For Beginners:</b> "You only pay for limits that are actually limiting you." Spare
    /// capacity is worth nothing at the margin.
    /// </para>
    /// </remarks>
    public T ComplementarySlackness { get; }

    /// <summary>Gets the largest of the four residuals.</summary>
    public T Worst { get; }

    /// <summary>
    /// Creates a KKT residual report.
    /// </summary>
    /// <param name="stationarity">Stationarity residual.</param>
    /// <param name="primalFeasibility">Primal feasibility residual.</param>
    /// <param name="dualFeasibility">Dual feasibility residual.</param>
    /// <param name="complementarySlackness">Complementary slackness residual.</param>
    /// <param name="worst">The largest of the four.</param>
    public KktResidual(
        T stationarity, T primalFeasibility, T dualFeasibility, T complementarySlackness, T worst)
    {
        Stationarity = stationarity;
        PrimalFeasibility = primalFeasibility;
        DualFeasibility = dualFeasibility;
        ComplementarySlackness = complementarySlackness;
        Worst = worst;
    }
}

/// <summary>
/// Verifies the Karush-Kuhn-Tucker optimality conditions for a constrained problem.
/// </summary>
/// <remarks>
/// <para>
/// The KKT conditions (Karush, 1939; Kuhn and Tucker, 1951) generalize "set the derivative to zero"
/// to problems with constraints. For a convex problem they are necessary and sufficient, so a point
/// satisfying them <b>is</b> the optimum — no further search can improve it.
/// </para>
/// <para>
/// This class exists so that a solver's answer can be audited independently of the solver. That
/// matters more than it sounds: an optimization bug typically produces a plausible-looking number
/// rather than an obvious failure, and the KKT residual is the cheapest way to catch one.
/// </para>
/// <para><b>For Beginners:</b> Think of it as a receipt-checker. Any solver can hand you an answer;
/// this tells you whether the answer really satisfies the four properties an optimum must have —
/// nowhere better to move, obeys the rules, prices are sensible, and you are only paying for limits
/// that actually bind.
/// </para>
/// </remarks>
public static class KktConditions
{
    /// <summary>
    /// Evaluates the KKT residuals for a quadratic objective <c>½·xᵀQx + cᵀx</c> subject to
    /// <c>A_ub·x ≤ b_ub</c> and <c>A_eq·x = b_eq</c>.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="quadratic">
    /// The matrix <c>Q</c>, or <c>null</c> for a purely linear objective (which makes this the KKT
    /// check for a linear program).
    /// </param>
    /// <param name="linear">The linear objective coefficients <c>c</c>.</param>
    /// <param name="solution">The candidate point <c>x</c>.</param>
    /// <param name="inequalityMatrix">Rows of <c>A_ub</c>, or null.</param>
    /// <param name="inequalityBounds">Right-hand side <c>b_ub</c>, or null.</param>
    /// <param name="inequalityMultipliers">Multipliers <c>λ</c> for the inequalities, or null.</param>
    /// <param name="equalityMatrix">Rows of <c>A_eq</c>, or null.</param>
    /// <param name="equalityBounds">Right-hand side <c>b_eq</c>, or null.</param>
    /// <param name="equalityMultipliers">Multipliers <c>ν</c> for the equalities, or null.</param>
    /// <returns>The four residuals and the worst of them.</returns>
    /// <exception cref="ArgumentNullException">
    /// Thrown when <paramref name="linear"/> or <paramref name="solution"/> is null.
    /// </exception>
    public static KktResidual<T> Evaluate<T>(
        Matrix<T>? quadratic,
        Vector<T> linear,
        Vector<T> solution,
        Matrix<T>? inequalityMatrix = null,
        Vector<T>? inequalityBounds = null,
        Vector<T>? inequalityMultipliers = null,
        Matrix<T>? equalityMatrix = null,
        Vector<T>? equalityBounds = null,
        Vector<T>? equalityMultipliers = null)
    {
        if (linear is null) throw new ArgumentNullException(nameof(linear));
        if (solution is null) throw new ArgumentNullException(nameof(solution));

        var numOps = MathHelper.GetNumericOperations<T>();
        int variableCount = solution.Length;
        if (variableCount == 0)
        {
            throw new ArgumentException("Solution must contain at least one variable.", nameof(solution));
        }
        if (linear.Length != variableCount)
        {
            throw new ArgumentException(
                $"Linear objective length ({linear.Length}) must match solution length ({variableCount}).",
                nameof(linear));
        }
        if (quadratic is not null &&
            (quadratic.Rows != variableCount || quadratic.Columns != variableCount))
        {
            throw new ArgumentException(
                $"Quadratic objective must be {variableCount}x{variableCount}, but was " +
                $"{quadratic.Rows}x{quadratic.Columns}.",
                nameof(quadratic));
        }

        ValidateConstraintBlock(
            inequalityMatrix, inequalityBounds, inequalityMultipliers,
            variableCount, nameof(inequalityMatrix), nameof(inequalityBounds),
            nameof(inequalityMultipliers));
        ValidateConstraintBlock(
            equalityMatrix, equalityBounds, equalityMultipliers,
            variableCount, nameof(equalityMatrix), nameof(equalityBounds),
            nameof(equalityMultipliers));

        // Stationarity: gradient of the objective plus the constraint normals weighted by their
        // multipliers must vanish.
        var gradient = new Vector<T>(variableCount);
        for (int i = 0; i < variableCount; i++)
        {
            T value = linear[i];
            if (quadratic is not null)
            {
                for (int j = 0; j < variableCount; j++)
                {
                    value = numOps.Add(value, numOps.Multiply(quadratic[i, j], solution[j]));
                }
            }

            gradient[i] = value;
        }

        if (equalityMatrix is not null && equalityMultipliers is not null)
        {
            for (int r = 0; r < equalityMatrix.Rows; r++)
            {
                for (int c = 0; c < variableCount; c++)
                {
                    gradient[c] = numOps.Add(
                        gradient[c], numOps.Multiply(equalityMatrix[r, c], equalityMultipliers[r]));
                }
            }
        }

        if (inequalityMatrix is not null && inequalityMultipliers is not null)
        {
            for (int r = 0; r < inequalityMatrix.Rows; r++)
            {
                for (int c = 0; c < variableCount; c++)
                {
                    gradient[c] = numOps.Add(
                        gradient[c], numOps.Multiply(inequalityMatrix[r, c], inequalityMultipliers[r]));
                }
            }
        }

        T stationarity = numOps.Zero;
        for (int i = 0; i < variableCount; i++)
        {
            stationarity = Max(numOps, stationarity, numOps.Abs(gradient[i]));
        }

        // Primal feasibility: constraint violations.
        T primalFeasibility = numOps.Zero;

        if (inequalityMatrix is not null && inequalityBounds is not null)
        {
            for (int r = 0; r < inequalityMatrix.Rows; r++)
            {
                T rowValue = RowDot(numOps, inequalityMatrix, r, solution);
                T violation = numOps.Subtract(rowValue, inequalityBounds[r]);
                if (numOps.GreaterThan(violation, primalFeasibility)) primalFeasibility = violation;
            }
        }

        if (equalityMatrix is not null && equalityBounds is not null)
        {
            for (int r = 0; r < equalityMatrix.Rows; r++)
            {
                T rowValue = RowDot(numOps, equalityMatrix, r, solution);
                T violation = numOps.Abs(numOps.Subtract(rowValue, equalityBounds[r]));
                primalFeasibility = Max(numOps, primalFeasibility, violation);
            }
        }

        // Dual feasibility: inequality multipliers must be non-negative.
        T dualFeasibility = numOps.Zero;
        if (inequalityMultipliers is not null)
        {
            for (int r = 0; r < inequalityMultipliers.Length; r++)
            {
                if (numOps.LessThan(inequalityMultipliers[r], numOps.Zero))
                {
                    dualFeasibility = Max(
                        numOps, dualFeasibility, numOps.Abs(inequalityMultipliers[r]));
                }
            }
        }

        // Complementary slackness: each inequality is tight or priced at zero.
        T complementarySlackness = numOps.Zero;
        if (inequalityMatrix is not null && inequalityBounds is not null
            && inequalityMultipliers is not null)
        {
            for (int r = 0; r < inequalityMatrix.Rows; r++)
            {
                T slack = numOps.Subtract(
                    RowDot(numOps, inequalityMatrix, r, solution), inequalityBounds[r]);
                T product = numOps.Abs(numOps.Multiply(inequalityMultipliers[r], slack));
                complementarySlackness = Max(numOps, complementarySlackness, product);
            }
        }

        T worst = Max(numOps, Max(numOps, stationarity, primalFeasibility),
            Max(numOps, dualFeasibility, complementarySlackness));

        return new KktResidual<T>(
            stationarity, primalFeasibility, dualFeasibility, complementarySlackness, worst);
    }

    private static void ValidateConstraintBlock<T>(
        Matrix<T>? matrix,
        Vector<T>? bounds,
        Vector<T>? multipliers,
        int variableCount,
        string matrixName,
        string boundsName,
        string multipliersName)
    {
        bool anySupplied = matrix is not null || bounds is not null || multipliers is not null;
        bool allSupplied = matrix is not null && bounds is not null && multipliers is not null;
        if (anySupplied && !allSupplied)
        {
            throw new ArgumentException(
                $"Constraint block must supply {matrixName}, {boundsName}, and {multipliersName} together.");
        }
        if (!allSupplied)
        {
            return;
        }

        if (matrix!.Columns != variableCount)
        {
            throw new ArgumentException(
                $"{matrixName} must have {variableCount} columns, but had {matrix.Columns}.",
                matrixName);
        }
        if (bounds!.Length != matrix.Rows)
        {
            throw new ArgumentException(
                $"{boundsName} length ({bounds.Length}) must match {matrixName} rows ({matrix.Rows}).",
                boundsName);
        }
        if (multipliers!.Length != matrix.Rows)
        {
            throw new ArgumentException(
                $"{multipliersName} length ({multipliers.Length}) must match {matrixName} rows ({matrix.Rows}).",
                multipliersName);
        }
    }

    private static T RowDot<T>(
        INumericOperations<T> numOps, Matrix<T> matrix, int row, Vector<T> vector)
    {
        T sum = numOps.Zero;
        for (int c = 0; c < matrix.Columns; c++)
        {
            sum = numOps.Add(sum, numOps.Multiply(matrix[row, c], vector[c]));
        }

        return sum;
    }

    private static T Max<T>(INumericOperations<T> numOps, T left, T right)
    {
        return numOps.GreaterThan(left, right) ? left : right;
    }
}
