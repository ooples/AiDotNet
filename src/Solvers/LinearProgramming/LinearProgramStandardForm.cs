using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Solvers.LinearProgramming;

/// <summary>
/// A <see cref="LinearProgram{T}"/> rewritten over non-negative variables with a non-negative
/// right-hand side, which is the shape every linear-programming algorithm expects as its input.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Three rewrites get any problem into this shape:
/// </para>
/// <list type="number">
/// <item><b>Bounds become shifts, reflections and rows.</b> A variable with a finite lower bound is
/// shifted so its floor is zero. One bounded above only is reflected. One bounded on neither side
/// is split into a difference of two non-negative parts. A finite upper bound then becomes an
/// ordinary inequality row.</item>
/// <item><b>Constant terms move to the right-hand side.</b> Shifting a variable leaves a constant
/// in every row that referenced it, and a constant in the objective.</item>
/// <item><b>Negative right-hand sides are negated,</b> which flips that row's sense.</item>
/// </list>
/// <para>
/// What each algorithm adds afterwards differs — the simplex method needs slack, surplus and
/// artificial columns to build a starting basis, while an interior-point method needs only slacks —
/// so this stops at the point where they diverge. Sharing it keeps the fiddly bound handling in one
/// place rather than reimplemented per solver.
/// </para>
/// </remarks>
internal sealed class LinearProgramStandardForm<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// How one original variable is reconstructed from the non-negative standard-form variables.
    /// </summary>
    internal readonly struct VariableMapping
    {
        /// <summary>Column of the primary non-negative variable.</summary>
        public int PrimaryColumn { get; }

        /// <summary>
        /// Column of the negative part for a free variable, or -1 when the variable is not free.
        /// </summary>
        public int NegativePartColumn { get; }

        /// <summary>Constant added after scaling: the shift or the reflection origin.</summary>
        public T Offset { get; }

        /// <summary>+1 for a shifted variable, -1 for one reflected about its upper bound.</summary>
        public T Scale { get; }

        public VariableMapping(int primaryColumn, int negativePartColumn, T offset, T scale)
        {
            PrimaryColumn = primaryColumn;
            NegativePartColumn = negativePartColumn;
            Offset = offset;
            Scale = scale;
        }
    }

    /// <summary>Coefficient rows over the non-negative variables.</summary>
    public List<Vector<T>> Rows { get; }

    /// <summary>Right-hand side of each row, guaranteed non-negative.</summary>
    public List<T> RightHandSides { get; }

    /// <summary>True where the row is an equality, false where it is <c>≤</c> before negation.</summary>
    public List<bool> IsEquality { get; }

    /// <summary>True where the row was negated to make its right-hand side non-negative.</summary>
    public bool[] RowWasNegated { get; }

    /// <summary>Objective coefficients over the non-negative variables.</summary>
    public Vector<T> Objective { get; }

    /// <summary>Constant term the variable shifts contributed to the objective.</summary>
    public T ObjectiveOffset { get; }

    /// <summary>Number of non-negative variables.</summary>
    public int VariableCount { get; }

    /// <summary>Number of variables in the caller's original problem.</summary>
    public int OriginalVariableCount { get; }

    /// <summary>Number of rows that came from the original inequality block.</summary>
    public int InequalityRowCount { get; }

    /// <summary>Number of rows that came from the original equality block.</summary>
    public int EqualityRowCount { get; }

    private readonly List<VariableMapping> _mappings;

    private LinearProgramStandardForm(
        List<Vector<T>> rows,
        List<T> rightHandSides,
        List<bool> isEquality,
        bool[] rowWasNegated,
        Vector<T> objective,
        T objectiveOffset,
        int variableCount,
        int originalVariableCount,
        int inequalityRowCount,
        int equalityRowCount,
        List<VariableMapping> mappings)
    {
        Rows = rows;
        RightHandSides = rightHandSides;
        IsEquality = isEquality;
        RowWasNegated = rowWasNegated;
        Objective = objective;
        ObjectiveOffset = objectiveOffset;
        VariableCount = variableCount;
        OriginalVariableCount = originalVariableCount;
        InequalityRowCount = inequalityRowCount;
        EqualityRowCount = equalityRowCount;
        _mappings = mappings;
    }

    /// <summary>
    /// Converts a linear program to standard form.
    /// </summary>
    public static LinearProgramStandardForm<T> Build(LinearProgram<T> program)
    {
        int originalVariableCount = program.VariableCount;

        var mappings = new List<VariableMapping>(originalVariableCount);
        var upperBoundRows = new List<(int MappingIndex, T Limit)>();
        int nextColumn = 0;

        for (int i = 0; i < originalVariableCount; i++)
        {
            T lower = program.LowerBounds is null ? NumOps.Zero : program.LowerBounds[i];
            T upper = program.UpperBounds is null
                ? NumOps.FromDouble(double.PositiveInfinity)
                : program.UpperBounds[i];

            bool lowerIsFinite = IsFinite(lower);
            bool upperIsFinite = IsFinite(upper);

            if (lowerIsFinite)
            {
                mappings.Add(new VariableMapping(nextColumn++, -1, lower, NumOps.One));
                if (upperIsFinite)
                {
                    upperBoundRows.Add((mappings.Count - 1, NumOps.Subtract(upper, lower)));
                }
            }
            else if (upperIsFinite)
            {
                mappings.Add(new VariableMapping(
                    nextColumn++, -1, upper, NumOps.Negate(NumOps.One)));
            }
            else
            {
                int positiveColumn = nextColumn++;
                int negativeColumn = nextColumn++;
                mappings.Add(new VariableMapping(
                    positiveColumn, negativeColumn, NumOps.Zero, NumOps.One));
            }
        }

        int variableCount = nextColumn;

        int inequalityRowCount = program.InequalityMatrix?.Rows ?? 0;
        int equalityRowCount = program.EqualityMatrix?.Rows ?? 0;
        int constraintCount = inequalityRowCount + equalityRowCount + upperBoundRows.Count;

        var rows = new List<Vector<T>>(constraintCount);
        var rightHandSides = new List<T>(constraintCount);
        var isEquality = new List<bool>(constraintCount);

        for (int r = 0; r < inequalityRowCount; r++)
        {
            var (row, shift) = ProjectRow(program.InequalityMatrix!, r, mappings, variableCount);
            rows.Add(row);
            rightHandSides.Add(NumOps.Subtract(program.InequalityBounds![r], shift));
            isEquality.Add(false);
        }

        for (int r = 0; r < equalityRowCount; r++)
        {
            var (row, shift) = ProjectRow(program.EqualityMatrix!, r, mappings, variableCount);
            rows.Add(row);
            rightHandSides.Add(NumOps.Subtract(program.EqualityBounds![r], shift));
            isEquality.Add(true);
        }

        foreach (var (mappingIndex, limit) in upperBoundRows)
        {
            var row = new Vector<T>(variableCount);
            row[mappings[mappingIndex].PrimaryColumn] = NumOps.One;
            rows.Add(row);
            rightHandSides.Add(limit);
            isEquality.Add(false);
        }

        var rowWasNegated = new bool[constraintCount];
        for (int r = 0; r < constraintCount; r++)
        {
            if (NumOps.LessThan(rightHandSides[r], NumOps.Zero))
            {
                rows[r] = Negate(rows[r]);
                rightHandSides[r] = NumOps.Negate(rightHandSides[r]);
                rowWasNegated[r] = true;
            }
        }

        var objective = new Vector<T>(variableCount);
        T objectiveOffset = NumOps.Zero;
        for (int i = 0; i < originalVariableCount; i++)
        {
            T coefficient = program.Objective[i];
            var mapping = mappings[i];

            objectiveOffset = NumOps.Add(objectiveOffset, NumOps.Multiply(coefficient, mapping.Offset));
            objective[mapping.PrimaryColumn] = NumOps.Add(
                objective[mapping.PrimaryColumn], NumOps.Multiply(coefficient, mapping.Scale));

            if (mapping.NegativePartColumn >= 0)
            {
                objective[mapping.NegativePartColumn] = NumOps.Subtract(
                    objective[mapping.NegativePartColumn],
                    NumOps.Multiply(coefficient, mapping.Scale));
            }
        }

        return new LinearProgramStandardForm<T>(
            rows, rightHandSides, isEquality, rowWasNegated, objective, objectiveOffset,
            variableCount, originalVariableCount, inequalityRowCount, equalityRowCount, mappings);
    }

    /// <summary>
    /// Recovers the caller's variables from a standard-form solution.
    /// </summary>
    public Vector<T> RecoverOriginalVariables(Vector<T> standardValues)
    {
        var solution = new Vector<T>(OriginalVariableCount);
        for (int i = 0; i < OriginalVariableCount; i++)
        {
            var mapping = _mappings[i];
            T value = standardValues[mapping.PrimaryColumn];

            if (mapping.NegativePartColumn >= 0)
            {
                value = NumOps.Subtract(value, standardValues[mapping.NegativePartColumn]);
            }

            solution[i] = NumOps.Add(mapping.Offset, NumOps.Multiply(mapping.Scale, value));
        }

        return solution;
    }

    /// <summary>
    /// Rewrites a quadratic objective term <c>½ xᵀQx</c> over the non-negative variables.
    /// </summary>
    /// <param name="quadratic">The original <c>Q</c>, indexed by the caller's variables.</param>
    /// <returns>
    /// The projected quadratic, the linear coefficients the shift contributes, and the constant it
    /// contributes.
    /// </returns>
    /// <remarks>
    /// <para>
    /// The variable rewrite is the affine map <c>x = u + Mz</c>, where <c>u</c> holds the offsets and
    /// <c>M</c> holds the scales (with a second, negated entry for each free variable's negative
    /// part). Substituting gives
    /// <c>½xᵀQx = ½zᵀ(MᵀQM)z + (MᵀQu)ᵀz + ½uᵀQu</c>, so a shifted variable leaves both a linear and
    /// a constant term behind — exactly as it does in the linear part of the objective.
    /// </para>
    /// </remarks>
    public (Matrix<T> Projected, Vector<T> LinearCorrection, T Constant) ProjectQuadratic(
        Matrix<T> quadratic)
    {
        // qu[i] = (Q u)_i, over the caller's variables.
        var quadraticTimesOffset = new Vector<T>(OriginalVariableCount);
        for (int i = 0; i < OriginalVariableCount; i++)
        {
            T accumulator = NumOps.Zero;
            for (int j = 0; j < OriginalVariableCount; j++)
            {
                accumulator = NumOps.Add(
                    accumulator, NumOps.Multiply(quadratic[i, j], _mappings[j].Offset));
            }

            quadraticTimesOffset[i] = accumulator;
        }

        T constant = NumOps.Zero;
        for (int i = 0; i < OriginalVariableCount; i++)
        {
            constant = NumOps.Add(
                constant, NumOps.Multiply(_mappings[i].Offset, quadraticTimesOffset[i]));
        }

        constant = NumOps.Divide(constant, NumOps.FromDouble(2.0));

        var linearCorrection = new Vector<T>(VariableCount);
        var projected = new Matrix<T>(VariableCount, VariableCount);

        for (int i = 0; i < OriginalVariableCount; i++)
        {
            var rowMapping = _mappings[i];
            ScatterScaled(linearCorrection, rowMapping, quadraticTimesOffset[i]);

            for (int j = 0; j < OriginalVariableCount; j++)
            {
                T entry = quadratic[i, j];
                var columnMapping = _mappings[j];
                AccumulateOuter(projected, rowMapping, columnMapping, entry);
            }
        }

        return (projected, linearCorrection, constant);
    }

    /// <summary>
    /// Adds <paramref name="value"/> scaled by one variable's mapping into <paramref name="target"/>.
    /// </summary>
    private static void ScatterScaled(Vector<T> target, VariableMapping mapping, T value)
    {
        T scaled = NumOps.Multiply(mapping.Scale, value);
        target[mapping.PrimaryColumn] = NumOps.Add(target[mapping.PrimaryColumn], scaled);

        if (mapping.NegativePartColumn >= 0)
        {
            target[mapping.NegativePartColumn] =
                NumOps.Subtract(target[mapping.NegativePartColumn], scaled);
        }
    }

    /// <summary>
    /// Adds <c>entry · scaleRow · scaleColumn</c> into every standard-form cell the pair of original
    /// variables maps onto, negating the entries that belong to a free variable's negative part.
    /// </summary>
    private static void AccumulateOuter(
        Matrix<T> target, VariableMapping row, VariableMapping column, T entry)
    {
        T scaled = NumOps.Multiply(entry, NumOps.Multiply(row.Scale, column.Scale));

        target[row.PrimaryColumn, column.PrimaryColumn] =
            NumOps.Add(target[row.PrimaryColumn, column.PrimaryColumn], scaled);

        if (column.NegativePartColumn >= 0)
        {
            target[row.PrimaryColumn, column.NegativePartColumn] =
                NumOps.Subtract(target[row.PrimaryColumn, column.NegativePartColumn], scaled);
        }

        if (row.NegativePartColumn >= 0)
        {
            target[row.NegativePartColumn, column.PrimaryColumn] =
                NumOps.Subtract(target[row.NegativePartColumn, column.PrimaryColumn], scaled);

            if (column.NegativePartColumn >= 0)
            {
                target[row.NegativePartColumn, column.NegativePartColumn] =
                    NumOps.Add(target[row.NegativePartColumn, column.NegativePartColumn], scaled);
            }
        }
    }

    private static (Vector<T> Row, T Shift) ProjectRow(
        Matrix<T> matrix, int rowIndex, List<VariableMapping> mappings, int variableCount)
    {
        var row = new Vector<T>(variableCount);
        T shift = NumOps.Zero;

        for (int i = 0; i < mappings.Count; i++)
        {
            T coefficient = matrix[rowIndex, i];
            var mapping = mappings[i];

            shift = NumOps.Add(shift, NumOps.Multiply(coefficient, mapping.Offset));
            row[mapping.PrimaryColumn] = NumOps.Add(
                row[mapping.PrimaryColumn], NumOps.Multiply(coefficient, mapping.Scale));

            if (mapping.NegativePartColumn >= 0)
            {
                row[mapping.NegativePartColumn] = NumOps.Subtract(
                    row[mapping.NegativePartColumn], NumOps.Multiply(coefficient, mapping.Scale));
            }
        }

        return (row, shift);
    }

    private static Vector<T> Negate(Vector<T> vector)
    {
        var result = new Vector<T>(vector.Length);
        for (int i = 0; i < vector.Length; i++) result[i] = NumOps.Negate(vector[i]);
        return result;
    }

    private static bool IsFinite(T value)
    {
        double asDouble = NumOps.ToDouble(value);
        return !double.IsInfinity(asDouble) && !double.IsNaN(asDouble);
    }
}
