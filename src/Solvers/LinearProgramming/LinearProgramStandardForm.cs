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
    public static LinearProgramStandardForm<T> Build(
        LinearProgram<T> program,
        bool treatMissingLowerBoundsAsUnbounded = false)
    {
        int originalVariableCount = program.VariableCount;
        var engine = SolverTensorOps<T>.Engine;

        // Classify every bound at once; the mapping loop below only reads plain arrays.
        bool[] lowerFinite = FiniteFlags(program.LowerBounds, originalVariableCount, !treatMissingLowerBoundsAsUnbounded);
        bool[] upperFinite = FiniteFlags(program.UpperBounds, originalVariableCount, whenMissing: false);
        // A missing bound reads as zero, as it did per element before.
        var zeros = Tensor<T>.CreateDefault(new[] { originalVariableCount }, NumOps.Zero);
        var lowerTensor = program.LowerBounds is null ? zeros : SolverTensorOps<T>.AsTensor(program.LowerBounds);
        var upperTensor = program.UpperBounds is null ? zeros : SolverTensorOps<T>.AsTensor(program.UpperBounds);
        T[] lowers = lowerTensor.ToArray();
        T[] uppers = upperTensor.ToArray();
        T[] widths = engine.TensorSubtract(upperTensor, lowerTensor).ToArray();
        T negativeOne = NumOps.Negate(NumOps.One);

        var mappings = new List<VariableMapping>(originalVariableCount);
        var upperBoundRows = new List<(int MappingIndex, T Limit)>();
        int nextColumn = 0;

        for (int i = 0; i < originalVariableCount; i++)
        {
            if (lowerFinite[i])
            {
                mappings.Add(new VariableMapping(nextColumn++, -1, lowers[i], NumOps.One));
                if (upperFinite[i])
                {
                    upperBoundRows.Add((mappings.Count - 1, widths[i]));
                }
            }
            else if (upperFinite[i])
            {
                mappings.Add(new VariableMapping(nextColumn++, -1, uppers[i], negativeOne));
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
        var layout = new MappingLayout(mappings, variableCount);

        int inequalityRowCount = program.InequalityMatrix?.Rows ?? 0;
        int equalityRowCount = program.EqualityMatrix?.Rows ?? 0;
        int constraintCount = inequalityRowCount + equalityRowCount + upperBoundRows.Count;

        var rowBlocks = new List<Tensor<T>>(3);
        var rightHandSideBlocks = new List<Tensor<T>>(3);
        if (inequalityRowCount > 0)
        {
            var matrix = SolverTensorOps<T>.AsTensor(program.InequalityMatrix ?? throw new InvalidOperationException("program.InequalityMatrix is null where it is required."));
            var bounds = SolverTensorOps<T>.AsTensor(program.InequalityBounds ?? throw new InvalidOperationException("program.InequalityBounds is null where it is required."));
            rowBlocks.Add(layout.ToStandardColumns(matrix));
            rightHandSideBlocks.Add(engine.TensorSubtract(bounds, SolverTensorOps<T>.MatVec(matrix, layout.Offsets)));
        }

        if (equalityRowCount > 0)
        {
            var matrix = SolverTensorOps<T>.AsTensor(program.EqualityMatrix ?? throw new InvalidOperationException("program.EqualityMatrix is null where it is required."));
            var bounds = SolverTensorOps<T>.AsTensor(program.EqualityBounds ?? throw new InvalidOperationException("program.EqualityBounds is null where it is required."));
            rowBlocks.Add(layout.ToStandardColumns(matrix));
            rightHandSideBlocks.Add(engine.TensorSubtract(bounds, SolverTensorOps<T>.MatVec(matrix, layout.Offsets)));
        }

        if (upperBoundRows.Count > 0)
        {
            var columns = new int[upperBoundRows.Count];
            var limits = new T[upperBoundRows.Count];
            for (int k = 0; k < columns.Length; k++)
            {
                columns[k] = mappings[upperBoundRows[k].MappingIndex].PrimaryColumn;
                limits[k] = upperBoundRows[k].Limit;
            }

            // Unit rows e_column only (a dense identity would be [n, n] to select a few rows).
            rowBlocks.Add(engine.TensorOneHot<T>(new Tensor<int>(new[] { columns.Length }, new Vector<int>(columns)), variableCount));
            rightHandSideBlocks.Add(new Tensor<T>(new[] { limits.Length }, new Vector<T>(limits)));
        }

        var rows = new List<Vector<T>>(constraintCount);
        var rightHandSides = new List<T>(constraintCount);
        var isEquality = new List<bool>(constraintCount);
        var rowWasNegated = new bool[constraintCount];
        for (int r = 0; r < inequalityRowCount; r++) isEquality.Add(false);
        for (int r = 0; r < equalityRowCount; r++) isEquality.Add(true);
        for (int r = 0; r < upperBoundRows.Count; r++) isEquality.Add(false);

        if (constraintCount > 0)
        {
            var allRows = rowBlocks.Count == 1 ? rowBlocks[0] : engine.TensorConcatenate(rowBlocks.ToArray(), 0);
            var allRightHandSides = rightHandSideBlocks.Count == 1
                ? rightHandSideBlocks[0]
                : engine.TensorConcatenate(rightHandSideBlocks.ToArray(), 0);

            // A negative right-hand side flips its whole row so every b stays non-negative.
            var negative = engine.TensorLessThan(allRightHandSides, NumOps.Zero);
            var rowMask = engine.TensorBroadcastTo(engine.Reshape(negative, new[] { constraintCount, 1 }), new[] { constraintCount, variableCount });
            allRows = engine.TensorWhere(rowMask, engine.TensorNegate(allRows), allRows);
            allRightHandSides = engine.TensorWhere(negative, engine.TensorNegate(allRightHandSides), allRightHandSides);
            foreach (int r in engine.TensorNonzero(negative).ToArray()) rowWasNegated[r] = true;

            for (int r = 0; r < constraintCount; r++) rows.Add(engine.TensorSliceAxis(allRows, 0, r).ToVector());
            rightHandSides.AddRange(allRightHandSides.ToArray());
        }

        var coefficients = engine.Reshape(SolverTensorOps<T>.AsTensor(program.Objective), new[] { 1, originalVariableCount });
        var objective = engine.Reshape(layout.ToStandardColumns(coefficients), new[] { variableCount }).ToVector();
        T objectiveOffset = engine.TensorSum(engine.TensorMultiply(SolverTensorOps<T>.AsTensor(program.Objective), layout.Offsets));

        return new LinearProgramStandardForm<T>(
            rows, rightHandSides, isEquality, rowWasNegated, objective, objectiveOffset,
            variableCount, originalVariableCount, inequalityRowCount, equalityRowCount, mappings);
    }

    /// <summary>
    /// Recovers the caller's variables from a standard-form solution.
    /// </summary>
    public Vector<T> RecoverOriginalVariables(Vector<T> standardValues)
    {
        var engine = SolverTensorOps<T>.Engine;
        var layout = new MappingLayout(_mappings, VariableCount);
        var value = layout.FromStandard(SolverTensorOps<T>.AsTensor(standardValues));
        return engine.TensorAdd(layout.Offsets, engine.TensorMultiply(layout.Scales, value)).ToVector();
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
        var engine = SolverTensorOps<T>.Engine;
        var layout = new MappingLayout(_mappings, VariableCount);
        var q = SolverTensorOps<T>.AsTensor(quadratic);

        // qu[i] = (Q u)_i, over the caller's variables.
        var quadraticTimesOffset = SolverTensorOps<T>.MatVec(q, layout.Offsets);

        T constant = engine.TensorSum(engine.TensorMultiply(layout.Offsets, quadraticTimesOffset));
        constant = NumOps.Divide(constant, NumOps.FromDouble(2.0));

        var linearCorrection = engine.Reshape(
            layout.ToStandardColumns(engine.Reshape(quadraticTimesOffset, new[] { 1, OriginalVariableCount })),
            new[] { VariableCount }).ToVector();

        // Every (i, j) pair lands on its own standard-form cells, so mapping the columns and then the rows
        // places entry * scaleRow * scaleColumn exactly where the per-entry scatter did (scales are +-1).
        var columnsMapped = layout.ToStandardColumns(q);
        var projected = engine.TensorTranspose(layout.ToStandardColumns(engine.TensorTranspose(columnsMapped)).Contiguous());

        return (projected.Contiguous().ToMatrix(), linearCorrection, constant);
    }

    /// <summary>
    /// The variable map <c>x = u + Mz</c> as index tensors: each original variable's primary column, and for a
    /// free variable the column of its negative part.
    /// </summary>
    private sealed class MappingLayout
    {
        private readonly Tensor<int> _primary;
        private readonly Tensor<int>? _negativeSource;
        private readonly Tensor<int>? _negativeColumn;
        private readonly int _variableCount;

        public MappingLayout(List<VariableMapping> mappings, int variableCount)
        {
            _variableCount = variableCount;
            var primary = new int[mappings.Count];
            var offsets = new T[mappings.Count];
            var scales = new T[mappings.Count];
            var negativeSource = new List<int>();
            var negativeColumn = new List<int>();
            for (int i = 0; i < mappings.Count; i++)
            {
                var mapping = mappings[i];
                primary[i] = mapping.PrimaryColumn;
                offsets[i] = mapping.Offset;
                scales[i] = mapping.Scale;
                if (mapping.NegativePartColumn >= 0)
                {
                    negativeSource.Add(i);
                    negativeColumn.Add(mapping.NegativePartColumn);
                }
            }

            _primary = new Tensor<int>(new[] { primary.Length }, new Vector<int>(primary));
            Offsets = new Tensor<T>(new[] { offsets.Length }, new Vector<T>(offsets));
            Scales = new Tensor<T>(new[] { scales.Length }, new Vector<T>(scales));
            if (negativeSource.Count > 0)
            {
                _negativeSource = new Tensor<int>(new[] { negativeSource.Count }, new Vector<int>(negativeSource.ToArray()));
                _negativeColumn = new Tensor<int>(new[] { negativeColumn.Count }, new Vector<int>(negativeColumn.ToArray()));
            }
        }

        /// <summary>The offsets <c>u</c>, one per original variable.</summary>
        public Tensor<T> Offsets { get; }

        /// <summary>The scales (+1 or -1), one per original variable.</summary>
        public Tensor<T> Scales { get; }

        /// <summary>
        /// Rewrites the columns of <paramref name="x"/> ([rows, original]) over the standard-form variables:
        /// <c>x[:, i] * scale_i</c> into the primary column and its negation into the negative-part column.
        /// </summary>
        public Tensor<T> ToStandardColumns(Tensor<T> x)
        {
            var engine = SolverTensorOps<T>.Engine;
            int rows = x.Shape[0];
            // Explicit broadcast: equal shapes keep the product valid under ShapePolicy.Strict().
            var scaled = engine.TensorMultiply(
                x, engine.TensorBroadcastTo(engine.Reshape(Scales, new[] { 1, Scales.Length }), x.Shape.ToArray()));
            var result = engine.TensorIndexAdd(
                Tensor<T>.CreateDefault(new[] { rows, _variableCount }, NumOps.Zero), 1, _primary, scaled);
            if (_negativeSource is not null && _negativeColumn is not null)
            {
                result = engine.TensorIndexAdd(
                    result, 1, _negativeColumn, engine.TensorNegate(engine.TensorIndexSelect(scaled, _negativeSource, 1)));
            }

            return result;
        }

        /// <summary>
        /// The mapped value of each original variable from a standard-form point: its primary column, minus its
        /// negative-part column for a free variable.
        /// </summary>
        public Tensor<T> FromStandard(Tensor<T> standardValues)
        {
            var engine = SolverTensorOps<T>.Engine;
            var value = engine.TensorTake(standardValues, _primary);
            if (_negativeSource is not null && _negativeColumn is not null)
            {
                value = engine.TensorIndexAdd(
                    value, 0, _negativeSource, engine.TensorNegate(engine.TensorTake(standardValues, _negativeColumn)));
            }

            return value;
        }
    }

    private static bool[] FiniteFlags(Vector<T>? bounds, int count, bool whenMissing)
    {
        var flags = new bool[count];
        if (bounds is null)
        {
            if (whenMissing)
            {
                for (int i = 0; i < count; i++) flags[i] = true;
            }

            return flags;
        }

        for (int i = 0; i < count; i++) flags[i] = true;
        if (count == 0) return flags;
        var nonFinite = SolverTensorOps<T>.NonFinite(SolverTensorOps<T>.AsTensor(bounds));
        foreach (int i in SolverTensorOps<T>.Engine.TensorNonzero(nonFinite).ToArray()) flags[i] = false;
        return flags;
    }
}
