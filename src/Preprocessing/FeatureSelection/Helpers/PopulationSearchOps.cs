using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Helpers;

/// <summary>
/// Whole-array engine helpers shared by the population-based feature selectors (swarm and evolutionary
/// metaheuristics, multi-objective searches, annealing): target correlations, batched subset fitness, row updates
/// and random draws that consume the caller's <see cref="Random"/> in the same order as a scalar loop.
/// </summary>
internal static class PopulationSearchOps
{
    /// <summary>
    /// |Pearson r| of every column of <paramref name="data"/> with <paramref name="target"/>. A column whose sum of
    /// squared deviations is at most 1e-10, or a target whose sum is, gets <paramref name="undefined"/>.
    /// </summary>
    public static Tensor<T> AbsTargetCorrelations<T>(Matrix<T> data, Vector<T> target, T undefined)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = data.Rows;
        int p = data.Columns;
        var x = Tensor<T>.FromMatrix(data);
        var y = engine.Reshape(Tensor<T>.FromVector(target), new[] { n, 1 });
        var xc = engine.TensorSubtract(x, engine.ReduceMean(x, new[] { 0 }, true));
        var yc = engine.TensorSubtract(y, engine.ReduceMean(y, new[] { 0 }, true));
        var sxy = engine.ReduceSum(engine.TensorMultiply(xc, engine.TensorBroadcastTo(yc, new[] { n, p })), new[] { 0 }, false);
        var sxx = engine.ReduceSum(engine.TensorSquare(xc), new[] { 0 }, false);
        var syy = engine.TensorBroadcastTo(engine.ReduceSum(engine.TensorSquare(yc), new[] { 0 }, false), new[] { p });
        T tiny = ops.FromDouble(1e-10);
        var defined = engine.TensorMultiply(engine.TensorGreaterThan(sxx, tiny), engine.TensorGreaterThan(syy, tiny));
        var safe = engine.TensorWhere(
            defined, engine.TensorMultiply(engine.TensorSqrt(sxx), engine.TensorSqrt(syy)), PreprocessingTensorOps.Filled(new[] { p }, ops.One));
        return engine.TensorWhere(
            defined, engine.TensorAbs(engine.TensorDivide(sxy, safe)), PreprocessingTensorOps.Filled(new[] { p }, undefined));
    }

    /// <summary>
    /// Fitness of every row of a 0/1 selection mask [m, p]: the mean score of the selected features minus
    /// <paramref name="penalty"/> times |count - <paramref name="targetCount"/>|, or <paramref name="emptyValue"/>
    /// for a row that selects nothing. Returns [m].
    /// </summary>
    public static Tensor<T> SubsetFitness<T>(Tensor<T> mask, Tensor<T> scores, double targetCount, T penalty, T emptyValue)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int m = mask.Shape[0];
        int p = mask.Shape[1];
        var total = engine.Reshape(engine.TensorMatMul(mask, engine.Reshape(scores, new[] { p, 1 })), new[] { m });
        // counted in int: a 0/1 sum in T stops counting at 2048 for Half
        var count = engine.ReduceSum(mask.Cast<int>(), new[] { 1 }, false).Cast<T>();
        var any = engine.TensorGreaterThan(count, ops.Zero);
        var safeCount = engine.TensorWhere(any, count, PreprocessingTensorOps.Filled(new[] { m }, ops.One));
        var deviation = engine.TensorAbs(engine.TensorAddScalar(count, ops.FromDouble(-targetCount)));
        var fitness = engine.TensorSubtract(engine.TensorDivide(total, safeCount), engine.TensorMultiplyScalar(deviation, penalty));
        return engine.TensorWhere(any, fitness, PreprocessingTensorOps.Filled(new[] { m }, emptyValue));
    }

    /// <summary>
    /// <see cref="SubsetFitness{T}"/> of continuous positions [m, p], where a feature is selected when its position
    /// exceeds 0.5 and an empty selection scores 0.
    /// </summary>
    public static Tensor<T> ThresholdFitness<T>(Tensor<T> positions, Tensor<T> scores, double targetCount, double penalty)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        var mask = engine.TensorGreaterThan(positions, ops.FromDouble(0.5));
        return SubsetFitness(mask, scores, targetCount, ops.FromDouble(penalty), ops.Zero);
    }

    /// <summary>
    /// Uniform [0, 1) draws of the given shape, taken from <paramref name="random"/> with one
    /// <see cref="Random.NextDouble"/> per element in row-major order, so the stream matches a scalar loop.
    /// </summary>
    public static Tensor<T> Uniform<T>(Random random, params int[] shape) => Tensor<T>.CreateRandom(random, shape);

    /// <summary>Element-wise <c>x * scale + shift</c>.</summary>
    public static Tensor<T> Affine<T>(Tensor<T> x, double scale, double shift)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        return engine.TensorAddScalar(engine.TensorMultiplyScalar(x, ops.FromDouble(scale)), ops.FromDouble(shift));
    }

    /// <summary>Element-wise clamp to [0, 1].</summary>
    public static Tensor<T> Clamp01<T>(Tensor<T> x)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        return AiDotNetEngine.Current.TensorClamp(x, ops.Zero, ops.One);
    }

    /// <summary>Row <paramref name="row"/> of a [m, p] tensor as a [p] tensor.</summary>
    public static Tensor<T> Row<T>(Tensor<T> x, int row)
    {
        var engine = AiDotNetEngine.Current;
        int p = x.Shape[1];
        return engine.Reshape(engine.TensorSlice(x, new[] { row, 0 }, new[] { 1, p }), new[] { p });
    }

    /// <summary>The rows <paramref name="rows"/> of a [m, p] tensor, in that order.</summary>
    public static Tensor<T> Rows<T>(Tensor<T> x, int[] rows) =>
        AiDotNetEngine.Current.TensorIndexSelect(x, new Tensor<int>((int[])rows.Clone(), new[] { rows.Length }), 0);

    /// <summary>Element <paramref name="index"/> of a [m] tensor as a [1] tensor.</summary>
    public static Tensor<T> At<T>(Tensor<T> v, int index) =>
        AiDotNetEngine.Current.TensorSlice(v, new[] { index }, new[] { 1 });

    /// <summary>
    /// Replaces the rows of <paramref name="x"/> [m, p] where <paramref name="rowMask"/> [m] is nonzero with the
    /// matching rows of <paramref name="values"/> ([m, p], or one [p] row for all of them).
    /// </summary>
    public static Tensor<T> ReplaceRows<T>(Tensor<T> x, Tensor<T> rowMask, Tensor<T> values)
    {
        var engine = AiDotNetEngine.Current;
        int m = x.Shape[0];
        int p = x.Shape[1];
        var shape = new[] { m, p };
        var mask = engine.TensorBroadcastTo(engine.Reshape(rowMask, new[] { m, 1 }), shape);
        return engine.TensorWhere(mask, engine.TensorBroadcastTo(values, shape), x);
    }

    /// <summary>Replaces row <paramref name="row"/> of <paramref name="x"/> [m, p] with <paramref name="value"/> [p].</summary>
    public static Tensor<T> ReplaceRow<T>(Tensor<T> x, int row, Tensor<T> value) =>
        ReplaceRows(x, FeatureSelectionTensorOps.OneHot<T>(row, x.Shape[0]), value);

    /// <summary>Replaces element <paramref name="index"/> of <paramref name="v"/> [m] with <paramref name="value"/> [1].</summary>
    public static Tensor<T> ReplaceAt<T>(Tensor<T> v, int index, Tensor<T> value)
    {
        var engine = AiDotNetEngine.Current;
        int m = v.Shape[0];
        return engine.TensorWhere(
            FeatureSelectionTensorOps.OneHot<T>(index, m), engine.TensorBroadcastTo(value, new[] { m }), v);
    }

    /// <summary>
    /// Writes row k of <paramref name="values"/> [k, p] into row <paramref name="rows"/>[k] of <paramref name="x"/>
    /// [m, p]. The target rows must be distinct.
    /// </summary>
    public static Tensor<T> ScatterRows<T>(Tensor<T> x, int[] rows, Tensor<T> values)
    {
        // One masked select per row (not a one-hot matmul) so a non-finite value never leaks into another row.
        for (int k = 0; k < rows.Length; k++)
            x = ReplaceRow(x, rows[k], Row(values, k));
        return x;
    }

    /// <summary>
    /// Writes <paramref name="values"/>[k] into element <paramref name="indices"/>[k] of <paramref name="v"/> [m].
    /// The target indices must be distinct.
    /// </summary>
    public static Tensor<T> ScatterAt<T>(Tensor<T> v, int[] indices, Tensor<T> values)
    {
        if (indices.Length == 0)
            return v;

        var engine = AiDotNetEngine.Current;
        int m = v.Shape[0];
        int k = indices.Length;
        var placed = ScatterRows(engine.Reshape(v, new[] { m, 1 }), indices, engine.Reshape(values, new[] { k, 1 }));
        return engine.Reshape(placed, new[] { m });
    }

    /// <summary>Multiplies every row i of <paramref name="x"/> [m, p] by <paramref name="scale"/>[i].</summary>
    public static Tensor<T> ScaleRows<T>(Tensor<T> x, Tensor<T> scale)
    {
        var engine = AiDotNetEngine.Current;
        int m = x.Shape[0];
        return engine.TensorMultiply(x, engine.TensorBroadcastTo(engine.Reshape(scale, new[] { m, 1 }), x.Shape.ToArray()));
    }

    /// <summary>Multiplies every element of <paramref name="x"/> by the single value held in <paramref name="scalar"/> [1].</summary>
    public static Tensor<T> Scale<T>(Tensor<T> x, Tensor<T> scalar) =>
        AiDotNetEngine.Current.TensorMultiply(x, AiDotNetEngine.Current.TensorBroadcastTo(scalar, x.Shape.ToArray()));

    /// <summary>Index of the first maximum of a [m] tensor.</summary>
    public static int ArgMax<T>(Tensor<T> v) => PreprocessingTensorOps.StableOrder(v, descending: true)[0];

    /// <summary>Index of the first minimum of a [m] tensor.</summary>
    public static int ArgMin<T>(Tensor<T> v) => PreprocessingTensorOps.StableOrder(v, descending: false)[0];

    /// <summary>True when the 0/1 tensor <paramref name="mask"/> has any nonzero element.</summary>
    public static bool Any<T>(Tensor<T> mask) => AiDotNetEngine.Current.TensorCountNonzero(mask) > 0;

    /// <summary>True when <paramref name="a"/>[0] &gt; <paramref name="b"/>[0] for two [1] tensors.</summary>
    public static bool Greater<T>(Tensor<T> a, Tensor<T> b) => Any(AiDotNetEngine.Current.TensorGreaterThan(a, b));

    /// <summary>
    /// Int <paramref name="counts"/> divided by <paramref name="total"/>, divided in double so counts and totals
    /// beyond the exact integer range of T (2048 for Half) or its largest value (65504 for Half) stay exact.
    /// </summary>
    public static Tensor<T> Frequency<T>(Tensor<int> counts, double total) =>
        AiDotNetEngine.Current.TensorDivideScalar(counts.Cast<double>(), total).Cast<T>();

    /// <summary>
    /// <paramref name="value"/> in T as a positive guard (added to a denominator or a log argument), raised to
    /// 2^-24 when it rounds to 0 in T: 1e-10 underflows to 0 in Half, which turns x / (0 + eps) into 0 / 0 = NaN.
    /// </summary>
    public static T Tiny<T>(double value)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        T tiny = ops.FromDouble(value);
        return ops.Equals(tiny, ops.Zero) ? ops.FromDouble(Math.Pow(2, -24)) : tiny;
    }

    /// <summary>The single value of a one-element tensor.</summary>
    public static T Value<T>(Tensor<T> oneElement) => AiDotNetEngine.Current.TensorSum(oneElement);

    /// <summary>A 0/1 [p] mask with ones at <paramref name="indices"/>.</summary>
    public static Tensor<T> IndexMask<T>(int[] indices, int length)
    {
        var engine = AiDotNetEngine.Current;
        if (indices.Length == 0)
            return new Tensor<T>(new[] { length });

        var oneHot = engine.TensorOneHot<T>(new Tensor<int>((int[])indices.Clone(), new[] { indices.Length }), length);
        return engine.TensorClamp(
            engine.ReduceSum(oneHot, new[] { 0 }, false),
            MathHelper.GetNumericOperations<T>().Zero,
            MathHelper.GetNumericOperations<T>().One);
    }

    /// <summary>
    /// Picks, from a [p] gather index per column, the element of <paramref name="x"/> [m, p] at row
    /// <paramref name="rows"/>[j] in column j. Returns [p].
    /// </summary>
    public static Tensor<T> GatherPerColumn<T>(Tensor<T> x, int[] rows)
    {
        var engine = AiDotNetEngine.Current;
        int m = x.Shape[0];
        int p = x.Shape[1];
        var oneHot = engine.TensorOneHot<T>(new Tensor<int>((int[])rows.Clone(), new[] { p }), m); // [p, m]
        return engine.ReduceSum(engine.TensorMultiply(engine.TensorTranspose(oneHot), x), new[] { 0 }, false);
    }
}
