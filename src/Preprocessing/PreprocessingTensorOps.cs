using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing;

/// <summary>
/// Whole-array engine helpers shared by the preprocessing transformers (column/row gathers, filled tensors,
/// ranked index selection and the scalar decisions iterative fits make once per iteration).
/// </summary>
internal static class PreprocessingTensorOps
{
    /// <summary>Gathers the given columns of <paramref name="data"/> into a new matrix, in the given order.</summary>
    public static Matrix<T> SelectColumns<T>(Matrix<T> data, int[] columns)
    {
        if (columns.Length == 0 || data.Rows == 0)
            return new Matrix<T>(data.Rows, columns.Length);

        var indices = new Tensor<int>((int[])columns.Clone(), new[] { columns.Length });
        return AiDotNetEngine.Current.TensorIndexSelect(Tensor<T>.FromMatrix(data), indices, 1).ToMatrix();
    }

    /// <summary>Gathers the given rows of <paramref name="data"/> into a new matrix, in the given order.</summary>
    public static Matrix<T> SelectRows<T>(Matrix<T> data, int[] rows)
    {
        if (rows.Length == 0 || data.Columns == 0)
            return new Matrix<T>(rows.Length, data.Columns);

        var indices = new Tensor<int>((int[])rows.Clone(), new[] { rows.Length });
        return AiDotNetEngine.Current.TensorIndexSelect(Tensor<T>.FromMatrix(data), indices, 0).ToMatrix();
    }

    /// <summary>Gathers the given elements of <paramref name="values"/> into a new vector, in the given order.</summary>
    public static Vector<T> SelectElements<T>(Vector<T> values, int[] positions)
    {
        if (positions.Length == 0)
            return new Vector<T>(0);

        var indices = new Tensor<int>((int[])positions.Clone(), new[] { positions.Length });
        return AiDotNetEngine.Current.TensorIndexSelect(Tensor<T>.FromVector(values), indices, 0).ToVector();
    }

    /// <summary>A tensor of the given shape with every element set to <paramref name="value"/>.</summary>
    public static Tensor<T> Filled<T>(int[] shape, T value)
    {
        var result = new Tensor<T>(shape);
        AiDotNetEngine.Current.TensorFill(result, value);
        return result;
    }

    /// <summary>
    /// The positions of the <paramref name="k"/> largest entries of a 1-D score tensor, returned in ascending
    /// index order (the usual "select top-k features, keep column order" step).
    /// </summary>
    public static int[] TopKIndicesSorted<T>(Tensor<T> scores, int k)
    {
        int count = Math.Min(Math.Max(k, 0), scores.Length);
        if (count == 0)
            return Array.Empty<int>();

        var order = AiDotNetEngine.Current.TensorArgsort(scores, 0, true);
        var selected = order.AsSpan().Slice(0, count).ToArray();
        Array.Sort(selected);
        return selected;
    }

    /// <summary>
    /// The positions of the <paramref name="k"/> smallest entries of a 1-D score tensor, returned in ascending
    /// index order.
    /// </summary>
    public static int[] BottomKIndicesSorted<T>(Tensor<T> scores, int k)
    {
        int count = Math.Min(Math.Max(k, 0), scores.Length);
        if (count == 0)
            return Array.Empty<int>();

        var order = AiDotNetEngine.Current.TensorArgsort(scores, 0, false);
        var selected = order.AsSpan().Slice(0, count).ToArray();
        Array.Sort(selected);
        return selected;
    }

    /// <summary>
    /// The positions of the (at most <paramref name="k"/>) largest entries that are at least <paramref name="minimum"/>,
    /// returned in ascending index order.
    /// </summary>
    public static int[] TopKAtLeastSorted<T>(Tensor<T> scores, T minimum, int k)
    {
        int below = CountNonZero(AiDotNetEngine.Current.TensorLessThan(scores, minimum));
        return TopKIndicesSorted(scores, Math.Min(k, scores.Length - below));
    }

    /// <summary>
    /// Pearson correlation of every column of <paramref name="x"/> ([n, p]) with <paramref name="y"/> ([n]).
    /// A column (or target) whose centred sum of squares is not above <paramref name="epsilon"/> gets 0.
    /// </summary>
    public static Tensor<T> TargetCorrelations<T>(Tensor<T> x, Tensor<T> y, double epsilon = 1e-10)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = x.Shape[0];
        T eps = ops.FromDouble(epsilon);
        var xc = engine.TensorSubtract(x, engine.ReduceMean(x, new[] { 0 }, true));
        var yc = engine.Reshape(engine.TensorSubtractScalar(y, engine.TensorMean(y)), new[] { n, 1 });
        var sxy = engine.Reshape(engine.TensorMatMul(engine.TensorTranspose(xc), yc), new[] { x.Shape[1] });
        var sxx = engine.ReduceSum(engine.TensorSquare(xc), new[] { 0 }, false);
        T syy = engine.TensorSumOfSquares(yc);
        var zeros = new Tensor<T>(new[] { x.Shape[1] });
        if (!ops.GreaterThan(syy, eps))
            return zeros;
        var r = engine.TensorDivide(sxy, engine.TensorSqrt(engine.TensorMultiplyScalar(sxx, syy)));
        return engine.TensorWhere(engine.TensorGreaterThan(sxx, eps), r, zeros);
    }

    /// <summary>
    /// Pearson correlation of every column of <paramref name="x"/> ([n, p]) with <paramref name="y"/> ([n]),
    /// 0 where <c>sqrt(sxx * syy)</c> is not above <paramref name="epsilon"/> (both inputs are centred first).
    /// </summary>
    public static Tensor<T> CorrelationsWithDenominatorGuard<T>(Tensor<T> x, Tensor<T> y, double epsilon = 1e-10)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = x.Shape[0];
        int p = x.Shape[1];
        var xc = engine.TensorSubtract(x, engine.ReduceMean(x, new[] { 0 }, true));
        var yc = engine.Reshape(engine.TensorSubtractScalar(y, engine.TensorMean(y)), new[] { n, 1 });
        var sxy = engine.Reshape(engine.TensorMatMul(engine.TensorTranspose(xc), yc), new[] { p });
        var sxx = engine.ReduceSum(engine.TensorSquare(xc), new[] { 0 }, false);
        var denom = engine.TensorSqrt(engine.TensorMultiplyScalar(sxx, engine.TensorSumOfSquares(yc)));
        return engine.TensorWhere(
            engine.TensorGreaterThan(denom, ops.FromDouble(epsilon)), engine.TensorDivide(sxy, denom), new Tensor<T>(new[] { p }));
    }

    /// <summary>Column-wise minimum of a 2-D tensor ([n, p] to [p]).</summary>
    public static Tensor<T> ColumnMin<T>(Tensor<T> x)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorNegate(engine.ReduceMax(engine.TensorNegate(x), new[] { 0 }, false));
    }

    /// <summary>The <paramref name="k"/>-th smallest value (0-based) of every column of a 2-D tensor ([n, p] to [p]).</summary>
    public static Tensor<T> ColumnOrderStatistic<T>(Tensor<T> x, int k)
    {
        var engine = AiDotNetEngine.Current;
        var (sorted, _) = engine.TensorSort(x, 0, false);
        return engine.TensorSliceAxis(sorted, 0, k);
    }

    /// <summary>Element-wise <c>x &gt;= row</c> for an [n, p] tensor against a [p] row, as a 0/1 [n, p] tensor.</summary>
    public static Tensor<T> AtLeastRow<T>(Tensor<T> x, Tensor<T> row)
    {
        var engine = AiDotNetEngine.Current;
        var broadcast = engine.TensorBroadcastTo(engine.Reshape(row, new[] { 1, row.Length }), x.Shape.ToArray());
        return Not(engine.TensorLessThan(x, broadcast));
    }

    /// <summary>Column-wise maximum of a 2-D tensor ([n, p] to [p]).</summary>
    public static Tensor<T> ColumnMax<T>(Tensor<T> x) => AiDotNetEngine.Current.ReduceMax(x, new[] { 0 }, false);

    /// <summary>
    /// Equal-width histogram bin of every element of <paramref name="x"/> ([n, p]): each column is split into
    /// <paramref name="bins"/> bins of width <c>(max - min) / bins + 1e-10</c> and the bin index is
    /// <c>min(floor((x - min) / width), bins - 1)</c>. Returns the bins one-hot as [n, p * bins] (column j owns
    /// entries j * bins .. j * bins + bins - 1) and, in <paramref name="varying"/>, a [p] 0/1 mask of the columns
    /// whose range is at least 1e-10.
    /// </summary>
    public static Tensor<T> EqualWidthBinOneHot<T>(Tensor<T> x, int bins, out Tensor<T> varying)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = x.Shape[0];
        int p = x.Shape[1];
        var min = ColumnMin(x);
        var range = engine.TensorSubtract(ColumnMax(x), min);
        var width = engine.TensorAddScalar(engine.TensorDivideScalar(range, ops.FromDouble(bins)), ops.FromDouble(1e-10));
        var scaled = engine.TensorDivide(
            engine.TensorSubtract(x, engine.Reshape(min, new[] { 1, p })), engine.Reshape(width, new[] { 1, p }));
        var binIndex = engine.TensorMin(engine.TensorFloor(scaled), ops.FromDouble(bins - 1));
        varying = engine.TensorAddScalar(
            engine.TensorNegate(engine.TensorLessThan(range, ops.FromDouble(1e-10))), ops.One);
        return OneHotBins(binIndex, bins);
    }

    /// <summary>
    /// Range-scaled bins of every column of <paramref name="x"/> ([n, p]):
    /// <c>bin = min(floor((x - min) / (max - min) * (bins - 1)), bins - 1)</c>, 0 for a column whose range is not
    /// above 1e-10. Returns the bins one-hot as [n, p * bins].
    /// </summary>
    public static Tensor<T> RangeScaledBinOneHot<T>(Tensor<T> x, int bins)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = x.Shape[0];
        int p = x.Shape[1];
        var min = engine.Reshape(ColumnMin(x), new[] { 1, p });
        var range = engine.TensorSubtract(engine.Reshape(ColumnMax(x), new[] { 1, p }), min);
        var scaled = engine.TensorMultiplyScalar(engine.TensorDivide(engine.TensorSubtract(x, min), range), ops.FromDouble(bins - 1));
        var binIndex = engine.TensorMin(engine.TensorFloor(scaled), ops.FromDouble(bins - 1));
        var varying = engine.TensorBroadcastTo(engine.TensorGreaterThan(range, ops.FromDouble(1e-10)), new[] { n, p });
        return OneHotBins(engine.TensorWhere(varying, binIndex, new Tensor<T>(new[] { n, p })), bins);
    }

    /// <summary>
    /// One-hot of integral bin indices held in T (each element of the [n, p] tensor is in 0..bins-1):
    /// returns [n, p * bins] where column j owns entries j * bins .. j * bins + bins - 1.
    /// </summary>
    public static Tensor<T> OneHotBins<T>(Tensor<T> binIndex, int bins)
    {
        int n = binIndex.Shape[0];
        int p = binIndex.Rank > 1 ? binIndex.Shape[1] : 1;
        return AiDotNetEngine.Current.Reshape(
            AiDotNetEngine.Current.TensorOneHot<T>(binIndex.Cast<int>(), bins), new[] { n, p * bins });
    }

    /// <summary>Element-wise <c>a &gt;= b</c> as a 0/1 tensor.</summary>
    public static Tensor<T> AtLeast<T>(Tensor<T> a, T b)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorAddScalar(engine.TensorNegate(engine.TensorLessThan(a, b)), MathHelper.GetNumericOperations<T>().One);
    }

    /// <summary>Element-wise logical not of a 0/1 tensor.</summary>
    public static Tensor<T> Not<T>(Tensor<T> mask) => OneMinus(mask);

    /// <summary>Element-wise <c>1 - x</c>.</summary>
    public static Tensor<T> OneMinus<T>(Tensor<T> x)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorAddScalar(engine.TensorNegate(x), MathHelper.GetNumericOperations<T>().One);
    }

    /// <summary>
    /// Histogram mutual information (in bits) between every column of A and every column of B, from their
    /// one-hot bins (<see cref="EqualWidthBinOneHot{T}"/>): <c>sum pxy * log(pxy / (px * py) + logEpsilon)</c> over
    /// the non-empty joint cells, in bits (log2) or nats, optionally clamped at 0. Returns [pa, pb].
    /// </summary>
    public static Tensor<T> HistogramMutualInformation<T>(
        Tensor<T> oneHotA, Tensor<T> oneHotB, int bins, bool bits = true, double logEpsilon = 1e-10, bool clampAtZero = true)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = oneHotA.Shape[0];
        int pa = oneHotA.Shape[1] / bins;
        int pb = oneHotB.Shape[1] / bins;
        T count = ops.FromDouble(n);

        var joint = engine.TensorMatMul(engine.TensorTranspose(oneHotA), oneHotB);
        var marginalA = engine.ReduceSum(oneHotA, new[] { 0 }, false);
        var marginalB = engine.ReduceSum(oneHotB, new[] { 0 }, false);
        var pxy = engine.TensorDivideScalar(joint, count);
        var pxpy = engine.TensorDivideScalar(engine.TensorOuter(marginalA, marginalB), ops.Multiply(count, count));
        var ratio = engine.TensorDivide(pxy, pxpy);
        if (logEpsilon != 0)
            ratio = engine.TensorAddScalar(ratio, ops.FromDouble(logEpsilon));
        var terms = engine.TensorMultiply(pxy, engine.TensorLog(ratio));
        if (bits)
            terms = engine.TensorDivideScalar(terms, ops.FromDouble(Math.Log(2)));
        var cells = engine.TensorWhere(engine.TensorGreaterThan(joint, ops.Zero), terms, new Tensor<T>(joint.Shape.ToArray()));
        var mi = engine.ReduceSum(engine.Reshape(cells, new[] { pa, bins, pb, bins }), new[] { 1, 3 }, false);
        return clampAtZero ? engine.TensorMax(mi, ops.Zero) : mi;
    }

    /// <summary>
    /// Integer class labels of a target (each value rounded to the nearest integer, ties to even, like
    /// <c>(int)Math.Round(value)</c>): the sorted distinct classes and, per row, the position of its class.
    /// </summary>
    public static int[] ClassLabels<T>(Vector<T> target, out int[] classes, out int[] classIndex)
    {
        var labels = AiDotNetEngine.Current.TensorRound(Tensor<T>.FromVector(target)).Cast<int>().AsSpan().ToArray();
        classes = labels.Distinct().OrderBy(c => c).ToArray();
        classIndex = new int[labels.Length];
        for (int i = 0; i < labels.Length; i++)
            classIndex[i] = Array.BinarySearch(classes, labels[i]);
        return labels;
    }

    /// <summary>One-hot class membership [n, classCount] (0/1 in T) from per-row class positions.</summary>
    public static Tensor<T> ClassOneHot<T>(int[] classIndex, int classCount)
    {
        if (classIndex.Length == 0)
            return new Tensor<T>(new[] { 0, classCount });
        return AiDotNetEngine.Current.TensorOneHot<T>(
            new Tensor<int>((int[])classIndex.Clone(), new[] { classIndex.Length }), classCount);
    }

    /// <summary>Row <paramref name="row"/> of a 2-D tensor as a [1, columns] tensor.</summary>
    public static Tensor<T> Row<T>(Tensor<T> x, int row) =>
        AiDotNetEngine.Current.TensorSlice(x, new[] { row, 0 }, new[] { 1, x.Shape[1] });

    /// <summary>Row <paramref name="row"/> of a 2-D tensor as a [columns] tensor.</summary>
    public static Tensor<T> RowVector<T>(Tensor<T> x, int row) =>
        AiDotNetEngine.Current.Reshape(Row(x, row), new[] { x.Shape[1] });

    /// <summary>Column <paramref name="column"/> of a 2-D tensor as an [n] tensor.</summary>
    public static Tensor<T> ColumnVector<T>(Tensor<T> x, int column) =>
        AiDotNetEngine.Current.Reshape(
            AiDotNetEngine.Current.TensorSlice(x, new[] { 0, column }, new[] { x.Shape[0], 1 }), new[] { x.Shape[0] });

    /// <summary>
    /// Column-wise minimum and maximum of the rows of <paramref name="x"/> ([n, p]) selected by a 0/1 row mask
    /// ([n]); the selection must be non-empty.
    /// </summary>
    public static (Tensor<T> Min, Tensor<T> Max) MaskedColumnRange<T>(Tensor<T> x, Tensor<T> rowMask)
    {
        var engine = AiDotNetEngine.Current;
        var shape = x.Shape.ToArray();
        var mask = engine.TensorBroadcastTo(engine.Reshape(rowMask, new[] { shape[0], 1 }), shape);
        var high = engine.TensorBroadcastTo(engine.Reshape(ColumnMax(x), new[] { 1, shape[1] }), shape);
        var low = engine.TensorBroadcastTo(engine.Reshape(ColumnMin(x), new[] { 1, shape[1] }), shape);
        return (ColumnMin(engine.TensorWhere(mask, x, high)), ColumnMax(engine.TensorWhere(mask, x, low)));
    }

    /// <summary>The positions where a 0/1 mask tensor is non-zero, in ascending order.</summary>
    public static int[] MaskPositions<T>(Tensor<T> mask)
    {
        var flat = mask.Rank == 1 ? mask : AiDotNetEngine.Current.Reshape(mask, new[] { mask.Length });
        var coordinates = AiDotNetEngine.Current.TensorNonzero(flat);
        return coordinates.AsSpan().ToArray();
    }

    /// <summary>The number of non-zero entries of a 0/1 mask tensor.</summary>
    public static int CountNonZero<T>(Tensor<T> mask) => AiDotNetEngine.Current.TensorCountNonzero(mask);

    /// <summary>A per-iteration scalar decision: whether <paramref name="value"/> is below <paramref name="limit"/>.</summary>
    public static bool IsBelow<T>(T value, T limit) => MathHelper.GetNumericOperations<T>().LessThan(value, limit);

    /// <summary>A per-iteration scalar decision: whether <paramref name="value"/> is above <paramref name="limit"/>.</summary>
    public static bool IsAbove<T>(T value, T limit) => MathHelper.GetNumericOperations<T>().GreaterThan(value, limit);

    /// <summary>A count (or other integer) as a scalar of type T.</summary>
    public static T CountAsScalar<T>(int value) => MathHelper.GetNumericOperations<T>().FromDouble(value);

    /// <summary><c>max(a, b)</c> of two scalars.</summary>
    public static T ScalarMax<T>(T a, T b)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        return ops.GreaterThan(a, b) ? a : b;
    }
}
