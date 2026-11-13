using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing;

/// <summary>
/// Whole-array engine helpers shared by the preprocessing transformers (column/row gathers, filled tensors,
/// ranked index selection and the scalar decisions iterative fits make once per iteration).
/// </summary>
internal static class PreprocessingTensorOps
{
    /// <summary>The smallest positive normal Half, 2^-14.</summary>
    private const double HalfSmallestNormal = 6.103515625e-05;

    /// <summary>
    /// <paramref name="epsilon"/> as a T, raised to the smallest positive normal value of T where it is below that. Half
    /// cannot hold 1e-10 (it rounds to 0), which turns "below epsilon" guards off and "+ epsilon" denominators back
    /// into divisions by zero; float, double and decimal keep <paramref name="epsilon"/> unchanged.
    /// </summary>
    public static T Tolerance<T>(double epsilon) =>
        MathHelper.GetNumericOperations<T>().FromDouble(typeof(T) == typeof(Half) ? Math.Max(epsilon, HalfSmallestNormal) : epsilon);

    /// <summary>
    /// Whether <typeparamref name="T"/> is a 16-bit float (Half or BFloat16). Sums of squares and their products
    /// overflow Half's 65504 maximum and lose BFloat16's 8-bit mantissa, so second-moment statistics are evaluated in
    /// float for these types and cast back once.
    /// </summary>
    public static bool IsNarrowFloat<T>() =>
        typeof(T) == typeof(Half) || typeof(T) == typeof(AiDotNet.Tensors.NumericOperations.BFloat16);

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

        var order = StableOrder(scores, descending: true);
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

        var order = StableOrder(scores, descending: false);
        var selected = order.AsSpan().Slice(0, count).ToArray();
        Array.Sort(selected);
        return selected;
    }

    /// <summary>
    /// The indices that sort a 1-D tensor ascending (or descending), with tied values kept in their original
    /// index order, like LINQ <c>OrderBy</c>/<c>OrderByDescending</c>; with <paramref name="lastAmongTies"/>
    /// the tied values come in reverse index order instead. The engine argsort is not documented as stable, so
    /// it sorts a distinct composite key (rank of the value, then the index) whose order is unique.
    /// </summary>
    /// <remarks>
    /// NaN orders as the worst value in both directions: after every number (infinities included) when
    /// ascending, so a NaN p-value is never the most significant, and after every number when descending, so
    /// a NaN score is never the best. This matches scikit-learn, whose univariate selectors never pick a NaN.
    /// </remarks>
    public static int[] StableOrder<T>(Tensor<T> values, bool descending, bool lastAmongTies = false)
    {
        int count = values.Length;
        if (count == 0)
            return Array.Empty<int>();

        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        var flat = values.Rank == 1 ? values : engine.Reshape(values, new[] { count });

        // NaN takes a placeholder 0 for the rank search (ranks stay monotone in the real values) and is then
        // given rank `count`, past every real value's rank, whichever way the order runs.
        var isNaN = engine.TensorIsNan(flat);
        var zeros = Filled(new[] { count }, ops.Zero);
        var nanPositions = MaskPositions(engine.TensorWhere(isNaN, Filled(new[] { count }, ops.One), zeros));
        var ranked = nanPositions.Length == 0 ? flat : engine.TensorWhere(isNaN, zeros, flat);
        var sorted = engine.TensorSort(ranked, 0, false).Item1;

        // ascending rank: #{m : v[m] < v[j]}; descending rank: #{m : v[m] > v[j]} = count - #{m : v[m] <= v[j]}
        var less = engine.TensorSearchSorted(sorted, ranked, false).AsSpan().ToArray();
        var atMost = engine.TensorSearchSorted(sorted, ranked, true).AsSpan().ToArray();
        var notANumber = new bool[count];
        foreach (int position in nanPositions)
            notANumber[position] = true;

        var keys = new long[count];
        for (int j = 0; j < count; j++)
        {
            long rank = notANumber[j] ? count : descending ? count - atMost[j] : less[j];
            keys[j] = rank * count + (lastAmongTies ? count - 1 - j : j);
        }

        return engine.TensorArgsort(new Tensor<long>(keys, new[] { count }), 0, false).AsSpan().ToArray();
    }

    /// <summary>
    /// The index of the first (lowest-index) largest entry of a 1-D tensor, a NaN ranking below every number: the
    /// first entry of <see cref="StableOrder{T}"/>, the one ranking every FirstArgMax shares.
    /// </summary>
    public static int FirstArgMax<T>(Tensor<T> values) => StableOrder(values, descending: true)[0];

    /// <summary>
    /// The positions of the (at most <paramref name="k"/>) largest entries that are at least <paramref name="minimum"/>,
    /// returned in ascending index order.
    /// </summary>
    public static int[] TopKAtLeastSorted<T>(Tensor<T> scores, T minimum, int k)
    {
        // Only entries at least the minimum qualify; a NaN score never does (NaN >= minimum is false, as in master)
        var qualifies = AtLeast(scores, minimum).Cast<int>().AsSpan().ToArray();
        var selected = StableOrder(scores, descending: true).Where(i => qualifies[i] != 0).Take(Math.Max(k, 0)).ToArray();
        Array.Sort(selected);
        return selected;
    }

    /// <summary>
    /// Pearson correlation of every column of <paramref name="x"/> ([n, p]) with <paramref name="y"/> ([n]).
    /// A column (or target) whose centred sum of squares is not above <paramref name="epsilon"/> gets 0.
    /// </summary>
    public static Tensor<T> TargetCorrelations<T>(Tensor<T> x, Tensor<T> y, double epsilon = 1e-10)
    {
        if (IsNarrowFloat<T>())
            return TargetCorrelations(x.Cast<float>(), y.Cast<float>(), epsilon).Cast<T>();

        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = x.Shape[0];
        T eps = Tolerance<T>(epsilon);
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
        if (IsNarrowFloat<T>())
            return CorrelationsWithDenominatorGuard(x.Cast<float>(), y.Cast<float>(), epsilon).Cast<T>();

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
            engine.TensorGreaterThan(denom, Tolerance<T>(epsilon)), engine.TensorDivide(sxy, denom), new Tensor<T>(new[] { p }));
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
        return engine.TensorGreaterEqual(x, broadcast); // NaN >= row is false, as in master
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
        var width = engine.TensorAddScalar(engine.TensorDivideScalar(range, ops.FromDouble(bins)), Tolerance<T>(1e-10));
        var scaled = engine.TensorDivide(
            engine.TensorSubtract(x, engine.Reshape(min, new[] { 1, p })), engine.Reshape(width, new[] { 1, p }));
        var binIndex = engine.TensorMin(engine.TensorFloor(scaled), ops.FromDouble(bins - 1));
        varying = engine.TensorAddScalar(
            engine.TensorNegate(engine.TensorLessThan(range, Tolerance<T>(1e-10))), ops.One);
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
        var varying = engine.TensorBroadcastTo(engine.TensorGreaterThan(range, Tolerance<T>(1e-10)), new[] { n, p });
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
        // Not 1 - (a < b): that is 1 for a NaN, where master's a >= b is false
        return AiDotNetEngine.Current.TensorGreaterEqual(a, Filled(a.Shape.ToArray(), b));
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
        int n = oneHotA.Shape[0];
        int pa = oneHotA.Shape[1] / bins;
        int pb = oneHotB.Shape[1] / bins;
        double count = n;

        // Joint and marginal bin counts in double (exact up to 2^53): 0/1 sums in a narrow T such as Half stop
        // growing at 2048, and the marginal product overflows Half. Only the finished scores are converted to T.
        var a = oneHotA.Cast<double>();
        var b = oneHotB.Cast<double>();
        var joint = engine.TensorMatMul(engine.TensorTranspose(a), b);
        var marginalA = engine.ReduceSum(a, new[] { 0 }, false);
        var marginalB = engine.ReduceSum(b, new[] { 0 }, false);
        var pxy = engine.TensorDivideScalar(joint, count);
        var pxpy = engine.TensorDivideScalar(engine.TensorOuter(marginalA, marginalB), count * count);
        var ratio = engine.TensorDivide(pxy, pxpy);
        if (logEpsilon != 0)
            ratio = engine.TensorAddScalar(ratio, logEpsilon);
        var terms = engine.TensorMultiply(pxy, engine.TensorLog(ratio));
        if (bits)
            terms = engine.TensorDivideScalar(terms, Math.Log(2));
        var cells = engine.TensorWhere(engine.TensorGreaterThan(joint, 0.0), terms, new Tensor<double>(joint.Shape.ToArray()));
        var mi = engine.ReduceSum(engine.Reshape(cells, new[] { pa, bins, pb, bins }), new[] { 1, 3 }, false);
        return (clampAtZero ? engine.TensorMax(mi, 0.0) : mi).Cast<T>();
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

    /// <summary>
    /// 0/1 indicator [<paramref name="length"/>] of a single position, built from the int position (comparing a
    /// <see cref="Range{T}"/> in T against the position fails in Half above 2048, where integers are not all exact).
    /// </summary>
    public static Tensor<T> PositionOneHot<T>(int position, int length) =>
        AiDotNetEngine.Current.Reshape(
            AiDotNetEngine.Current.TensorOneHot<T>(new Tensor<int>(new[] { position }, new[] { 1 }), length), new[] { length });

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

    /// <summary>The sorted distinct values of a 1-D tensor.</summary>
    public static Tensor<T> SortedUnique<T>(Tensor<T> values) =>
        values.Length == 0 ? new Tensor<T>(new[] { 0 }) : AiDotNetEngine.Current.TensorUnique(values, true);

    /// <summary>
    /// Looks every value up in a sorted list of distinct categories: returns each value's category position
    /// (clamped into range for values that are not categories) and, in <paramref name="found"/>, a 0/1 mask of
    /// the values that are categories.
    /// </summary>
    public static Tensor<int> LookupSorted<T>(Tensor<T> sortedCategories, Tensor<T> values, out Tensor<T> found)
    {
        var engine = AiDotNetEngine.Current;
        int k = sortedCategories.Length;
        if (k == 0 || values.Length == 0)
        {
            found = new Tensor<T>(values.Shape.ToArray());
            return new Tensor<int>(values.Shape.ToArray());
        }

        var positions = engine.TensorSearchSorted(sortedCategories, values, false).AsSpan().ToArray();
        for (int i = 0; i < positions.Length; i++)
            positions[i] = Math.Min(positions[i], k - 1);
        var index = new Tensor<int>(positions, values.Shape.ToArray());
        found = engine.TensorEquals(engine.TensorIndexSelect(sortedCategories, index, 0), values);
        return index;
    }

    /// <summary>
    /// Inverse of a label lookup: each label (truncated toward zero) in 0..categories-1 becomes that category;
    /// any other label keeps its value.
    /// </summary>
    public static Tensor<T> GatherCategories<T>(Tensor<T> categories, Tensor<T> labels)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int k = categories.Length;
        if (k == 0 || labels.Length == 0)
            return labels;

        var truncated = engine.TensorTrunc(labels);
        var valid = engine.TensorMultiply(
            AtLeast(truncated, ops.Zero), engine.TensorLessThan(truncated, ops.FromDouble(k)));
        var index = engine.TensorClamp(truncated, ops.Zero, ops.FromDouble(k - 1)).Cast<int>();
        return engine.TensorWhere(valid, engine.TensorIndexSelect(categories, index, 0), labels);
    }

    /// <summary>Per-category sums of <paramref name="values"/> ([n]) for category positions <paramref name="index"/>.</summary>
    public static Tensor<T> CategorySums<T>(Tensor<int> index, Tensor<T> values, int categories)
    {
        var engine = AiDotNetEngine.Current;
        if (values.Length == 0)
            return new Tensor<T>(new[] { categories });
        var oneHot = engine.TensorOneHot<T>(index, categories); // [n, k]
        return engine.Reshape(
            engine.TensorMatMul(engine.Reshape(values, new[] { 1, values.Length }), oneHot), new[] { categories });
    }

    /// <summary>
    /// Per-category row counts for category positions <paramref name="index"/> (positions outside
    /// [0, <paramref name="categories"/>) are not counted). Counted in int and converted to T at the end, so a narrow
    /// T such as Half does not stop counting at 2048.
    /// </summary>
    public static Tensor<T> CategoryCounts<T>(Tensor<int> index, int categories)
    {
        var engine = AiDotNetEngine.Current;
        if (index.Length == 0)
            return new Tensor<T>(new[] { categories });
        var flat = index.Rank == 1 ? index : engine.Reshape(index, new[] { index.Length });
        var valid = engine.TensorMultiply(engine.TensorGreaterThan(flat, -1), engine.TensorLessThan(flat, categories));
        var binned = engine.TensorWhere(valid, flat, Filled(new[] { flat.Length }, categories));
        return engine.TensorNarrow(engine.TensorBinCount(binned, categories + 1), 0, 0, categories).Cast<T>();
    }

    /// <summary>Stacks [n] column tensors into an [n, columns] matrix.</summary>
    public static Matrix<T> StackColumns<T>(IReadOnlyList<Tensor<T>> columns, int rows)
    {
        if (columns.Count == 0)
            return new Matrix<T>(rows, 0);
        return AiDotNetEngine.Current.TensorStack(columns.ToArray(), 1).ToMatrix();
    }

    /// <summary>Column <paramref name="column"/> of a matrix as an [n] tensor.</summary>
    public static Tensor<T> Column<T>(Matrix<T> data, int column) => Tensor<T>.FromVector(data.GetColumn(column));

    /// <summary>
    /// For each row of a 0/1 mask ([n, w]), the column of its first non-zero entry, or <c>w</c> when the row is all zero.
    /// </summary>
    public static int[] FirstNonZeroPerRow<T>(Tensor<T> mask)
    {
        var engine = AiDotNetEngine.Current;
        int n = mask.Shape[0];
        int w = mask.Shape[1];
        if (n == 0)
            return Array.Empty<int>();
        if (w == 0)
            return new int[n];

        var columnIndex = engine.TensorBroadcastTo(engine.Reshape(Range<T>(w), new[] { 1, w }), new[] { n, w });
        var candidates = engine.TensorWhere(mask, columnIndex, Filled(new[] { n, w }, MathHelper.GetNumericOperations<T>().FromDouble(w)));
        var first = engine.TensorNegate(engine.ReduceMax(engine.TensorNegate(candidates), new[] { 1 }, false));
        return first.Cast<int>().AsSpan().ToArray();
    }

    /// <summary>
    /// The 1-based position of each value among <paramref name="sortedCategories"/>, or 0 for values that are
    /// not categories; <paramref name="found"/> is the 0/1 mask of the values that are categories.
    /// </summary>
    public static int[] OrdinalCodes<T>(Tensor<T> sortedCategories, Tensor<T> values, out Tensor<T> found)
    {
        var position = LookupSorted(sortedCategories, values, out found).AsSpan().ToArray();
        var known = found.Cast<int>().AsSpan().ToArray();
        for (int i = 0; i < position.Length; i++)
            position[i] = known[i] != 0 ? position[i] + 1 : 0;
        return position;
    }

    /// <summary>
    /// The base-<paramref name="radix"/> digits (most significant first) of each integer code, as a
    /// <c>[codes, digits]</c> tensor.
    /// </summary>
    public static Tensor<T> DigitBlock<T>(int[] codes, int digits, int radix)
    {
        var result = new int[codes.Length * digits];
        for (int i = 0; i < codes.Length; i++)
        {
            int value = codes[i];
            for (int d = digits - 1; d >= 0; d--)
            {
                result[i * digits + d] = value % radix;
                value /= radix;
            }
        }

        return new Tensor<int>(result, new[] { codes.Length, digits }).Cast<T>();
    }

    /// <summary>
    /// Reads back the binary code of each row of a <c>[rows, bits]</c> block (most significant bit first),
    /// treating entries above 0.5 as set bits.
    /// </summary>
    public static int[] ReadBinaryCodes<T>(Tensor<T> block)
    {
        int rows = block.Shape[0];
        int bits = block.Shape[1];
        var set = AiDotNetEngine.Current
            .TensorGreaterThan(block, MathHelper.GetNumericOperations<T>().FromDouble(0.5))
            .Cast<int>().AsSpan().ToArray();
        var codes = new int[rows];
        for (int i = 0; i < rows; i++)
            for (int b = 0; b < bits; b++)
                if (set[i * bits + b] != 0)
                    codes[i] |= 1 << (bits - 1 - b);
        return codes;
    }

    /// <summary>
    /// A <c>[rows, cols]</c> matrix with entry <c>numerator(row, col) / denominator(col)</c>, for contrast
    /// matrices whose entries are ratios of small integers.
    /// </summary>
    public static Tensor<T> RationalMatrix<T>(int rows, int cols, Func<int, int, int> numerator, Func<int, int> denominator)
    {
        var num = new int[rows * cols];
        var den = new int[cols];
        for (int c = 0; c < cols; c++)
            den[c] = denominator(c);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                num[r * cols + c] = numerator(r, c);

        var numerators = new Tensor<int>(num, new[] { rows, cols }).Cast<T>();
        if (rows == 0 || cols == 0)
            return numerators;

        var engine = AiDotNetEngine.Current;
        var denominators = new Tensor<int>(den, new[] { 1, cols }).Cast<T>();
        return engine.TensorDivide(numerators, engine.TensorBroadcastTo(denominators, new[] { rows, cols }));
    }

    /// <summary>
    /// Contrast-codes a column: each known category becomes its row of <paramref name="contrast"/> and unknown
    /// values become zeros, or throw <see cref="ArgumentException"/> when <paramref name="throwOnUnknown"/> is set.
    /// </summary>
    public static Tensor<T> ContrastBlock<T>(
        Tensor<T> sortedCategories, Tensor<T> contrast, Tensor<T> column, bool throwOnUnknown, int columnIndex)
    {
        var engine = AiDotNetEngine.Current;
        int rows = column.Length;
        int cols = contrast.Shape[1];
        var position = LookupSorted(sortedCategories, column, out var found);
        if (throwOnUnknown)
        {
            var missing = MaskPositions(Not(found));
            if (missing.Length > 0)
                throw new ArgumentException($"Unknown category value: {column.GetFlat(missing[0])} in column {columnIndex}");
        }

        if (rows == 0 || sortedCategories.Length == 0)
            return new Tensor<T>(new[] { rows, cols });

        return engine.TensorMultiply(
            engine.TensorIndexSelect(contrast, position, 0),
            engine.TensorBroadcastTo(engine.Reshape(found, new[] { rows, 1 }), new[] { rows, cols }));
    }

    /// <summary>Column <paramref name="column"/> of <paramref name="data"/> truncated toward zero to integer ids.</summary>
    public static int[] IntColumn<T>(Matrix<T> data, int column) =>
        data.Rows == 0
            ? Array.Empty<int>()
            : AiDotNetEngine.Current.TensorTrunc(Column(data, column)).Cast<int>().AsSpan().ToArray();

    /// <summary>
    /// <c>(x - min(x)) / (max(x) - min(x))</c>, dividing by 1 instead when the range is not positive.
    /// </summary>
    public static Tensor<T> MinMaxNormalize<T>(Tensor<T> x)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        T min = engine.TensorMinValue(x);
        T range = ops.Subtract(engine.TensorMaxValue(x), min);
        if (!ops.GreaterThan(range, ops.Zero))
            range = ops.One;
        return engine.TensorDivideScalar(engine.TensorSubtractScalar(x, min), range);
    }

    /// <summary>
    /// The equal-width block of each value: <c>min(trunc((x - min) / width), blocks - 1)</c> with
    /// <c>width = (max - min) / blocks</c>, or width 1 when that is not positive.
    /// </summary>
    public static int[] EqualWidthBlockIndex<T>(Tensor<T> x, int blocks)
    {
        if (x.Length == 0)
            return Array.Empty<int>();

        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        T min = engine.TensorMinValue(x);
        T width = ops.Divide(ops.Subtract(engine.TensorMaxValue(x), min), ops.FromDouble(blocks));
        if (!ops.GreaterThan(width, ops.Zero))
            width = ops.One;

        var block = engine.TensorTrunc(engine.TensorDivideScalar(engine.TensorSubtractScalar(x, min), width));
        return engine.TensorClamp(block, ops.Zero, ops.FromDouble(blocks - 1)).Cast<int>().AsSpan().ToArray();
    }

    /// <summary>
    /// The Euclidean distance of every row of <paramref name="data"/> from the column means. With
    /// <paramref name="standardize"/>, each column difference is first divided by the column's population
    /// standard deviation (1 where that is below 1e-10).
    /// </summary>
    public static Tensor<T> CentroidDistances<T>(Matrix<T> data, bool standardize)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        var x = Tensor<T>.FromMatrix(data);
        var diff = engine.TensorSubtract(x, engine.ReduceMean(x, new[] { 0 }, true));
        if (standardize)
        {
            var std = engine.TensorSqrt(engine.ReduceMean(engine.TensorMultiply(diff, diff), new[] { 0 }, true));
            var tiny = engine.TensorLessThan(std, Tolerance<T>(1e-10));
            std = engine.TensorWhere(tiny, Filled(std.Shape.ToArray(), ops.One), std);
            diff = engine.TensorDivide(diff, std);
        }

        return engine.TensorSqrt(engine.ReduceSum(engine.TensorMultiply(diff, diff), new[] { 1 }, false));
    }

    /// <summary>The values 0, 1, ..., count - 1 as a 1-D tensor.</summary>
    public static Tensor<T> Range<T>(int count) =>
        new Tensor<int>(Enumerable.Range(0, count).ToArray(), new[] { count }).Cast<T>();

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
