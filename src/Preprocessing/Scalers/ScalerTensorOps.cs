using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.Scalers;

/// <summary>
/// Whole-matrix engine helpers shared by the column-wise scalers, clippers and power transforms: per-column
/// parameter rows, NaN-ignoring column statistics (scikit-learn treats NaN as missing in <c>fit</c> and keeps it
/// in <c>transform</c>) and the broadcast affine maps the transforms apply.
/// </summary>
internal static class ScalerTensorOps
{
    /// <summary>float64 machine epsilon, the constant scikit-learn uses in its constant-feature test.</summary>
    internal const double Float64Epsilon = 2.220446049250313e-16;

    private static IEngine Engine => AiDotNetEngine.Current;

    /// <summary>A 0/1 tensor [numColumns] that is 1 for the given column positions.</summary>
    public static Tensor<T> ColumnMask<T>(int numColumns, int[] columns)
    {
        var flags = new int[numColumns];
        foreach (int column in columns)
            flags[column] = 1;
        return new Tensor<int>(flags, new[] { numColumns }).Cast<T>();
    }

    /// <summary>
    /// Per-column parameters: <paramref name="values"/> on the masked columns, <paramref name="fallback"/> on the others.
    /// </summary>
    public static Vector<T> MaskedParameters<T>(Tensor<T> mask, Tensor<T> values, T fallback) =>
        Engine.TensorWhere(mask, values, PreprocessingTensorOps.Filled(values.Shape.ToArray(), fallback)).ToVector();

    /// <summary>A vector of <paramref name="length"/> copies of <paramref name="value"/>.</summary>
    public static Vector<T> FilledVector<T>(int length, T value) =>
        PreprocessingTensorOps.Filled(new[] { length }, value).ToVector();

    /// <summary>Broadcasts a [p] row to an [n, p] tensor.</summary>
    public static Tensor<T> BroadcastRow<T>(Tensor<T> row, int rows) =>
        Engine.TensorBroadcastTo(Engine.Reshape(row, new[] { 1, row.Length }), new[] { rows, row.Length });

    /// <summary>Broadcasts a per-column parameter vector to an [n, p] tensor.</summary>
    public static Tensor<T> BroadcastRow<T>(Vector<T> row, int rows) => BroadcastRow(Tensor<T>.FromVector(row), rows);

    /// <summary>Throws when <paramref name="data"/> does not have the fitted column count.</summary>
    public static void EnsureColumnCount<T>(Matrix<T> data, int fittedColumns)
    {
        if (data.Columns != fittedColumns)
        {
            throw new ArgumentException(
                $"Data has {data.Columns} columns, but the transformer was fitted on {fittedColumns} columns.",
                nameof(data));
        }
    }

    /// <summary><c>(x - shift) / scale</c> column-wise. A column with shift 0 and scale 1 is returned unchanged.</summary>
    public static Matrix<T> ShiftThenScale<T>(Matrix<T> data, Vector<T> shift, Vector<T> scale)
    {
        EnsureColumnCount(data, shift.Length);
        if (data.Rows == 0 || data.Columns == 0)
            return new Matrix<T>(data.Rows, data.Columns);

        var x = Tensor<T>.FromMatrix(data);
        var centered = Engine.TensorSubtract(x, BroadcastRow(shift, data.Rows));
        return Engine.TensorDivide(centered, BroadcastRow(scale, data.Rows)).ToMatrix();
    }

    /// <summary><c>x * scale + shift</c> column-wise (the inverse of <see cref="ShiftThenScale{T}"/>).</summary>
    public static Matrix<T> ScaleThenShift<T>(Matrix<T> data, Vector<T> scale, Vector<T> shift)
    {
        EnsureColumnCount(data, shift.Length);
        if (data.Rows == 0 || data.Columns == 0)
            return new Matrix<T>(data.Rows, data.Columns);

        var x = Tensor<T>.FromMatrix(data);
        var scaled = Engine.TensorMultiply(x, BroadcastRow(scale, data.Rows));
        return Engine.TensorAdd(scaled, BroadcastRow(shift, data.Rows)).ToMatrix();
    }

    /// <summary>
    /// Keeps <paramref name="transformed"/> on the masked columns and <paramref name="original"/> on the others.
    /// </summary>
    public static Matrix<T> MergeColumns<T>(Tensor<T> columnMask, Tensor<T> transformed, Matrix<T> original)
    {
        if (original.Rows == 0 || original.Columns == 0)
            return new Matrix<T>(original.Rows, original.Columns);

        return Engine.TensorWhere(
            BroadcastRow(columnMask, original.Rows), transformed, Tensor<T>.FromMatrix(original)).ToMatrix();
    }

    /// <summary>A 0/1 tensor that is 1 where <paramref name="x"/> is NaN.</summary>
    public static Tensor<T> NanMask<T>(Tensor<T> x)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var shape = x.Shape.ToArray();
        return Engine.TensorWhere(
            Engine.TensorIsNan(x), PreprocessingTensorOps.Filled(shape, ops.One), new Tensor<T>(shape));
    }

    /// <summary>Replaces NaN entries of <paramref name="x"/> with <paramref name="value"/>.</summary>
    public static Tensor<T> FillNan<T>(Tensor<T> x, T value) =>
        Engine.TensorWhere(Engine.TensorIsNan(x), PreprocessingTensorOps.Filled(x.Shape.ToArray(), value), x);

    /// <summary>
    /// Whether the statistics of <typeparamref name="T"/> are reduced in float. numpy accumulates float16 sums,
    /// means and variances in float32; a Half accumulator stops counting at 2048 (2048 + 1 rounds back to 2048),
    /// its sums overflow past 65504 and the squared deviations of values 256 apart already do. The statistics are
    /// reduced in float and rounded to Half once.
    /// </summary>
    internal static bool ReducesInFloat<T>() => typeof(T) == typeof(Half);

    /// <summary>Number of non-NaN entries per column of an [n, p] tensor, as [p] (counted in int).</summary>
    public static Tensor<T> NanColumnCount<T>(Tensor<T> x) =>
        new Tensor<int>(CountPerColumn(PreprocessingTensorOps.OneMinus(NanMask(x))), new[] { x.Shape[1] }).Cast<T>();

    /// <summary>Mean of the non-NaN entries per column (NaN for an all-NaN column), as [p].</summary>
    public static Tensor<T> NanColumnMean<T>(Tensor<T> x) =>
        ReducesInFloat<T>()
            ? Engine.TensorNanMean(x.Cast<float>(), new[] { 0 }, false).Cast<T>()
            : Engine.TensorNanMean(x, new[] { 0 }, false);

    /// <summary>Mean of every column of an [n, p] tensor (NaN propagates), as [p].</summary>
    public static Tensor<T> ColumnMean<T>(Tensor<T> x) =>
        ReducesInFloat<T>()
            ? Engine.ReduceMean(x.Cast<float>(), new[] { 0 }, false).Cast<T>()
            : Engine.ReduceMean(x, new[] { 0 }, false);

    /// <summary>The mean of all the entries of a tensor.</summary>
    public static T Mean<T>(Tensor<T> x) =>
        ReducesInFloat<T>()
            ? MathHelper.GetNumericOperations<T>().FromDouble(Engine.TensorMean(x.Cast<float>()))
            : Engine.TensorMean(x);

    /// <summary>The sum of all the entries of a tensor, as a double.</summary>
    public static double Sum<T>(Tensor<T> x) =>
        ReducesInFloat<T>()
            ? Engine.TensorSum(x.Cast<float>())
            : MathHelper.GetNumericOperations<T>().ToDouble(Engine.TensorSum(x));

    /// <summary>The population (ddof = 0) variance of all the entries of a tensor, as a double (numpy <c>var</c>).</summary>
    public static double Variance<T>(Tensor<T> x)
    {
        if (ReducesInFloat<T>())
            return VarianceCore(x.Cast<float>());
        return MathHelper.GetNumericOperations<T>().ToDouble(VarianceCore(x));
    }

    private static TW VarianceCore<TW>(Tensor<TW> x)
    {
        TW mean = Engine.TensorMean(x);
        return Engine.TensorMean(Engine.TensorSquare(Engine.TensorSubtractScalar(x, mean)));
    }

    /// <summary>
    /// Per column of an [n, p] tensor, ignoring NaN: the mean, the population (ddof = 0) standard deviation
    /// (numpy <c>nanmean</c> / <c>nanstd</c>) and scikit-learn's constant-feature mask (see
    /// <see cref="ConstantFeatureMask{T}"/>), each [p]. Half is reduced in float (see <see cref="ReducesInFloat{T}"/>);
    /// its standard deviation is rounded to Half after the square root, so a variance past 65504 does not overflow.
    /// </summary>
    public static (Tensor<T> Mean, Tensor<T> Std, Tensor<T> Constant) NanColumnMoments<T>(Tensor<T> x)
    {
        if (ReducesInFloat<T>())
        {
            var (mean, std, constant) = NanColumnMomentsCore(x.Cast<float>());
            return (mean.Cast<T>(), std.Cast<T>(), constant.Cast<T>());
        }

        return NanColumnMomentsCore(x);
    }

    private static (Tensor<TW> Mean, Tensor<TW> Std, Tensor<TW> Constant) NanColumnMomentsCore<TW>(Tensor<TW> x)
    {
        var mean = Engine.TensorNanMean(x, new[] { 0 }, false);
        var deviation = Engine.TensorSubtract(x, BroadcastRow(mean, x.Shape[0]));
        var variance = Engine.TensorNanMean(Engine.TensorSquare(deviation), new[] { 0 }, false);
        var constant = ConstantFeatureMask(variance, mean, NanColumnCount(x));
        var std = Engine.TensorSqrt(Engine.TensorMax(variance, MathHelper.GetNumericOperations<TW>().Zero));
        return (mean, std, constant);
    }

    /// <summary>Maximum of the non-NaN entries per column, as [p] (the type's minimum for an all-NaN column).</summary>
    public static Tensor<T> NanColumnMax<T>(Tensor<T> x) =>
        Engine.ReduceMax(FillNan(x, MathHelper.GetNumericOperations<T>().MinValue), new[] { 0 }, false);

    /// <summary>Minimum of the non-NaN entries per column, as [p] (the type's maximum for an all-NaN column).</summary>
    public static Tensor<T> NanColumnMin<T>(Tensor<T> x) =>
        PreprocessingTensorOps.ColumnMin(FillNan(x, MathHelper.GetNumericOperations<T>().MaxValue));

    /// <summary>
    /// scikit-learn's <c>_is_constant_feature</c>: a column is constant when its variance is within the
    /// floating-point error bound <c>n * eps * var + (n * mean * eps)^2</c>. Returns a 0/1 [p] tensor.
    /// </summary>
    public static Tensor<T> ConstantFeatureMask<T>(Tensor<T> variance, Tensor<T> mean, Tensor<T> count)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        T eps = ops.FromDouble(Float64Epsilon);
        var nEps = Engine.TensorMultiplyScalar(count, eps);
        var bound = Engine.TensorAdd(
            Engine.TensorMultiply(nEps, variance), Engine.TensorSquare(Engine.TensorMultiply(nEps, mean)));
        return Engine.TensorLessEqual(variance, bound);
    }

    /// <summary>
    /// The machine epsilon of <typeparamref name="T"/> (numpy <c>finfo(dtype).eps</c>): 2^-23 for float, 2^-10 for
    /// Half and the float64 epsilon otherwise.
    /// </summary>
    public static double MachineEpsilon<T>() =>
        typeof(T) == typeof(float) ? 1.1920928955078125e-7
        : typeof(T) == typeof(Half) ? 0.0009765625
        : Float64Epsilon;

    /// <summary>
    /// scikit-learn's <c>_handle_zeros_in_scale</c>: without a constant mask, scales below <c>10 * eps</c> (eps of the
    /// data's dtype, <see cref="MachineEpsilon{T}"/>, or <paramref name="epsilon"/> when the scale was computed in a
    /// wider type) become 1; with one, exactly the masked constant columns become 1 (scikit-learn then skips the
    /// <c>10 * eps</c> test, so a tiny but varying scale is kept).
    /// </summary>
    public static Tensor<T> HandleZerosInScale<T>(Tensor<T> scale, Tensor<T>? constantMask = null, double? epsilon = null)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var ones = PreprocessingTensorOps.Filled(scale.Shape.ToArray(), ops.One);
        var replaced = constantMask
            ?? Engine.TensorLessThan(scale, ops.FromDouble(10 * (epsilon ?? MachineEpsilon<T>())));
        return Engine.TensorWhere(replaced, ones, scale);
    }

    /// <summary>
    /// Clips every column of <paramref name="x"/> ([n, p]) to [lower, upper]; NaN passes through unchanged
    /// (like <c>Math.Max(lower, Math.Min(upper, NaN))</c>).
    /// </summary>
    public static Tensor<T> ClipColumns<T>(Tensor<T> x, Vector<T> lower, Vector<T> upper)
    {
        int rows = x.Shape[0];
        var low = BroadcastRow(lower, rows);
        var high = BroadcastRow(upper, rows);
        var raised = Engine.TensorWhere(Engine.TensorGreaterThan(x, high), high, x);
        return Engine.TensorWhere(Engine.TensorLessThan(raised, low), low, raised);
    }

    /// <summary>A 0/1 [n, p] mask of the entries below <paramref name="lower"/> or above <paramref name="upper"/>.</summary>
    public static Tensor<T> OutsideMask<T>(Tensor<T> x, Vector<T> lower, Vector<T> upper)
    {
        int rows = x.Shape[0];
        return Engine.TensorMax(
            Engine.TensorLessThan(x, BroadcastRow(lower, rows)), Engine.TensorGreaterThan(x, BroadcastRow(upper, rows)));
    }

    /// <summary>Copies a 0/1 [n, p] mask into a <c>bool[n, p]</c>.</summary>
    public static bool[,] ToBoolMatrix<T>(Tensor<T> mask)
    {
        int rows = mask.Shape[0];
        int columns = mask.Shape[1];
        var result = new bool[rows, columns];
        if (rows == 0 || columns == 0)
            return result;

        var flags = mask.Cast<int>().AsSpan();
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < columns; j++)
                result[i, j] = flags[i * columns + j] != 0;
        return result;
    }

    /// <summary>
    /// Number of non-zero entries in each column of a 0/1 [n, p] mask. The mask is summed as int: a sum in T
    /// stops counting at 2048 for Half (2048 + 1 rounds back to 2048).
    /// </summary>
    public static int[] CountPerColumn<T>(Tensor<T> mask) =>
        mask.Shape[0] == 0
            ? new int[mask.Shape[1]]
            : Engine.ReduceSum(mask.Cast<int>(), new[] { 0 }, false).AsSpan().ToArray();

    /// <summary>
    /// The most frequent non-NaN value of every column in <paramref name="columns"/> (scipy / scikit-learn's mode:
    /// the smallest value among ties), as [p]; the other columns, and columns with no observed value, get 0.
    /// </summary>
    public static Tensor<T> ColumnModes<T>(Matrix<T> data, IEnumerable<int> columns)
    {
        var modes = new Tensor<T>(new[] { data.Columns });
        foreach (int col in columns)
        {
            var column = PreprocessingTensorOps.Column(data, col);
            var observed = PreprocessingTensorOps.MaskPositions(PreprocessingTensorOps.Not(NanMask(column)));
            if (observed.Length == 0)
                continue;

            var values = Engine.TensorIndexSelect(column, new Tensor<int>(observed, new[] { observed.Length }), 0);
            modes = Engine.TensorSetSlice(modes, PreprocessingTensorOps.Filled(new[] { 1 }, Mode(values)), new[] { col });
        }

        return modes;
    }

    /// <summary>
    /// The most frequent value of a non-empty 1-D tensor, the smallest among ties, in O(n log n): each distinct
    /// value's count is the width of its run in the sorted values (the engine's TensorMode compares every value
    /// with every distinct value seen so far, which is quadratic for a continuous column).
    /// </summary>
    public static T Mode<T>(Tensor<T> values)
    {
        var sorted = Engine.TensorSort(values, 0, false).Item1;
        var distinct = PreprocessingTensorOps.SortedUnique(values);
        var start = Engine.TensorSearchSorted(sorted, distinct, false);
        var end = Engine.TensorSearchSorted(sorted, distinct, true);
        var counts = Engine.TensorSubtract(end, start).Cast<double>();

        // distinct is ascending, so the first largest count is the smallest of the most frequent values
        int best = PreprocessingTensorOps.FirstArgMax(counts);
        return distinct.AsSpan()[best];
    }

    /// <summary>Converts a double hyper-parameter to T.</summary>
    public static T Scalar<T>(double value) => MathHelper.GetNumericOperations<T>().FromDouble(value);
}
