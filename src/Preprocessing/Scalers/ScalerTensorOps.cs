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

    /// <summary>Number of non-NaN entries per column of an [n, p] tensor, as [p].</summary>
    public static Tensor<T> NanColumnCount<T>(Tensor<T> x) =>
        Engine.ReduceSum(PreprocessingTensorOps.OneMinus(NanMask(x)), new[] { 0 }, false);

    /// <summary>Mean of the non-NaN entries per column (NaN for an all-NaN column), as [p].</summary>
    public static Tensor<T> NanColumnMean<T>(Tensor<T> x) => Engine.TensorNanMean(x, new[] { 0 }, false);

    /// <summary>
    /// Population (ddof = 0) variance of the non-NaN entries per column around <paramref name="mean"/>, as [p]
    /// (numpy <c>nanvar</c>).
    /// </summary>
    public static Tensor<T> NanColumnVariance<T>(Tensor<T> x, Tensor<T> mean)
    {
        var deviation = Engine.TensorSubtract(x, BroadcastRow(mean, x.Shape[0]));
        return Engine.TensorNanMean(Engine.TensorSquare(deviation), new[] { 0 }, false);
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
    /// scikit-learn's <c>_handle_zeros_in_scale</c>: scales below <c>10 * eps</c>, and the masked constant
    /// columns, become 1 so they are left unscaled.
    /// </summary>
    public static Tensor<T> HandleZerosInScale<T>(Tensor<T> scale, Tensor<T>? constantMask = null)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var ones = PreprocessingTensorOps.Filled(scale.Shape.ToArray(), ops.One);
        var tiny = Engine.TensorLessThan(scale, ops.FromDouble(10 * Float64Epsilon));
        var result = Engine.TensorWhere(tiny, ones, scale);
        return constantMask is null ? result : Engine.TensorWhere(constantMask, ones, result);
    }
}
