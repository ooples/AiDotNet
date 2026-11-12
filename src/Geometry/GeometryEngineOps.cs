namespace AiDotNet.Geometry;

/// <summary>
/// Whole-array mesh and point-cloud primitives shared by the geometry types, expressed as engine operations.
/// </summary>
/// <typeparam name="T">The numeric type of the coordinates.</typeparam>
internal static class GeometryEngineOps<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private static IEngine Engine => AiDotNetEngine.Current;

    /// <summary>
    /// Returns one column of a [rows, columns] tensor as a rank-1 tensor.
    /// </summary>
    internal static Tensor<TValue> Column<TValue>(Tensor<TValue> table, int column)
    {
        int rows = table.Shape[0];
        return Engine.TensorSlice(table, new[] { 0, column }, new[] { rows, 1 }).Reshape(rows);
    }

    /// <summary>
    /// Returns the given corner of every face as a [faces, 3] tensor of positions.
    /// </summary>
    internal static Tensor<T> FaceCorner(Tensor<T> vertices, Tensor<int> faces, int corner)
    {
        return Engine.TensorIndexSelect(vertices, Column(faces, corner), 0);
    }

    /// <summary>
    /// Returns the unnormalized face normals (v1 - v0) x (v2 - v0) as a [faces, 3] tensor.
    /// </summary>
    internal static Tensor<T> FaceCrossProducts(Tensor<T> vertices, Tensor<int> faces)
    {
        var p0 = FaceCorner(vertices, faces, 0);
        var edge1 = Engine.TensorSubtract(FaceCorner(vertices, faces, 1), p0);
        var edge2 = Engine.TensorSubtract(FaceCorner(vertices, faces, 2), p0);
        return Engine.TensorCross(edge1, edge2, 1);
    }

    /// <summary>
    /// Returns the squared Euclidean length of every row of a [rows, 3] tensor as a [rows, 1] tensor.
    /// </summary>
    internal static Tensor<T> RowSquaredNorms(Tensor<T> rows)
    {
        return Engine.ReduceSum(Engine.TensorSquare(rows), new[] { 1 }, true);
    }

    /// <summary>
    /// Multiplies every row of a [rows, columns] tensor by the matching entry of a [rows, 1] tensor.
    /// </summary>
    internal static Tensor<T> ScaleRows(Tensor<T> rows, Tensor<T> factors)
    {
        return Engine.TensorMultiply(rows, Engine.TensorBroadcastTo(factors, rows.Shape));
    }

    /// <summary>
    /// Scales every row to unit length; rows whose squared length is not positive become zero.
    /// </summary>
    internal static Tensor<T> NormalizeRowsOrZero(Tensor<T> rows)
    {
        var lengthSquared = RowSquaredNorms(rows);
        var positive = Engine.TensorGreaterThan(lengthSquared, NumOps.Zero);
        var inverseLength = Engine.TensorReciprocal(Engine.TensorSqrt(lengthSquared));
        var factors = Engine.TensorWhere(positive, inverseLength, Engine.TensorZerosLike(lengthSquared));
        return ScaleRows(rows, factors);
    }

    /// <summary>
    /// Divides every row by its length when that length exceeds <paramref name="minLength"/>; shorter rows are
    /// returned unchanged.
    /// </summary>
    internal static Tensor<T> NormalizeRowsAbove(Tensor<T> rows, T minLength)
    {
        var length = Engine.TensorBroadcastTo(Engine.TensorSqrt(RowSquaredNorms(rows)), rows.Shape);
        var usable = Engine.TensorGreaterThan(length, minLength);
        return Engine.TensorWhere(usable, Engine.TensorDivide(rows, length), rows);
    }

    /// <summary>
    /// Sums each face's row of <paramref name="faceValues"/> into its three vertices, in face then corner order.
    /// </summary>
    internal static Tensor<T> AccumulateToVertices(Tensor<T> faceValues, Tensor<int> faces, int numVertices)
    {
        int numFaces = faces.Shape[0];
        int width = faceValues.Shape[1];
        var sums = new Tensor<T>(new[] { numVertices, width });
        if (numFaces == 0)
        {
            return sums;
        }

        var cornerVertices = faces.Reshape(numFaces * 3);
        var cornerValues = Engine.TensorRepeatInterleave(faceValues, 3, 0);
        return Engine.TensorIndexAdd(sums, 0, cornerVertices, cornerValues);
    }

    /// <summary>
    /// Returns the per-column minimum and maximum of a [rows, columns] tensor, ignoring NaN entries. The results
    /// start from <c>MaxValue</c> and <c>MinValue</c>, as a running comparison would, so they never exceed the
    /// type's finite range and an all-NaN column reports those start values.
    /// </summary>
    internal static (Tensor<T> min, Tensor<T> max) ColumnBounds(Tensor<T> rows)
    {
        var isNan = Engine.TensorIsNan(rows);
        var forMin = Engine.TensorWhere(isNan, Engine.TensorFullLike(rows, NumOps.MaxValue), rows);
        var forMax = Engine.TensorWhere(isNan, Engine.TensorFullLike(rows, NumOps.MinValue), rows);
        var min = Engine.TensorNegate(Engine.ReduceMax(Engine.TensorNegate(forMin), new[] { 0 }, false));
        var max = Engine.ReduceMax(forMax, new[] { 0 }, false);
        return (Engine.TensorMin(min, NumOps.MaxValue), Engine.TensorMax(max, NumOps.MinValue));
    }
}
