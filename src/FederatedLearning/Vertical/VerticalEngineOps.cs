namespace AiDotNet.FederatedLearning.Vertical;

/// <summary>
/// Whole-tensor building blocks for the small dense networks used by the vertical FL parties
/// (row-major flat storage, as the original per-element loops read it).
/// </summary>
internal static class VerticalEngineOps
{
    /// <summary>a[m,k] * b[k,n].</summary>
    internal static Tensor<T> MatMul<T>(Tensor<T> a, Tensor<T> b, int m, int k, int n)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorMatMul(Rows(a, m, k), Rows(b, k, n));
    }

    /// <summary>a[m,k]^T * b[m,n], a <c>[k, n]</c> result.</summary>
    internal static Tensor<T> MatMulTransposeA<T>(Tensor<T> a, Tensor<T> b, int m, int k, int n)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorMatMul(engine.TensorTranspose(Rows(a, m, k)), Rows(b, m, n));
    }

    /// <summary>a[m,k] * b[n,k]^T, a <c>[m, n]</c> result.</summary>
    internal static Tensor<T> MatMulTransposeB<T>(Tensor<T> a, Tensor<T> b, int m, int k, int n)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorMatMul(Rows(a, m, k), engine.TensorTranspose(Rows(b, n, k)));
    }

    /// <summary>tensor[b, d] + bias[d] for every row b.</summary>
    internal static Tensor<T> AddBias<T>(Tensor<T> tensor, Tensor<T> bias, int batchSize, int dim)
    {
        var engine = AiDotNetEngine.Current;
        var tiled = engine.TensorTile(Rows(bias, 1, dim), new[] { batchSize, 1 });
        return engine.TensorAdd(Rows(tensor, batchSize, dim), tiled);
    }

    /// <summary>Negative entries become zero; NaN is kept.</summary>
    internal static Tensor<T> Relu<T>(Tensor<T> tensor)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        var negative = engine.TensorLessThan(tensor, numOps.Zero);
        return engine.TensorWhere(negative, new Tensor<T>(tensor.Shape.ToArray()), tensor);
    }

    /// <summary>
    /// Zeroes the gradient where the activation is not positive. A NaN activation zeroes the gradient
    /// when <paramref name="zeroWhereActivationIsNaN"/> is set and keeps it otherwise.
    /// </summary>
    internal static Tensor<T> ReluGradient<T>(Tensor<T> grad, Tensor<T> activation, bool zeroWhereActivationIsNaN)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        var zeros = new Tensor<T>(grad.Shape.ToArray());
        var act = activation.Reshape(grad.Shape.ToArray());
        if (zeroWhereActivationIsNaN)
        {
            // keep only where activation > 0
            return engine.TensorWhere(engine.TensorGreaterThan(act, numOps.Zero), grad, zeros);
        }

        // zero only where activation <= 0
        return engine.TensorWhere(engine.TensorLessEqual(act, new Tensor<T>(grad.Shape.ToArray())), zeros, grad);
    }

    /// <summary>Column sums of a <c>[batchSize, dim]</c> tensor.</summary>
    internal static Tensor<T> SumOverBatch<T>(Tensor<T> tensor, int batchSize, int dim)
    {
        return AiDotNetEngine.Current.ReduceSum(Rows(tensor, batchSize, dim), new[] { 0 }, false);
    }

    /// <summary>param -= lr * grad, written into <paramref name="param"/> (its storage is shared with callers).</summary>
    internal static void UpdateInPlace<T>(Tensor<T> param, Tensor<T> grad, double lr)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        int total = param.Length;
        var flatGrad = engine.TensorSlice(grad.Reshape(new[] { grad.Length }), new[] { 0 }, new[] { total });
        var updated = engine.TensorSubtract(param.Reshape(new[] { total }), engine.TensorMultiplyScalar(flatGrad, numOps.FromDouble(lr)));
        updated.AsSpan().CopyTo(param.AsWritableSpan());
    }

    /// <summary>The first <c>rows * cols</c> entries of <paramref name="tensor"/> as a new <c>[rows, cols]</c> tensor.</summary>
    internal static Tensor<T> Rows<T>(Tensor<T> tensor, int rows, int cols)
    {
        int count = rows * cols;
        if (tensor.Length == count)
        {
            return tensor.Reshape(new[] { rows, cols });
        }

        var engine = AiDotNetEngine.Current;
        return engine.TensorSlice(tensor.Reshape(new[] { tensor.Length }), new[] { 0 }, new[] { count }).Reshape(new[] { rows, cols });
    }
}
