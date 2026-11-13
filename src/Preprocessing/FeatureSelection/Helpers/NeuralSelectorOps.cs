using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Helpers;

/// <summary>
/// Engine-op building blocks of the small neural feature selectors trained by per-sample gradient descent:
/// column standardization, row splitting, vector-matrix products and outer products.
/// </summary>
internal static class NeuralSelectorOps
{
    /// <summary>
    /// Columns of x [n, p] centered and divided by the population standard deviation, using 1 for a column
    /// whose deviation is below 1e-10.
    /// </summary>
    public static Tensor<T> StandardizePopulation<T>(Tensor<T> x)
    {
        var engine = AiDotNetEngine.Current;
        int n = x.Shape[0];
        int p = x.Shape[1];
        var centered = engine.TensorSubtract(x, engine.TensorBroadcastTo(engine.ReduceMean(x, new[] { 0 }, true), new[] { n, p }));
        return engine.TensorDivide(centered, engine.TensorBroadcastTo(PopulationStdOrOne(centered), new[] { n, p }));
    }

    /// <summary>
    /// The population standard deviation [1, p] of every column of the centred <paramref name="centered"/> [n, p],
    /// with 1 for a column whose deviation is below 1e-10 (the smallest normal value in Half: 1e-10 itself is 0
    /// in Half, so a constant column kept std 0 and divided 0 by 0).
    /// </summary>
    public static Tensor<T> PopulationStdOrOne<T>(Tensor<T> centered)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int p = centered.Shape[1];
        var std = engine.TensorSqrt(engine.ReduceMean(engine.TensorSquare(centered), new[] { 0 }, true));
        return engine.TensorWhere(
            engine.TensorLessThan(std, PreprocessingTensorOps.Tolerance<T>(1e-10)),
            PreprocessingTensorOps.Filled(new[] { 1, p }, ops.One),
            std);
    }

    /// <summary>Columns of x [n, p] centered and divided by (sample standard deviation + 1e-10).</summary>
    public static Tensor<T> StandardizeSample<T>(Tensor<T> x)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = x.Shape[0];
        int p = x.Shape[1];
        var centered = engine.TensorSubtract(x, engine.TensorBroadcastTo(engine.ReduceMean(x, new[] { 0 }, true), new[] { n, p }));
        var std = engine.TensorAddScalar(
            engine.TensorSqrt(engine.TensorDivideScalar(
                engine.ReduceSum(engine.TensorSquare(centered), new[] { 0 }, true), ops.FromDouble(n - 1))),
            PopulationSearchOps.Tiny<T>(1e-10));
        return engine.TensorDivide(centered, engine.TensorBroadcastTo(std, new[] { n, p }));
    }

    /// <summary>The rows of x [n, p] as n tensors [p].</summary>
    public static Tensor<T>[] SplitRows<T>(Tensor<T> x)
    {
        int n = x.Shape[0];
        var rows = new Tensor<T>[n];
        for (int i = 0; i < n; i++)
            rows[i] = PopulationSearchOps.Row(x, i);
        return rows;
    }

    /// <summary>The elements of v [n] as n tensors [1].</summary>
    public static Tensor<T>[] SplitValues<T>(Tensor<T> v)
    {
        int n = v.Length;
        var values = new Tensor<T>[n];
        for (int i = 0; i < n; i++)
            values[i] = PopulationSearchOps.At(v, i);
        return values;
    }

    /// <summary>v W for v [m] and W [m, k] ([k]).</summary>
    public static Tensor<T> VecMat<T>(Tensor<T> v, Tensor<T> w)
    {
        var engine = AiDotNetEngine.Current;
        return engine.Reshape(engine.TensorMatMul(engine.Reshape(v, new[] { 1, v.Length }), w), new[] { w.Shape[1] });
    }

    /// <summary>W v for W [m, k] and v [k] ([m]).</summary>
    public static Tensor<T> MatVec<T>(Tensor<T> w, Tensor<T> v)
    {
        var engine = AiDotNetEngine.Current;
        return engine.Reshape(engine.TensorMatMul(w, engine.Reshape(v, new[] { v.Length, 1 })), new[] { w.Shape[0] });
    }

    /// <summary>The outer product a b^T of a [m] and b [k] ([m, k]).</summary>
    public static Tensor<T> Outer<T>(Tensor<T> a, Tensor<T> b)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorMatMul(engine.Reshape(a, new[] { a.Length, 1 }), engine.Reshape(b, new[] { 1, b.Length }));
    }

    /// <summary>w - rate * gradient.</summary>
    public static Tensor<T> Step<T>(Tensor<T> w, Tensor<T> gradient, T rate)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorSubtract(w, engine.TensorMultiplyScalar(gradient, rate));
    }

    /// <summary>Uniform draws of the given shape mapped to [-scale, scale).</summary>
    public static Tensor<T> SymmetricUniform<T>(Random random, double scale, params int[] shape) =>
        PopulationSearchOps.Affine(PopulationSearchOps.Uniform<T>(random, shape), 2 * scale, -scale);
}
