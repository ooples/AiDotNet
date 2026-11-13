using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.FederatedLearning.Infrastructure;

/// <summary>
/// Reductions shared by the federated-learning components that stay correct for 16-bit floats.
/// </summary>
/// <remarks>
/// A sum, dot product or sum of squares accumulated in <see cref="Half"/> stops growing at 2048 (2048 + 1 rounds back
/// to 2048) and overflows past 65504, so norms, distances and client averages of Half or BFloat16 values are
/// evaluated in float and returned (or cast back) once. Every other T uses the engine op in T directly.
/// </remarks>
internal static class FederatedReductions
{
    /// <summary>Smallest normal Half (2^-14): epsilons below it are flushed to 0 or subnormal in Half.</summary>
    private const double HalfSmallestNormal = 6.103515625E-05;

    private static IEngine Engine => AiDotNetEngine.Current;

    /// <summary>Whether <typeparamref name="T"/> is a 16-bit float (Half or BFloat16).</summary>
    public static bool IsNarrowFloat<T>() =>
        typeof(T) == typeof(Half) || typeof(T) == typeof(AiDotNet.Tensors.NumericOperations.BFloat16);

    /// <summary>
    /// <paramref name="epsilon"/> as a T guard that stays positive: for Half it is raised to Half's smallest normal.
    /// </summary>
    public static T Tolerance<T>(double epsilon) =>
        MathHelper.GetNumericOperations<T>().FromDouble(
            typeof(T) == typeof(Half) ? Math.Max(epsilon, HalfSmallestNormal) : epsilon);

    /// <summary>Dot product a . b.</summary>
    public static double Dot<T>(Vector<T> a, Vector<T> b)
    {
        if (IsNarrowFloat<T>())
        {
            return Engine.DotProduct(a.AsTensor().Cast<float>().ToVector(), b.AsTensor().Cast<float>().ToVector());
        }

        return MathHelper.GetNumericOperations<T>().ToDouble(Engine.DotProduct(a, b));
    }

    /// <summary>Squared L2 norm ||a||^2.</summary>
    public static double SquaredNorm<T>(Vector<T> a) => Dot(a, a);

    /// <summary>Squared L2 distance ||a - b||^2.</summary>
    public static double SquaredDistance<T>(Vector<T> a, Vector<T> b)
    {
        if (IsNarrowFloat<T>())
        {
            var diff = Engine.Subtract(a.AsTensor().Cast<float>().ToVector(), b.AsTensor().Cast<float>().ToVector());
            return Engine.DotProduct(diff, diff);
        }

        var d = Engine.Subtract(a, b);
        return MathHelper.GetNumericOperations<T>().ToDouble(Engine.DotProduct(d, d));
    }

    /// <summary>Cosine similarity a . b / (||a|| ||b||), or 0 when that denominator is not above <paramref name="floor"/>.</summary>
    public static double Cosine<T>(Vector<T> a, Vector<T> b, double floor)
    {
        double denom = Math.Sqrt(SquaredNorm(a)) * Math.Sqrt(SquaredNorm(b));
        return denom > floor ? Dot(a, b) / denom : 0.0;
    }

    /// <summary>Sum of squares of every element.</summary>
    public static double SumOfSquares<T>(Tensor<T> x)
    {
        if (IsNarrowFloat<T>())
        {
            return Engine.TensorSumOfSquares(x.Cast<float>());
        }

        return MathHelper.GetNumericOperations<T>().ToDouble(Engine.TensorSumOfSquares(x));
    }

    /// <summary>Sum of squared element-wise differences sum (a - b)^2 of equal-shape tensors.</summary>
    public static double SumOfSquaredDifferences<T>(Tensor<T> a, Tensor<T> b)
    {
        if (IsNarrowFloat<T>())
        {
            return Engine.TensorSumOfSquares(Engine.TensorSubtract(a.Cast<float>(), b.Cast<float>()));
        }

        return MathHelper.GetNumericOperations<T>().ToDouble(Engine.TensorSumOfSquares(Engine.TensorSubtract(a, b)));
    }

    /// <summary>Sum of absolute values of every element.</summary>
    public static double SumAbs<T>(Tensor<T> x)
    {
        if (IsNarrowFloat<T>())
        {
            return Engine.TensorSum(Engine.TensorAbs(x.Cast<float>()));
        }

        return MathHelper.GetNumericOperations<T>().ToDouble(Engine.TensorSum(Engine.TensorAbs(x)));
    }

    /// <summary>
    /// sum_r weights[r] * rows[r] for an [m, P] tensor of rows, as a length-P vector. For a narrow T the rows are
    /// combined in float and the result is cast back once, so a client average does not saturate or overflow.
    /// </summary>
    public static Vector<T> WeightedRowSum<T>(Tensor<T> rows, double[] weights)
    {
        int m = rows.Shape[0];
        int p = rows.Shape[1];
        if (weights.Length != m)
        {
            throw new ArgumentException($"Expected {m} row weights, got {weights.Length}.", nameof(weights));
        }

        var w = new Tensor<double>(weights, new[] { 1, m });
        if (IsNarrowFloat<T>())
        {
            return Engine.TensorMatMul(w.Cast<float>(), rows.Cast<float>()).Reshape(p).Cast<T>().ToVector();
        }

        return Engine.TensorMatMul(w.Cast<T>(), rows).Reshape(p).ToVector();
    }

    /// <summary>sum_i weights[i] * vectors[i] for equal-length vectors (see <see cref="WeightedRowSum{T}"/>).</summary>
    public static Vector<T> WeightedSum<T>(IReadOnlyList<Vector<T>> vectors, double[] weights)
    {
        if (vectors.Count == 0)
        {
            throw new ArgumentException("At least one vector is required.", nameof(vectors));
        }

        var rows = Engine.TensorStack(vectors.Select(v => v.AsTensor()).ToArray(), 0);
        return WeightedRowSum(rows, weights);
    }

    /// <summary>||rows[r] - center||^2 for every row r of an [m, P] tensor.</summary>
    public static double[] RowSquaredDistances<T>(Tensor<T> rows, Vector<T> center)
    {
        if (IsNarrowFloat<T>())
        {
            return RowSquaredDistancesCore(rows.Cast<float>(), center.AsTensor().Cast<float>()).Cast<double>().ToArray();
        }

        return RowSquaredDistancesCore(rows, center.AsTensor()).Cast<double>().ToArray();
    }

    private static Tensor<TV> RowSquaredDistancesCore<TV>(Tensor<TV> rows, Tensor<TV> center)
    {
        int m = rows.Shape[0];
        int p = rows.Shape[1];
        var diff = Engine.TensorSubtract(rows, Engine.TensorBroadcastTo(center.Reshape(1, p), new[] { m, p }));
        return Engine.ReduceSum(Engine.TensorSquare(diff), new[] { 1 }, false);
    }

    /// <summary>Unweighted mean of equal-length vectors.</summary>
    public static Vector<T> Mean<T>(IReadOnlyList<Vector<T>> vectors) =>
        WeightedSum(vectors, Enumerable.Repeat(1.0 / vectors.Count, vectors.Count).ToArray());
}
