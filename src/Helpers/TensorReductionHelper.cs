using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Helpers;

/// <summary>
/// Scalar tensor reductions that stand in for an engine op with a known defect.
/// </summary>
internal static class TensorReductionHelper
{
    /// <summary>
    /// The sum of the squares of every element of <paramref name="tensor"/>, as <see cref="IEngine.TensorSumOfSquares{T}"/>
    /// is documented to return it.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Use this instead of <c>engine.TensorSumOfSquares(x)</c>. In AiDotNet.Tensors 0.150.0 the engine op is wrong for a
    /// <c>Tensor&lt;double&gt;</c> of 16 or more elements on a CPU with AVX2/FMA (net5+ only): the horizontal sum of its
    /// vector kernel passes a <c>shufps</c> mask to <c>shufpd</c>, so within each block of 16 elements the odd-index
    /// squares are dropped and the even-index ones counted twice. 1..16 gives 1360 for 1496, and a tensor whose only
    /// non-zero elements sit at odd indices gives 0. The error is deterministic and data-dependent, between -100% and
    /// +100%. float, Half, decimal, .NET Framework and CPUs without AVX2 are not affected.
    /// </para>
    /// <para>
    /// For double this takes the dot product of the tensor's span with itself, which is correct, vectorized and
    /// allocates nothing (a non-contiguous view is copied first, as the engine op does). Every other element type goes
    /// to the engine op unchanged: it is correct for them, runs on the device for float, and accumulates Half in
    /// float32 where squaring element by element would round, or overflow, each square in Half.
    /// </para>
    /// <para>
    /// Fixed upstream in AiDotNet.Tensors #1733, which no released package contains yet. Once the pinned package has
    /// it, the double branch can go. <c>TensorSumOfSquaresDoubleTests</c> pins the defect and fails when it is gone.
    /// </para>
    /// </remarks>
    internal static T SumOfSquares<T>(IEngine engine, Tensor<T> tensor)
    {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));

        if (typeof(T) != typeof(double) || tensor is SparseTensor<T>)
        {
            return engine.TensorSumOfSquares(tensor);
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        if (tensor.Length == 0)
        {
            return numOps.Zero;
        }

        var values = (tensor.IsContiguous ? tensor : tensor.Contiguous()).AsSpan();
        return numOps.Dot(values, values);
    }
}
