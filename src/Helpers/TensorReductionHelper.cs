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
    /// allocates nothing (a non-contiguous view is copied first, as the engine op does, and a sparse tensor is
    /// densified, since the engine op runs the same faulty kernel over it). Every other element type goes
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
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));

        if (typeof(T) != typeof(double))
        {
            return engine.TensorSumOfSquares(tensor);
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        if (tensor.Length == 0)
        {
            return numOps.Zero;
        }

        // A sparse double tensor goes through the same defective kernel, and its span is not readable: densify it
        // (ToDense accumulates duplicate coordinates, so the sum is the dense tensor's).
        var dense = tensor is SparseTensor<T> sparse ? sparse.ToDense() : tensor;
        var values = (dense.IsContiguous ? dense : dense.Contiguous()).AsSpan();
        return numOps.Dot(values, values);
    }

    /// <summary>
    /// The sum of the squares of every element of <paramref name="tensor"/>, returned as a double that stays finite
    /// whenever the true sum is finite in double, even where it is past the range of <typeparamref name="T"/>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Use this wherever the sum feeds a standard deviation, an RMS, a norm, a mean or a ratio rather than being the
    /// result itself. <see cref="SumOfSquares{T}"/> returns <typeparamref name="T"/>, so a Half sum past 65504 is
    /// infinite even though the square root or the mean it feeds is representable: 40 Half returns of 200 have a sum
    /// of squares of 1.6e6 and a standard deviation of 200, and the infinite sum made that standard deviation
    /// infinite and every ratio over it 0.
    /// </para>
    /// <para>
    /// The direct sum is used when it is finite (every double and float input in practice). When it is not, the
    /// elements are divided by m = max |x| and by sqrt(n), so every square and every partial sum is at most 1 in
    /// <typeparamref name="T"/>, and the scale is restored in double: sum x^2 = m^2 * n * sum (x / m / sqrt(n))^2.
    /// An infinite element keeps the infinite sum and a NaN element the NaN one.
    /// </para>
    /// </remarks>
    internal static double SumOfSquaresAsDouble<T>(IEngine engine, Tensor<T> tensor)
    {
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));

        int length = tensor.Length;
        if (length == 0)
        {
            return 0.0;
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        T direct = SumOfSquares(engine, tensor);
        if (!numOps.IsInfinity(direct) || numOps.IsNaN(direct))
        {
            return numOps.ToDouble(direct);
        }

        var source = tensor is SparseTensor<T> sparse ? sparse.ToDense() : tensor;
        T maxAbs = engine.TensorMaxValue(engine.TensorAbs(source));
        if (numOps.IsInfinity(maxAbs) || numOps.IsNaN(maxAbs) || !numOps.GreaterThan(maxAbs, numOps.Zero))
        {
            // An infinite element makes the true sum infinite; NaN stays NaN; a zero maximum cannot overflow.
            return numOps.ToDouble(direct);
        }

        var normalized = engine.TensorDivideScalar(
            engine.TensorDivideScalar(source, maxAbs), numOps.FromDouble(Math.Sqrt(length)));
        double scale = numOps.ToDouble(maxAbs);
        return scale * scale * length * numOps.ToDouble(SumOfSquares(engine, normalized));
    }

    /// <summary>
    /// The sum of the absolute values of every element of <paramref name="tensor"/>, returned as a double that stays
    /// finite whenever the true sum is, even where it is past the range of <typeparamref name="T"/>.
    /// </summary>
    /// <remarks>
    /// The direct engine sum is used when it is finite. When it is not, the absolute values are divided by
    /// m = max |x| and by sqrt(n), so every partial sum is at most sqrt(n) (65504 only past 4e9 elements),
    /// and the scale is restored in double:
    /// sum |x| = m * sqrt(n) * sum (|x| / m / sqrt(n)). An infinite element keeps the infinite sum and NaN stays NaN.
    /// </remarks>
    internal static double SumAbsoluteAsDouble<T>(IEngine engine, Tensor<T> tensor)
    {
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));

        int length = tensor.Length;
        if (length == 0)
        {
            return 0.0;
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var source = tensor is SparseTensor<T> sparse ? sparse.ToDense() : tensor;
        var absolute = engine.TensorAbs(source);
        T direct = engine.TensorSum(absolute);
        if (!numOps.IsInfinity(direct) || numOps.IsNaN(direct))
        {
            return numOps.ToDouble(direct);
        }

        T maxAbs = engine.TensorMaxValue(absolute);
        if (numOps.IsInfinity(maxAbs) || numOps.IsNaN(maxAbs) || !numOps.GreaterThan(maxAbs, numOps.Zero))
        {
            return numOps.ToDouble(direct);
        }

        double rootLength = Math.Sqrt(length);
        var normalized = engine.TensorDivideScalar(engine.TensorDivideScalar(absolute, maxAbs), numOps.FromDouble(rootLength));
        return numOps.ToDouble(maxAbs) * rootLength * numOps.ToDouble(engine.TensorSum(normalized));
    }
}
