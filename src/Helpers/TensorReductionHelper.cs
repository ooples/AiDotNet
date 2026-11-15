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

    /// <summary>
    /// The sum of every element of <paramref name="tensor"/>, returned as a double that stays finite whenever the true
    /// sum is, even where it is past the range of <typeparamref name="T"/>.
    /// </summary>
    /// <remarks>
    /// The direct engine sum is used when it is finite. When it is not, the elements are divided by m = max |x| and
    /// by sqrt(n), so every partial sum is at most sqrt(n), and the scale is restored in double. An infinite element
    /// keeps the infinite (or NaN) sum and NaN stays NaN.
    /// </remarks>
    internal static double SumAsDouble<T>(IEngine engine, Tensor<T> tensor)
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
        T direct = engine.TensorSum(source);
        if (!numOps.IsInfinity(direct) || numOps.IsNaN(direct))
        {
            return numOps.ToDouble(direct);
        }

        T maxAbs = engine.TensorMaxValue(engine.TensorAbs(source));
        if (numOps.IsInfinity(maxAbs) || numOps.IsNaN(maxAbs) || !numOps.GreaterThan(maxAbs, numOps.Zero))
        {
            return numOps.ToDouble(direct);
        }

        double rootLength = Math.Sqrt(length);
        var normalized = engine.TensorDivideScalar(engine.TensorDivideScalar(source, maxAbs), numOps.FromDouble(rootLength));
        return numOps.ToDouble(maxAbs) * rootLength * numOps.ToDouble(engine.TensorSum(normalized));
    }

    /// <summary>
    /// The mean and the population variance of every element of <paramref name="tensor"/>, in double, finite
    /// whenever they are, even where the sums behind them are past the range of <typeparamref name="T"/>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// In Half, 40 values near 200 with a spread near 150 have a sum of squared deviations near 9e5: the variance
    /// taken in <typeparamref name="T"/> was infinite although the standard deviation (150) fits. Here the mean is
    /// <see cref="SumAsDouble{T}"/> / n and the variance <see cref="SumOfSquaresAsDouble{T}"/> of the centred values
    /// / n. Should a centred value itself overflow (a spread past the range of <typeparamref name="T"/>), the values
    /// are centred in units of max |x| instead.
    /// </para>
    /// <para>NaN for both when the tensor is empty.</para>
    /// </remarks>
    internal static (double Mean, double Variance) MeanAndPopulationVarianceAsDouble<T>(IEngine engine, Tensor<T> tensor)
    {
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));

        int length = tensor.Length;
        if (length == 0)
        {
            return (double.NaN, double.NaN);
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var source = tensor is SparseTensor<T> sparse ? sparse.ToDense() : tensor;
        double mean = SumAsDouble(engine, source) / length;
        double sumSquares = SumOfSquaresAsDouble(engine, engine.TensorSubtractScalar(source, numOps.FromDouble(mean)));
        if (double.IsInfinity(sumSquares) && !double.IsInfinity(mean))
        {
            T maxAbs = engine.TensorMaxValue(engine.TensorAbs(source));
            if (!numOps.IsInfinity(maxAbs) && !numOps.IsNaN(maxAbs) && numOps.GreaterThan(maxAbs, numOps.Zero))
            {
                double scale = numOps.ToDouble(maxAbs);
                var scaled = engine.TensorDivideScalar(source, maxAbs);
                sumSquares = scale * scale
                    * SumOfSquaresAsDouble(engine, engine.TensorSubtractScalar(scaled, numOps.FromDouble(mean / scale)));
            }
        }

        return (mean, sumSquares / length);
    }

    /// <summary>
    /// The dot product of two tensors of the same length, in double, finite whenever it is, even where the sum is past
    /// the range of <typeparamref name="T"/>.
    /// </summary>
    /// <remarks>
    /// The direct engine sum of the products is used when it is finite. When it is not, each tensor is divided by its
    /// max |x| and both by sqrt(n), so every product and partial sum is at most 1, and the scale is restored in double.
    /// </remarks>
    internal static double DotAsDouble<T>(IEngine engine, Tensor<T> left, Tensor<T> right)
    {
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        if (left is null) throw new ArgumentNullException(nameof(left));
        if (right is null) throw new ArgumentNullException(nameof(right));
        if (left.Length != right.Length)
            throw new ArgumentException($"Lengths differ ({left.Length} and {right.Length}).", nameof(right));

        int length = left.Length;
        if (length == 0)
        {
            return 0.0;
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        T direct = engine.TensorSum(engine.TensorMultiply(left, right));
        if (!numOps.IsInfinity(direct) || numOps.IsNaN(direct))
        {
            return numOps.ToDouble(direct);
        }

        T leftMax = engine.TensorMaxValue(engine.TensorAbs(left));
        T rightMax = engine.TensorMaxValue(engine.TensorAbs(right));
        if (!IsPositiveFinite(numOps, leftMax) || !IsPositiveFinite(numOps, rightMax))
        {
            return numOps.ToDouble(direct);
        }

        T root = numOps.FromDouble(Math.Sqrt(length));
        var leftUnit = engine.TensorDivideScalar(engine.TensorDivideScalar(left, leftMax), root);
        var rightUnit = engine.TensorDivideScalar(engine.TensorDivideScalar(right, rightMax), root);
        return numOps.ToDouble(leftMax) * numOps.ToDouble(rightMax) * length
            * numOps.ToDouble(engine.TensorSum(engine.TensorMultiply(leftUnit, rightUnit)));
    }

    private static bool IsPositiveFinite<T>(INumericOperations<T> numOps, T value)
        => !numOps.IsInfinity(value) && !numOps.IsNaN(value) && numOps.GreaterThan(value, numOps.Zero);

    /// <summary>Whether every element of <paramref name="tensor"/> is finite (neither infinite nor NaN).</summary>
    internal static bool AllFinite<T>(IEngine engine, Tensor<T> tensor)
    {
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));
        var numOps = MathHelper.GetNumericOperations<T>();
        return engine.TensorCountNonzero(engine.TensorEquals(engine.TensorSubtract(tensor, tensor), numOps.Zero)) == tensor.Length;
    }

    /// <summary>
    /// Standardizes <paramref name="x"/> along <paramref name="axes"/> in units of each slice's max |x|, so that no
    /// sum behind the statistics can pass the range of <typeparamref name="T"/>: the overflow-safe form of
    /// (x - mean) / sqrt(variance + epsilon).
    /// </summary>
    /// <param name="engine">The engine to run on. Every op is an engine op, so the result stays on any active tape.</param>
    /// <param name="x">The values.</param>
    /// <param name="axes">The axes the statistics are taken over.</param>
    /// <param name="epsilon">Added to the variance before the square root.</param>
    /// <param name="divisor">What the sum of squared deviations is divided by: the count for the population variance,
    /// the count less one for the sample variance.</param>
    /// <returns>The standardized values (the shape of <paramref name="x"/>), and the mean and the standard deviation
    /// (the shape of <paramref name="x"/> with each of <paramref name="axes"/> kept at size 1).</returns>
    /// <remarks>
    /// <para>
    /// Callers keep their direct computation and fall back to this one when its statistics are not finite while the
    /// input is: in Half 40 values near 200 have a sum of squared deviations near 9e5, so the variance was infinite,
    /// the standard deviation infinite and every standardized value 0, although the standard deviation (150) fits.
    /// </para>
    /// <para>
    /// Each slice is divided by c = (max |x| + 1) * sqrt(count), a detached constant, so every unit value is at most
    /// 1 / sqrt(count), every partial sum of them at most sqrt(count) and the sum of squared deviations at most 4. The
    /// mean is c * mean(u), the standard deviation (max |x| + 1) * sqrt(count / divisor * sum (u - mean(u))^2 +
    /// epsilon / (max |x| + 1)^2), and the standardized value sqrt(count) * (u - mean(u)) / that same root. These are
    /// the direct formulas with the constant scale factored out, so the values and, on a tape, the gradients are those
    /// of the direct form.
    /// </para>
    /// <para>
    /// A slice with no spread and no epsilon (a zero standard deviation) standardizes to 0, as its deviations are 0.
    /// </para>
    /// </remarks>
    internal static (Tensor<T> Standardized, Tensor<T> Mean, Tensor<T> Std) StandardizeRescaled<T>(
        IEngine engine, Tensor<T> x, int[] axes, double epsilon, double divisor)
    {
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        if (x is null) throw new ArgumentNullException(nameof(x));
        if (axes is null) throw new ArgumentNullException(nameof(axes));

        var numOps = MathHelper.GetNumericOperations<T>();
        var shape = (int[])x._shape.Clone();
        int count = 1;
        foreach (int axis in axes) count *= shape[axis];
        T root = numOps.FromDouble(Math.Sqrt(count));

        Tensor<T> magnitude;
        using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
        {
            // A constant: the standardized values do not depend on it, so it carries no gradient.
            magnitude = engine.TensorAddScalar(engine.ReduceMax(engine.TensorAbs(x), axes, true), numOps.One);
        }

        var unit = engine.TensorDivideScalar(engine.TensorDivide(x, engine.TensorBroadcastTo(magnitude, shape)), root);
        var unitMean = engine.TensorDivideScalar(engine.TensorDivideScalar(engine.ReduceSum(unit, axes, true), root), root);
        var deviation = engine.TensorSubtract(unit, engine.TensorBroadcastTo(unitMean, shape));
        var sumSquares = engine.ReduceSum(engine.TensorMultiply(deviation, deviation), axes, true);
        var scaledVariance = engine.TensorMultiplyScalar(sumSquares, numOps.FromDouble(count / divisor));
        var scaledEpsilon = engine.TensorDivide(
            engine.TensorFullLike(magnitude, numOps.FromDouble(epsilon)), engine.TensorMultiply(magnitude, magnitude));
        var scaledStd = engine.TensorSqrt(engine.TensorAdd(scaledVariance, scaledEpsilon));
        var safeStd = engine.TensorWhere(
            engine.TensorEquals(scaledStd, numOps.Zero), engine.TensorFullLike(scaledStd, numOps.One), scaledStd);

        var standardized = engine.TensorMultiplyScalar(
            engine.TensorDivide(deviation, engine.TensorBroadcastTo(safeStd, shape)), root);
        var mean = engine.TensorMultiplyScalar(engine.TensorMultiply(unitMean, magnitude), root);
        var std = engine.TensorMultiply(scaledStd, magnitude);
        return (standardized, mean, std);
    }
}
