using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Helpers;

/// <summary>
/// Scalar tensor reductions with widened accumulation when an intermediate exceeds the input type's range.
/// </summary>
internal static class TensorReductionHelper
{
    /// <summary>
    /// The sum of the squares of every element of <paramref name="tensor"/>, as <see cref="IEngine.TensorSumOfSquares{T}"/>
    /// is documented to return it.
    /// </summary>
    internal static T SumOfSquares<T>(IEngine engine, Tensor<T> tensor)
    {
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));

        return engine.TensorSumOfSquares(tensor);
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
    /// elements are widened to double and summed there, so no square or partial sum is rounded to
    /// <typeparamref name="T"/>: a rescaled sum taken in <typeparamref name="T"/> could overflow at one end and, in
    /// Half, underflow the small squares at the other (2^20 values with two near 256 rounded each scaled square to a
    /// multiple of 2^-24). An infinite element keeps the infinite sum and a NaN element the NaN one.
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

        return SumOfSquares(engine, Widen(tensor));
    }

    /// <summary>
    /// <paramref name="tensor"/> as a dense <c>Tensor&lt;double&gt;</c>: every element of every supported numeric type
    /// is exact in double, so the overflow fallbacks sum the true values.
    /// </summary>
    private static Tensor<double> Widen<T>(Tensor<T> tensor)
    {
        var source = tensor is SparseTensor<T> sparse ? sparse.ToDense() : tensor;
        return (source.IsContiguous ? source : source.Contiguous()).Cast<double>();
    }

    /// <summary>
    /// The sum of the absolute values of every element of <paramref name="tensor"/>, returned as a double that stays
    /// finite whenever the true sum is, even where it is past the range of <typeparamref name="T"/>.
    /// </summary>
    /// <remarks>
    /// The direct engine sum is used when it is finite. When it is not, the values are widened to double and summed
    /// there. An infinite element keeps the infinite sum and NaN stays NaN.
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

        return engine.TensorSum(engine.TensorAbs(Widen(source)));
    }

    /// <summary>
    /// The sum of every element of <paramref name="tensor"/>, returned as a double that stays finite whenever the true
    /// sum is, even where it is past the range of <typeparamref name="T"/>.
    /// </summary>
    /// <remarks>
    /// The direct engine sum is used when it is finite. When it is not, the values are widened to double and summed
    /// there. An infinite element keeps the infinite (or NaN) sum and NaN stays NaN.
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

        return engine.TensorSum(Widen(source));
    }

    /// <summary>
    /// The mean and the population variance of every element of <paramref name="tensor"/>, in double, finite
    /// whenever they are, even where the sums behind them are past the range of <typeparamref name="T"/>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// In Half, 40 values near 200 with a spread near 150 have a sum of squared deviations near 9e5: the variance
    /// taken in <typeparamref name="T"/> was infinite although the standard deviation (150) fits. Here the values are
    /// widened to double and both statistics taken there, so neither a sum nor the centring is rounded to
    /// <typeparamref name="T"/>: centring Half 2048 and 2050 at their mean rounded to Half (2048) gave a standard
    /// deviation of 1.41 for 1.
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

        var values = Widen(tensor);
        double mean = engine.TensorSum(values) / length;
        double sumSquares = SumOfSquares(engine, engine.TensorSubtractScalar(values, mean));
        return (mean, sumSquares / length);
    }

    /// <summary>
    /// The dot product of two tensors of the same length, in double, finite whenever it is, even where the sum is past
    /// the range of <typeparamref name="T"/>.
    /// </summary>
    /// <remarks>
    /// The direct engine sum of the products is used when it is finite. When it is not, both tensors are widened to
    /// double and the products summed there.
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

        return engine.TensorSum(engine.TensorMultiply(Widen(left), Widen(right)));
    }

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
    /// Each slice is first centered on its midpoint s = max / 2 + min / 2, and x - s is divided by
    /// c = (m + 1) * sqrt(count) with m = max |x - s|; both are detached constants. Every unit value is then at most
    /// 1 / sqrt(count), every partial sum of them at most sqrt(count) and the sum of squared deviations at most 4. The
    /// mean is s + c * mean(u), the standard deviation (m + 1) * sqrt(count / divisor * sum (u - mean(u))^2 +
    /// epsilon / (m + 1) / (m + 1)) (divided twice, as (m + 1)^2 overflows Half), and the standardized value sqrt(count) * (u - mean(u)) / that same root. These are
    /// the direct formulas with the constant scale factored out, so the values and, on a tape, the gradients are those
    /// of the direct form.
    /// </para>
    /// <para>
    /// A slice with no spread standardizes to 0, as its deviations are exactly 0 (its midpoint is its value), and its
    /// standard deviation is sqrt(epsilon), as m = 0 keeps epsilon unscaled: the values of the direct form, also in Half.
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

        Tensor<T> shift;
        using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
        {
            // The slice's midpoint max / 2 + min / 2, a detached constant. Centering on it makes a constant slice's
            // deviations exactly 0 (the midpoint of equal values is that value), and halves the largest |x| the
            // division below sees. Halving each bound first keeps the sum finite.
            var half = numOps.FromDouble(0.5);
            var high = engine.ReduceMax(x, axes, true);
            var low = engine.TensorNegate(engine.ReduceMax(engine.TensorNegate(x), axes, true));
            shift = engine.TensorAdd(engine.TensorMultiplyScalar(high, half), engine.TensorMultiplyScalar(low, half));
        }

        var centered = engine.TensorSubtract(x, engine.TensorBroadcastTo(shift, shape));
        Tensor<T> magnitude;
        using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
        {
            // A constant: the standardized values do not depend on it, so it carries no gradient.
            magnitude = engine.TensorAddScalar(engine.ReduceMax(engine.TensorAbs(centered), axes, true), numOps.One);
        }

        var unit = engine.TensorDivideScalar(engine.TensorDivide(centered, engine.TensorBroadcastTo(magnitude, shape)), root);
        var unitMean = engine.TensorDivideScalar(engine.TensorDivideScalar(engine.ReduceSum(unit, axes, true), root), root);
        var deviation = engine.TensorSubtract(unit, engine.TensorBroadcastTo(unitMean, shape));
        var sumSquares = engine.ReduceSum(engine.TensorMultiply(deviation, deviation), axes, true);
        var scaledVariance = engine.TensorMultiplyScalar(sumSquares, numOps.FromDouble(count / divisor));
        // epsilon / magnitude, then / magnitude again: magnitude^2 itself overflows Half once max |x - shift| >= 255.
        // A constant slice has magnitude 1, so its epsilon floor is kept exactly.
        var scaledEpsilon = engine.TensorDivide(
            engine.TensorDivide(engine.TensorFullLike(magnitude, numOps.FromDouble(epsilon)), magnitude), magnitude);
        var scaledStd = engine.TensorSqrt(engine.TensorAdd(scaledVariance, scaledEpsilon));
        var safeStd = engine.TensorWhere(
            engine.TensorEquals(scaledStd, numOps.Zero), engine.TensorFullLike(scaledStd, numOps.One), scaledStd);

        var standardized = engine.TensorMultiplyScalar(
            engine.TensorDivide(deviation, engine.TensorBroadcastTo(safeStd, shape)), root);
        var mean = engine.TensorAdd(engine.TensorMultiplyScalar(engine.TensorMultiply(unitMean, magnitude), root), shift);
        var std = engine.TensorMultiply(scaledStd, magnitude);
        return (standardized, mean, std);
    }
}
