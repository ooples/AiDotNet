using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Augmentation;

/// <summary>
/// Draws random tensors of <typeparamref name="T"/> from an augmentation context's generator,
/// consuming it in exactly the order an element-by-element loop over the same shape would.
/// </summary>
/// <typeparam name="T">The numeric type of the tensors.</typeparam>
internal static class AugmentationRandom<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// The smallest positive value of <typeparamref name="T"/> used to floor the Box-Muller u1 draw.
    /// <c>double.Epsilon</c> converts to zero for narrower types (float, Half, decimal), which would let
    /// a zero draw reach log(0), so the first candidate that survives conversion is used.
    /// </summary>
    private static readonly T MinPositive = SmallestPositive();

    private static T SmallestPositive()
    {
        // double subnormal min, float subnormal min, Half subnormal min (2^-24), decimal min scale.
        T value = NumOps.FromDouble(double.Epsilon);
        if (IsPositive(value)) return value;
        value = NumOps.FromDouble(float.Epsilon);
        if (IsPositive(value)) return value;
        value = NumOps.FromDouble(5.9604644775390625e-8);
        if (IsPositive(value)) return value;
        value = NumOps.FromDouble(1e-28);
        return IsPositive(value) ? value : NumOps.One;
    }

    private static bool IsPositive(T value) => NumOps.GreaterThan(value, NumOps.Zero);

    /// <summary>
    /// One <see cref="Random.NextDouble"/> per element in row-major order.
    /// </summary>
    internal static Tensor<T> Unit(AugmentationContext<T> context, int[] shape)
    {
        return Tensor<T>.CreateRandom(context.Random, shape);
    }

    /// <summary>
    /// <see cref="AugmentationContext{T}.GetRandomDouble"/>(min, max) per element in row-major order.
    /// </summary>
    internal static Tensor<T> Uniform(AugmentationContext<T> context, int[] shape, double min, double max)
    {
        var engine = AiDotNetEngine.Current;
        var unit = Unit(context, shape);
        return engine.TensorAddScalar(engine.TensorMultiplyScalar(unit, NumOps.FromDouble(max - min)), NumOps.FromDouble(min));
    }

    /// <summary>
    /// <see cref="AugmentationContext{T}.SampleGaussian"/>(0, 1) per element in row-major order
    /// (Box-Muller with two uniform draws per element, as the context does).
    /// </summary>
    internal static Tensor<T> StandardNormal(AugmentationContext<T> context, int[] shape)
    {
        var engine = AiDotNetEngine.Current;
        int count = 1;
        foreach (int dim in shape)
        {
            count *= dim;
        }

        var pairs = Unit(context, [count, 2]);

        // u1 is floored at the smallest positive value of T to avoid log(0)
        var u1 = engine.TensorClamp(engine.TensorSliceAxis(pairs, 1, 0), MinPositive, NumOps.One);
        var u2 = engine.TensorSliceAxis(pairs, 1, 1);
        var radius = engine.TensorSqrt(engine.TensorMultiplyScalar(engine.TensorLog(u1), NumOps.FromDouble(-2.0)));
        var angle = engine.TensorSin(engine.TensorMultiplyScalar(u2, NumOps.FromDouble(2.0 * Math.PI)));
        return engine.Reshape(engine.TensorMultiply(radius, angle), shape);
    }

    /// <summary>
    /// <see cref="AugmentationContext{T}.SampleGaussian"/>(mean, stdDev) per element in row-major order.
    /// </summary>
    internal static Tensor<T> Gaussian(AugmentationContext<T> context, int[] shape, double mean, double stdDev)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorAddScalar(
            engine.TensorMultiplyScalar(StandardNormal(context, shape), NumOps.FromDouble(stdDev)), NumOps.FromDouble(mean));
    }
}
