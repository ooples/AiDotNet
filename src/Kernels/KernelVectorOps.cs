namespace AiDotNet.Kernels;

/// <summary>
/// Whole-vector engine helpers shared by kernels whose per-dimension terms need a guard.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
internal static class KernelVectorOps<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Computes sum_i (x1_i - x2_i)^2 / (x1_i + x2_i), skipping dimensions where x1_i + x2_i is zero.
    /// </summary>
    public static T SafeChiSquaredSum(Vector<T> x1, Vector<T> x2)
    {
        var engine = AiDotNetEngine.Current;
        var diff = engine.Subtract(x1, x2).AsTensor();
        var numerator = engine.TensorMultiply(diff, diff);
        var denominator = engine.Add(x1, x2).AsTensor();
        var isZero = engine.TensorEquals(denominator, NumOps.Zero);
        var ones = Tensor<T>.CreateDefault(new[] { x1.Length }, NumOps.One);
        var zeros = Tensor<T>.CreateDefault(new[] { x1.Length }, NumOps.Zero);
        var safeDenominator = engine.TensorWhere(isZero, ones, denominator);
        var terms = engine.TensorWhere(isZero, zeros, engine.TensorDivide(numerator, safeDenominator));
        return engine.TensorSum(terms);
    }
}
