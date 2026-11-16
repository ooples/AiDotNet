namespace AiDotNet.WindowFunctions;

/// <summary>
/// Shared engine-op building blocks for the symmetric window functions.
/// </summary>
/// <remarks>
/// Every window is built as whole-tensor engine operations over the sample index n = 0..M-1, generic in T.
/// Length 0 and length 1 follow scipy.signal.windows (an empty window and the single value 1). The sample
/// geometry (n / (M-1), 2n / (M-1) - 1, n - (M-1)/2) is formed in double and converted to T once: Half holds
/// integers exactly only up to 2048, so an index formed in Half merges neighbouring samples of a longer window.
/// </remarks>
internal static class WindowFunctionHelper
{
    /// <summary>
    /// Handles the degenerate lengths: returns true with the empty window for 0 and [1] for 1.
    /// A negative length throws from the vector constructor, as before.
    /// </summary>
    internal static bool TryCreateDegenerate<T>(int windowSize, out Vector<T> window)
    {
        if (windowSize > 1)
        {
            window = new Vector<T>(0);
            return false;
        }

        window = new Vector<T>(windowSize);
        if (windowSize == 1)
        {
            window[0] = MathHelper.GetNumericOperations<T>().One;
        }

        return true;
    }

    /// <summary>The sample indices n = 0, 1, ..., M-1.</summary>
    internal static Tensor<T> SampleIndices<T>(int windowSize)
        => AiDotNetEngine.Current.TensorArange<T>(0, windowSize, 1);

    /// <summary>The phase n / (M-1), running from 0 to 1.</summary>
    internal static Tensor<T> UnitPhase<T>(int windowSize)
        => TensorCopyHelper.CastLogical<double, T>(UnitPhaseInDouble(windowSize));

    private static Tensor<double> UnitPhaseInDouble(int windowSize)
        => AiDotNetEngine.Current.TensorDivideScalar(SampleIndices<double>(windowSize), windowSize - 1.0);

    /// <summary>The centred coordinate 2n / (M-1) - 1, running from -1 to 1.</summary>
    internal static Tensor<T> SymmetricCoordinate<T>(int windowSize)
    {
        var engine = AiDotNetEngine.Current;
        return TensorCopyHelper.CastLogical<double, T>(
            engine.TensorSubtractScalar(engine.TensorMultiplyScalar(UnitPhaseInDouble(windowSize), 2.0), 1.0));
    }

    /// <summary>
    /// The generalized cosine window a0 - a1 cos(x) + a2 cos(2x) - a3 cos(3x) + a4 cos(4x), x = 2 pi n / (M-1)
    /// (scipy.signal.windows.general_cosine with alternating signs).
    /// </summary>
    internal static Vector<T> GeneralCosine<T>(int windowSize, double a0, double a1, double a2 = 0.0, double a3 = 0.0, double a4 = 0.0)
    {
        if (TryCreateDegenerate(windowSize, out Vector<T> degenerate))
        {
            return degenerate;
        }

        // The whole sum is formed in double and converted once: the window is a fixed function of n, and a
        // narrow T would round both the phase and every cosine term.
        var engine = AiDotNetEngine.Current;
        Tensor<double> x = engine.TensorMultiplyScalar(UnitPhaseInDouble(windowSize), 2.0 * Math.PI);

        Tensor<double> result = engine.ScalarMinusTensor(a0, CosineTerm(x, 1.0, a1));
        if (a2 != 0.0)
        {
            result = engine.TensorAdd(result, CosineTerm(x, 2.0, a2));
        }

        if (a3 != 0.0)
        {
            result = engine.TensorSubtract(result, CosineTerm(x, 3.0, a3));
        }

        if (a4 != 0.0)
        {
            result = engine.TensorAdd(result, CosineTerm(x, 4.0, a4));
        }

        return TensorCopyHelper.CastLogical<double, T>(result).ToVector();
    }

    private static Tensor<double> CosineTerm(Tensor<double> x, double harmonic, double coefficient)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorMultiplyScalar(engine.TensorCos(engine.TensorMultiplyScalar(x, harmonic)), coefficient);
    }
}
