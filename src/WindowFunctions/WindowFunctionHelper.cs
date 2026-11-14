namespace AiDotNet.WindowFunctions;

/// <summary>
/// Shared engine-op building blocks for the symmetric window functions.
/// </summary>
/// <remarks>
/// Every window is built as whole-tensor engine operations over the sample index n = 0..M-1, generic in T.
/// Length 0 and length 1 follow scipy.signal.windows (an empty window and the single value 1).
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
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        return AiDotNetEngine.Current.TensorDivideScalar(SampleIndices<T>(windowSize), numOps.FromDouble(windowSize - 1));
    }

    /// <summary>The centred coordinate 2n / (M-1) - 1, running from -1 to 1.</summary>
    internal static Tensor<T> SymmetricCoordinate<T>(int windowSize)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        return engine.TensorSubtractScalar(
            engine.TensorMultiplyScalar(UnitPhase<T>(windowSize), numOps.FromDouble(2.0)), numOps.One);
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

        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        Tensor<T> x = engine.TensorMultiplyScalar(UnitPhase<T>(windowSize), numOps.FromDouble(2.0 * Math.PI));

        Tensor<T> result = engine.ScalarMinusTensor(numOps.FromDouble(a0), CosineTerm(x, 1.0, a1));
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

        return result.ToVector();
    }

    private static Tensor<T> CosineTerm<T>(Tensor<T> x, double harmonic, double coefficient)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        return engine.TensorMultiplyScalar(
            engine.TensorCos(engine.TensorMultiplyScalar(x, numOps.FromDouble(harmonic))), numOps.FromDouble(coefficient));
    }
}
