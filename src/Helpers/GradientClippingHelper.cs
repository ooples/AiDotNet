namespace AiDotNet.Helpers;

/// <summary>
/// Provides gradient clipping utilities to prevent exploding gradients during training.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> During neural network training, gradients tell us how to adjust
/// weights. Sometimes gradients become extremely large ("exploding gradients"), which can
/// destabilize training. Gradient clipping limits the magnitude of gradients to keep
/// training stable.
///
/// There are two main approaches:
/// - **Clip by Value**: Limits each gradient element to a range (e.g., -1 to 1)
/// - **Clip by Norm**: Scales the entire gradient vector if its norm exceeds a threshold
///
/// The "by norm" approach is generally preferred as it preserves gradient direction.
/// </para>
/// </remarks>
public static class GradientClippingHelper
{
    /// <summary>
    /// Default maximum gradient norm for clipping.
    /// </summary>
    public const double DefaultMaxNorm = 1.0;

    /// <summary>
    /// Default maximum gradient value for value clipping.
    /// </summary>
    public const double DefaultMaxValue = 1.0;

    /// <summary>
    /// Clips gradient values to a specified range [-maxValue, maxValue].
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="gradients">The gradient vector to clip.</param>
    /// <param name="maxValue">Maximum absolute value for any gradient element.</param>
    /// <returns>A new vector with clipped gradients.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the simplest form of gradient clipping.
    /// Each gradient value is independently limited to the range [-maxValue, maxValue].
    /// For example, with maxValue=1.0, a gradient of 5.0 becomes 1.0, and -3.0 becomes -1.0.
    /// </para>
    /// </remarks>
    public static Vector<T>? ClipByValue<T>(Vector<T>? gradients, double maxValue = DefaultMaxValue)
    {
        if (gradients == null) return null;

        var numOps = MathHelper.GetNumericOperations<T>();
        T maxVal = numOps.FromDouble(maxValue);
        T minVal = numOps.FromDouble(-maxValue);

        var clipped = new Vector<T>(gradients.Length);
        for (int i = 0; i < gradients.Length; i++)
        {
            clipped[i] = MathHelper.Clamp(gradients[i], minVal, maxVal);
        }

        return clipped;
    }

    /// <summary>
    /// Clips gradient values to a specified range [-maxValue, maxValue] in place.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="gradients">The gradient vector to clip (modified in place).</param>
    /// <param name="maxValue">Maximum absolute value for any gradient element.</param>
    public static void ClipByValueInPlace<T>(Vector<T> gradients, double maxValue = DefaultMaxValue)
    {
        if (gradients == null) return;

        var numOps = MathHelper.GetNumericOperations<T>();
        T maxVal = numOps.FromDouble(maxValue);
        T minVal = numOps.FromDouble(-maxValue);

        for (int i = 0; i < gradients.Length; i++)
        {
            gradients[i] = MathHelper.Clamp(gradients[i], minVal, maxVal);
        }
    }

    /// <summary>
    /// Clips gradients by their L2 norm (global norm clipping).
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="gradients">The gradient vector to clip.</param>
    /// <param name="maxNorm">Maximum L2 norm for the gradient vector.</param>
    /// <returns>A new vector with clipped gradients.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the preferred gradient clipping method.
    /// Instead of clipping each value independently, we look at the total "length"
    /// (norm) of the gradient vector. If it exceeds maxNorm, we scale the entire
    /// vector down proportionally.
    ///
    /// This preserves the direction of the gradient while limiting its magnitude,
    /// which typically leads to better training behavior.
    ///
    /// Formula: if ||g|| > maxNorm, then g = g * (maxNorm / ||g||)
    /// </para>
    /// </remarks>
    public static Vector<T>? ClipByNorm<T>(Vector<T>? gradients, double maxNorm = DefaultMaxNorm)
    {
        if (gradients == null) return null;

        var numOps = MathHelper.GetNumericOperations<T>();

        // Compute L2 norm
        T sumSquares = numOps.Zero;
        for (int i = 0; i < gradients.Length; i++)
        {
            sumSquares = numOps.Add(sumSquares, numOps.Multiply(gradients[i], gradients[i]));
        }
        T norm = numOps.Sqrt(sumSquares);

        // If norm is below threshold, return unchanged
        T maxNormT = numOps.FromDouble(maxNorm);
        if (!numOps.GreaterThan(norm, maxNormT))
        {
            return gradients.Clone();
        }

        // Scale gradients
        T scale = numOps.Divide(maxNormT, norm);
        var clipped = new Vector<T>(gradients.Length);
        for (int i = 0; i < gradients.Length; i++)
        {
            clipped[i] = numOps.Multiply(gradients[i], scale);
        }

        return clipped;
    }

    /// <summary>
    /// Clips gradients by their L2 norm in place.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="gradients">The gradient vector to clip (modified in place).</param>
    /// <param name="maxNorm">Maximum L2 norm for the gradient vector.</param>
    /// <returns>True if clipping was applied, false otherwise.</returns>
    public static bool ClipByNormInPlace<T>(Vector<T> gradients, double maxNorm = DefaultMaxNorm)
    {
        if (gradients == null) return false;

        var numOps = MathHelper.GetNumericOperations<T>();

        // Compute L2 norm
        T sumSquares = numOps.Zero;
        for (int i = 0; i < gradients.Length; i++)
        {
            sumSquares = numOps.Add(sumSquares, numOps.Multiply(gradients[i], gradients[i]));
        }
        T norm = numOps.Sqrt(sumSquares);

        // If norm is below threshold, no clipping needed
        T maxNormT = numOps.FromDouble(maxNorm);
        if (!numOps.GreaterThan(norm, maxNormT))
        {
            return false;
        }

        // Scale gradients in place
        T scale = numOps.Divide(maxNormT, norm);
        for (int i = 0; i < gradients.Length; i++)
        {
            gradients[i] = numOps.Multiply(gradients[i], scale);
        }

        return true;
    }

    /// <summary>
    /// Clips gradients by global norm across multiple gradient vectors.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="gradientsList">List of gradient vectors to clip together.</param>
    /// <param name="maxNorm">Maximum global L2 norm.</param>
    /// <returns>A list of clipped gradient vectors.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> When training a neural network with multiple layers,
    /// each layer has its own gradients. Global norm clipping computes the norm across
    /// ALL gradients and scales them all together. This ensures consistent clipping
    /// behavior across the entire network.
    /// </para>
    /// </remarks>
    public static List<Vector<T>>? ClipByGlobalNorm<T>(List<Vector<T>>? gradientsList, double maxNorm = DefaultMaxNorm)
    {
        if (gradientsList == null || gradientsList.Count == 0)
            return gradientsList;

        var numOps = MathHelper.GetNumericOperations<T>();

        // Compute global L2 norm
        T globalSumSquares = numOps.Zero;
        foreach (var gradients in gradientsList)
        {
            if (gradients == null) continue;
            for (int i = 0; i < gradients.Length; i++)
            {
                globalSumSquares = numOps.Add(globalSumSquares,
                    numOps.Multiply(gradients[i], gradients[i]));
            }
        }
        T globalNorm = numOps.Sqrt(globalSumSquares);

        // If global norm is below threshold, return clones
        T maxNormT = numOps.FromDouble(maxNorm);
        if (!numOps.GreaterThan(globalNorm, maxNormT))
        {
            return gradientsList.Select(g => g?.Clone()).ToList()!;
        }

        // Scale all gradients
        T scale = numOps.Divide(maxNormT, globalNorm);
        var clippedList = new List<Vector<T>>();
        foreach (var gradients in gradientsList)
        {
            if (gradients == null)
            {
                continue;
            }

            var clipped = new Vector<T>(gradients.Length);
            for (int i = 0; i < gradients.Length; i++)
            {
                clipped[i] = numOps.Multiply(gradients[i], scale);
            }
            clippedList.Add(clipped);
        }

        return clippedList;
    }

    /// <summary>
    /// Clips tensor gradients by their L2 norm.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="gradients">The gradient tensor to clip.</param>
    /// <param name="maxNorm">Maximum L2 norm.</param>
    /// <returns>A new tensor with clipped gradients.</returns>
    public static Tensor<T>? ClipByNorm<T>(Tensor<T>? gradients, double maxNorm = DefaultMaxNorm)
    {
        if (gradients == null) return null;

        var numOps = MathHelper.GetNumericOperations<T>();
        int length = gradients.Length;

        // Compute L2 norm
        T sumSquares = numOps.Zero;
        for (int i = 0; i < length; i++)
        {
            var val = gradients.GetFlatIndexValue(i);
            sumSquares = numOps.Add(sumSquares, numOps.Multiply(val, val));
        }
        T norm = numOps.Sqrt(sumSquares);

        // If norm is below threshold, return clone
        T maxNormT = numOps.FromDouble(maxNorm);
        if (!numOps.GreaterThan(norm, maxNormT))
        {
            return (Tensor<T>)gradients.Clone();
        }

        // Scale gradients
        T scale = numOps.Divide(maxNormT, norm);
        var clipped = new Tensor<T>(gradients.Shape);
        for (int i = 0; i < length; i++)
        {
            clipped.SetFlatIndexValue(i, numOps.Multiply(gradients.GetFlatIndexValue(i), scale));
        }

        return clipped;
    }

    /// <summary>
    /// Computes the L2 norm of a gradient vector.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="gradients">The gradient vector.</param>
    /// <returns>The L2 norm.</returns>
    public static T ComputeNorm<T>(Vector<T> gradients)
    {
        if (gradients == null)
        {
            var numOps = MathHelper.GetNumericOperations<T>();
            return numOps.Zero;
        }

        var ops = MathHelper.GetNumericOperations<T>();
        T sumSquares = ops.Zero;
        for (int i = 0; i < gradients.Length; i++)
        {
            sumSquares = ops.Add(sumSquares, ops.Multiply(gradients[i], gradients[i]));
        }
        return ops.Sqrt(sumSquares);
    }

    /// <summary>
    /// Computes the global L2 norm across multiple gradient vectors.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="gradientsList">List of gradient vectors.</param>
    /// <returns>The global L2 norm.</returns>
    public static T ComputeGlobalNorm<T>(List<Vector<T>> gradientsList)
    {
        var numOps = MathHelper.GetNumericOperations<T>();

        if (gradientsList == null || gradientsList.Count == 0)
            return numOps.Zero;

        T globalSumSquares = numOps.Zero;
        foreach (var gradients in gradientsList)
        {
            if (gradients == null) continue;
            for (int i = 0; i < gradients.Length; i++)
            {
                globalSumSquares = numOps.Add(globalSumSquares,
                    numOps.Multiply(gradients[i], gradients[i]));
            }
        }

        return numOps.Sqrt(globalSumSquares);
    }

    /// <summary>
    /// Applies adaptive gradient clipping based on parameter norm.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="gradients">The gradient vector.</param>
    /// <param name="parameters">The corresponding parameter vector.</param>
    /// <param name="clipRatio">Ratio threshold for clipping (e.g., 0.01 means gradient norm should not exceed 1% of parameter norm).</param>
    /// <returns>Clipped gradients.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Adaptive gradient clipping (AGC) scales the clipping threshold
    /// based on the magnitude of the parameters themselves. This is useful because large parameters
    /// can tolerate larger gradients without destabilizing, while small parameters need tighter
    /// gradient bounds.
    ///
    /// This technique was introduced in the NFNet paper and can help train very deep networks
    /// without batch normalization.
    /// </para>
    /// </remarks>
    public static Vector<T>? ClipAdaptive<T>(Vector<T>? gradients, Vector<T>? parameters, double clipRatio = 0.01)
    {
        if (gradients == null || parameters == null)
            return gradients;

        if (gradients.Length != parameters.Length)
            throw new ArgumentException("Gradients and parameters must have the same length");

        var numOps = MathHelper.GetNumericOperations<T>();

        // Compute parameter norm
        T paramNorm = ComputeNorm(parameters);
        T gradNorm = ComputeNorm(gradients);

        // Compute adaptive threshold
        T clipRatioT = numOps.FromDouble(clipRatio);
        T maxGradNorm = numOps.Multiply(paramNorm, clipRatioT);

        // Ensure minimum threshold
        T minThreshold = numOps.FromDouble(1e-3);
        if (numOps.LessThan(maxGradNorm, minThreshold))
            maxGradNorm = minThreshold;

        // Clip if needed
        if (!numOps.GreaterThan(gradNorm, maxGradNorm))
            return gradients.Clone();

        T scale = numOps.Divide(maxGradNorm, gradNorm);
        var clipped = new Vector<T>(gradients.Length);
        for (int i = 0; i < gradients.Length; i++)
        {
            clipped[i] = numOps.Multiply(gradients[i], scale);
        }

        return clipped;
    }

    /// <summary>
    /// Detects if gradients are exploding (have very large values).
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="gradients">The gradient vector to check.</param>
    /// <param name="threshold">Threshold for considering gradients as exploding.</param>
    /// <returns>True if gradients appear to be exploding.</returns>
    public static bool AreGradientsExploding<T>(Vector<T> gradients, double threshold = 1e6)
    {
        if (gradients == null) return false;

        var numOps = MathHelper.GetNumericOperations<T>();
        T norm = ComputeNorm(gradients);

        return numOps.GreaterThan(norm, numOps.FromDouble(threshold)) ||
               NumericalStabilityHelper.ContainsNaN(gradients) ||
               NumericalStabilityHelper.ContainsInfinity(gradients);
    }

    /// <summary>
    /// Detects if gradients are vanishing (have very small values).
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="gradients">The gradient vector to check.</param>
    /// <param name="threshold">Threshold for considering gradients as vanishing.</param>
    /// <returns>True if gradients appear to be vanishing.</returns>
    public static bool AreGradientsVanishing<T>(Vector<T> gradients, double threshold = 1e-7)
    {
        if (gradients == null) return true;

        var numOps = MathHelper.GetNumericOperations<T>();
        T norm = ComputeNorm(gradients);

        return numOps.LessThan(norm, numOps.FromDouble(threshold));
    }
}
