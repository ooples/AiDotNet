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

    /// <summary>Sum of squares accumulated in <c>double</c>, whatever <typeparamref name="T"/> is.</summary>
    /// <remarks>
    /// ONE ACCUMULATOR FOR THE WHOLE FILE. Summing squares in T overflows for a large float gradient
    /// vector: the norm becomes +Infinity, the threshold test passes, the scale becomes
    /// maxNorm/Infinity = 0, and any non-finite element then yields Infinity * 0 = NaN -- so enabling
    /// clipping POISONS the gradients it was turned on to protect. Measured on TableTransformer:
    /// green without clipping, "L2 distance = NaN ... collapsed to a uniform-output state" with it.
    ///
    /// This existed as a fix in two of the six norm computations in this file, which left the same
    /// failure reachable through the other four -- including ClipByGlobalNorm, the most exposed of all
    /// because it sums across every layer. One helper is what stops them drifting apart again.
    /// </remarks>
    private static double SumSquares<T>(Vector<T> gradients, INumericOperations<T> ops)
    {
        double total = 0.0;
        for (int i = 0; i < gradients.Length; i++)
        {
            double v = ops.ToDouble(gradients[i]);
            total += v * v;
        }

        return total;
    }

    /// <inheritdoc cref="SumSquares{T}(Vector{T}, INumericOperations{T})"/>
    private static double SumSquares<T>(Tensor<T> gradients, INumericOperations<T> ops)
    {
        double total = 0.0;
        for (int i = 0; i < gradients.Length; i++)
        {
            double v = ops.ToDouble(gradients.GetFlatIndexValue(i));
            total += v * v;
        }

        return total;
    }

    /// <summary>True when a norm cannot be used to scale, having reported why.</summary>
    /// <remarks>
    /// The non-finite branch used to return the gradients untouched and say nothing. Its caller in
    /// this codebase, GradientBasedOptimizerBase.ApplyGradientClipping, has no guard of its own, so a
    /// poisoned step went straight into the update with no signal that the safeguard had declined to
    /// act. Clipping cannot repair gradients that are already non-finite -- but it can say so.
    /// </remarks>
    private static bool IsUnusableNorm(double norm, string operation)
    {
        if (!double.IsNaN(norm) && !double.IsInfinity(norm)) return false;

        System.Diagnostics.Trace.TraceWarning(
            $"AiDotNet.GradientClippingHelper.{operation}: gradient norm is {norm}, so the gradients " +
            "were left unclipped. They are already non-finite before clipping; scaling by a degenerate " +
            "factor would turn that into silently zeroed or NaN updates.");

        return true;
    }

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

        // ACCUMULATE IN DOUBLE, NOT IN T. Summing squares in T overflows for a
        // large float gradient vector: norm becomes +Infinity, the threshold test
        // below PASSES, scale becomes maxNorm/Infinity = 0, and any non-finite
        // element then yields Infinity * 0 = NaN -- so enabling clipping POISONED
        // the gradients that it was turned on to protect. Measured on
        // TableTransformer: green without clipping, "L2 distance = NaN ... collapsed
        // to a uniform-output state" with it. The tape-path clipper in
        // GradientBasedOptimizerBase already accumulates in double and guards the
        // non-finite cases; this is the same computation and needs the same care.
        double norm = Math.Sqrt(SumSquares(gradients, numOps));

        // A norm that is zero or non-finite has no meaningful direction to
        // preserve, so scaling is skipped rather than applied with a degenerate
        // factor. Returning the gradients untouched leaves the caller exactly as
        // well off as clipping being disabled.
        // `norm <= 0.0` rather than `norm == 0.0`: the norm is a square root of a sum of squares, so
        // it is never negative, which makes the two tests equivalent here -- without the exact
        // float-equality that CodeQL flags. The clause is not redundant with `norm <= maxNorm`; it is
        // what stops a caller-supplied negative maxNorm from producing a -Infinity scale.
        if (IsUnusableNorm(norm, nameof(ClipByNorm)) || norm <= maxNorm || norm <= 0.0)
        {
            return gradients.Clone();
        }

        // Scale gradients
        T scale = numOps.FromDouble(maxNorm / norm);
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

        // Accumulated in double for the reason given in ClipByNorm above: a T-typed
        // sum of squares overflows on large float gradients, and the resulting
        // Infinity norm turns clipping into a NaN generator instead of a safeguard.
        double norm = Math.Sqrt(SumSquares(gradients, numOps));

        // No meaningful direction to preserve -- leave the gradients alone.
        // See ClipByNorm above for why this is `norm <= 0.0` and not `norm == 0.0`.
        if (IsUnusableNorm(norm, nameof(ClipByNormInPlace)) || norm <= maxNorm || norm <= 0.0)
        {
            return false;
        }

        // Scale gradients in place
        T scale = numOps.FromDouble(maxNorm / norm);
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
        double globalSumSquares = 0.0;
        foreach (var gradients in gradientsList)
        {
            if (gradients == null) continue;
            globalSumSquares += SumSquares(gradients, numOps);
        }
        double globalNorm = Math.Sqrt(globalSumSquares);

        // If global norm is below threshold, or is not a number that can scale anything, return clones.
        // Nulls are dropped here, exactly as the scaling path below drops them: the two paths used to
        // disagree -- this one preserved a null as a null entry, the other skipped it -- so the length
        // of the returned list depended on whether clipping happened to fire, and a caller indexing it
        // against its layers got a different answer each way.
        if (IsUnusableNorm(globalNorm, nameof(ClipByGlobalNorm)) || globalNorm <= maxNorm)
        {
            var clones = new List<Vector<T>>(gradientsList.Count);
            foreach (var gradients in gradientsList)
            {
                if (gradients == null) continue;
                clones.Add(gradients.Clone());
            }

            return clones;
        }

        // Scale all gradients
        T scale = numOps.FromDouble(maxNorm / globalNorm);
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
        double norm = Math.Sqrt(SumSquares(gradients, numOps));

        // If norm is below threshold, or cannot scale anything, return clone
        if (IsUnusableNorm(norm, nameof(ClipByNorm)) || norm <= maxNorm)
        {
            return (Tensor<T>)gradients.Clone();
        }

        // Scale gradients
        T scale = numOps.FromDouble(maxNorm / norm);
        var clipped = new Tensor<T>(gradients._shape);
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

        // Double accumulation for the reason given on SumSquares. AreGradientsExploding and
        // ClipAdaptive both build on this, so a T-typed sum that overflowed made the explosion
        // DETECTOR report on its own overflow rather than on the gradients.
        return ops.FromDouble(Math.Sqrt(SumSquares(gradients, ops)));
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

        double globalSumSquares = 0.0;
        foreach (var gradients in gradientsList)
        {
            if (gradients == null) continue;
            globalSumSquares += SumSquares(gradients, numOps);
        }

        return numOps.FromDouble(Math.Sqrt(globalSumSquares));
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
