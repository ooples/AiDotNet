using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;

namespace AiDotNet.Helpers;

/// <summary>
/// Computes the Kaiming/He initialization gain for a given activation function.
/// Mirrors <c>torch.nn.init.calculate_gain</c> from PyTorch — the gain is the
/// per-activation variance-preservation factor used in <c>kaiming_uniform_</c>
/// / <c>kaiming_normal_</c> weight initialization (He et al. 2015 "Delving
/// Deep into Rectifiers", §2.2).
/// </summary>
/// <remarks>
/// <para>
/// The Kaiming-uniform bound for a layer with fan_in inputs is
/// <c>bound = gain * sqrt(3 / fan_in)</c> so that the resulting weights
/// have variance <c>gain² / fan_in</c> — preserving the variance of the
/// pre-activation through the layer when the activation has the matching
/// nonlinearity.
/// </para>
/// <para>
/// Using the wrong gain on a deep network produces single-step gradient
/// explosions: a 53-layer convnet (GraFPrint, Bhattacharjee 2023) initialized
/// with the ReLU gain (<c>sqrt(2)</c>) when the actual nonlinearity is
/// LeakyReLU(0.2) drives forward variance and backward gradient norms ~2%
/// higher than they should be, which is enough to make the first Adam step
/// move the loss by orders of magnitude on small-batch training.
/// </para>
/// </remarks>
internal static class KaimingInitHelper
{
    /// <summary>
    /// Returns the Kaiming gain for the given activation function. Returns
    /// the linear gain (1.0) when <paramref name="activation"/> is null,
    /// identity, or any activation not specifically tabulated. Mirrors
    /// PyTorch's <c>calculate_gain</c> table.
    /// </summary>
    /// <typeparam name="T">Numeric type (gain is computed in double and the
    /// caller folds it into the per-T initialization arithmetic).</typeparam>
    /// <param name="activation">The activation function the layer feeds into.
    /// Pass <c>null</c> when the layer has no activation in its own slot AND
    /// no information about the downstream activation (treated as linear).</param>
    /// <returns>The variance-preservation gain (a scalar &gt;= 1).</returns>
    public static double GainFor<T>(IActivationFunction<T>? activation)
    {
        if (activation is null) return 1.0;

        if (activation is LeakyReLUActivation<T> leaky)
        {
            // gain = sqrt(2 / (1 + a²)) where a is the negative slope.
            double a = MathHelper.GetNumericOperations<T>().ToDouble(leaky.Alpha);
            return Math.Sqrt(2.0 / (1.0 + a * a));
        }
        if (activation is ReLUActivation<T>)
            return Math.Sqrt(2.0);
        if (activation is TanhActivation<T>)
            return 5.0 / 3.0;                 // PyTorch convention.
        if (activation is SELUActivation<T>)
            return 3.0 / 4.0;                 // PyTorch convention.
        // Sigmoid, identity, and unknown activations all use linear gain.
        return 1.0;
    }

    /// <summary>
    /// Returns the Kaiming-uniform bound for the given fan-in and activation.
    /// </summary>
    public static double UniformBoundFor<T>(int fanIn, IActivationFunction<T>? activation)
    {
        if (fanIn <= 0)
            throw new ArgumentOutOfRangeException(nameof(fanIn),
                $"fan_in must be positive; got {fanIn}.");
        double gain = GainFor(activation);
        return gain * Math.Sqrt(3.0 / fanIn);
    }
}
