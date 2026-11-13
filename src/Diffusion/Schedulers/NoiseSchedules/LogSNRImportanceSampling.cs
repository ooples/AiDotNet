using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.Diffusion.Schedulers.NoiseSchedules;

/// <summary>
/// Log-SNR importance sampling for efficient timestep selection during diffusion training.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Instead of uniformly sampling timesteps during training, samples proportionally to the
/// gradient magnitude at each timestep. Approximates this by sampling from a distribution
/// that is uniform in log-SNR space, focusing training on the most informative timesteps.
/// </para>
/// <para>
/// <b>For Beginners:</b> During training, some timesteps are more important than others
/// for the model to learn from. This utility picks timesteps that give the model the
/// most useful learning signal, making training more efficient.
/// </para>
/// <para>
/// Reference: Hang et al., "Efficient Diffusion Training via Min-SNR Weighting Strategy", ICCV 2023
/// </para>
/// </remarks>
[ComponentType(ComponentType.Scheduler)]
[PipelineStage(PipelineStage.Training)]
public class LogSNRImportanceSampling<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// The largest |log-SNR| used for sampling, ln(1e10): the value log(alpha / (1 - alpha)) takes at alpha = 0 and
    /// alpha = 1 under a 1e-10 guard.
    /// </summary>
    internal static readonly double LogSnrBound = Math.Log(1e10);

    /// <summary>
    /// Samples timesteps with importance weighting based on log-SNR distribution.
    /// </summary>
    /// <param name="batchSize">Number of timesteps to sample.</param>
    /// <param name="alphasCumprod">Alpha cumulative product values.</param>
    /// <param name="random">Random number generator.</param>
    /// <returns>Array of sampled timestep indices.</returns>
    public int[] SampleTimesteps(int batchSize, Vector<T> alphasCumprod, Random random)
    {
        var engine = AiDotNetEngine.Current;
        int numTimesteps = alphasCumprod.Length;

        // log-SNR = log(alpha / (1 - alpha)) per timestep, clamped to +-ln(1e10) so zero-terminal-SNR schedules
        // (alpha = 0, log-SNR = -inf) and alpha = 1 give a finite range. The bound is the log-SNR the 1e-10 guard
        // assigns to alpha = 0 and alpha = 1, fixed in value rather than tied to the element type's epsilon, so
        // the sampled distribution does not change with T.
        var alphas = alphasCumprod.AsTensor();
        var logSnr = engine.TensorClamp(
            engine.TensorSubtract(engine.TensorLog(alphas), engine.TensorLog(engine.ScalarMinusTensor(NumOps.One, alphas))),
            NumOps.FromDouble(-LogSnrBound), NumOps.FromDouble(LogSnrBound));

        // Sample uniformly in log-SNR space. The seeded uniform draws stay one per batch element in master's order
        // (no engine op consumes a caller Random).
        var uniform = new Vector<T>(batchSize);
        for (int b = 0; b < batchSize; b++)
            uniform[b] = NumOps.FromDouble(random.NextDouble());
        T minLogSnr = logSnr[numTimesteps - 1];
        T maxLogSnr = logSnr[0];
        var targets = engine.TensorAddScalar(
            engine.TensorMultiplyScalar(uniform.AsTensor(), NumOps.Subtract(maxLogSnr, minLogSnr)), minLogSnr);

        // Closest timestep per target: argmin over |logSnr[i] - target| (first index on ties).
        var shape = new[] { batchSize, numTimesteps };
        var distances = engine.TensorAbs(engine.TensorSubtract(
            engine.TensorBroadcastTo(logSnr.Reshape(new[] { 1, numTimesteps }), shape),
            engine.TensorBroadcastTo(targets.Reshape(new[] { batchSize, 1 }), shape)));
        return engine.TensorArgMin(distances, 1).ToArray();
    }
}
