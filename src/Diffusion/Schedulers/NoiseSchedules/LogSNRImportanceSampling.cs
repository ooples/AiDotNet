using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;

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
public class LogSNRImportanceSampling<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Samples timesteps with importance weighting based on log-SNR distribution.
    /// </summary>
    /// <param name="batchSize">Number of timesteps to sample.</param>
    /// <param name="alphasCumprod">Alpha cumulative product values.</param>
    /// <param name="random">Random number generator.</param>
    /// <returns>Array of sampled timestep indices.</returns>
    public int[] SampleTimesteps(int batchSize, Vector<T> alphasCumprod, Random random)
    {
        int numTimesteps = alphasCumprod.Length;
        var timesteps = new int[batchSize];

        // Compute log-SNR for each timestep
        var logSNR = new double[numTimesteps];
        for (int i = 0; i < numTimesteps; i++)
        {
            double alpha = NumOps.ToDouble(alphasCumprod[i]);
            alpha = Math.Max(alpha, 1e-10);
            logSNR[i] = Math.Log(alpha / (1.0 - alpha + 1e-10));
        }

        // Sample uniformly in log-SNR space
        double minLogSNR = logSNR[numTimesteps - 1];
        double maxLogSNR = logSNR[0];

        for (int b = 0; b < batchSize; b++)
        {
            double targetLogSNR = minLogSNR + random.NextDouble() * (maxLogSNR - minLogSNR);

            // Find closest timestep
            int bestIdx = 0;
            double bestDist = double.MaxValue;
            for (int i = 0; i < numTimesteps; i++)
            {
                double dist = Math.Abs(logSNR[i] - targetLogSNR);
                if (dist < bestDist) { bestDist = dist; bestIdx = i; }
            }

            timesteps[b] = bestIdx;
        }

        return timesteps;
    }
}
