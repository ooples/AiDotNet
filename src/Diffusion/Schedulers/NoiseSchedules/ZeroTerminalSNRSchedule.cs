using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.Diffusion.Schedulers.NoiseSchedules;

/// <summary>
/// Zero Terminal SNR noise schedule ensuring signal-to-noise ratio reaches exactly zero at the final timestep.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Standard beta schedules often have non-zero SNR at the final timestep, meaning the model
/// never sees pure noise during training. Zero Terminal SNR rescales the schedule so that
/// alpha_cumprod[T] = 0, ensuring the final timestep is pure noise.
/// </para>
/// <para>
/// <b>For Beginners:</b> This fixes a common issue in diffusion training where the model
/// never sees completely noisy images. By ensuring SNR reaches zero, the model learns to
/// generate from pure noise, improving overall sample quality.
/// </para>
/// <para>
/// Reference: Lin et al., "Common Diffusion Noise Schedules and Sample Steps are Flawed", WACV 2024
/// </para>
/// </remarks>
[ComponentType(ComponentType.Scheduler)]
[PipelineStage(PipelineStage.Training)]
public class ZeroTerminalSNRSchedule<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Rescales alpha cumulative products to enforce zero terminal SNR.
    /// </summary>
    /// <param name="alphasCumprod">Original alpha cumulative product values.</param>
    /// <returns>Rescaled values where the last element is exactly zero.</returns>
    public Vector<T> Apply(Vector<T> alphasCumprod)
    {
        int len = alphasCumprod.Length;
        if (len == 0) return alphasCumprod;

        // Lin et al. (2024), Algorithm 1: rescale sqrt(alpha_cumprod) so the last value is 0 and the first is
        // unchanged, then square back.
        var engine = AiDotNetEngine.Current;
        var sqrtAlphas = engine.Sqrt(alphasCumprod);
        T first = sqrtAlphas[0];
        T last = sqrtAlphas[len - 1];
        var shifted = engine.Subtract(sqrtAlphas, SchedulerTensorOps.Full(len, last));
        var rescaled = engine.Multiply(shifted, NumOps.Divide(first, NumOps.Subtract(first, last)));
        return engine.Multiply(rescaled, rescaled);
    }
}
