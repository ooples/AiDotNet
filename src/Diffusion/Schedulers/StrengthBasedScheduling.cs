using AiDotNet.Helpers;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.Schedulers;

/// <summary>
/// Strength-based scheduling for img2img and inpainting denoising control.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Controls how much of the original image is preserved vs. regenerated in img2img and
/// inpainting pipelines. Maps a user-facing "strength" parameter (0.0-1.0) to the
/// appropriate starting timestep in the noise schedule, determining how many denoising
/// steps to run and at what noise level to begin.
/// </para>
/// <para>
/// <b>For Beginners:</b> In image editing with diffusion models, "strength" controls how
/// much the AI changes the original image. Low strength (0.2) makes small tweaks, high
/// strength (0.9) almost completely regenerates. This class handles the math of converting
/// that simple 0-1 slider into the right technical settings for the diffusion process.
/// </para>
/// </remarks>
public class StrengthBasedScheduling<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly double _defaultStrength;
    private readonly int _totalTimesteps;

    /// <summary>
    /// Gets the default denoising strength.
    /// </summary>
    public double DefaultStrength => _defaultStrength;

    /// <summary>
    /// Gets the total number of timesteps in the full schedule.
    /// </summary>
    public int TotalTimesteps => _totalTimesteps;

    /// <summary>
    /// Initializes strength-based scheduling.
    /// </summary>
    /// <param name="totalTimesteps">Total timesteps in the noise schedule (default: 1000).</param>
    /// <param name="defaultStrength">Default denoising strength (default: 0.8).</param>
    public StrengthBasedScheduling(int totalTimesteps = 1000, double defaultStrength = 0.8)
    {
        _totalTimesteps = totalTimesteps;
        _defaultStrength = Math.Min(1.0, Math.Max(0.0, defaultStrength));
    }

    /// <summary>
    /// Gets the starting timestep for a given strength.
    /// </summary>
    /// <param name="strength">Denoising strength (0.0 = no change, 1.0 = full regeneration).</param>
    /// <returns>Starting timestep index.</returns>
    public int GetStartTimestep(double strength)
    {
        strength = Math.Min(1.0, Math.Max(0.0, strength));
        return (int)Math.Round(strength * _totalTimesteps);
    }

    /// <summary>
    /// Gets the number of denoising steps to perform for a given strength.
    /// </summary>
    /// <param name="strength">Denoising strength.</param>
    /// <param name="numInferenceSteps">Total inference steps configured.</param>
    /// <returns>Number of steps to actually run.</returns>
    public int GetEffectiveSteps(double strength, int numInferenceSteps)
    {
        strength = Math.Min(1.0, Math.Max(0.0, strength));
        return Math.Max(1, (int)Math.Round(strength * numInferenceSteps));
    }

    /// <summary>
    /// Gets the noise level (alpha_bar) at the starting timestep.
    /// </summary>
    /// <param name="strength">Denoising strength.</param>
    /// <returns>Noise level as a value of T.</returns>
    public T GetStartNoiseLevel(double strength)
    {
        strength = Math.Min(1.0, Math.Max(0.0, strength));
        // Linear noise schedule approximation: alpha_bar ≈ 1 - strength
        return NumOps.FromDouble(1.0 - strength);
    }

    /// <summary>
    /// Truncates a full timestep schedule to start from the strength-determined point.
    /// </summary>
    /// <param name="fullSchedule">Full array of timesteps from the scheduler.</param>
    /// <param name="strength">Denoising strength.</param>
    /// <returns>Truncated timestep array.</returns>
    public T[] TruncateSchedule(T[] fullSchedule, double strength)
    {
        int startIdx = fullSchedule.Length - GetEffectiveSteps(strength, fullSchedule.Length);
        startIdx = Math.Max(0, Math.Min(startIdx, fullSchedule.Length - 1));

        var truncated = new T[fullSchedule.Length - startIdx];
        Array.Copy(fullSchedule, startIdx, truncated, 0, truncated.Length);
        return truncated;
    }
}
