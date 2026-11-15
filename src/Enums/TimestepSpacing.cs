namespace AiDotNet.Enums;

/// <summary>
/// How a scheduler spreads its inference timesteps over the training timesteps (diffusers'
/// <c>timestep_spacing</c>; Lin et al. 2024, "Common Diffusion Noise Schedules and Sample Steps are Flawed", Table 2).
/// </summary>
/// <remarks>
/// <para>
/// With <c>T</c> training steps, <c>n</c> inference steps and the ratio <c>r = T / n</c> (integer division):
/// <list type="bullet">
/// <item><description><see cref="Trailing"/>: <c>T − 1, T − 1 − r, …</c>, ending at <c>r − 1</c>.</description></item>
/// <item><description><see cref="Leading"/>: <c>(n − 1)·r, …, r, 0</c>, each plus the scheduler's
/// <c>StepsOffset</c> (Stable Diffusion 1.x and Wonder3D: offset 1, so <c>981, 961, …, 1</c> for 50 of 1000).</description></item>
/// <item><description><see cref="Linspace"/>: <c>round(linspace(0, T − 1, n))</c>, descending.</description></item>
/// </list>
/// </para>
/// <para><b>For Beginners:</b> A sampler only visits some of the noise levels it was trained on; this picks which
/// ones. A model gives its best results with the spacing it was released with.</para>
/// </remarks>
public enum TimestepSpacing
{
    /// <summary>Count down from the last training step by the step ratio.</summary>
    Trailing,

    /// <summary>Multiples of the step ratio from zero, plus the steps offset (diffusers' default).</summary>
    Leading,

    /// <summary>Evenly spaced from the first to the last training step, rounded.</summary>
    Linspace,
}
