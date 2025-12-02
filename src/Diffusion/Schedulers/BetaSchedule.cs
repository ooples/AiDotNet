namespace AiDotNet.Diffusion.Schedulers;

/// <summary>
/// Defines the types of beta (noise variance) schedules available for diffusion models.
/// </summary>
/// <remarks>
/// <para>
/// The beta schedule controls how noise variance changes across timesteps during the
/// diffusion process. Different schedules have different characteristics and are suited
/// for different applications.
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of this like choosing how to gradually add static to a TV signal.
///
/// - Linear: Add static evenly - each step adds about the same amount
/// - ScaledLinear: Start slow, then add more - common in image generation (Stable Diffusion)
/// - SquaredCosine: Smooth S-curve - often produces better quality results
///
/// The choice affects both training efficiency and generation quality.
/// </para>
/// </remarks>
public enum BetaSchedule
{
    /// <summary>
    /// Linear interpolation between beta start and end values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The simplest schedule: beta increases linearly from start to end.
    /// This is the original schedule used in the DDPM paper.
    /// </para>
    /// <para>
    /// <b>Default values:</b> beta_start=0.0001, beta_end=0.02 (from DDPM paper)
    /// </para>
    /// </remarks>
    Linear,

    /// <summary>
    /// Scaled linear schedule commonly used in latent diffusion models.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Uses the square root of linearly interpolated values.
    /// This is the default schedule used by Stable Diffusion and similar models.
    /// </para>
    /// <para>
    /// <b>Default values:</b> beta_start=0.00085, beta_end=0.012 (from Stable Diffusion)
    /// </para>
    /// </remarks>
    ScaledLinear,

    /// <summary>
    /// Squared cosine schedule for improved diffusion models.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Based on the "Improved Denoising Diffusion Probabilistic Models" paper.
    /// Provides smoother noise progression and often better generation quality.
    /// </para>
    /// <para>
    /// <b>Why use this:</b> The squared cosine schedule is designed to maintain
    /// more signal (clearer images) for longer during the forward process, which
    /// can improve the model's ability to learn fine details.
    /// </para>
    /// </remarks>
    SquaredCosine
}
