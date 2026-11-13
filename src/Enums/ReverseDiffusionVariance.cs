namespace AiDotNet.Enums;

/// <summary>
/// Selects the fixed variance sigma_t^2 used by a Gaussian reverse-diffusion step.
/// </summary>
/// <remarks>
/// <para>
/// Ho et al. (2020, "Denoising Diffusion Probabilistic Models", §3.2) report that both
/// sigma_t^2 = beta_t and sigma_t^2 = beta_tilde_t (the true posterior variance) work well.
/// TabDDPM (Kotelnikov et al. 2023, reference code yandex-research/tab-ddpm,
/// <c>gaussian_p_mean_variance</c>) samples with sigma_t^2 = beta_t, which is the default here.
/// </para>
/// <para>
/// <b>For Beginners:</b> When the model removes noise one step at a time, it adds back a
/// little fresh randomness. This setting picks how much: the larger "beta" amount or the
/// smaller "posterior" amount.
/// </para>
/// </remarks>
public enum ReverseDiffusionVariance
{
    /// <summary>
    /// sigma_t^2 = beta_t ("fixedlarge"), the TabDDPM reference default.
    /// </summary>
    FixedLarge,

    /// <summary>
    /// sigma_t^2 = beta_tilde_t = beta_t (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t) ("fixedsmall").
    /// </summary>
    FixedSmall
}
