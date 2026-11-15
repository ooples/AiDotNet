using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Models.Options;

/// <summary>
/// The MINE grid parameters of the maximal information coefficient (Reshef et al. 2011, "Detecting Novel
/// Associations in Large Data Sets", Science 334), as minepy's <c>mine_parameter</c>.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> MIC tries many grids over a scatter plot of a feature against the target. Alpha sets
/// how many grid cells it may use, and C how finely each column may first be split before columns are merged.
/// The defaults are the paper's.</para>
/// </remarks>
public sealed class MaximalInformationOptions
{
    /// <summary>
    /// The grid budget exponent: <c>B(n) = max(n^alpha, 4)</c> for alpha in (0, 1], or <c>B = min(alpha, n)</c> for
    /// alpha of at least 4 (paper and minepy default 0.6).
    /// </summary>
    public double Alpha { get; set; } = MaximalInformationCoefficient.DefaultAlpha;

    /// <summary>The clump factor: the x axis is split into at most <c>c</c> times as many superclumps as columns (default 15).</summary>
    public double C { get; set; } = MaximalInformationCoefficient.DefaultClumps;
}
