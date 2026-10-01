namespace AiDotNet.Enums;

/// <summary>
/// Selects how DeepAR rescales a series before the recurrent trunk sees it.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Real series live on wildly different magnitudes - one stock trades at $10
/// and another at $1000, one product sells five units a day and another five thousand. A network
/// that sees the raw numbers spends its capacity learning the magnitudes instead of the shapes.
/// DeepAR divides each series by a single number that summarizes its own level, learns on the
/// result, and multiplies the prediction back afterwards, so every series arrives at the network
/// looking roughly the same size.
/// </para>
/// <para>
/// Salinas et al. 2020, section 3.3 ("Scale handling") defines that number as
/// <c>nu = 1 + mean(z)</c> over the conditioning range, which is <see cref="PaperMean"/> and the
/// default. The alternatives exist because that definition assumes non-negative data: a series
/// that is already centered on zero, such as a return series, has a mean near zero and gains
/// nothing from it.
/// </para>
/// </remarks>
public enum DeepARScaleHandling
{
    /// <summary>
    /// The paper's scale factor, <c>nu = 1 + mean(z)</c> over the conditioning range
    /// (Salinas et al. 2020, section 3.3).
    /// </summary>
    PaperMean,

    /// <summary>
    /// <c>nu = mean(|z|)</c> over the conditioning range. Appropriate for a signed series centered
    /// on zero, where the paper's mean carries no magnitude information.
    /// </summary>
    MeanAbsolute,

    /// <summary>
    /// No rescaling: the network sees the series as supplied. Appropriate when the caller has
    /// already normalized the data.
    /// </summary>
    None
}
