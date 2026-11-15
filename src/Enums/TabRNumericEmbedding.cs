namespace AiDotNet.Enums;

/// <summary>
/// How TabR embeds its numeric features before the encoder's first linear map: the reference implementation's
/// <c>num_embeddings</c>.
/// </summary>
/// <remarks>
/// <para>
/// Gorishniy et al. (2023) report TabR-S without numeric embeddings and their main TabR results with the PLR
/// embedding in its lite form (<c>lib.deep.PLREmbeddings</c>, configured by <c>n_frequencies</c>,
/// <c>frequency_scale</c>, <c>d_embedding</c> and <c>lite</c>).
/// </para>
/// <para><b>For Beginners:</b> keep <see cref="None"/> for the simple TabR-S model; choose
/// <see cref="PeriodicLinearReLU"/> for the stronger variant the paper reports as TabR.</para>
/// </remarks>
public enum TabRNumericEmbedding
{
    /// <summary><c>num_embeddings = None</c>: the raw features enter the encoder, as in TabR-S.</summary>
    None = 0,

    /// <summary>
    /// <c>PLREmbeddings</c>: every feature becomes <c>ReLU(Linear([cos(2 pi c x), sin(2 pi c x)]))</c> with learned
    /// frequencies <c>c</c>.
    /// </summary>
    PeriodicLinearReLU = 1
}
