namespace AiDotNet.HarmonicEngine.Options;

/// <summary>
/// Configuration options for spectral sparsity masking (top-K frequency selection).
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Spectral sparsity controls how aggressively the model compresses
/// its internal representation. Only the K strongest frequency components are kept;
/// the rest are zeroed out. This serves as a regularizer — preventing overfitting by
/// limiting model capacity — and as compression — reducing model storage.
/// </para>
/// </remarks>
public class SpectralSparsityOptions
{
    /// <summary>
    /// Number of spectral coefficients to retain. Set to 0 for automatic selection via MDL.
    /// </summary>
    public int K { get; set; } = 16;

    /// <summary>
    /// Whether to use MDL (Minimum Description Length) for automatic K selection.
    /// When true, the K property is ignored.
    /// </summary>
    public bool UseMDLAutoK { get; set; }

    /// <summary>
    /// Minimum energy ratio required for the selected K components.
    /// If the top-K components capture less than this fraction of total energy,
    /// K is increased until the threshold is met.
    /// </summary>
    public double MinEnergyRatio { get; set; } = 0.95;

    /// <summary>
    /// Maximum K value when using MDL auto-selection, to prevent overfitting.
    /// </summary>
    public int MaxK { get; set; } = 256;

    /// <summary>
    /// Whether to apply sparsity during training (acts as regularization)
    /// or only during inference (acts as compression).
    /// </summary>
    public bool ApplyDuringTraining { get; set; } = true;
}
