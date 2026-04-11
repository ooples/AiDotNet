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
    private int _k = 16;
    private int _maxK = 256;
    private double _minEnergyRatio = 0.95;

    /// <summary>
    /// Number of spectral coefficients to retain. Must be non-negative.
    /// Set to 0 for automatic selection via MDL.
    /// </summary>
    public int K
    {
        get => _k;
        set
        {
            if (value < 0) throw new ArgumentOutOfRangeException(nameof(value), "K must be non-negative.");
            _k = value;
        }
    }

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
    public double MinEnergyRatio
    {
        get => _minEnergyRatio;
        set
        {
            if (value < 0.0 || value > 1.0)
                throw new ArgumentOutOfRangeException(nameof(value), "MinEnergyRatio must be in [0, 1].");
            _minEnergyRatio = value;
        }
    }

    /// <summary>
    /// Maximum K value when using MDL auto-selection, to prevent overfitting.
    /// </summary>
    public int MaxK
    {
        get => _maxK;
        set
        {
            if (value <= 0) throw new ArgumentOutOfRangeException(nameof(value), "MaxK must be positive.");
            _maxK = value;
        }
    }

    /// <summary>
    /// Whether to apply sparsity during training (acts as regularization)
    /// or only during inference (acts as compression).
    /// </summary>
    public bool ApplyDuringTraining { get; set; } = true;
}
