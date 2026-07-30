using System;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for RWKV-based time series forecasting.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// RWKV (Receptance Weighted Key Value) is a linear-complexity sequence model that combines
/// the efficient training parallelism of Transformers with the constant-memory inference of RNNs.
/// This options class configures an RWKV model for time series forecasting tasks.
/// </para>
/// <para><b>For Beginners:</b> RWKV combines the best of both worlds:
///
/// <b>Key Properties:</b>
/// 1. <b>Linear Complexity:</b> O(n) for training and inference (vs O(n^2) for Transformers)
/// 2. <b>Constant Memory:</b> O(1) per-token generation memory
/// 3. <b>Parallel Training:</b> Can be computed as a convolution for efficient parallel training
/// 4. <b>Multi-head:</b> Multiple attention heads for better capacity
///
/// <b>Architecture:</b>
/// - Time mixing: WKV attention mechanism with learned decay
/// - Channel mixing: FFN with gating
/// - Residual connections and layer normalization
/// </para>
/// <para>
/// <b>Reference:</b> Peng et al., "RWKV: Reinventing RNNs for the Transformer Era", 2023.
/// </para>
/// </remarks>
public class RWKVForecastingOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public RWKVForecastingOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public RWKVForecastingOptions(RWKVForecastingOptions<T> other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        ContextLength = other.ContextLength;
        ForecastHorizon = other.ForecastHorizon;
        ModelDimension = other.ModelDimension;
        NumHeads = other.NumHeads;
        NumLayers = other.NumLayers;
        DropoutRate = other.DropoutRate;
        LearningRate = other.LearningRate;
        GlobalIclrMultiplier = other.GlobalIclrMultiplier;
        AdamBeta1 = other.AdamBeta1;
        AdamBeta2 = other.AdamBeta2;
        AdamEpsilon = other.AdamEpsilon;
    }

    /// <summary>
    /// Gets or sets the context length (input sequence length). Default: 512.
    /// </summary>
    public int ContextLength { get; set; } = 512;

    /// <summary>
    /// Gets or sets the forecast horizon (prediction length). Default: 96.
    /// </summary>
    public int ForecastHorizon { get; set; } = 96;

    /// <summary>
    /// Gets or sets the model dimension (d_model). Default: 256.
    /// </summary>
    public int ModelDimension { get; set; } = 256;

    /// <summary>
    /// Gets or sets the number of RWKV heads. Default: 8.
    /// </summary>
    public int NumHeads { get; set; } = 8;

    /// <summary>
    /// Gets or sets the number of RWKV layers. Default: 4.
    /// </summary>
    public int NumLayers { get; set; } = 4;

    /// <summary>
    /// Gets or sets the dropout rate for regularization. Default: 0.1.
    /// </summary>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets Adam's initial learning rate. Default: 6e-4, matching the
    /// smallest-model training recipe reported by the RWKV paper.
    /// </summary>
    public double LearningRate { get; set; } = 6e-4;

    /// <summary>
    /// Gets or sets RWKV-7's "Global ICLR Multiplier" c in the state transition
    /// A_t = diag(w_t) - c * kappaHat^T(a_t (*) kappaHat).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Peng et al. 2025 (arXiv 2503.14456) Appendix C, Theorem 1 bounds all eigenvalues of A in
    /// (-1, 1) for c in (0, 1 + u), where u = exp(-e^-0.5) = 0.5452..., and states that c "is set to
    /// 1 in the current implementations of RWKV-7 language modeling" -- hence the default of 1.
    /// </para>
    /// <para>
    /// Long-context forecasting can restrict the range the way the paper's own authors do. Theorem 1
    /// item 4 only guarantees the PRODUCT of transitions is bounded when a_t is time-independent,
    /// which RWKV-7 violates by design, and the paper's footnote 4 records that they "allow only part
    /// of the range of possible negative eigenvalues in our pre-trained large language models due to
    /// experimentally observed training instabilities". A negative eigenvalue arises exactly when
    /// c*a > w, so any c &lt;= u keeps w - c*a positive for every admissible w and a and the long
    /// product bounded. The default is exactly that bound, u = exp(-e^-0.5), the same constant Eq 12
    /// clamps the decay to; set this to 1.0 to reproduce RWKV-7 language modeling verbatim.
    /// </para>
    /// </remarks>
    public double GlobalIclrMultiplier { get; set; } = Math.Exp(-Math.Exp(-0.5));

    /// <summary>
    /// Gets or sets the variance floor used by RevIN's per-instance standard deviation,
    /// std = sqrt(var + eps). Default: 1e-5, the value Kim et al. 2022 use.
    /// </summary>
    /// <remarks>
    /// This is what keeps a constant (zero-variance) series from dividing by zero. It is exposed
    /// because a series whose natural scale is far from unity may need a different floor.
    /// </remarks>
    public double RevInEpsilon { get; set; } = 1e-5;

    /// <summary>
    /// Gets or sets whether reversible instance normalization (RevIN) is applied around the forecast.
    /// Default: true, which is the published behaviour.
    /// </summary>
    /// <remarks>
    /// RevIN is an optional wrapper in Kim et al. 2022, not part of the RWKV architecture itself, so it
    /// is switchable. Turning it off makes the model consume the raw series and forecast on the raw
    /// scale.
    /// </remarks>
    public bool UseReversibleNormalization { get; set; } = true;

    /// <summary>
    /// Gets or sets Adam's first-moment decay. Default: 0.9.
    /// </summary>
    public double AdamBeta1 { get; set; } = 0.9;

    /// <summary>
    /// Gets or sets Adam's second-moment decay. Default: 0.99.
    /// </summary>
    public double AdamBeta2 { get; set; } = 0.99;

    /// <summary>
    /// Gets or sets Adam's numerical-stability epsilon. Default: 1e-8.
    /// </summary>
    public double AdamEpsilon { get; set; } = 1e-8;
}
