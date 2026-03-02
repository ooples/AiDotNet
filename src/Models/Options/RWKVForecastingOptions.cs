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
}
