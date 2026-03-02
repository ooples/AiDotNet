using System;
using AiDotNet.Enums;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Kronos (Foundation Model for the Language of Financial Markets).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Kronos is a decoder-only foundation model pre-trained on 12B+ K-line (candlestick) records
/// across 45 global exchanges. It natively understands OHLCV (Open, High, Low, Close, Volume)
/// candlestick patterns for financial market forecasting.
/// </para>
/// <para><b>For Beginners:</b> Kronos is purpose-built for financial markets:
///
/// <b>Candlestick-Native Architecture:</b>
/// Unlike general time series models that treat financial data as simple numbers,
/// Kronos understands candlestick (K-line) patterns directly. Each time step has
/// 5 features: Open, High, Low, Close, Volume (OHLCV).
///
/// <b>Key Advantages:</b>
/// - Pretrained on 12B+ financial records from 45 exchanges
/// - Natively handles multi-feature candlestick data
/// - Decoder-only architecture (efficient autoregressive generation)
///
/// <b>When to Use:</b>
/// - Financial market forecasting with candlestick data
/// - When you need a model that understands OHLCV patterns
/// </para>
/// <para>
/// <b>Reference:</b> "Kronos: A Foundation Model for the Language of Financial Markets", 2025.
/// https://arxiv.org/abs/2508.02739
/// </para>
/// </remarks>
public class KronosOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public KronosOptions() { }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The instance to copy from.</param>
    public KronosOptions(KronosOptions<T> other)
    {
        if (other == null) throw new ArgumentNullException(nameof(other));
        ContextLength = other.ContextLength;
        ForecastHorizon = other.ForecastHorizon;
        PatchLength = other.PatchLength;
        HiddenDimension = other.HiddenDimension;
        NumLayers = other.NumLayers;
        NumHeads = other.NumHeads;
        IntermediateSize = other.IntermediateSize;
        DropoutRate = other.DropoutRate;
        ModelSize = other.ModelSize;
        NumCandlestickFeatures = other.NumCandlestickFeatures;
    }

    /// <summary>
    /// Gets or sets the number of historical candlestick steps used as input context.
    /// </summary>
    /// <value>Defaults to 1024.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many past candlestick bars the model sees.
    /// 1024 bars of daily data ≈ 4 years of history.</para>
    /// </remarks>
    public int ContextLength { get; set; } = 1024;

    /// <summary>
    /// Gets or sets the number of future candlestick steps to forecast.
    /// </summary>
    /// <value>Defaults to 96.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many future candlestick bars to predict.</para>
    /// </remarks>
    public int ForecastHorizon { get; set; } = 96;

    /// <summary>
    /// Gets or sets the patch length for input tokenization.
    /// </summary>
    /// <value>Defaults to 32.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Groups consecutive candlestick bars into patches.
    /// Each patch becomes one token for the transformer.</para>
    /// </remarks>
    public int PatchLength { get; set; } = 32;

    /// <summary>
    /// Gets or sets the hidden dimension of the transformer layers.
    /// </summary>
    /// <value>Defaults to 768.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Controls the model's capacity. Must be divisible
    /// by <see cref="NumHeads"/>.</para>
    /// </remarks>
    public int HiddenDimension { get; set; } = 768;

    /// <summary>
    /// Gets or sets the number of transformer layers.
    /// </summary>
    /// <value>Defaults to 12.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> More layers capture deeper patterns in financial data.</para>
    /// </remarks>
    public int NumLayers { get; set; } = 12;

    /// <summary>
    /// Gets or sets the number of attention heads.
    /// </summary>
    /// <value>Defaults to 12.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each head can focus on different candlestick patterns
    /// (e.g., trend direction, volatility, volume correlation).</para>
    /// </remarks>
    public int NumHeads { get; set; } = 12;

    /// <summary>
    /// Gets or sets the intermediate size for the feed-forward network.
    /// </summary>
    /// <value>Defaults to 3072 (4x hidden dimension).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Size of the hidden layer in the feed-forward network
    /// inside each transformer block. Typically 4x the hidden dimension.</para>
    /// </remarks>
    public int IntermediateSize { get; set; } = 3072;

    /// <summary>
    /// Gets or sets the dropout rate for regularization.
    /// </summary>
    /// <value>Defaults to 0.1 (10%).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Prevents overfitting to training patterns.</para>
    /// </remarks>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the model size variant.
    /// </summary>
    /// <value>Defaults to <see cref="FoundationModelSize.Base"/>.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Controls the overall model size. Larger variants
    /// have more parameters and capacity.</para>
    /// </remarks>
    public FoundationModelSize ModelSize { get; set; } = FoundationModelSize.Base;

    /// <summary>
    /// Gets or sets the number of candlestick features (OHLCV = 5).
    /// </summary>
    /// <value>Defaults to 5 (Open, High, Low, Close, Volume).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Standard candlestick data has 5 features per time step.
    /// Set to fewer if you only have Close prices (1) or OHLC without volume (4).</para>
    /// </remarks>
    public int NumCandlestickFeatures { get; set; } = 5;
}
