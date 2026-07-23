using System;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for TOTEM (TOkenized Time Series EMbeddings).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// TOTEM learns discrete tokenized representations for time series via VQ-VAE,
/// enabling the use of discrete token-based methods (like LLMs) on continuous time series data.
/// </para>
/// <para><b>For Beginners:</b> TOTEM bridges continuous time series and discrete tokens:
///
/// <b>Vector Quantization (VQ-VAE):</b>
/// TOTEM maintains a learned codebook of discrete "patterns". Each segment of your time
/// series is matched to the nearest codebook entry, converting continuous values into
/// discrete tokens. This allows LLM-style methods to work on time series.
///
/// <b>Key Advantages:</b>
/// - Converts continuous time series to discrete tokens for LLM compatibility
/// - Multiple codebooks capture different aspects of temporal patterns
/// - Commitment loss keeps encoder outputs close to codebook entries
/// </para>
/// <para>
/// <b>Reference:</b> Talukder et al., "TOTEM: TOkenized Time Series EMbeddings for General Time Series Analysis", 2024.
/// </para>
/// </remarks>
public class TOTEMOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public TOTEMOptions() { }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The instance to copy from.</param>
    public TOTEMOptions(TOTEMOptions<T> other)
    {
        if (other == null) throw new ArgumentNullException(nameof(other));
        ContextLength = other.ContextLength; ForecastHorizon = other.ForecastHorizon;
        HiddenDimension = other.HiddenDimension; NumLayers = other.NumLayers;
        NumHeads = other.NumHeads; CodebookSize = other.CodebookSize;
        CodebookDimension = other.CodebookDimension; NumCodebooks = other.NumCodebooks;
        DropoutRate = other.DropoutRate; CommitmentWeight = other.CommitmentWeight;
    }

    /// <summary>
    /// Gets or sets the number of historical time steps used as input context.
    /// </summary>
    /// <value>Defaults to 512.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How much history the VQ-VAE encoder processes.</para>
    /// </remarks>
    public int ContextLength { get; set; } = 512;

    /// <summary>
    /// Gets or sets the number of future time steps to forecast.
    /// </summary>
    /// <value>Defaults to 96.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many future steps the decoder reconstructs.</para>
    /// </remarks>
    public int ForecastHorizon { get; set; } = 96;

    /// <summary>
    /// Gets or sets the hidden dimension of the transformer layers.
    /// </summary>
    /// <value>Defaults to 256.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Internal representation size. Must be divisible by <see cref="NumHeads"/>.</para>
    /// </remarks>
    public int HiddenDimension { get; set; } = 256;

    /// <summary>
    /// Gets or sets the number of transformer layers.
    /// </summary>
    /// <value>Defaults to 6.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Depth of the encoder/decoder stacks.</para>
    /// </remarks>
    public int NumLayers { get; set; } = 6;

    /// <summary>
    /// Gets or sets the number of attention heads.
    /// </summary>
    /// <value>Defaults to 8.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each head captures different temporal relationships.</para>
    /// </remarks>
    public int NumHeads { get; set; } = 8;

    /// <summary>
    /// Gets or sets the number of entries in each codebook.
    /// </summary>
    /// <value>Defaults to 1024.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many discrete "patterns" each codebook can represent.
    /// Larger codebooks capture more patterns but are harder to train. 1024 is standard.</para>
    /// </remarks>
    public int CodebookSize { get; set; } = 1024;

    /// <summary>
    /// Gets or sets the dimension of each codebook entry.
    /// </summary>
    /// <value>Defaults to 64.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The vector size of each codebook entry. Should be
    /// a fraction of <see cref="HiddenDimension"/> divided by <see cref="NumCodebooks"/>.</para>
    /// </remarks>
    public int CodebookDimension { get; set; } = 64;

    /// <summary>
    /// Gets or sets the number of parallel codebooks (product quantization).
    /// </summary>
    /// <value>Defaults to 4.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Multiple codebooks work together to represent each
    /// time step. More codebooks = more expressive representations but more parameters.</para>
    /// </remarks>
    public int NumCodebooks { get; set; } = 4;

    /// <summary>
    /// Gets or sets the dropout rate for regularization.
    /// </summary>
    /// <value>Defaults to 0.1 (10%).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Prevents overfitting during training.</para>
    /// </remarks>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the commitment loss weight for VQ training.
    /// </summary>
    /// <value>Defaults to 0.25.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Controls how strongly encoder outputs are pulled toward
    /// their nearest codebook entry. Higher values produce tighter clusters. The original
    /// VQ-VAE paper recommends 0.25.</para>
    /// </remarks>
    public double CommitmentWeight { get; set; } = 0.25;
}
