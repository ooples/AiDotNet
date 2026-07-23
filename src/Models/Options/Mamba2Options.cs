using System;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Mamba-2 (Structured State Space Duality) forecasting.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// Mamba-2 improves upon Mamba by discovering the connection between selective state space models
/// and structured masked attention (State Space Duality). This enables a more efficient SSD algorithm
/// using matrix multiplications rather than associative scans, achieving 2-8x faster training.
/// </para>
/// <para><b>For Beginners:</b> Mamba-2 is an evolution of the Mamba architecture:
///
/// <b>Key Improvements over Mamba-1:</b>
/// 1. <b>SSD Algorithm:</b> Uses matrix multiply instead of associative scan — much faster on GPUs
/// 2. <b>Multi-head Structure:</b> Like multi-head attention, enabling better capacity per parameter
/// 3. <b>Chunk-wise Processing:</b> Processes sequences in chunks for better hardware utilization
/// 4. <b>2-8x Faster Training:</b> Due to better hardware mapping
///
/// <b>Architecture:</b>
/// - Input projection to expanded dimension
/// - Multi-head structured state space blocks
/// - Chunk-wise parallel processing
/// - Output projection back to model dimension
/// </para>
/// <para>
/// <b>Reference:</b> Dao and Gu, "Transformers are SSMs: Generalized Models and Efficient Algorithms
/// Through Structured State Space Duality", 2024.
/// </para>
/// </remarks>
public class Mamba2Options<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public Mamba2Options()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public Mamba2Options(Mamba2Options<T> other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        ContextLength = other.ContextLength;
        ForecastHorizon = other.ForecastHorizon;
        ModelDimension = other.ModelDimension;
        StateDimension = other.StateDimension;
        NumHeads = other.NumHeads;
        ExpandFactor = other.ExpandFactor;
        ConvKernelSize = other.ConvKernelSize;
        ChunkSize = other.ChunkSize;
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
    /// Gets or sets the state dimension per head. Default: 64.
    /// </summary>
    public int StateDimension { get; set; } = 64;

    /// <summary>
    /// Gets or sets the number of heads for multi-head SSD. Default: 8.
    /// </summary>
    public int NumHeads { get; set; } = 8;

    /// <summary>
    /// Gets or sets the expansion factor for the inner dimension. Default: 2.
    /// </summary>
    public int ExpandFactor { get; set; } = 2;

    /// <summary>
    /// Gets or sets the convolution kernel size. Default: 4.
    /// </summary>
    public int ConvKernelSize { get; set; } = 4;

    /// <summary>
    /// Gets or sets the chunk size for SSD computation. Default: 64.
    /// </summary>
    public int ChunkSize { get; set; } = 64;

    /// <summary>
    /// Gets or sets the number of Mamba-2 layers. Default: 4.
    /// </summary>
    public int NumLayers { get; set; } = 4;

    /// <summary>
    /// Gets or sets the dropout rate for regularization. Default: 0.1.
    /// </summary>
    public double DropoutRate { get; set; } = 0.1;
}
