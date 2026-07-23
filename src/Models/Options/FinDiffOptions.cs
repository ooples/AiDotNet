namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for FinDiff, a diffusion model specialized for generating
/// realistic financial tabular data with temporal correlation preservation.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// FinDiff extends standard diffusion models with financial domain knowledge:
/// - <b>Temporal correlation loss</b>: Preserves autocorrelation and cross-correlation
/// - <b>Volatility-aware noise</b>: Adapts noise schedule to financial volatility patterns
/// - <b>Financial constraints</b>: Enforces domain rules (positive prices, valid ranges)
/// </para>
/// <para>
/// <b>For Beginners:</b> FinDiff generates realistic financial data (stocks, portfolios, etc.)
/// by understanding that financial data has special properties:
///
/// 1. Values change gradually over time (temporal correlation)
/// 2. Some values must always be positive (stock prices)
/// 3. Volatility clusters (periods of high/low market turbulence)
///
/// Example:
/// <code>
/// var options = new FinDiffOptions&lt;double&gt;
/// {
///     NumTimesteps = 500,
///     TemporalWeight = 5.0,     // How much to preserve time patterns
///     EnforcePositive = true,   // All values must be positive
///     Epochs = 300
/// };
/// var findiff = new FinDiffGenerator&lt;double&gt;(options);
/// </code>
/// </para>
/// <para>
/// Reference: "Diffusion Models for Financial Tabular Data" (2024)
/// </para>
/// </remarks>
public class FinDiffOptions<T> : RiskModelOptions<T>
{
    /// <summary>
    /// Gets or sets the number of diffusion timesteps.
    /// </summary>
    /// <value>Number of timesteps, defaulting to 500.</value>
    public int NumTimesteps { get; set; } = 500;

    /// <summary>
    /// Gets or sets the weight for the temporal correlation loss.
    /// </summary>
    /// <value>Temporal loss weight, defaulting to 5.0.</value>
    public double TemporalWeight { get; set; } = 5.0;

    /// <summary>
    /// Gets or sets whether to enforce positive values in generated data.
    /// </summary>
    /// <value>True to enforce positive values; defaults to false.</value>
    public bool EnforcePositive { get; set; } = false;

    /// <summary>
    /// Gets or sets the minimum beta for the noise schedule.
    /// </summary>
    /// <value>Beta start, defaulting to 1e-4.</value>
    public double BetaStart { get; set; } = 1e-4;

    /// <summary>
    /// Gets or sets the maximum beta for the noise schedule.
    /// </summary>
    /// <value>Beta end, defaulting to 0.02.</value>
    public double BetaEnd { get; set; } = 0.02;

    /// <summary>
    /// Gets or sets the hidden layer dimensions for the denoiser MLP.
    /// </summary>
    /// <value>Array of layer dimensions. Defaults to [256, 256].</value>
    public int[] MLPDimensions { get; set; } = [256, 256];

    /// <summary>
    /// Gets or sets the timestep embedding dimension.
    /// </summary>
    /// <value>Embedding dimension, defaulting to 128.</value>
    public int TimestepEmbeddingDimension { get; set; } = 128;

    /// <summary>
    /// Gets or sets the training batch size.
    /// </summary>
    /// <value>The batch size, defaulting to 256.</value>
    public int BatchSize { get; set; } = 256;

    /// <summary>
    /// Gets or sets the number of training epochs.
    /// </summary>
    /// <value>Number of epochs, defaulting to 300.</value>
    public int Epochs { get; set; } = 300;

    /// <summary>
    /// Gets or sets the learning rate.
    /// </summary>
    /// <value>The learning rate, defaulting to 1e-3.</value>
    public double LearningRate { get; set; } = 1e-3;

    /// <summary>
    /// Gets or sets the number of VGM modes for continuous column transformation.
    /// </summary>
    /// <value>Number of modes, defaulting to 10.</value>
    public int VGMModes { get; set; } = 10;
}
