using System;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration for <see cref="AiDotNet.Finance.Portfolio.GraphAttentionPortfolio{T}"/>.
/// </summary>
/// <remarks>
/// <para>
/// Defaults follow Kamesh Korangi, Christophe Mues and Cristián Bravo, "Large-scale Time-Varying
/// Portfolio Optimisation using Graph Attention Networks" (arXiv:2407.15532): attention feature width
/// 24, 8 heads, a 30-day volatility lookback, and a three-year correlation window.
/// </para>
/// <para><b>For Beginners:</b> This model builds a network of which companies move together, learns
/// over that network which ones to hold, and is trained to maximize return per unit of risk.</para>
/// </remarks>
/// <typeparam name="T">The numeric type.</typeparam>
public class GraphAttentionPortfolioOptions<T> : NeuralNetworkOptions
{
    /// <summary>Gets or sets the number of assets. Default 30.</summary>
    /// <remarks>
    /// The paper runs at ~5,000 firms per window out of 16,793 over 1990-2021; 30 is a tractable
    /// default here, not a paper value.
    /// </remarks>
    public int NumAssets { get; set; } = 30;

    /// <summary>
    /// Gets or sets the volatility lookback in days. Default 30, the paper's value.
    /// </summary>
    /// <remarks>
    /// Correlations are computed between return-VOLATILITY series rather than returns, following
    /// Diebold and Yilmaz. Volatility series co-move sharply in risk-off periods and weakly otherwise,
    /// which is the structure the graph is meant to capture.
    /// </remarks>
    public int VolatilityLookback { get; set; } = 30;

    /// <summary>
    /// Gets or sets the correlation window in trading days. Default 756, about the paper's three years.
    /// </summary>
    public int CorrelationWindow { get; set; } = 756;

    /// <summary>
    /// Gets or sets T', the per-head attention feature width. Default 24, the paper's value.
    /// </summary>
    public int AttentionFeatureDimension { get; set; } = 24;

    /// <summary>Gets or sets K, the number of attention heads. Default 8, the paper's value.</summary>
    /// <remarks>
    /// Head outputs are CONCATENATED, so the representation after attention is
    /// <c>K * AttentionFeatureDimension</c> wide rather than <c>AttentionFeatureDimension</c>.
    /// </remarks>
    public int NumHeads { get; set; } = 8;

    /// <summary>Gets or sets the LeakyReLU negative slope in the attention. Default 0.2.</summary>
    /// <remarks>
    /// From the original GAT formulation. A slope of 0 makes it a plain ReLU and zeroes the gradient
    /// for negatively scored pairs, so those edges can never recover.
    /// </remarks>
    public double LeakyReLUSlope { get; set; } = 0.2;

    /// <summary>Gets or sets the dropout rate applied after the first feed-forward block. Default 0.1.</summary>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the L1 (LASSO) strength on the final scoring layer. Default 1e-3.
    /// </summary>
    /// <remarks>
    /// This is the SPARSITY driver. The paper applies LASSO to the second feed-forward network "to
    /// further shrink and eliminate unnecessary weights"; because the allocation layer merely
    /// sum-normalizes, a score driven to zero becomes a weight of exactly zero and the firm leaves the
    /// portfolio. With this at zero the scores stay dense and the allocation spreads across everything,
    /// which is the outcome the paper chose its allocation layer to avoid. The paper does not state a
    /// numeric strength, so this default is ours.
    /// </remarks>
    public double L1Regularization { get; set; } = 1e-3;

    /// <summary>Gets or sets the learning rate. Default 1e-3.</summary>
    /// <remarks>The paper does not state one; this is a conventional Adam default.</remarks>
    public double LearningRate { get; set; } = 1e-3;

    /// <summary>Gets or sets the batch size. Default 64.</summary>
    /// <remarks>The paper does not state one; this is a conventional default.</remarks>
    public int BatchSize { get; set; } = 64;

    /// <summary>Gets or sets the maximum training epochs. Default 100.</summary>
    public int MaxEpochs { get; set; } = 100;

    /// <summary>Validates the configuration.</summary>
    /// <exception cref="ArgumentOutOfRangeException">A value cannot describe a usable model.</exception>
    public void Validate()
    {
        RequirePositive(NumAssets, nameof(NumAssets));
        RequirePositive(CorrelationWindow, nameof(CorrelationWindow));
        RequirePositive(AttentionFeatureDimension, nameof(AttentionFeatureDimension));
        RequirePositive(NumHeads, nameof(NumHeads));
        RequirePositive(BatchSize, nameof(BatchSize));
        RequirePositive(MaxEpochs, nameof(MaxEpochs));

        if (VolatilityLookback < 2)
            throw new ArgumentOutOfRangeException(nameof(VolatilityLookback), VolatilityLookback,
                "A standard deviation needs at least 2 observations.");

        // TMFG needs a triangular face to grow from, so a graph of fewer than 3 nodes cannot be built.
        if (NumAssets < 3)
            throw new ArgumentOutOfRangeException(nameof(NumAssets), NumAssets,
                "At least 3 assets are required to build a filtered graph.");

        if (LeakyReLUSlope < 0.0 || double.IsNaN(LeakyReLUSlope))
            throw new ArgumentOutOfRangeException(nameof(LeakyReLUSlope), LeakyReLUSlope,
                "LeakyReLUSlope cannot be negative or NaN.");
        if (DropoutRate is < 0.0 or >= 1.0 || double.IsNaN(DropoutRate))
            throw new ArgumentOutOfRangeException(nameof(DropoutRate), DropoutRate,
                "DropoutRate must be in [0, 1).");
        if (L1Regularization < 0.0 || double.IsNaN(L1Regularization))
            throw new ArgumentOutOfRangeException(nameof(L1Regularization), L1Regularization,
                "L1Regularization cannot be negative.");
        if (LearningRate <= 0.0 || double.IsNaN(LearningRate))
            throw new ArgumentOutOfRangeException(nameof(LearningRate), LearningRate,
                "LearningRate must be positive.");
        if (CorrelationWindow < VolatilityLookback)
            throw new ArgumentOutOfRangeException(nameof(CorrelationWindow), CorrelationWindow,
                $"CorrelationWindow must be at least VolatilityLookback ({VolatilityLookback}); " +
                "otherwise there is no volatility series to correlate.");
    }

    private static void RequirePositive(int value, string name)
    {
        if (value <= 0)
            throw new ArgumentOutOfRangeException(name, value, $"{name} must be positive.");
    }
}
