using AiDotNet.Interfaces;

namespace AiDotNet.Finance.Interfaces;

/// <summary>
/// Interface for financial factor models that learn latent factors from market data.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// Factor models decompose asset returns into systematic factors and idiosyncratic noise.
/// This interface extends <see cref="IFinancialModel{T}"/> with factor-specific capabilities
/// for extracting, analyzing, and using latent factors in quantitative finance.
/// </para>
/// <para>
/// <b>For Beginners:</b> Factor models help explain why asset prices move together.
///
/// <b>The Key Insight:</b>
/// Asset returns can be broken down into:
/// - <b>Factor returns:</b> Movements explained by common factors (market, size, value, momentum, etc.)
/// - <b>Idiosyncratic returns:</b> Stock-specific movements not explained by factors
///
/// For example, if tech stocks all go up together, that's likely a "tech sector factor."
/// If just Apple goes up because of a product launch, that's idiosyncratic to Apple.
///
/// <b>Why Use Factor Models:</b>
/// - Risk decomposition: Understand what risks drive your portfolio
/// - Alpha generation: Find factors that predict future returns
/// - Portfolio construction: Build portfolios with desired factor exposures
/// - Risk management: Hedge specific factor risks
///
/// <b>Common Factor Categories:</b>
/// - Style factors: Value, growth, momentum, quality, volatility
/// - Macro factors: Interest rates, inflation, GDP growth
/// - Statistical factors: PCA-derived factors, machine-learned factors
/// - Fundamental factors: Earnings, book value, leverage
/// </para>
/// </remarks>
public interface IFactorModel<T> : IFinancialModel<T>
{
    /// <summary>
    /// Gets the number of latent factors the model learns or uses.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is how many underlying "drivers" the model identifies.
    /// More factors can capture more nuance but may overfit. Common values: 3-20 factors.
    /// </para>
    /// </remarks>
    int NumFactors { get; }

    /// <summary>
    /// Gets the number of assets (securities) the model can handle.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the universe of stocks/assets the model analyzes.
    /// Could be 500 for S&amp;P 500, 3000 for Russell 3000, etc.
    /// </para>
    /// </remarks>
    int NumAssets { get; }

    /// <summary>
    /// Extracts latent factors from market data.
    /// </summary>
    /// <param name="returns">Asset returns tensor of shape [batch_size, sequence_length, num_assets].</param>
    /// <returns>Factor values tensor of shape [batch_size, sequence_length, num_factors].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Given historical returns data, this method identifies the
    /// underlying factors that explain the returns. Think of it like finding the
    /// "hidden causes" behind price movements.
    /// </para>
    /// </remarks>
    Tensor<T> ExtractFactors(Tensor<T> returns);

    /// <summary>
    /// Computes factor loadings (exposures) for each asset.
    /// </summary>
    /// <param name="returns">Asset returns tensor of shape [batch_size, sequence_length, num_assets].</param>
    /// <returns>Factor loadings matrix of shape [num_assets, num_factors].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Factor loadings tell you how sensitive each stock is to each factor.
    /// A high loading on the "momentum" factor means the stock follows momentum trends closely.
    /// A negative loading means the stock moves opposite to the factor.
    /// </para>
    /// </remarks>
    Tensor<T> GetFactorLoadings(Tensor<T> returns);

    /// <summary>
    /// Predicts expected returns based on factor exposures.
    /// </summary>
    /// <param name="factorExposures">Factor exposure tensor of shape [batch_size, num_factors].</param>
    /// <returns>Expected returns tensor of shape [batch_size, num_assets].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> If you know the factor values for tomorrow, this predicts
    /// what each stock's return should be based on its factor loadings.
    ///
    /// Expected Return = Sum(Factor_i * Loading_i) for each factor i
    /// </para>
    /// </remarks>
    Tensor<T> PredictReturns(Tensor<T> factorExposures);

    /// <summary>
    /// Computes the factor covariance matrix.
    /// </summary>
    /// <param name="returns">Historical returns tensor of shape [batch_size, sequence_length, num_assets].</param>
    /// <returns>Factor covariance matrix of shape [num_factors, num_factors].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The factor covariance matrix shows how factors move together.
    /// This is crucial for risk management - if two factors are highly correlated,
    /// a portfolio exposed to both has concentrated risk.
    /// </para>
    /// </remarks>
    Tensor<T> GetFactorCovariance(Tensor<T> returns);

    /// <summary>
    /// Computes alpha (expected excess return) for each asset.
    /// </summary>
    /// <param name="returns">Historical returns tensor of shape [batch_size, sequence_length, num_assets].</param>
    /// <param name="factorReturns">Factor returns tensor of shape [batch_size, sequence_length, num_factors].</param>
    /// <returns>Alpha values tensor of shape [num_assets].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Alpha is the return a stock generates beyond what's explained
    /// by its factor exposures. Positive alpha means the stock outperforms expectations.
    /// Finding stocks with high alpha is the holy grail of active investing.
    /// </para>
    /// </remarks>
    Tensor<T> ComputeAlpha(Tensor<T> returns, Tensor<T> factorReturns);

    /// <summary>
    /// Gets factor-specific performance metrics.
    /// </summary>
    /// <returns>Dictionary containing factor model metrics.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Metrics specific to factor models include:
    /// - <b>R-squared:</b> How much of the return variance is explained by factors
    /// - <b>Factor IC:</b> Information coefficient - how predictive are the factors
    /// - <b>Factor turnover:</b> How much factor exposures change over time
    /// - <b>Factor correlation:</b> How independent are the factors
    /// </para>
    /// </remarks>
    Dictionary<string, T> GetFactorMetrics();
}
