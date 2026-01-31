using AiDotNet.Tensors;

namespace AiDotNet.Finance.Interfaces;

/// <summary>
/// Interface for financial risk models that estimate potential losses.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// Risk models are essential for financial risk management, providing estimates of potential
/// losses under various market conditions. This interface defines common methods for risk
/// measurement, stress testing, and risk decomposition.
/// </para>
/// <para>
/// <b>For Beginners:</b> Risk models help answer: "How much could I lose?"
///
/// <b>Key Risk Measures:</b>
/// - <b>Value at Risk (VaR):</b> Maximum loss at a given confidence level
/// - <b>Conditional VaR (CVaR):</b> Average loss when VaR is exceeded
/// - <b>Expected Shortfall:</b> Same as CVaR, also called tail risk
///
/// <b>Example:</b>
/// If 95% VaR is $1M, there's a 95% chance you won't lose more than $1M today.
/// The 5% CVaR tells you the average loss in the worst 5% of cases.
///
/// <b>Why Use Neural Risk Models:</b>
/// - Capture non-linear risk patterns
/// - Handle complex portfolio structures
/// - Adapt to changing market conditions
/// - Process high-dimensional data
/// </para>
/// </remarks>
public interface IRiskModel<T> : IFinancialModel<T>
{
    /// <summary>
    /// Gets the confidence level for risk calculations (e.g., 0.95 for 95% VaR).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is how confident you want to be in your risk estimate.
    /// 95% is standard (5% chance of exceeding VaR), 99% is conservative.
    /// </para>
    /// </remarks>
    double ConfidenceLevel { get; }

    /// <summary>
    /// Gets the time horizon for risk calculations in days.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> How far ahead to measure risk. 1 day is common for trading,
    /// 10 days for regulatory capital, 1 year for strategic planning.
    /// </para>
    /// </remarks>
    int TimeHorizon { get; }

    /// <summary>
    /// Calculates Value at Risk (VaR) for the given portfolio.
    /// </summary>
    /// <param name="portfolioReturns">Historical or simulated returns tensor of shape [samples, assets].</param>
    /// <param name="weights">Portfolio weights tensor of shape [assets].</param>
    /// <returns>VaR estimate as a single value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> VaR answers: "With X% confidence, what's the maximum I could lose?"
    ///
    /// Example: If 95% VaR = $10,000, then:
    /// - 95% of the time, you'll lose less than $10,000
    /// - 5% of the time, you could lose more (the "tail risk")
    ///
    /// VaR is like a speed limit for losses - most days you're under it, but occasionally you exceed it.
    /// </para>
    /// </remarks>
    T CalculateVaR(Tensor<T> portfolioReturns, Tensor<T> weights);

    /// <summary>
    /// Calculates Conditional Value at Risk (CVaR), also known as Expected Shortfall.
    /// </summary>
    /// <param name="portfolioReturns">Historical or simulated returns tensor of shape [samples, assets].</param>
    /// <param name="weights">Portfolio weights tensor of shape [assets].</param>
    /// <returns>CVaR estimate as a single value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> CVaR answers: "When things go really bad, how bad on average?"
    ///
    /// CVaR is always worse than VaR because it's the average of the worst losses.
    ///
    /// Example: If 95% VaR = $10,000 and 95% CVaR = $15,000:
    /// - 5% of days you exceed VaR
    /// - On those bad days, you lose $15,000 on average
    ///
    /// CVaR is better for risk management because it considers the severity of extreme losses.
    /// </para>
    /// </remarks>
    T CalculateCVaR(Tensor<T> portfolioReturns, Tensor<T> weights);

    /// <summary>
    /// Performs stress testing under specified scenarios.
    /// </summary>
    /// <param name="portfolioWeights">Current portfolio weights tensor of shape [assets].</param>
    /// <param name="stressScenarios">Stress scenarios tensor of shape [scenarios, assets].</param>
    /// <returns>Portfolio loss under each scenario, shape [scenarios].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Stress testing asks: "What if something extreme happens?"
    ///
    /// Scenarios might include:
    /// - 2008 financial crisis repeat
    /// - Interest rate spike
    /// - Currency crisis
    /// - Tech stock crash
    ///
    /// Unlike VaR (which uses historical data), stress testing imagines specific bad events.
    /// </para>
    /// </remarks>
    Tensor<T> StressTest(Tensor<T> portfolioWeights, Tensor<T> stressScenarios);

    /// <summary>
    /// Decomposes total risk into component contributions.
    /// </summary>
    /// <param name="portfolioReturns">Historical returns tensor of shape [samples, assets].</param>
    /// <param name="weights">Portfolio weights tensor of shape [assets].</param>
    /// <returns>Risk contribution per asset, shape [assets].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Risk decomposition answers: "Which assets contribute most to my risk?"
    ///
    /// This helps identify:
    /// - Concentration risk (too much in one asset)
    /// - Diversification benefits
    /// - Which positions to trim for risk reduction
    ///
    /// Contributions sum to total portfolio risk.
    /// </para>
    /// </remarks>
    Tensor<T> DecomposeRisk(Tensor<T> portfolioReturns, Tensor<T> weights);

    /// <summary>
    /// Estimates the probability of a given loss being exceeded.
    /// </summary>
    /// <param name="portfolioReturns">Historical returns tensor.</param>
    /// <param name="weights">Portfolio weights.</param>
    /// <param name="lossThreshold">The loss amount to test against.</param>
    /// <returns>Probability of exceeding the loss threshold.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This answers: "What's the chance I lose more than $X?"
    ///
    /// Useful for setting risk limits and capital requirements.
    /// </para>
    /// </remarks>
    T EstimateExceedanceProbability(Tensor<T> portfolioReturns, Tensor<T> weights, T lossThreshold);

    /// <summary>
    /// Gets risk-specific metrics for model evaluation.
    /// </summary>
    /// <returns>Dictionary containing risk model metrics.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Metrics for evaluating risk models include:
    /// - <b>Backtesting exceptions:</b> How often VaR was exceeded
    /// - <b>Kupiec test:</b> Statistical test for VaR accuracy
    /// - <b>Christoffersen test:</b> Tests for clustered exceptions
    /// - <b>Lopez score:</b> Measures exception severity
    /// </para>
    /// </remarks>
    Dictionary<string, T> GetRiskMetrics();
}
