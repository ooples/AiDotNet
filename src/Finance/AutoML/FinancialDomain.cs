namespace AiDotNet.Finance.AutoML;

/// <summary>
/// Defines which finance sub-domain AutoML should focus on.
/// </summary>
/// <remarks>
/// <para>
/// This helps AutoML choose appropriate candidate models and default metrics.
/// </para>
/// <para>
/// <b>For Beginners:</b> Pick the area you care about:
/// - Forecasting: predict future time series values
/// - Risk: estimate risk measures like VaR
/// </para>
/// </remarks>
public enum FinancialDomain
{
    /// <summary>
    /// Time series forecasting models (predict future values).
    /// </summary>
    Forecasting,

    /// <summary>
    /// Risk management models (estimate risk metrics).
    /// </summary>
    Risk
}
