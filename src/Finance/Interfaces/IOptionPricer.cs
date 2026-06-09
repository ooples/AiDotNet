namespace AiDotNet.Finance.Interfaces;

/// <summary>
/// A European-option pricing engine: fair value, first-order Greeks, and implied volatility.
/// </summary>
/// <remarks>
/// <para>
/// This is a customization point, not a trainable model. Models that need to price options or
/// hedge option exposure (e.g. an options-focused RL trading agent, a covered-call portfolio model)
/// depend on this interface and default to the closed-form <see cref="AiDotNet.Finance.Options.BlackScholes{T}"/>
/// implementation, but callers can substitute their own pricer (binomial tree, Heston, a learned
/// surface, …) without changing the consuming model.
/// </para>
/// <para><b>For Beginners:</b> An "option pricer" answers "what is this option worth, and how does that
/// value move when the market moves?" The default is the Black-Scholes formula; swap in your own if you
/// need a different model.</para>
/// </remarks>
/// <typeparam name="T">Numeric type (float/double).</typeparam>
public interface IOptionPricer<T>
{
    /// <summary>Fair value of a European call (<paramref name="isCall"/> = true) or put.</summary>
    T Price(T spot, T strike, T timeToExpiry, T riskFreeRate, T volatility, bool isCall);

    /// <summary>Delta — sensitivity of price to the underlying spot.</summary>
    T Delta(T spot, T strike, T timeToExpiry, T riskFreeRate, T volatility, bool isCall);

    /// <summary>Gamma — sensitivity of delta to the underlying spot (same for calls and puts).</summary>
    T Gamma(T spot, T strike, T timeToExpiry, T riskFreeRate, T volatility);

    /// <summary>Vega — sensitivity of price to volatility (same for calls and puts).</summary>
    T Vega(T spot, T strike, T timeToExpiry, T riskFreeRate, T volatility);

    /// <summary>Theta — sensitivity of price to the passage of time.</summary>
    T Theta(T spot, T strike, T timeToExpiry, T riskFreeRate, T volatility, bool isCall);

    /// <summary>Rho — sensitivity of price to the risk-free rate.</summary>
    T Rho(T spot, T strike, T timeToExpiry, T riskFreeRate, T volatility, bool isCall);

    /// <summary>The implied volatility that reproduces <paramref name="marketPrice"/> under this pricer.</summary>
    T ImpliedVolatility(
        T marketPrice, T spot, T strike, T timeToExpiry, T riskFreeRate, bool isCall,
        int maxIterations = 100, double tolerance = 1e-8);
}
