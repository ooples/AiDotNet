using System;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Finance.Options;

/// <summary>
/// Closed-form Black-Scholes-Merton European option pricing, the full first-order Greeks
/// (delta, gamma, vega, theta, rho) and implied-volatility solve.
/// </summary>
/// <remarks>
/// <para>
/// AiDotNet ships <see cref="AiDotNet.PhysicsInformed.PDEs.BlackScholesEquation{T}"/> — the PDE
/// <i>residual</i> for a PINN solver — but no analytic pricer. This is the cheap, exact closed form for
/// European options: pricing + Greeks in O(1) with no training, which is what trading/sizing/hedging
/// actually needs. Feed a GARCH / forecast volatility in as <c>volatility</c>.
/// </para>
/// <para><b>For Beginners:</b> The Black-Scholes formula gives the fair price of a European call or put
/// option from five inputs: the current price (spot), the strike, time to expiry (in years), the
/// risk-free interest rate, and the volatility. The "Greeks" are its sensitivities — delta (price vs.
/// spot), gamma (delta vs. spot), vega (price vs. volatility), theta (price vs. time), rho (price vs.
/// rate) — which traders use to size positions and hedge risk.</para>
/// </remarks>
/// <typeparam name="T">Numeric type (float/double).</typeparam>
public static class BlackScholes<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>Black-Scholes price of a European call (<paramref name="isCall"/> = true) or put.</summary>
    /// <param name="spot">Current underlying price S.</param>
    /// <param name="strike">Strike price K.</param>
    /// <param name="timeToExpiry">Time to expiry T in years.</param>
    /// <param name="riskFreeRate">Continuously-compounded risk-free rate r.</param>
    /// <param name="volatility">Annualized volatility σ.</param>
    /// <param name="isCall">True for a call, false for a put.</param>
    public static T Price(T spot, T strike, T timeToExpiry, T riskFreeRate, T volatility, bool isCall)
    {
        // At/after expiry the value collapses to the (undiscounted) intrinsic.
        if (!NumOps.GreaterThan(timeToExpiry, NumOps.Zero))
        {
            return Intrinsic(spot, strike, isCall);
        }

        // Zero-volatility limit at positive maturity: the underlying grows
        // deterministically at the risk-free rate, so the payoff is the forward
        // intrinsic against the DISCOUNTED strike, max(S - K·e^(-rT), 0) for a
        // call (put analog). Returning the undiscounted intrinsic here is wrong
        // whenever r != 0.
        if (!NumOps.GreaterThan(volatility, NumOps.Zero))
        {
            var discounted = NumOps.Multiply(strike, Discount(riskFreeRate, timeToExpiry));
            var payoff = isCall ? NumOps.Subtract(spot, discounted) : NumOps.Subtract(discounted, spot);
            return NumOps.GreaterThan(payoff, NumOps.Zero) ? payoff : NumOps.Zero;
        }

        var (d1, d2) = D1D2(spot, strike, timeToExpiry, riskFreeRate, volatility);
        var discountedStrike = NumOps.Multiply(strike, Discount(riskFreeRate, timeToExpiry));

        if (isCall)
        {
            // C = S·N(d1) − K·e^(−rT)·N(d2)
            return NumOps.Subtract(
                NumOps.Multiply(spot, NormalCdf(d1)),
                NumOps.Multiply(discountedStrike, NormalCdf(d2)));
        }

        // P = K·e^(−rT)·N(−d2) − S·N(−d1)
        return NumOps.Subtract(
            NumOps.Multiply(discountedStrike, NormalCdf(NumOps.Negate(d2))),
            NumOps.Multiply(spot, NormalCdf(NumOps.Negate(d1))));
    }

    /// <summary>Delta — ∂Price/∂Spot. Call: N(d1); Put: N(d1) − 1.</summary>
    public static T Delta(T spot, T strike, T timeToExpiry, T riskFreeRate, T volatility, bool isCall)
    {
        if (!NumOps.GreaterThan(timeToExpiry, NumOps.Zero) || !NumOps.GreaterThan(volatility, NumOps.Zero))
        {
            var itm = isCall ? NumOps.GreaterThan(spot, strike) : NumOps.LessThan(spot, strike);
            return itm ? (isCall ? NumOps.One : NumOps.Negate(NumOps.One)) : NumOps.Zero;
        }

        var (d1, _) = D1D2(spot, strike, timeToExpiry, riskFreeRate, volatility);
        var nd1 = NormalCdf(d1);
        return isCall ? nd1 : NumOps.Subtract(nd1, NumOps.One);
    }

    /// <summary>Gamma — ∂²Price/∂Spot² (same for calls and puts): φ(d1) / (S·σ·√T).</summary>
    public static T Gamma(T spot, T strike, T timeToExpiry, T riskFreeRate, T volatility)
    {
        if (!NumOps.GreaterThan(timeToExpiry, NumOps.Zero) || !NumOps.GreaterThan(volatility, NumOps.Zero))
        {
            return NumOps.Zero;
        }

        var (d1, _) = D1D2(spot, strike, timeToExpiry, riskFreeRate, volatility);
        var denom = NumOps.Multiply(NumOps.Multiply(spot, volatility), NumOps.Sqrt(timeToExpiry));
        return NumOps.Divide(NormalPdf(d1), denom);
    }

    /// <summary>Vega — ∂Price/∂σ (per 1.0 of vol, same for calls and puts): S·φ(d1)·√T.</summary>
    public static T Vega(T spot, T strike, T timeToExpiry, T riskFreeRate, T volatility)
    {
        if (!NumOps.GreaterThan(timeToExpiry, NumOps.Zero) || !NumOps.GreaterThan(volatility, NumOps.Zero))
        {
            return NumOps.Zero;
        }

        var (d1, _) = D1D2(spot, strike, timeToExpiry, riskFreeRate, volatility);
        return NumOps.Multiply(NumOps.Multiply(spot, NormalPdf(d1)), NumOps.Sqrt(timeToExpiry));
    }

    /// <summary>Theta — ∂Price/∂t (per year; typically negative). Call/put differ in the rate term.</summary>
    public static T Theta(T spot, T strike, T timeToExpiry, T riskFreeRate, T volatility, bool isCall)
    {
        if (!NumOps.GreaterThan(timeToExpiry, NumOps.Zero) || !NumOps.GreaterThan(volatility, NumOps.Zero))
        {
            return NumOps.Zero;
        }

        var (d1, d2) = D1D2(spot, strike, timeToExpiry, riskFreeRate, volatility);
        // −S·φ(d1)·σ / (2·√T)
        var term1 = NumOps.Negate(NumOps.Divide(
            NumOps.Multiply(NumOps.Multiply(spot, NormalPdf(d1)), volatility),
            NumOps.Multiply(NumOps.FromDouble(2.0), NumOps.Sqrt(timeToExpiry))));
        var rateTerm = NumOps.Multiply(NumOps.Multiply(riskFreeRate, strike), Discount(riskFreeRate, timeToExpiry));

        return isCall
            ? NumOps.Subtract(term1, NumOps.Multiply(rateTerm, NormalCdf(d2)))
            : NumOps.Add(term1, NumOps.Multiply(rateTerm, NormalCdf(NumOps.Negate(d2))));
    }

    /// <summary>Rho — ∂Price/∂r (per 1.0 of rate). Call: K·T·e^(−rT)·N(d2); Put: −K·T·e^(−rT)·N(−d2).</summary>
    public static T Rho(T spot, T strike, T timeToExpiry, T riskFreeRate, T volatility, bool isCall)
    {
        if (!NumOps.GreaterThan(timeToExpiry, NumOps.Zero) || !NumOps.GreaterThan(volatility, NumOps.Zero))
        {
            return NumOps.Zero;
        }

        var (_, d2) = D1D2(spot, strike, timeToExpiry, riskFreeRate, volatility);
        var ktDiscount = NumOps.Multiply(NumOps.Multiply(strike, timeToExpiry), Discount(riskFreeRate, timeToExpiry));
        return isCall
            ? NumOps.Multiply(ktDiscount, NormalCdf(d2))
            : NumOps.Negate(NumOps.Multiply(ktDiscount, NormalCdf(NumOps.Negate(d2))));
    }

    /// <summary>
    /// Implied volatility from a market option price, via Newton-Raphson seeded with Brenner-Subrahmanyam,
    /// falling back to bisection if a Newton step leaves the bracket. Returns the σ such that
    /// <see cref="Price"/> matches <paramref name="marketPrice"/> within tolerance.
    /// </summary>
    public static T ImpliedVolatility(
        T marketPrice, T spot, T strike, T timeToExpiry, T riskFreeRate, bool isCall,
        int maxIterations = 100, double tolerance = 1e-8)
    {
        if (!NumOps.GreaterThan(timeToExpiry, NumOps.Zero))
            throw new ArgumentOutOfRangeException(nameof(timeToExpiry), "timeToExpiry must be > 0 for implied volatility.");

        double price = NumOps.ToDouble(marketPrice);
        double s = NumOps.ToDouble(spot), k = NumOps.ToDouble(strike), t = NumOps.ToDouble(timeToExpiry);
        double r = NumOps.ToDouble(riskFreeRate);

        // Reject prices outside the no-arbitrage bounds up front: a European option
        // price must lie in [max(±(S − K·e^(−rT)), 0), upper] where upper is S for a
        // call and K·e^(−rT) for a put — otherwise no real σ reproduces it.
        double discountedK = k * Math.Exp(-r * t);
        double lowerBound = isCall ? Math.Max(s - discountedK, 0.0) : Math.Max(discountedK - s, 0.0);
        double upperBound = isCall ? s : discountedK;
        if (double.IsNaN(price) || double.IsInfinity(price) || price < lowerBound || price > upperBound)
            throw new ArgumentOutOfRangeException(nameof(marketPrice),
                "marketPrice violates the European no-arbitrage bounds; no implied volatility exists.");

        // Brenner-Subrahmanyam closed-form seed for at-the-money, bracketed to a sane vol range.
        // (Math.Max/Min instead of Math.Clamp, explicit finite check instead of double.IsFinite: net471.)
        double sigma = Math.Sqrt(2.0 * Math.PI / Math.Max(t, 1e-12)) * (price / Math.Max(s, 1e-12));
        var seed = !double.IsNaN(sigma) && !double.IsInfinity(sigma) ? sigma : 0.2;
        sigma = Math.Max(1e-4, Math.Min(5.0, seed));
        double low = 1e-4, high = 5.0;
        var converged = false;

        for (var i = 0; i < maxIterations; i++)
        {
            double modelPrice = NumOps.ToDouble(Price(spot, strike, timeToExpiry,
                riskFreeRate, NumOps.FromDouble(sigma), isCall));
            double diff = modelPrice - price;
            if (Math.Abs(diff) < tolerance)
            {
                converged = true;
                break;
            }

            // Bracket update (price is monotone increasing in σ).
            if (diff > 0) high = sigma; else low = sigma;

            double vega = NumOps.ToDouble(Vega(spot, strike, timeToExpiry, riskFreeRate, NumOps.FromDouble(sigma)));
            double next = vega > 1e-12 ? sigma - diff / vega : (low + high) / 2.0;
            sigma = (next > low && next < high) ? next : (low + high) / 2.0;
        }

        // Fail loudly rather than silently returning an unconverged σ in production.
        if (!converged)
            throw new InvalidOperationException(
                "Implied-volatility solver did not converge within maxIterations; the price may be too close to a bound.");

        return NumOps.FromDouble(sigma);
    }

    private static (T d1, T d2) D1D2(T spot, T strike, T timeToExpiry, T riskFreeRate, T volatility)
    {
        if (!NumOps.GreaterThan(spot, NumOps.Zero))
            throw new ArgumentOutOfRangeException(nameof(spot), "spot must be > 0.");
        if (!NumOps.GreaterThan(strike, NumOps.Zero))
            throw new ArgumentOutOfRangeException(nameof(strike), "strike must be > 0.");
        var sqrtT = NumOps.Sqrt(timeToExpiry);
        var volSqrtT = NumOps.Multiply(volatility, sqrtT);
        // d1 = [ln(S/K) + (r + σ²/2)·T] / (σ·√T)
        var lnMoneyness = NumOps.Log(NumOps.Divide(spot, strike));
        var drift = NumOps.Multiply(
            NumOps.Add(riskFreeRate, NumOps.Multiply(NumOps.FromDouble(0.5), NumOps.Multiply(volatility, volatility))),
            timeToExpiry);
        var d1 = NumOps.Divide(NumOps.Add(lnMoneyness, drift), volSqrtT);
        var d2 = NumOps.Subtract(d1, volSqrtT);
        return (d1, d2);
    }

    private static T Discount(T riskFreeRate, T timeToExpiry) =>
        NumOps.Exp(NumOps.Negate(NumOps.Multiply(riskFreeRate, timeToExpiry)));

    private static T Intrinsic(T spot, T strike, bool isCall)
    {
        var diff = isCall ? NumOps.Subtract(spot, strike) : NumOps.Subtract(strike, spot);
        return NumOps.GreaterThan(diff, NumOps.Zero) ? diff : NumOps.Zero;
    }

    /// <summary>Standard normal PDF φ(x) = e^(−x²/2) / √(2π).</summary>
    private static T NormalPdf(T x)
    {
        var invSqrt2Pi = NumOps.FromDouble(0.3989422804014327); // 1/√(2π)
        var exponent = NumOps.Negate(NumOps.Multiply(NumOps.FromDouble(0.5), NumOps.Multiply(x, x)));
        return NumOps.Multiply(invSqrt2Pi, NumOps.Exp(exponent));
    }

    /// <summary>Standard normal CDF N(x) = ½·(1 + erf(x/√2)).</summary>
    private static T NormalCdf(T x)
    {
        var arg = NumOps.Divide(x, NumOps.FromDouble(1.4142135623730951)); // √2
        return NumOps.Multiply(NumOps.FromDouble(0.5), NumOps.Add(NumOps.One, Erf(arg)));
    }

    /// <summary>Abramowitz &amp; Stegun 7.1.26 erf approximation (|error| &lt; 1.5e-7).</summary>
    private static T Erf(T x)
    {
        var sign = NumOps.GreaterThanOrEquals(x, NumOps.Zero) ? NumOps.One : NumOps.Negate(NumOps.One);
        var ax = NumOps.Abs(x);

        var t = NumOps.Divide(NumOps.One,
            NumOps.Add(NumOps.One, NumOps.Multiply(NumOps.FromDouble(0.3275911), ax)));

        // Horner over (a1..a5)
        var poly = NumOps.FromDouble(1.061405429);
        poly = NumOps.Add(NumOps.FromDouble(-1.453152027), NumOps.Multiply(poly, t));
        poly = NumOps.Add(NumOps.FromDouble(1.421413741), NumOps.Multiply(poly, t));
        poly = NumOps.Add(NumOps.FromDouble(-0.284496736), NumOps.Multiply(poly, t));
        poly = NumOps.Add(NumOps.FromDouble(0.254829592), NumOps.Multiply(poly, t));
        poly = NumOps.Multiply(poly, t);

        var expTerm = NumOps.Exp(NumOps.Negate(NumOps.Multiply(ax, ax)));
        var erf = NumOps.Subtract(NumOps.One, NumOps.Multiply(poly, expTerm));
        return NumOps.Multiply(sign, erf);
    }
}
