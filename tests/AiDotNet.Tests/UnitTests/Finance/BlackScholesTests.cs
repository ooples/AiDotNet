using System;
using AiDotNet.Finance.Options;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Finance;

/// <summary>
/// Verifies the closed-form Black-Scholes pricer/Greeks against well-known textbook values for the
/// canonical case S=100, K=100, T=1, r=0.05, σ=0.20 (e.g. Hull, "Options, Futures, and Other Derivatives").
/// </summary>
public class BlackScholesTests
{
    private const double S = 100.0, K = 100.0, T = 1.0, R = 0.05, Sigma = 0.20;
    private const double Tol = 1e-3;

    [Fact]
    public void Call_price_matches_textbook()
    {
        var call = BlackScholes<double>.Price(S, K, T, R, Sigma, isCall: true);
        Assert.Equal(10.4506, call, 3);
    }

    [Fact]
    public void Put_price_matches_textbook()
    {
        var put = BlackScholes<double>.Price(S, K, T, R, Sigma, isCall: false);
        Assert.Equal(5.5735, put, 3);
    }

    [Fact]
    public void Put_call_parity_holds()
    {
        var call = BlackScholes<double>.Price(S, K, T, R, Sigma, isCall: true);
        var put = BlackScholes<double>.Price(S, K, T, R, Sigma, isCall: false);
        // C - P = S - K·e^(-rT)
        var parity = S - K * Math.Exp(-R * T);
        Assert.Equal(parity, call - put, 6);
    }

    [Fact]
    public void Greeks_match_textbook()
    {
        Assert.Equal(0.6368, BlackScholes<double>.Delta(S, K, T, R, Sigma, isCall: true), 3);
        Assert.Equal(-0.3632, BlackScholes<double>.Delta(S, K, T, R, Sigma, isCall: false), 3); // N(d1) - 1
        Assert.Equal(0.018762, BlackScholes<double>.Gamma(S, K, T, R, Sigma), 5);
        Assert.Equal(37.524, BlackScholes<double>.Vega(S, K, T, R, Sigma), 2); // per 1.0 of vol
    }

    [Fact]
    public void Implied_vol_round_trips()
    {
        var price = BlackScholes<double>.Price(S, K, T, R, Sigma, isCall: true);
        var iv = BlackScholes<double>.ImpliedVolatility(price, S, K, T, R, isCall: true);
        Assert.Equal(Sigma, iv, 4);
    }

    [Fact]
    public void Expired_option_is_intrinsic()
    {
        Assert.Equal(5.0, BlackScholes<double>.Price(105, 100, 0.0, R, Sigma, isCall: true), Tol);
        Assert.Equal(0.0, BlackScholes<double>.Price(95, 100, 0.0, R, Sigma, isCall: true), Tol);
    }

    [Fact]
    public void Zero_volatility_at_positive_maturity_is_discounted_forward_intrinsic()
    {
        // With σ = 0 and T > 0 the underlying grows deterministically at r, so the payoff is the
        // intrinsic against the DISCOUNTED strike: call = max(S - K·e^(-rT), 0), put = max(K·e^(-rT) - S, 0).
        // (Returning the UNDISCOUNTED intrinsic here would be wrong whenever r != 0.)
        var discountedK = K * Math.Exp(-R * T); // 100·e^(-0.05) ≈ 95.1229

        // ITM call: 105 - 95.1229 ≈ 9.8771 (NOT the undiscounted 5.0).
        Assert.Equal(105.0 - discountedK, BlackScholes<double>.Price(105, K, T, R, 0.0, isCall: true), Tol);
        // OTM call (spot below discounted strike) floors at 0.
        Assert.Equal(0.0, BlackScholes<double>.Price(90, K, T, R, 0.0, isCall: true), Tol);
        // ITM put: 95.1229 - 90 ≈ 5.1229.
        Assert.Equal(discountedK - 90.0, BlackScholes<double>.Price(90, K, T, R, 0.0, isCall: false), Tol);
    }
}
