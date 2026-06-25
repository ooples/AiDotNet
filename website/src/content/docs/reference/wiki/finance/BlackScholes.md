---
title: "BlackScholes<T>"
description: "Closed-form Black-Scholes-Merton European option pricing, the full first-order Greeks (delta, gamma, vega, theta, rho) and implied-volatility solve."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Options`

Closed-form Black-Scholes-Merton European option pricing, the full first-order Greeks
(delta, gamma, vega, theta, rho) and implied-volatility solve.

## For Beginners

The Black-Scholes formula gives the fair price of a European call or put
option from five inputs: the current price (spot), the strike, time to expiry (in years), the
risk-free interest rate, and the volatility. The "Greeks" are its sensitivities — delta (price vs.
spot), gamma (delta vs. spot), vega (price vs. volatility), theta (price vs. time), rho (price vs.
rate) — which traders use to size positions and hedge risk.

## How It Works

AiDotNet ships `BlackScholesEquation` — the PDE
*residual* for a PINN solver — but no analytic pricer. This is the cheap, exact closed form for
European options: pricing + Greeks in O(1) with no training, which is what trading/sizing/hedging
actually needs. Feed a GARCH / forecast volatility in as `volatility`.

## Properties

| Property | Summary |
|:-----|:--------|
| `Default` | Shared stateless default instance for injection as an `IOptionPricer`. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Delta(,,,,,Boolean)` | Delta — ∂Price/∂Spot. |
| `Erf()` | Abramowitz & Stegun 7.1.26 erf approximation (\|error\| < 1.5e-7). |
| `Gamma(,,,,)` | Gamma — ∂²Price/∂Spot² (same for calls and puts): φ(d1) / (S·σ·√T). |
| `ImpliedVolatility(,,,,,Boolean,Int32,Double)` | Implied volatility from a market option price, via Newton-Raphson seeded with Brenner-Subrahmanyam, falling back to bisection if a Newton step leaves the bracket. |
| `NormalCdf()` | Standard normal CDF N(x) = ½·(1 + erf(x/√2)). |
| `NormalPdf()` | Standard normal PDF φ(x) = e^(−x²/2) / √(2π). |
| `Price(,,,,,Boolean)` | Black-Scholes price of a European call (`isCall` = true) or put. |
| `Rho(,,,,,Boolean)` | Rho — ∂Price/∂r (per 1.0 of rate). |
| `Theta(,,,,,Boolean)` | Theta — ∂Price/∂t (per year; typically negative). |
| `Vega(,,,,)` | Vega — ∂Price/∂σ (per 1.0 of vol, same for calls and puts): S·φ(d1)·√T. |

