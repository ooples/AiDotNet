---
title: "BlackScholesEquation<T>"
description: "Represents the Black-Scholes Equation for option pricing: ∂V/∂t + ½σ²S²∂²V/∂S² + rS∂V/∂S - rV = 0"
section: "API Reference"
---

`Models & Types` · `AiDotNet.PhysicsInformed.PDEs`

Represents the Black-Scholes Equation for option pricing:
∂V/∂t + ½σ²S²∂²V/∂S² + rS∂V/∂S - rV = 0

## How It Works

For Beginners:
The Black-Scholes equation is the fundamental equation in mathematical finance for pricing options.

Variables:

- V(S,t) = Option price (value of the derivative)
- S = Current stock price (the underlying asset)
- t = Time (usually time to expiration)
- σ (sigma) = Volatility of the stock (how much the price fluctuates)
- r = Risk-free interest rate

Physical/Financial Interpretation:

- The equation balances the change in option value over time with:
* The effect of stock price changes (delta hedging)
* The effect of volatility (gamma)
* The time value of money (discounting)

Key Insight:
Under certain assumptions (no arbitrage, continuous trading, no dividends),
any derivative can be perfectly hedged, leading to this equation.

Historical Note:
Developed by Fischer Black and Myron Scholes in 1973, earning Scholes
a Nobel Prize in Economics in 1997 (Black had passed away by then).

Example: Pricing a European call option on a stock trading at $100
with strike $105, volatility 20%, risk-free rate 5%, expiring in 1 year.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BlackScholesEquation(,)` | Initializes a new instance of the Black-Scholes Equation. |
| `BlackScholesEquation(Double,Double)` | Initializes a new instance of the Black-Scholes Equation with double parameters. |

## Properties

| Property | Summary |
|:-----|:--------|
| `InputDimension` |  |
| `Name` |  |
| `OutputDimension` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeResidual(Vector<>,Vector<>,PDEDerivatives<>)` |  |
| `ComputeResidualGradient(Vector<>,Vector<>,PDEDerivatives<>)` |  |

