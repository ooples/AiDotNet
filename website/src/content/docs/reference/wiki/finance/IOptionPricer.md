---
title: "IOptionPricer<T>"
description: "A European-option pricing engine: fair value, first-order Greeks, and implied volatility."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Finance.Interfaces`

A European-option pricing engine: fair value, first-order Greeks, and implied volatility.

## For Beginners

An "option pricer" answers "what is this option worth, and how does that
value move when the market moves?" The default is the Black-Scholes formula; swap in your own if you
need a different model.

## How It Works

This is a customization point, not a trainable model. Models that need to price options or
hedge option exposure (e.g. an options-focused RL trading agent, a covered-call portfolio model)
depend on this interface and default to the closed-form `BlackScholes`
implementation, but callers can substitute their own pricer (binomial tree, Heston, a learned
surface, …) without changing the consuming model.

## Methods

| Method | Summary |
|:-----|:--------|
| `Delta(,,,,,Boolean)` | Delta — sensitivity of price to the underlying spot. |
| `Gamma(,,,,)` | Gamma — sensitivity of delta to the underlying spot (same for calls and puts). |
| `ImpliedVolatility(,,,,,Boolean,Int32,Double)` | The implied volatility that reproduces `marketPrice` under this pricer. |
| `Price(,,,,,Boolean)` | Fair value of a European call (`isCall` = true) or put. |
| `Rho(,,,,,Boolean)` | Rho — sensitivity of price to the risk-free rate. |
| `Theta(,,,,,Boolean)` | Theta — sensitivity of price to the passage of time. |
| `Vega(,,,,)` | Vega — sensitivity of price to volatility (same for calls and puts). |

