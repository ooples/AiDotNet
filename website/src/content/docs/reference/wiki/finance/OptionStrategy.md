---
title: "OptionStrategy"
description: "A named option strategy: a set of `OptionLeg`s (plus an optional `StockLeg`), with the broker approval level it requires, whether its risk is defined, and its expiry payoff profile (max loss, max gain, breakevens) computed by sampling the p…"
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Options`

A named option strategy: a set of `OptionLeg`s (plus an optional `StockLeg`),
with the broker approval level it requires, whether its risk is defined, and its expiry payoff
profile (max loss, max gain, breakevens) computed by sampling the payoff over a spot grid — robust
for ANY leg combination, not just the textbook ones. Net Greeks come from `BlackScholes`.

The `RequiredLevel` classifier is the gate that lets the platform enforce a broker's
approval tier: a strategy with an unhedged short call is `Uncovered`;
a vertical (short hedged by a long) is `Level3`; long-only is Level 2;
covered-by-stock / cash-secured is Level 1. It errs HIGH when unsure (fail-safe).

## Properties

| Property | Summary |
|:-----|:--------|
| `Class` | The broker-independent risk archetype (mapped to a broker level via `BrokerOptionsProfile`). |
| `IsDefinedRisk` | True when the worst-case loss is bounded (no naked/uncovered short exposure). |
| `IsIndexUnderlying` | True when the underlying is a (cash-settled) index — naked index options are the highest broker tier, distinct from naked equity options. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BearPutSpread(Double,Double,Double,Int32)` | Level 3: bear put (debit) spread — long higher-strike put, short lower-strike put. |
| `BullCallSpread(Double,Double,Double,Int32)` | Level 3: bull call (debit) spread — long lower-strike call, short higher-strike call. |
| `CashSecuredPut(Double,Double,Int32)` | Level 1: short put fully cash-secured (willing to buy the stock at the strike). |
| `CoveredCall(Double,Double,Int32)` | Level 1: long 100 shares + short 1 call (income on owned stock). |
| `IronCondor(Double,Double,Double,Double,Double,Int32)` | Level 3: iron condor — short put spread + short call spread (defined-risk, short volatility). |
| `LongCall(Double,Double,Int32)` | Level 2: long call (defined risk = premium). |
| `LongPut(Double,Double,Int32)` | Level 2: long put. |
| `LongStraddle(Double,Double,Int32)` | Level 2: long straddle (long call + long put at the same strike) — long volatility. |
| `NakedCall(Double,Double,Int32,Boolean)` | Naked short call (uncovered, large risk) — modeled so the gate can REJECT it (NakedEquity, or NakedIndex when `isIndex`). |
| `PayoffAtExpiry(Double)` | Per-share intrinsic payoff of all legs (+ stock) at a given expiry spot, excluding premium. |
| `PayoffProfile(Double,Int32,Double)` | Expiry payoff profile sampled over a spot grid spanning [0, gridMax×maxStrike]. |
| `ProtectivePut(Double,Double,Int32)` | Level 1: long 100 shares + long 1 put (downside protection). |

