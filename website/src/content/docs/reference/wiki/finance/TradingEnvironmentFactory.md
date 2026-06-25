---
title: "TradingEnvironmentFactory"
description: "Factory helpers for creating trading environments from market data."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Finance.Trading.Environments`

Factory helpers for creating trading environments from market data.

## For Beginners

If you already have price bars, these helpers build
the right trading simulator (stock, portfolio, or market making) without
manual tensor conversion.

## How It Works

This factory converts OHLCV series into the price tensors required by
trading environments, keeping environment setup concise and consistent.

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateMarketMakingEnvironment(IReadOnlyList<MarketDataPoint<>>,Int32,,,Double,Double,Int32,Double,Double,Boolean,Boolean,Int32,Nullable<Int32>,Func<MarketDataPoint<>,>)` | Creates a market making environment from a single asset series. |
| `CreateMarketMakingEnvironment(MarketDataProvider<>,Int32,,,Double,Double,Int32,Double,Double,Boolean,Boolean,Int32,Nullable<Int32>,Func<MarketDataPoint<>,>)` | Creates a market making environment from a market data provider. |
| `CreatePortfolioTradingEnvironment(IReadOnlyList<IReadOnlyList<MarketDataPoint<>>>,Int32,,Double,Boolean,Boolean,Int32,Nullable<Int32>,Func<MarketDataPoint<>,>)` | Creates a portfolio trading environment from multiple asset series. |
| `CreatePortfolioTradingEnvironment(IReadOnlyList<MarketDataProvider<>>,Int32,,Double,Boolean,Boolean,Int32,Nullable<Int32>,Func<MarketDataPoint<>,>)` | Creates a portfolio trading environment from multiple data providers. |
| `CreatePriceTensor(IReadOnlyList<IReadOnlyList<MarketDataPoint<>>>,Func<MarketDataPoint<>,>)` | Converts multiple asset series into a [time, assets] price tensor. |
| `CreatePriceTensor(IReadOnlyList<MarketDataPoint<>>,Func<MarketDataPoint<>,>)` | Converts a single asset series into a [time, 1] price tensor. |
| `CreateStockTradingEnvironment(IReadOnlyList<MarketDataPoint<>>,Int32,,,Double,Boolean,Boolean,Int32,Nullable<Int32>,Func<MarketDataPoint<>,>)` | Creates a stock trading environment from a single asset series. |
| `CreateStockTradingEnvironment(MarketDataProvider<>,Int32,,,Double,Boolean,Boolean,Int32,Nullable<Int32>,Func<MarketDataPoint<>,>)` | Creates a stock trading environment from a market data provider. |
| `ExtractSeriesFromProviders(IReadOnlyList<MarketDataProvider<>>)` | Extracts series lists from a collection of market data providers. |

