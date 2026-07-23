using System;
using System.Collections.Generic;
using AiDotNet.Finance.Data;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors;

namespace AiDotNet.Finance.Trading.Environments;

/// <summary>
/// Factory helpers for creating trading environments from market data.
/// </summary>
/// <remarks>
/// <para>
/// This factory converts OHLCV series into the price tensors required by
/// trading environments, keeping environment setup concise and consistent.
/// </para>
/// <para>
/// <b>For Beginners:</b> If you already have price bars, these helpers build
/// the right trading simulator (stock, portfolio, or market making) without
/// manual tensor conversion.
/// </para>
/// </remarks>
public static class TradingEnvironmentFactory
{
    /// <summary>
    /// Creates a stock trading environment from a single asset series.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="series">Market data series for one asset.</param>
    /// <param name="windowSize">Number of historical steps in each observation.</param>
    /// <param name="initialCapital">Starting cash for the portfolio.</param>
    /// <param name="tradeSize">Units to buy or sell per trade.</param>
    /// <param name="transactionCost">Transaction cost rate.</param>
    /// <param name="allowShortSelling">Whether short selling is allowed.</param>
    /// <param name="randomStart">Whether to start episodes at random indices.</param>
    /// <param name="maxEpisodeLength">Maximum steps per episode (0 = full data).</param>
    /// <param name="seed">Optional random seed.</param>
    /// <param name="priceSelector">Optional selector for which price to use (defaults to Close).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this when you want a simple buy/hold/sell environment
    /// driven by the close price of a single asset.
    /// </para>
    /// </remarks>
    public static StockTradingEnvironment<T> CreateStockTradingEnvironment<T>(
        IReadOnlyList<MarketDataPoint<T>> series,
        int windowSize,
        T initialCapital,
        T tradeSize,
        double transactionCost = 0.001,
        bool allowShortSelling = true,
        bool randomStart = false,
        int maxEpisodeLength = 0,
        int? seed = null,
        Func<MarketDataPoint<T>, T>? priceSelector = null)
    {
        var prices = CreatePriceTensor(series, priceSelector);
        return new StockTradingEnvironment<T>(
            prices,
            windowSize,
            initialCapital,
            tradeSize,
            transactionCost,
            allowShortSelling,
            randomStart,
            maxEpisodeLength,
            seed);
    }

    /// <summary>
    /// Creates a stock trading environment from a market data provider.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="provider">Provider containing OHLCV data.</param>
    /// <param name="windowSize">Number of historical steps in each observation.</param>
    /// <param name="initialCapital">Starting cash for the portfolio.</param>
    /// <param name="tradeSize">Units to buy or sell per trade.</param>
    /// <param name="transactionCost">Transaction cost rate.</param>
    /// <param name="allowShortSelling">Whether short selling is allowed.</param>
    /// <param name="randomStart">Whether to start episodes at random indices.</param>
    /// <param name="maxEpisodeLength">Maximum steps per episode (0 = full data).</param>
    /// <param name="seed">Optional random seed.</param>
    /// <param name="priceSelector">Optional selector for which price to use (defaults to Close).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This overload lets you build the environment directly
    /// from a MarketDataProvider without manually extracting the series.
    /// </para>
    /// </remarks>
    public static StockTradingEnvironment<T> CreateStockTradingEnvironment<T>(
        MarketDataProvider<T> provider,
        int windowSize,
        T initialCapital,
        T tradeSize,
        double transactionCost = 0.001,
        bool allowShortSelling = true,
        bool randomStart = false,
        int maxEpisodeLength = 0,
        int? seed = null,
        Func<MarketDataPoint<T>, T>? priceSelector = null)
    {
        if (provider is null)
        {
            throw new ArgumentNullException(nameof(provider));
        }

        return CreateStockTradingEnvironment(
            provider.GetAll(),
            windowSize,
            initialCapital,
            tradeSize,
            transactionCost,
            allowShortSelling,
            randomStart,
            maxEpisodeLength,
            seed,
            priceSelector);
    }

    /// <summary>
    /// Creates a portfolio trading environment from multiple asset series.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="assetSeries">Collection of per-asset market series.</param>
    /// <param name="windowSize">Number of historical steps in each observation.</param>
    /// <param name="initialCapital">Starting cash for the portfolio.</param>
    /// <param name="transactionCost">Transaction cost rate.</param>
    /// <param name="allowShortSelling">Whether short selling is allowed.</param>
    /// <param name="randomStart">Whether to start episodes at random indices.</param>
    /// <param name="maxEpisodeLength">Maximum steps per episode (0 = full data).</param>
    /// <param name="seed">Optional random seed.</param>
    /// <param name="priceSelector">Optional selector for which price to use (defaults to Close).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This builds a multi-asset environment where each column
    /// in the price tensor represents a different asset's price history.
    /// </para>
    /// </remarks>
    public static PortfolioTradingEnvironment<T> CreatePortfolioTradingEnvironment<T>(
        IReadOnlyList<IReadOnlyList<MarketDataPoint<T>>> assetSeries,
        int windowSize,
        T initialCapital,
        double transactionCost = 0.001,
        bool allowShortSelling = false,
        bool randomStart = false,
        int maxEpisodeLength = 0,
        int? seed = null,
        Func<MarketDataPoint<T>, T>? priceSelector = null)
    {
        var prices = CreatePriceTensor(assetSeries, priceSelector);
        return new PortfolioTradingEnvironment<T>(
            prices,
            windowSize,
            initialCapital,
            transactionCost,
            allowShortSelling,
            randomStart,
            maxEpisodeLength,
            seed);
    }

    /// <summary>
    /// Creates a portfolio trading environment from multiple data providers.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="providers">Providers containing OHLCV data per asset.</param>
    /// <param name="windowSize">Number of historical steps in each observation.</param>
    /// <param name="initialCapital">Starting cash for the portfolio.</param>
    /// <param name="transactionCost">Transaction cost rate.</param>
    /// <param name="allowShortSelling">Whether short selling is allowed.</param>
    /// <param name="randomStart">Whether to start episodes at random indices.</param>
    /// <param name="maxEpisodeLength">Maximum steps per episode (0 = full data).</param>
    /// <param name="seed">Optional random seed.</param>
    /// <param name="priceSelector">Optional selector for which price to use (defaults to Close).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This overload lets you pass multiple MarketDataProvider
    /// instances (one per asset) and get a multi-asset trading environment.
    /// </para>
    /// </remarks>
    public static PortfolioTradingEnvironment<T> CreatePortfolioTradingEnvironment<T>(
        IReadOnlyList<MarketDataProvider<T>> providers,
        int windowSize,
        T initialCapital,
        double transactionCost = 0.001,
        bool allowShortSelling = false,
        bool randomStart = false,
        int maxEpisodeLength = 0,
        int? seed = null,
        Func<MarketDataPoint<T>, T>? priceSelector = null)
    {
        if (providers is null)
        {
            throw new ArgumentNullException(nameof(providers));
        }

        var series = ExtractSeriesFromProviders(providers);
        return CreatePortfolioTradingEnvironment(
            series,
            windowSize,
            initialCapital,
            transactionCost,
            allowShortSelling,
            randomStart,
            maxEpisodeLength,
            seed,
            priceSelector);
    }

    /// <summary>
    /// Creates a market making environment from a single asset series.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="series">Market data series for one asset.</param>
    /// <param name="windowSize">Number of historical steps in each observation.</param>
    /// <param name="initialCapital">Starting cash for the portfolio.</param>
    /// <param name="tradeSize">Units per filled order.</param>
    /// <param name="baseSpread">Baseline bid/ask spread in price units.</param>
    /// <param name="orderArrivalRate">Base probability of order arrival per step.</param>
    /// <param name="maxInventory">Maximum absolute inventory allowed.</param>
    /// <param name="inventoryPenalty">Penalty per unit inventory.</param>
    /// <param name="transactionCost">Transaction cost rate.</param>
    /// <param name="allowShortSelling">Whether negative inventory is allowed.</param>
    /// <param name="randomStart">Whether to start episodes at random indices.</param>
    /// <param name="maxEpisodeLength">Maximum steps per episode (0 = full data).</param>
    /// <param name="seed">Optional random seed.</param>
    /// <param name="priceSelector">Optional selector for which price to use (defaults to Close).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This builds a market making simulator where the agent
    /// sets bid/ask offsets relative to the chosen price (default close).
    /// </para>
    /// </remarks>
    public static MarketMakingEnvironment<T> CreateMarketMakingEnvironment<T>(
        IReadOnlyList<MarketDataPoint<T>> series,
        int windowSize,
        T initialCapital,
        T tradeSize,
        double baseSpread = 0.01,
        double orderArrivalRate = 0.2,
        int maxInventory = 10,
        double inventoryPenalty = 0.001,
        double transactionCost = 0.0,
        bool allowShortSelling = true,
        bool randomStart = false,
        int maxEpisodeLength = 0,
        int? seed = null,
        Func<MarketDataPoint<T>, T>? priceSelector = null)
    {
        var prices = CreatePriceTensor(series, priceSelector);
        return new MarketMakingEnvironment<T>(
            prices,
            windowSize,
            initialCapital,
            tradeSize,
            baseSpread,
            orderArrivalRate,
            maxInventory,
            inventoryPenalty,
            transactionCost,
            allowShortSelling,
            randomStart,
            maxEpisodeLength,
            seed);
    }

    /// <summary>
    /// Creates a market making environment from a market data provider.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="provider">Provider containing OHLCV data.</param>
    /// <param name="windowSize">Number of historical steps in each observation.</param>
    /// <param name="initialCapital">Starting cash for the portfolio.</param>
    /// <param name="tradeSize">Units per filled order.</param>
    /// <param name="baseSpread">Baseline bid/ask spread in price units.</param>
    /// <param name="orderArrivalRate">Base probability of order arrival per step.</param>
    /// <param name="maxInventory">Maximum absolute inventory allowed.</param>
    /// <param name="inventoryPenalty">Penalty per unit inventory.</param>
    /// <param name="transactionCost">Transaction cost rate.</param>
    /// <param name="allowShortSelling">Whether negative inventory is allowed.</param>
    /// <param name="randomStart">Whether to start episodes at random indices.</param>
    /// <param name="maxEpisodeLength">Maximum steps per episode (0 = full data).</param>
    /// <param name="seed">Optional random seed.</param>
    /// <param name="priceSelector">Optional selector for which price to use (defaults to Close).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this overload when your data is stored in a
    /// MarketDataProvider and you want a market making environment quickly.
    /// </para>
    /// </remarks>
    public static MarketMakingEnvironment<T> CreateMarketMakingEnvironment<T>(
        MarketDataProvider<T> provider,
        int windowSize,
        T initialCapital,
        T tradeSize,
        double baseSpread = 0.01,
        double orderArrivalRate = 0.2,
        int maxInventory = 10,
        double inventoryPenalty = 0.001,
        double transactionCost = 0.0,
        bool allowShortSelling = true,
        bool randomStart = false,
        int maxEpisodeLength = 0,
        int? seed = null,
        Func<MarketDataPoint<T>, T>? priceSelector = null)
    {
        if (provider is null)
        {
            throw new ArgumentNullException(nameof(provider));
        }

        return CreateMarketMakingEnvironment(
            provider.GetAll(),
            windowSize,
            initialCapital,
            tradeSize,
            baseSpread,
            orderArrivalRate,
            maxInventory,
            inventoryPenalty,
            transactionCost,
            allowShortSelling,
            randomStart,
            maxEpisodeLength,
            seed,
            priceSelector);
    }

    /// <summary>
    /// Converts a single asset series into a [time, 1] price tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="series">Market data series for one asset.</param>
    /// <param name="priceSelector">Optional selector for which price to use.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Environments operate on plain price tensors, so we
    /// extract a single price (like Close) from each bar into a matrix.
    /// </para>
    /// </remarks>
    private static Tensor<T> CreatePriceTensor<T>(
        IReadOnlyList<MarketDataPoint<T>> series,
        Func<MarketDataPoint<T>, T>? priceSelector)
    {
        if (series is null)
        {
            throw new ArgumentNullException(nameof(series));
        }

        if (series.Count == 0)
        {
            throw new ArgumentException("Series must contain at least one data point.", nameof(series));
        }

        var selector = priceSelector ?? (point => point.Close);
        var data = new T[series.Count];

        for (int i = 0; i < series.Count; i++)
        {
            data[i] = selector(series[i]);
        }

        return new Tensor<T>(new[] { series.Count, 1 }, new Vector<T>(data));
    }

    /// <summary>
    /// Converts multiple asset series into a [time, assets] price tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="assetSeries">Collection of per-asset series.</param>
    /// <param name="priceSelector">Optional selector for which price to use.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This stacks each asset's price history into columns
    /// so the environment can simulate a multi-asset portfolio.
    /// </para>
    /// </remarks>
    private static Tensor<T> CreatePriceTensor<T>(
        IReadOnlyList<IReadOnlyList<MarketDataPoint<T>>> assetSeries,
        Func<MarketDataPoint<T>, T>? priceSelector)
    {
        if (assetSeries is null)
        {
            throw new ArgumentNullException(nameof(assetSeries));
        }

        if (assetSeries.Count == 0)
        {
            throw new ArgumentException("At least one asset series is required.", nameof(assetSeries));
        }

        int steps = assetSeries[0]?.Count ?? 0;
        if (steps == 0)
        {
            throw new ArgumentException("Asset series must contain at least one data point.", nameof(assetSeries));
        }

        for (int i = 0; i < assetSeries.Count; i++)
        {
            if (assetSeries[i] is null)
            {
                throw new ArgumentNullException(nameof(assetSeries), $"Asset series at index {i} is null.");
            }

            if (assetSeries[i].Count != steps)
            {
                throw new ArgumentException("All asset series must have the same length.", nameof(assetSeries));
            }
        }

        var selector = priceSelector ?? (point => point.Close);
        int assets = assetSeries.Count;
        var data = new T[steps * assets];

        for (int t = 0; t < steps; t++)
        {
            for (int a = 0; a < assets; a++)
            {
                data[(t * assets) + a] = selector(assetSeries[a][t]);
            }
        }

        return new Tensor<T>(new[] { steps, assets }, new Vector<T>(data));
    }

    /// <summary>
    /// Extracts series lists from a collection of market data providers.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="providers">Providers containing OHLCV data.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Providers store price bars; this helper extracts the
    /// raw lists so they can be converted into a multi-asset price tensor.
    /// </para>
    /// </remarks>
    private static IReadOnlyList<IReadOnlyList<MarketDataPoint<T>>> ExtractSeriesFromProviders<T>(
        IReadOnlyList<MarketDataProvider<T>> providers)
    {
        var series = new List<IReadOnlyList<MarketDataPoint<T>>>(providers.Count);
        for (int i = 0; i < providers.Count; i++)
        {
            if (providers[i] is null)
            {
                throw new ArgumentNullException(nameof(providers), $"Provider at index {i} is null.");
            }

            series.Add(providers[i].GetAll());
        }

        return series;
    }
}
