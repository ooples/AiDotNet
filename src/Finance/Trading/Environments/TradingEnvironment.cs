using System;
using System.Collections.Generic;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors;

namespace AiDotNet.Finance.Trading.Environments;

/// <summary>
/// Base environment for financial trading simulations.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// TradingEnvironment implements common portfolio bookkeeping for RL trading:
/// positions, cash balance, portfolio value, and windowed market observations.
/// Derived environments specialize how actions are interpreted.
/// </para>
/// <para>
/// <b>For Beginners:</b> This is the "market simulator" shared by all trading
/// environments. It feeds price data to agents, executes trades, and computes
/// rewards based on portfolio changes.
/// </para>
/// </remarks>
public abstract class TradingEnvironment<T> : IEnvironment<T>
{
    protected readonly INumericOperations<T> NumOps;
    protected readonly Tensor<T> MarketData;
    protected readonly int WindowSize;
    protected readonly int NumAssets;
    protected readonly T InitialCapital;
    protected readonly double TransactionCost;
    protected readonly bool AllowShortSelling;
    protected readonly bool RandomStart;
    protected readonly int MaxEpisodeLength;

    private Random _random;
    private int _currentStep;
    private int _episodeStep;
    protected Vector<T> _positions;
    protected T _cash;
    protected T _portfolioValue;

    /// <inheritdoc/>
    public int ObservationSpaceDimension => (WindowSize * NumAssets) + NumAssets + 1;

    /// <inheritdoc/>
    public abstract int ActionSpaceSize { get; }

    /// <inheritdoc/>
    public abstract bool IsContinuousActionSpace { get; }

    /// <summary>
    /// Gets the environment random number generator.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Environments sometimes need randomness (like random starts).
    /// This property exposes the RNG so derived environments can stay consistent with seeding.
    /// </para>
    /// </remarks>
    protected Random Random => _random;

    /// <summary>
    /// Initializes a new trading environment.
    /// </summary>
    /// <param name="marketData">Market data tensor shaped [time, assets].</param>
    /// <param name="windowSize">Number of historical steps in each observation.</param>
    /// <param name="initialCapital">Starting cash for the portfolio.</param>
    /// <param name="transactionCost">Transaction cost rate (e.g., 0.001).</param>
    /// <param name="allowShortSelling">Whether negative positions are allowed.</param>
    /// <param name="randomStart">Whether to start episodes at random indices.</param>
    /// <param name="maxEpisodeLength">Maximum steps per episode (0 = full data).</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> You provide the price history, choose how many
    /// past steps the agent sees, and set basic trading rules (capital, costs, shorts).
    /// </para>
    /// </remarks>
    protected TradingEnvironment(
        Tensor<T> marketData,
        int windowSize,
        T initialCapital,
        double transactionCost = 0.001,
        bool allowShortSelling = true,
        bool randomStart = false,
        int maxEpisodeLength = 0,
        int? seed = null)
    {
        MarketData = marketData ?? throw new ArgumentNullException(nameof(marketData));
        WindowSize = windowSize;
        InitialCapital = initialCapital;
        TransactionCost = transactionCost;
        AllowShortSelling = allowShortSelling;
        RandomStart = randomStart;
        MaxEpisodeLength = maxEpisodeLength;

        NumOps = MathHelper.GetNumericOperations<T>();
        _random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();

        ValidateMarketData();
        NumAssets = MarketData.Shape[1];
        _positions = new Vector<T>(NumAssets);
        _cash = InitialCapital;
        _portfolioValue = InitialCapital;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Reset starts a fresh trading episode. It clears
    /// positions, restores cash, and returns the initial market observation.
    /// </para>
    /// </remarks>
    public Vector<T> Reset()
    {
        _positions = new Vector<T>(NumAssets);
        _cash = InitialCapital;
        _portfolioValue = InitialCapital;
        _episodeStep = 0;

        _currentStep = WindowSize;
        if (RandomStart)
        {
            int maxStart = MarketData.Shape[0] - 1;
            if (maxStart > WindowSize)
            {
                _currentStep = _random.Next(WindowSize, maxStart);
            }
        }

        return BuildObservation(_currentStep);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Step executes one trading decision: it applies the
    /// agent's action, updates the portfolio, and returns the next observation
    /// with a reward and done flag.
    /// </para>
    /// </remarks>
    public (Vector<T> NextState, T Reward, bool Done, Dictionary<string, object> Info) Step(Vector<T> action)
    {
        if (action == null)
        {
            throw new ArgumentNullException(nameof(action));
        }

        var prices = GetPricesAt(_currentStep);
        T previousValue = _portfolioValue;

        ApplyAction(action, prices);
        UpdatePortfolioValue(prices);

        T reward = ComputeReward(previousValue, _portfolioValue);

        _currentStep++;
        _episodeStep++;

        bool done = _currentStep >= MarketData.Shape[0]
            || (MaxEpisodeLength > 0 && _episodeStep >= MaxEpisodeLength);

        int observationStep = Math.Min(_currentStep, MarketData.Shape[0] - 1);
        var nextState = BuildObservation(observationStep);

        var info = new Dictionary<string, object>
        {
            ["step"] = _currentStep,
            ["portfolioValue"] = _portfolioValue!,
            ["cash"] = _cash!,
            ["positions"] = _positions
        };

        return (nextState, reward, done, info);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Seeding lets you get reproducible randomness,
    /// which is helpful for debugging trading strategies.
    /// </para>
    /// </remarks>
    public void Seed(int seed)
    {
        _random = RandomHelper.CreateSeededRandom(seed);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is a placeholder for cleanup. Most simple
    /// environments do not need special disposal logic.
    /// </para>
    /// </remarks>
    public void Close()
    {
        // No unmanaged resources to release.
    }

    /// <summary>
    /// Applies the trading action to update positions and cash.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Each environment interprets actions differently.
    /// This method is where "buy/sell/hold" or "target weights" becomes real trades.
    /// </para>
    /// </remarks>
    protected abstract void ApplyAction(Vector<T> action, Vector<T> prices);

    /// <summary>
    /// Builds an observation vector for a specific time step.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The observation includes recent prices plus the
    /// current portfolio state so the agent knows what it holds.
    /// </para>
    /// </remarks>
    protected virtual Vector<T> BuildObservation(int step)
    {
        var observation = new Vector<T>(ObservationSpaceDimension);
        int index = 0;

        int start = step - WindowSize + 1;
        for (int t = 0; t < WindowSize; t++)
        {
            int sourceIndex = start + t;
            for (int asset = 0; asset < NumAssets; asset++)
            {
                T value = NumOps.Zero;
                if (sourceIndex >= 0 && sourceIndex < MarketData.Shape[0])
                {
                    value = MarketData.Data.Span[(sourceIndex * NumAssets) + asset];
                }
                observation[index++] = value;
            }
        }

        for (int asset = 0; asset < NumAssets; asset++)
        {
            observation[index++] = _positions[asset];
        }

        observation[index] = _cash;
        return observation;
    }

    /// <summary>
    /// Computes the reward from portfolio value changes.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> By default, reward is the percentage change in portfolio value.
    /// Positive rewards mean the strategy made money, negative rewards mean losses.
    /// </para>
    /// </remarks>
    protected virtual T ComputeReward(T previousValue, T currentValue)
    {
        if (NumOps.Compare(previousValue, NumOps.Zero) == 0)
        {
            return NumOps.Zero;
        }

        return NumOps.Divide(NumOps.Subtract(currentValue, previousValue), previousValue);
    }

    /// <summary>
    /// Gets current prices at the specified time step.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This extracts the price vector for all assets at a single time step.
    /// </para>
    /// </remarks>
    protected Vector<T> GetPricesAt(int step)
    {
        var prices = new Vector<T>(NumAssets);
        for (int asset = 0; asset < NumAssets; asset++)
        {
            prices[asset] = MarketData.Data.Span[(step * NumAssets) + asset];
        }
        return prices;
    }

    /// <summary>
    /// Updates the portfolio value based on current prices.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Portfolio value = cash + sum(positions * prices).
    /// This keeps track of how much the account is worth after each step.
    /// </para>
    /// </remarks>
    protected void UpdatePortfolioValue(Vector<T> prices)
    {
        T value = _cash;
        for (int asset = 0; asset < NumAssets; asset++)
        {
            value = NumOps.Add(value, NumOps.Multiply(_positions[asset], prices[asset]));
        }
        _portfolioValue = value;
    }

    /// <summary>
    /// Executes a trade for a specific asset.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This helper applies trading costs and adjusts cash/positions
    /// for a buy or sell action.
    /// </para>
    /// </remarks>
    protected void ExecuteTrade(int assetIndex, T quantity, T price)
    {
        T tradeValue = NumOps.Multiply(quantity, price);
        bool isBuy = NumOps.Compare(quantity, NumOps.Zero) > 0;

        if (!AllowShortSelling && !isBuy)
        {
            T newPosition = NumOps.Add(_positions[assetIndex], quantity);
            if (NumOps.Compare(newPosition, NumOps.Zero) < 0)
            {
                quantity = NumOps.Negate(_positions[assetIndex]);
                tradeValue = NumOps.Multiply(quantity, price);
            }
        }

        if (isBuy)
        {
            T cost = NumOps.Multiply(tradeValue, NumOps.FromDouble(1.0 + TransactionCost));
            if (NumOps.Compare(_cash, cost) < 0)
            {
                return;
            }
            _cash = NumOps.Subtract(_cash, cost);
            _positions[assetIndex] = NumOps.Add(_positions[assetIndex], quantity);
        }
        else
        {
            T proceeds = NumOps.Multiply(NumOps.Abs(tradeValue), NumOps.FromDouble(1.0 - TransactionCost));
            _cash = NumOps.Add(_cash, proceeds);
            _positions[assetIndex] = NumOps.Add(_positions[assetIndex], quantity);
        }
    }

    /// <summary>
    /// Validates that market data is usable for trading.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The environment needs a 2D tensor with time and asset dimensions.
    /// This check prevents shape errors during simulation.
    /// </para>
    /// </remarks>
    private void ValidateMarketData()
    {
        if (MarketData.Rank != 2)
        {
            throw new ArgumentException("Market data must be 2D [time, assets].", nameof(MarketData));
        }

        if (WindowSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(WindowSize));
        }

        if (MarketData.Shape[0] <= WindowSize)
        {
            throw new ArgumentException("Market data must have more rows than the window size.", nameof(MarketData));
        }
    }
}
