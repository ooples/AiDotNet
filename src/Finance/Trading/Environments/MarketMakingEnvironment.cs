using System;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors;

namespace AiDotNet.Finance.Trading.Environments;

/// <summary>
/// Market making environment that simulates bid/ask quoting and inventory risk.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// MarketMakingEnvironment models a single asset where the agent sets bid/ask offsets.
/// Orders arrive stochastically based on the spread, and the agent earns the spread
/// but accumulates inventory risk.
/// </para>
/// <para>
/// <b>For Beginners:</b> A market maker tries to profit from the bid/ask spread while
/// keeping inventory small. This environment teaches an agent to balance profit and risk.
/// </para>
/// </remarks>
public sealed class MarketMakingEnvironment<T> : TradingEnvironment<T>
{
    private readonly int _maxInventory;
    private readonly double _baseSpread;
    private readonly double _orderArrivalRate;
    private readonly double _inventoryPenalty;
    private readonly T _tradeSize;

    /// <inheritdoc/>
    public override int ActionSpaceSize => 2;

    /// <inheritdoc/>
    public override bool IsContinuousActionSpace => true;

    /// <summary>
    /// Creates a market making environment for a single asset.
    /// </summary>
    /// <param name="marketData">Market data tensor shaped [time, 1].</param>
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
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The action has two values: bid offset and ask offset.
    /// Smaller offsets mean tighter spreads (more trades, less profit per trade).
    /// Larger offsets mean wider spreads (fewer trades, more profit per trade).
    /// </para>
    /// </remarks>
    public MarketMakingEnvironment(
        Tensor<T> marketData,
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
        int? seed = null)
        : base(marketData, windowSize, initialCapital, transactionCost, allowShortSelling, randomStart, maxEpisodeLength, seed)
    {
        if (marketData.Shape[1] != 1)
        {
            throw new ArgumentException("MarketMakingEnvironment expects a single asset (marketData shape [time, 1]).",
                nameof(marketData));
        }

        _tradeSize = tradeSize;
        _baseSpread = baseSpread;
        _orderArrivalRate = orderArrivalRate;
        _maxInventory = maxInventory;
        _inventoryPenalty = inventoryPenalty;
    }

    /// <summary>
    /// Applies bid/ask offset actions and simulates order fills.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> If an order arrives, the agent earns the spread
    /// but its inventory changes. Inventory is capped to reduce risk.
    /// </para>
    /// </remarks>
    protected override void ApplyAction(Vector<T> action, Vector<T> prices)
    {
        if (action.Length != 2)
        {
            throw new ArgumentException("MarketMakingEnvironment expects two action values.", nameof(action));
        }

        double bidOffset = Math.Abs(NumOps.ToDouble(action[0]));
        double askOffset = Math.Abs(NumOps.ToDouble(action[1]));

        double maxOffset = Math.Max(_baseSpread * 5.0, 1e-6);
        double normalizedBid = Math.Min(1.0, bidOffset / maxOffset);
        double normalizedAsk = Math.Min(1.0, askOffset / maxOffset);

        double buyProb = _orderArrivalRate * (1.0 - normalizedAsk);
        double sellProb = _orderArrivalRate * (1.0 - normalizedBid);

        T midPrice = prices[0];
        T bidPrice = NumOps.Subtract(midPrice, NumOps.FromDouble(_baseSpread + bidOffset));
        T askPrice = NumOps.Add(midPrice, NumOps.FromDouble(_baseSpread + askOffset));

        bool buyHit = Random.NextDouble() < buyProb;
        bool sellHit = Random.NextDouble() < sellProb;

        if (buyHit && CanChangeInventory(NumOps.Negate(_tradeSize)))
        {
            ExecuteTrade(0, NumOps.Negate(_tradeSize), askPrice);
        }

        if (sellHit && CanChangeInventory(_tradeSize))
        {
            ExecuteTrade(0, _tradeSize, bidPrice);
        }
    }

    /// <summary>
    /// Computes reward with inventory penalty.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Rewards capture profits but subtract a penalty when
    /// inventory grows too large.
    /// </para>
    /// </remarks>
    protected override T ComputeReward(T previousValue, T currentValue)
    {
        T baseReward = base.ComputeReward(previousValue, currentValue);
        T penalty = NumOps.Multiply(NumOps.Abs(_positions[0]), NumOps.FromDouble(_inventoryPenalty));
        return NumOps.Subtract(baseReward, penalty);
    }

    /// <summary>
    /// Checks if inventory can move without breaching limits.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This prevents the agent from holding too much inventory
    /// in either direction (long or short).
    /// </para>
    /// </remarks>
    private bool CanChangeInventory(T delta)
    {
        T next = NumOps.Add(_positions[0], delta);
        double nextValue = NumOps.ToDouble(next);
        if (!AllowShortSelling && nextValue < 0)
        {
            return false;
        }

        return Math.Abs(nextValue) <= _maxInventory;
    }

}
