using AiDotNet.Tensors.Engines;
using System;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors;

namespace AiDotNet.Finance.Trading.Environments;

/// <summary>
/// Multi-asset portfolio trading environment with continuous weight actions.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// PortfolioTradingEnvironment interprets the action vector as target
/// portfolio weights for each asset. It rebalances positions accordingly.
/// </para>
/// <para>
/// <b>For Beginners:</b> Instead of buy/hold/sell, the agent says "I want
/// 40% in asset A and 60% in asset B" every step. The environment rebalances
/// the portfolio to match those weights.
/// </para>
/// </remarks>
public sealed class PortfolioTradingEnvironment<T> : TradingEnvironment<T>
{
    /// <inheritdoc/>
    public override int ActionSpaceSize => NumAssets;

    /// <inheritdoc/>
    public override bool IsContinuousActionSpace => true;

    /// <summary>
    /// Creates a portfolio trading environment.
    /// </summary>
    /// <param name="marketData">Market data tensor shaped [time, assets].</param>
    /// <param name="windowSize">Number of historical steps in each observation.</param>
    /// <param name="initialCapital">Starting cash for the portfolio.</param>
    /// <param name="transactionCost">Transaction cost rate.</param>
    /// <param name="allowShortSelling">Whether short selling is allowed.</param>
    /// <param name="randomStart">Whether to start episodes at random indices.</param>
    /// <param name="maxEpisodeLength">Maximum steps per episode (0 = full data).</param>
    /// <param name="seed">Optional random seed.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this when you want to allocate capital across
    /// multiple assets every step (like an automated portfolio manager).
    /// </para>
    /// </remarks>
    public PortfolioTradingEnvironment(
        Tensor<T> marketData,
        int windowSize,
        T initialCapital,
        double transactionCost = 0.001,
        bool allowShortSelling = false,
        bool randomStart = false,
        int maxEpisodeLength = 0,
        int? seed = null)
        : base(marketData, windowSize, initialCapital, transactionCost, allowShortSelling, randomStart, maxEpisodeLength, seed)
    {
    }

    /// <summary>
    /// Applies target weight actions by rebalancing the portfolio.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This converts the action vector into target weights,
    /// normalizes them, and buys/sells assets to match the target mix.
    /// </para>
    /// </remarks>
    protected override void ApplyAction(Vector<T> action, Vector<T> prices)
    {
        if (action.Length != NumAssets)
        {
            throw new ArgumentException($"Action length must be {NumAssets}.", nameof(action));
        }

        var targetWeights = NormalizeWeights(action);
        var engine = Engine;

        T portfolioValue = NumOps.Add(engine.DotProduct(_positions, prices), _cash);

        // desired_i = V * w_i / p_i; delta_i = desired_i - position_i.
        var desiredPositions = engine.Divide(engine.Multiply(targetWeights, portfolioValue), prices);
        var deltas = engine.Subtract(desiredPositions, _positions);

        // Without short selling a trade may only take a position down to zero: delta_i >= -position_i.
        if (!AllowShortSelling)
        {
            deltas = engine.Max(deltas, engine.Negate(_positions));
        }

        // Sells first: proceeds = |q * p| * (1 - c), credited before any buy is priced.
        var buys = engine.ReLU(deltas);
        var sells = engine.Subtract(deltas, buys);
        T sellValue = engine.DotProduct(sells, prices);
        T proceeds = NumOps.Multiply(NumOps.Negate(sellValue), NumOps.FromDouble(1.0 - TransactionCost));
        _cash = NumOps.Add(_cash, proceeds);
        _positions = engine.Add(_positions, sells);

        // Buys: scaled down together when their cost (q * p * (1 + c)) exceeds the cash after the sells.
        buys = ScaleBuysToCash(buys, prices);
        T buyCost = NumOps.Multiply(engine.DotProduct(buys, prices), NumOps.FromDouble(1.0 + TransactionCost));
        T remaining = NumOps.Subtract(_cash, buyCost);

        // A scaled buy set costs the whole cash up to rounding: execute every scaled buy and never let that
        // rounding leave the cash negative (the old per-asset loop dropped a whole buy on a one-ulp shortfall).
        _cash = NumOps.Compare(remaining, NumOps.Zero) < 0 ? NumOps.Zero : remaining;
        _positions = engine.Add(_positions, buys);
    }

    /// <summary>
    /// Normalizes the action vector into valid weights.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> We make sure weights sum to 1. If short selling is not allowed,
    /// negative weights are clamped to zero.
    /// </para>
    /// </remarks>
    private Vector<T> NormalizeWeights(Vector<T> action)
    {
        var engine = Engine;
        var weights = AllowShortSelling ? action : engine.ReLU(action);
        T sum = engine.Sum(weights);

        if (NumOps.Compare(sum, NumOps.Zero) == 0)
        {
            var uniform = new Vector<T>(NumAssets);
            uniform.Fill(NumOps.Divide(NumOps.One, NumOps.FromDouble(NumAssets)));
            return uniform;
        }

        return engine.Divide(weights, sum);
    }

    /// <summary>
    /// Scales buy orders if they exceed available cash.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> If the agent asks to buy more than the cash allows,
    /// we shrink the buy orders proportionally so the trades are affordable.
    /// </para>
    /// </remarks>
    private Vector<T> ScaleBuysToCash(Vector<T> buys, Vector<T> prices)
    {
        T totalBuyCost = NumOps.Multiply(Engine.DotProduct(buys, prices), NumOps.FromDouble(1.0 + TransactionCost));

        if (NumOps.Compare(totalBuyCost, _cash) <= 0)
        {
            return buys;
        }

        return Engine.Multiply(buys, NumOps.Divide(_cash, totalBuyCost));
    }
}
