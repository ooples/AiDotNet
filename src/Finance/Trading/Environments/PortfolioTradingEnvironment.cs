using System;
using System.Linq;
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

        T portfolioValue = NumOps.Zero;
        for (int asset = 0; asset < NumAssets; asset++)
        {
            portfolioValue = NumOps.Add(portfolioValue, NumOps.Multiply(_positions[asset], prices[asset]));
        }
        portfolioValue = NumOps.Add(portfolioValue, _cash);

        var desiredPositions = new T[NumAssets];
        for (int asset = 0; asset < NumAssets; asset++)
        {
            T targetValue = NumOps.Multiply(portfolioValue, targetWeights[asset]);
            desiredPositions[asset] = NumOps.Divide(targetValue, prices[asset]);
        }

        var deltas = new T[NumAssets];
        for (int asset = 0; asset < NumAssets; asset++)
        {
            deltas[asset] = NumOps.Subtract(desiredPositions[asset], _positions[asset]);
        }

        ScaleBuysToCash(deltas, prices);

        for (int asset = 0; asset < NumAssets; asset++)
        {
            if (!AllowShortSelling)
            {
                T newPosition = NumOps.Add(_positions[asset], deltas[asset]);
                if (NumOps.Compare(newPosition, NumOps.Zero) < 0)
                {
                    deltas[asset] = NumOps.Negate(_positions[asset]);
                }
            }

            if (NumOps.Compare(deltas[asset], NumOps.Zero) != 0)
            {
                ExecuteTrade(asset, deltas[asset], prices[asset]);
            }
        }
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
    private T[] NormalizeWeights(Vector<T> action)
    {
        var weights = new T[NumAssets];
        T sum = NumOps.Zero;

        for (int i = 0; i < NumAssets; i++)
        {
            T value = action[i];
            if (!AllowShortSelling && NumOps.Compare(value, NumOps.Zero) < 0)
            {
                value = NumOps.Zero;
            }
            weights[i] = value;
            sum = NumOps.Add(sum, value);
        }

        if (NumOps.Compare(sum, NumOps.Zero) == 0)
        {
            T uniform = NumOps.Divide(NumOps.One, NumOps.FromDouble(NumAssets));
            return Enumerable.Repeat(uniform, NumAssets).ToArray();
        }

        for (int i = 0; i < NumAssets; i++)
        {
            weights[i] = NumOps.Divide(weights[i], sum);
        }

        return weights;
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
    private void ScaleBuysToCash(T[] deltas, Vector<T> prices)
    {
        T totalBuyCost = NumOps.Zero;
        for (int asset = 0; asset < NumAssets; asset++)
        {
            if (NumOps.Compare(deltas[asset], NumOps.Zero) > 0)
            {
                T cost = NumOps.Multiply(deltas[asset], prices[asset]);
                cost = NumOps.Multiply(cost, NumOps.FromDouble(1.0 + TransactionCost));
                totalBuyCost = NumOps.Add(totalBuyCost, cost);
            }
        }

        if (NumOps.Compare(totalBuyCost, _cash) <= 0)
        {
            return;
        }

        T scale = NumOps.Divide(_cash, totalBuyCost);
        for (int asset = 0; asset < NumAssets; asset++)
        {
            if (NumOps.Compare(deltas[asset], NumOps.Zero) > 0)
            {
                deltas[asset] = NumOps.Multiply(deltas[asset], scale);
            }
        }
    }
}
