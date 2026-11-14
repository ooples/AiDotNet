using System;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors;

namespace AiDotNet.Finance.Trading.Environments;

/// <summary>
/// Simple single-asset trading environment with buy/hold/sell actions.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// StockTradingEnvironment models a single asset with discrete actions:
/// hold, buy, or sell. It uses the base TradingEnvironment for portfolio
/// bookkeeping and reward calculation.
/// </para>
/// <para>
/// <b>For Beginners:</b> This environment is the "hello world" of trading RL:
/// the agent decides whether to buy one unit, sell one unit, or do nothing.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var NumOps = MathHelper.GetNumericOperations&lt;double&gt;();
/// // Create a single-asset stock trading environment with buy/hold/sell actions
/// var marketData = new Tensor&lt;double&gt;(new[] { 1000, 1 }); // 1000 timesteps, 1 asset
/// var env = new StockTradingEnvironment&lt;double&gt;(
///     marketData, windowSize: 20, initialCapital: NumOps.FromDouble(10000.0),
///     tradeSize: NumOps.FromDouble(1.0), transactionCost: 0.001);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Finance)]
[ModelDomain(ModelDomain.ReinforcementLearning)]
[ModelCategory(ModelCategory.ReinforcementLearningAgent)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
// Citation corrected. arXiv 2011.09607 is the FinRL LIBRARY paper, a different work; this title is the
// ICAIF 2020 ensemble-strategy paper (Yang, Liu, Zhong & Walid), whose arXiv posting is 2511.12120 —
// the late id is genuine, the paper was only posted to arXiv well after publication.
[ResearchPaper("Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy",
    "https://arxiv.org/abs/2511.12120",
    Year = 2020,
    Authors = "Hongyang Yang, Xiao-Yang Liu, Shan Zhong, Anwar Walid")]
    [ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
public sealed class StockTradingEnvironment<T> : TradingEnvironment<T>
{
    private readonly T _tradeSize;

    /// <inheritdoc/>
    public override int ActionSpaceSize => 3;

    /// <inheritdoc/>
    public override bool IsContinuousActionSpace => false;

    /// <summary>
    /// Creates a stock trading environment for a single asset.
    /// </summary>
    /// <param name="marketData">Market data tensor shaped [time, 1].</param>
    /// <param name="windowSize">Number of historical steps in each observation.</param>
    /// <param name="initialCapital">Starting cash for the portfolio.</param>
    /// <param name="tradeSize">Units to buy or sell per trade.</param>
    /// <param name="transactionCost">Transaction cost rate.</param>
    /// <param name="allowShortSelling">Whether short selling is allowed.</param>
    /// <param name="randomStart">Whether to start episodes at random indices.</param>
    /// <param name="maxEpisodeLength">Maximum steps per episode (0 = full data).</param>
    /// <param name="seed">Optional random seed.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this when you want to train a trading agent on
    /// one asset (like a single stock). The agent chooses between buy, sell, and hold.
    /// </para>
    /// </remarks>
    public StockTradingEnvironment(
        Tensor<T> marketData,
        int windowSize,
        T initialCapital,
        T tradeSize,
        double transactionCost = 0.001,
        bool allowShortSelling = true,
        bool randomStart = false,
        int maxEpisodeLength = 0,
        int? seed = null)
        : base(marketData, windowSize, initialCapital, transactionCost, allowShortSelling, randomStart, maxEpisodeLength, seed)
    {
        if (marketData.Shape[1] != 1)
        {
            throw new ArgumentException("StockTradingEnvironment expects a single asset (marketData shape [time, 1]).",
                nameof(marketData));
        }

        _tradeSize = tradeSize;
    }

    /// <summary>
    /// Applies the discrete buy/hold/sell action.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Action 0 = hold, 1 = buy, 2 = sell.
    /// Each buy/sell trades a fixed number of units.
    /// </para>
    /// </remarks>
    protected override void ApplyAction(Vector<T> action, Vector<T> prices)
    {
        int actionIndex = GetActionIndex(action);
        T price = prices[0];

        if (actionIndex == 1)
        {
            ExecuteTrade(0, _tradeSize, price);
        }
        else if (actionIndex == 2)
        {
            ExecuteTrade(0, NumOps.Negate(_tradeSize), price);
        }
    }

    /// <summary>
    /// Converts a Vector action into a discrete action index.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The agent can provide either a single index
    /// (e.g., [2]) or a one-hot vector (e.g., [0, 1, 0]).
    /// </para>
    /// </remarks>
    private int GetActionIndex(Vector<T> action)
    {
        if (action.Length == 1)
        {
            return (int)NumOps.ToDouble(action[0]);
        }

        return FirstMaxIndex(action);
    }

    /// <summary>
    /// The scan <c>best = 0; if (Compare(v[i], v[best]) &gt; 0) best = i</c>: the first index of the largest
    /// score, where a NaN score ranks below every number (as <c>CompareTo</c> orders it) and every score
    /// being NaN gives 0.
    /// </summary>
    /// <remarks>
    /// NaN positions are excluded with an int mask rather than replaced by -infinity, so a real -infinity
    /// score still beats an earlier NaN, and types without infinity (decimal) never convert one.
    /// </remarks>
    internal static int FirstMaxIndex(Vector<T> action)
    {
        int n = action.Length;
        if (n == 0) return 0;
        var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        var scores = action.AsTensor();
        var isNumber = engine.ScalarMinusTensor(1, engine.TensorIsNan(scores).Cast<int>());
        var numberIndices = engine.TensorNonzero(isNumber).ToArray();
        if (numberIndices.Length == 0) return 0;
        var numbers = numberIndices.Length == n
            ? scores
            : engine.TensorIndexSelect(scores, new Tensor<int>(numberIndices, new[] { numberIndices.Length }), 0);
        T best = engine.TensorMaxValue(numbers);
        var isBest = engine.TensorEquals(numbers, best).Cast<int>();
        return numberIndices[engine.TensorNonzero(isBest).ToArray()[0]];
    }
}
