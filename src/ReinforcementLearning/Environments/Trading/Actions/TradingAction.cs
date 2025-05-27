using AiDotNet.Interfaces;
using AiDotNet.Helpers;

namespace AiDotNet.ReinforcementLearning.Environments.Trading.Actions
{
    /// <summary>
    /// Represents an action in a trading environment.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public abstract class TradingAction<T>
    {
        /// <summary>
        /// Gets the numeric operations for type T.
        /// </summary>
        protected INumericOperations<T> NumOps => MathHelper.GetNumericOperations<T>();
        
        /// <summary>
        /// Gets the symbol this action is for.
        /// </summary>
        public string Symbol { get; }

        /// <summary>
        /// Gets the timestamp when this action was taken.
        /// </summary>
        public DateTime Timestamp { get; }

        /// <summary>
        /// Initializes a new instance of the <see cref="TradingAction{T}"/> class.
        /// </summary>
        /// <param name="symbol">The symbol this action is for.</param>
        /// <param name="timestamp">The timestamp when this action is taken.</param>
        protected TradingAction(string symbol, DateTime timestamp)
        {
            Symbol = symbol ?? throw new ArgumentNullException(nameof(symbol));
            Timestamp = timestamp;
        }
    }

    /// <summary>
    /// Represents a market order action (buy or sell immediately at market price).
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class MarketOrderAction<T> : TradingAction<T>
    {
        /// <summary>
        /// Gets the quantity to buy (positive) or sell (negative).
        /// </summary>
        public T Quantity { get; }

        /// <summary>
        /// Initializes a new instance of the <see cref="MarketOrderAction{T}"/> class.
        /// </summary>
        /// <param name="symbol">The symbol to trade.</param>
        /// <param name="quantity">The quantity to buy (positive) or sell (negative).</param>
        /// <param name="timestamp">The timestamp when this action is taken.</param>
        public MarketOrderAction(string symbol, T quantity, DateTime timestamp) 
            : base(symbol, timestamp)
        {
            Quantity = quantity;
        }

        /// <summary>
        /// Returns a string representation of this action.
        /// </summary>
        /// <returns>A string representation of the action.</returns>
        public override string ToString()
        {
            string direction = NumOps.GreaterThan(Quantity, NumOps.Zero) ? "BUY" : "SELL";
            return $"{direction} {Math.Abs(Convert.ToDouble(Quantity))} {Symbol} @ MARKET ({Timestamp})";
        }
    }

    /// <summary>
    /// Represents a hold action (no trading).
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class HoldAction<T> : TradingAction<T>
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="HoldAction{T}"/> class.
        /// </summary>
        /// <param name="symbol">The symbol not being traded.</param>
        /// <param name="timestamp">The timestamp when this action is taken.</param>
        public HoldAction(string symbol, DateTime timestamp) 
            : base(symbol, timestamp)
        {
        }

        /// <summary>
        /// Returns a string representation of this action.
        /// </summary>
        /// <returns>A string representation of the action.</returns>
        public override string ToString()
        {
            return $"HOLD {Symbol} ({Timestamp})";
        }
    }

    /// <summary>
    /// Represents a limit order action (buy or sell at a specified price or better).
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class LimitOrderAction<T> : TradingAction<T>
    {
        /// <summary>
        /// Gets the quantity to buy (positive) or sell (negative).
        /// </summary>
        public T Quantity { get; }
        
        /// <summary>
        /// Gets the limit price.
        /// </summary>
        public T LimitPrice { get; }

        /// <summary>
        /// Initializes a new instance of the <see cref="LimitOrderAction{T}"/> class.
        /// </summary>
        /// <param name="symbol">The symbol to trade.</param>
        /// <param name="quantity">The quantity to buy (positive) or sell (negative).</param>
        /// <param name="limitPrice">The limit price.</param>
        /// <param name="timestamp">The timestamp when this action is taken.</param>
        public LimitOrderAction(string symbol, T quantity, T limitPrice, DateTime timestamp) 
            : base(symbol, timestamp)
        {
            Quantity = quantity;
            LimitPrice = limitPrice;
        }

        /// <summary>
        /// Returns a string representation of this action.
        /// </summary>
        /// <returns>A string representation of the action.</returns>
        public override string ToString()
        {
            string direction = NumOps.GreaterThan(Quantity, NumOps.Zero) ? "BUY" : "SELL";
            return $"{direction} {Math.Abs(Convert.ToDouble(Quantity))} {Symbol} @ LIMIT {Convert.ToDouble(LimitPrice)} ({Timestamp})";
        }
    }

    /// <summary>
    /// Represents a portfolio allocation action (rebalance the portfolio to match target weights).
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class PortfolioAllocationAction<T> : TradingAction<T>
    {
        /// <summary>
        /// Gets the target weight of this asset in the portfolio (0 to 1).
        /// </summary>
        public T TargetWeight { get; }

        /// <summary>
        /// Initializes a new instance of the <see cref="PortfolioAllocationAction{T}"/> class.
        /// </summary>
        /// <param name="symbol">The symbol to allocate.</param>
        /// <param name="targetWeight">The target weight in the portfolio (0 to 1).</param>
        /// <param name="timestamp">The timestamp when this action is taken.</param>
        public PortfolioAllocationAction(string symbol, T targetWeight, DateTime timestamp) 
            : base(symbol, timestamp)
        {
            // Ensure weight is between 0 and 1
            if (NumOps.LessThan(targetWeight, NumOps.Zero) || NumOps.GreaterThan(targetWeight, NumOps.One))
            {
                throw new ArgumentOutOfRangeException(nameof(targetWeight), "Target weight must be between 0 and 1.");
            }
            
            TargetWeight = targetWeight;
        }

        /// <summary>
        /// Returns a string representation of this action.
        /// </summary>
        /// <returns>A string representation of the action.</returns>
        public override string ToString()
        {
            return $"ALLOCATE {Symbol} {Convert.ToDouble(TargetWeight):P2} ({Timestamp})";
        }
    }
}