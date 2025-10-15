using AiDotNet.LinearAlgebra;
using AiDotNet.Interfaces;
using AiDotNet.Helpers;
using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.ReinforcementLearning.Environments.Trading.Portfolio
{
    /// <summary>
    /// Represents a portfolio of positions in financial instruments.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class Portfolio<T>
    {
        private readonly Dictionary<string, Position<T>> _positions = default!;
        private T _cash = default!;
        private T _initialCash = default!;
        private readonly List<Transaction<T>> _transactions = default!;
        private readonly T _tradingFeeRate = default!;
        /// <summary>
        /// Gets the numeric operations for type T.
        /// </summary>
        protected INumericOperations<T> NumOps => MathHelper.GetNumericOperations<T>();
        
        /// <summary>
        /// Gets the cash balance in the portfolio.
        /// </summary>
        public T Cash => _cash;
        
        /// <summary>
        /// Gets the total value of all positions (excluding cash).
        /// </summary>
        public T PositionsValue
        {
            get
            {
                T total = NumOps.Zero;
                foreach (var position in _positions.Values)
                {
                    total = NumOps.Add(total, position.CurrentValue);
                }
                return total;
            }
        }
        
        /// <summary>
        /// Gets the total portfolio value (positions + cash).
        /// </summary>
        public T TotalValue => NumOps.Add(PositionsValue, Cash);
        
        /// <summary>
        /// Gets the total unrealized profit/loss of all positions.
        /// </summary>
        public T TotalUnrealizedPnL
        {
            get
            {
                T total = NumOps.Zero;
                foreach (var position in _positions.Values)
                {
                    total = NumOps.Add(total, position.UnrealizedPnL);
                }
                return total;
            }
        }
        
        /// <summary>
        /// Gets the total realized profit/loss from all closed trades.
        /// </summary>
        public T TotalRealizedPnL
        {
            get
            {
                T total = NumOps.Zero;
                foreach (var position in _positions.Values)
                {
                    total = NumOps.Add(total, position.RealizedPnL);
                }
                return total;
            }
        }
        
        /// <summary>
        /// Gets the total profit/loss (realized + unrealized).
        /// </summary>
        public T TotalPnL => NumOps.Add(TotalRealizedPnL, TotalUnrealizedPnL);
        
        /// <summary>
        /// Gets the portfolio return percentage relative to the initial cash.
        /// </summary>
        public T ReturnPercentage
        {
            get
            {
                if (NumOps.Equals(_initialCash, NumOps.Zero)) return NumOps.Zero;
                return NumOps.Divide(
                    NumOps.Subtract(TotalValue, _initialCash), 
                    _initialCash);
            }
        }
        
        /// <summary>
        /// Gets a readonly collection of positions in the portfolio.
        /// </summary>
        public IReadOnlyCollection<Position<T>> Positions => _positions.Values;
        
        /// <summary>
        /// Gets a readonly collection of all transactions.
        /// </summary>
        public IReadOnlyCollection<Transaction<T>> Transactions => _transactions;
        
        /// <summary>
        /// Gets the trading fee rate as a decimal (e.g., 0.001 for 0.1%).
        /// </summary>
        public T TradingFeeRate => _tradingFeeRate;

        /// <summary>
        /// Initializes a new instance of the <see cref="Portfolio{T}"/> class.
        /// </summary>
        /// <param name="initialCash">The initial cash balance.</param>
        /// <param name="tradingFeeRate">The trading fee rate as a decimal (e.g., 0.001 for 0.1%).</param>
        public Portfolio(T initialCash, T? tradingFeeRate)
        {
            if (NumOps.LessThan(initialCash, NumOps.Zero))
            {
                throw new ArgumentException("Initial cash cannot be negative.", nameof(initialCash));
            }
            
            _cash = initialCash;
            _initialCash = initialCash;
            _tradingFeeRate = tradingFeeRate ?? NumOps.Zero;
            _positions = new Dictionary<string, Position<T>>();
            _transactions = new List<Transaction<T>>();
        }

        /// <summary>
        /// Gets a position by symbol.
        /// </summary>
        /// <param name="symbol">The symbol to get the position for.</param>
        /// <returns>The position, or null if no position exists for the symbol.</returns>
        public Position<T>? GetPosition(string symbol)
        {
            return _positions.TryGetValue(symbol, out var position) ? position : null;
        }

        /// <summary>
        /// Gets or creates a position for a symbol.
        /// </summary>
        /// <param name="symbol">The symbol to get or create a position for.</param>
        /// <param name="currentPrice">The current price to use if creating a new position.</param>
        /// <param name="timestamp">The timestamp to use if creating a new position.</param>
        /// <returns>The existing or newly created position.</returns>
        public Position<T> GetOrCreatePosition(string symbol, T currentPrice, DateTime timestamp)
        {
            if (!_positions.TryGetValue(symbol, out var position))
            {
                position = new Position<T>(symbol, NumOps.Zero, NumOps.Zero, currentPrice, timestamp);
                _positions[symbol] = position;
            }
            return position;
        }

        /// <summary>
        /// Updates the prices of all positions.
        /// </summary>
        /// <param name="prices">A dictionary mapping symbols to their current prices.</param>
        /// <param name="timestamp">The timestamp of the update.</param>
        public void UpdatePrices(Dictionary<string, T> prices, DateTime timestamp)
        {
            foreach (var price in prices)
            {
                if (_positions.TryGetValue(price.Key, out var position))
                {
                    position.UpdatePrice(price.Value, timestamp);
                }
                else
                {
                    // Create a position with zero quantity at the current price
                    _positions[price.Key] = new Position<T>(price.Key, NumOps.Zero, NumOps.Zero, price.Value, timestamp);
                }
            }
        }

        /// <summary>
        /// Executes a market trade.
        /// </summary>
        /// <param name="symbol">The symbol to trade.</param>
        /// <param name="quantity">The quantity to trade (positive for buy, negative for sell).</param>
        /// <param name="price">The execution price.</param>
        /// <param name="timestamp">The timestamp of the trade.</param>
        /// <returns>A tuple containing the transaction and the realized profit/loss, if any.</returns>
        /// <exception cref="InvalidOperationException">Thrown when there is insufficient cash for a buy.</exception>
        public (Transaction<T> Transaction, T RealizedPnL) ExecuteTrade(string symbol, T quantity, T price, DateTime timestamp)
        {
            if (NumOps.Equals(quantity, NumOps.Zero)) 
            {
                throw new ArgumentException("Quantity cannot be zero.", nameof(quantity));
            }
            
            if (NumOps.LessThanOrEquals(price, NumOps.Zero))
            {
                throw new ArgumentException("Price must be positive.", nameof(price));
            }
            
            // Calculate the total cost/proceeds including fees
            T tradeValue = NumOps.Multiply(quantity, price);
            T fee = NumOps.Multiply(NumOps.Abs(tradeValue), _tradingFeeRate);
            T totalCost = NumOps.Add(tradeValue, fee);
            
            // For buys, check if we have enough cash
            if (NumOps.GreaterThan(quantity, NumOps.Zero) && NumOps.GreaterThan(totalCost, _cash))
            {
                throw new InvalidOperationException($"Insufficient cash for trade. Required: {totalCost}, Available: {_cash}");
            }
            
            // Get or create the position
            var position = GetOrCreatePosition(symbol, price, timestamp);
            
            // Update the position and get the realized P&L
            T realizedPnL = position.UpdatePosition(quantity, price, timestamp);
            
            // Create a transaction record
            var transaction = new Transaction<T>(
                symbol,
                quantity,
                price,
                fee,
                timestamp,
                _cash,
                TotalValue);
            
            // Update cash balance
            _cash = NumOps.Subtract(_cash, totalCost);
            
            // Record the transaction
            _transactions.Add(transaction);
            
            return (transaction, realizedPnL);
        }

        /// <summary>
        /// Calculates the quantity that can be bought with a specified amount of cash.
        /// </summary>
        /// <param name="cash">The amount of cash to use.</param>
        /// <param name="price">The price of the asset.</param>
        /// <returns>The maximum quantity that can be bought.</returns>
        public T CalculateMaxBuyQuantity(T cash, T price)
        {
            if (NumOps.LessThanOrEquals(cash, NumOps.Zero) || NumOps.LessThanOrEquals(price, NumOps.Zero))
            {
                return NumOps.Zero;
            }
            
            // Account for trading fees
            T feeRate = NumOps.Add(NumOps.One, _tradingFeeRate);
            return NumOps.Divide(cash, NumOps.Multiply(price, feeRate));
        }

        /// <summary>
        /// Calculates the weight of a position in the portfolio.
        /// </summary>
        /// <param name="symbol">The symbol to calculate the weight for.</param>
        /// <returns>The weight of the position (0 to 1).</returns>
        public T GetPositionWeight(string symbol)
        {
            if (NumOps.Equals(TotalValue, NumOps.Zero))
            {
                return NumOps.Zero;
            }
            
            if (!_positions.TryGetValue(symbol, out var position))
            {
                return NumOps.Zero;
            }
            
            return NumOps.Divide(position.CurrentValue, TotalValue);
        }

        /// <summary>
        /// Rebalances the portfolio to match target weights.
        /// </summary>
        /// <param name="targetWeights">A dictionary mapping symbols to their target weights.</param>
        /// <param name="prices">A dictionary mapping symbols to their current prices.</param>
        /// <param name="timestamp">The timestamp of the rebalance.</param>
        /// <returns>A list of transactions executed during the rebalance.</returns>
        public List<Transaction<T>> Rebalance(Dictionary<string, T> targetWeights, Dictionary<string, T> prices, DateTime timestamp)
        {
            var transactions = new List<Transaction<T>>();
            
            // Ensure all weights are between 0 and 1
            foreach (var weight in targetWeights)
            {
                if (NumOps.LessThan(weight.Value, NumOps.Zero) || NumOps.GreaterThan(weight.Value, NumOps.One))
                {
                    throw new ArgumentException($"Target weight for {weight.Key} must be between 0 and 1.");
                }
            }
            
            // Check that weights sum to at most 1
            T totalWeight = NumOps.Zero;
            foreach (var weight in targetWeights.Values)
            {
                totalWeight = NumOps.Add(totalWeight, weight);
            }
            
            if (NumOps.GreaterThan(totalWeight, NumOps.One))
            {
                throw new ArgumentException("Sum of target weights cannot exceed 1.");
            }
            
            // Update prices first
            UpdatePrices(prices, timestamp);
            
            // Calculate target values for each position
            var targetValues = new Dictionary<string, T>();
            foreach (var weight in targetWeights)
            {
                targetValues[weight.Key] = NumOps.Multiply(TotalValue, weight.Value);
            }
            
            // Calculate the difference between current and target values
            var trades = new Dictionary<string, T>();
            foreach (var symbol in targetValues.Keys.Union(_positions.Keys))
            {
                T targetValue = targetValues.TryGetValue(symbol, out var value) ? value : NumOps.Zero;
                T currentValue = _positions.TryGetValue(symbol, out var position) ? position.CurrentValue : NumOps.Zero;
                T valueDiff = NumOps.Subtract(targetValue, currentValue);
                
                if (!NumOps.Equals(valueDiff, NumOps.Zero) && prices.TryGetValue(symbol, out var price))
                {
                    // Calculate the quantity to trade
                    trades[symbol] = NumOps.Divide(valueDiff, price);
                }
            }
            
            // Execute trades
            foreach (var trade in trades)
            {
                if (!NumOps.Equals(trade.Value, NumOps.Zero) && prices.TryGetValue(trade.Key, out var price))
                {
                    try
                    {
                        var (transaction, _) = ExecuteTrade(trade.Key, trade.Value, price, timestamp);
                        transactions.Add(transaction);
                    }
                    catch (InvalidOperationException)
                    {
                        // If we don't have enough cash, trade as much as we can
                        if (NumOps.GreaterThan(trade.Value, NumOps.Zero))
                        {
                            T maxQuantity = CalculateMaxBuyQuantity(_cash, price);
                            if (NumOps.GreaterThan(maxQuantity, NumOps.Zero))
                            {
                                var (transaction, _) = ExecuteTrade(trade.Key, maxQuantity, price, timestamp);
                                transactions.Add(transaction);
                            }
                        }
                    }
                }
            }
            
            return transactions;
        }

        /// <summary>
        /// Gets a feature vector representing the current state of the portfolio.
        /// </summary>
        /// <param name="symbols">The symbols to include in the feature vector.</param>
        /// <returns>A feature vector with normalized portfolio and position information.</returns>
        public Vector<T> GetFeatureVector(IEnumerable<string> symbols)
        {
            var features = new List<T>();
            
            // Normalize cash to initial cash
            features.Add(NumOps.Divide(_cash, _initialCash));
            
            // Portfolio return
            features.Add(ReturnPercentage);
            
            // Position features for each symbol
            foreach (var symbol in symbols)
            {
                if (_positions.TryGetValue(symbol, out var position))
                {
                    // Position weight in portfolio
                    features.Add(NumOps.Divide(position.CurrentValue, TotalValue));
                    
                    // Unrealized P&L percentage
                    features.Add(position.UnrealizedPnLPercent);
                }
                else
                {
                    // No position
                    features.Add(NumOps.Zero); // Weight
                    features.Add(NumOps.Zero); // P&L percentage
                }
            }
            
            return new Vector<T>(features.ToArray());
        }
    }
}