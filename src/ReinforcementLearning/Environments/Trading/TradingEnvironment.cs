using AiDotNet.LinearAlgebra;
using AiDotNet.ReinforcementLearning.Environments.Trading.Actions;
using AiDotNet.ReinforcementLearning.Environments.Trading.MarketData;
using AiDotNet.ReinforcementLearning.Environments.Trading.Portfolio;
using AiDotNet.ReinforcementLearning.Environments.Trading.Rewards;
using AiDotNet.Interfaces;
using AiDotNet.Extensions;

namespace AiDotNet.ReinforcementLearning.Environments.Trading
{
    /// <summary>
    /// A reinforcement learning environment for financial trading.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class TradingEnvironment<T> : EnvironmentBase<TradingEnvironmentState<T>, object, T> 
       
    {
        private readonly IMarketDataFeed<T> _marketData = default!;
        private readonly Portfolio<T> _portfolio = default!;
        private readonly IRewardFunction<T> _rewardFunction = default!;
        private readonly int _lookbackWindow;
        private readonly List<string> _symbols = default!;
        private readonly List<string> _marketFeatures = default!;
        private readonly bool _allowShort;
        private readonly bool _allowFractionalShares;
        private readonly int _warmupPeriod;
        private readonly List<(DateTime Timestamp, T Reward, T PortfolioValue)> _performanceHistory;
        private readonly INumericOperations<T> _numOps = default!;
        
        /// <summary>
        /// Gets the size of the state space.
        /// </summary>
        public override int StateSize => 
            (_lookbackWindow * _marketFeatures.Count * _symbols.Count) + // Market features
            (2 + 2 * _symbols.Count); // Account features: cash ratio, return + position weight and P&L for each symbol
        
        /// <summary>
        /// Gets the size of the action space.
        /// </summary>
        /// <remarks>
        /// For discrete action space: hold + buy + sell for each symbol.
        /// For continuous action space: target weight for each symbol.
        /// </remarks>
        public override int ActionSize => IsContinuous ? _symbols.Count : (2 * _symbols.Count) + 1;
        
        /// <summary>
        /// Gets a value indicating whether the action space is continuous.
        /// </summary>
        public override bool IsContinuous { get; }
        
        /// <summary>
        /// Gets the minimum values for each dimension of the action space (for continuous action spaces).
        /// </summary>
        public override Vector<T>? ActionLowerBound => 
            IsContinuous ? new Vector<T>(Enumerable.Repeat(_numOps.Zero, ActionSize).ToArray()) : null;
        
        /// <summary>
        /// Gets the maximum values for each dimension of the action space (for continuous action spaces).
        /// </summary>
        public override Vector<T>? ActionUpperBound => 
            IsContinuous ? new Vector<T>(Enumerable.Repeat(_numOps.One, ActionSize).ToArray()) : null;

        /// <summary>
        /// Gets the total return of the portfolio.
        /// </summary>
        public T TotalReturn => _portfolio.ReturnPercentage;
        
        /// <summary>
        /// Gets the current portfolio value.
        /// </summary>
        public T PortfolioValue => _portfolio.TotalValue;
        
        /// <summary>
        /// Gets a list of all transactions executed in the environment.
        /// </summary>
        public IReadOnlyCollection<Transaction<T>> Transactions => _portfolio.Transactions;
        
        /// <summary>
        /// Gets the performance history (timestamp, reward, portfolio value).
        /// </summary>
        public IReadOnlyCollection<(DateTime Timestamp, T Reward, T PortfolioValue)> PerformanceHistory => _performanceHistory;
        
        /// <summary>
        /// Gets the current timestamp in the environment.
        /// </summary>
        public DateTime CurrentTimestamp => _marketData.CurrentTimestamp;

        /// <summary>
        /// Initializes a new instance of the <see cref="TradingEnvironment{T}"/> class.
        /// </summary>
        /// <param name="marketData">The market data feed to use.</param>
        /// <param name="initialCash">The initial cash balance.</param>
        /// <param name="rewardFunction">The reward function to use.</param>
        /// <param name="numericOps">The numeric operations for type T.</param>
        /// <param name="lookbackWindow">The number of historical data points to include in the state.</param>
        /// <param name="allowShort">Whether to allow short selling.</param>
        /// <param name="allowFractionalShares">Whether to allow fractional shares.</param>
        /// <param name="tradingFeeRate">The trading fee rate as a decimal (e.g., 0.001 for 0.1%).</param>
        /// <param name="isContinuous">Whether to use a continuous action space (portfolio weights).</param>
        /// <param name="warmupPeriod">The number of steps to use for warmup (not included in training).</param>
        public TradingEnvironment(
            IMarketDataFeed<T> marketData,
            T initialCash,
            IRewardFunction<T> rewardFunction,
            INumericOperations<T> numericOps,
            int lookbackWindow = 10,
            bool allowShort = false,
            bool allowFractionalShares = true,
            T? tradingFeeRate = default,
            bool isContinuous = true,
            int warmupPeriod = 0)
        {
            _marketData = marketData ?? throw new ArgumentNullException(nameof(marketData));
            _rewardFunction = rewardFunction ?? throw new ArgumentNullException(nameof(rewardFunction));
            _numOps = numericOps ?? throw new ArgumentNullException(nameof(numericOps));
            _lookbackWindow = lookbackWindow;
            _allowShort = allowShort;
            _allowFractionalShares = allowFractionalShares;
            _portfolio = new Portfolio<T>(initialCash, tradingFeeRate);
            _symbols = marketData.Symbols.ToList();
            _marketFeatures = marketData.Features.ToList();
            IsContinuous = isContinuous;
            _warmupPeriod = warmupPeriod;
            _performanceHistory = new List<(DateTime, T, T)>();
        }

        /// <summary>
        /// Gets the current state of the environment.
        /// </summary>
        /// <returns>The current state observation.</returns>
        public override TradingEnvironmentState<T> GetState()
        {
            // Get current market data
            var currentData = _marketData.GetCurrentData();
            
            // Build market features
            var marketFeatures = new List<T>();
            
            // For each symbol
            foreach (var symbol in _symbols)
            {
                // For each feature
                foreach (var feature in _marketFeatures)
                {
                    // Get historical values with lookback
                    var values = _marketData.GetHistoricalValues(symbol, feature, _lookbackWindow);
                    marketFeatures.AddRange(values);
                }
            }
            
            // Build account features using portfolio
            var accountFeatures = _portfolio.GetFeatureVector(_symbols).ToArray();
            
            return new TradingEnvironmentState<T>(
                marketFeatures.ToArray(),
                accountFeatures,
                currentData.Timestamp,
                GetMarketFeatureNames(),
                GetAccountFeatureNames());
        }

        /// <summary>
        /// Takes an action in the environment and returns the result.
        /// </summary>
        /// <param name="action">The action to take.</param>
        /// <returns>A tuple containing the new state, the reward received, and a flag indicating if the episode is done.</returns>
        public override (TradingEnvironmentState<T> nextState, T reward, bool done) Step(object action)
        {
            // Get current prices
            var currentData = _marketData.GetCurrentData();
            var prices = new Dictionary<string, T>();
            
            foreach (var symbol in _symbols)
            {
                if (currentData.TryGetValue(symbol, "Close", out var price))
                {
                    prices[symbol] = price;
                }
            }
            
            // Update portfolio with current prices
            _portfolio.UpdatePrices(prices, currentData.Timestamp);
            
            // Record the portfolio value before the action
            T portfolioValueBefore = _portfolio.TotalValue;
            
            // Execute the action
            if (IsContinuous)
            {
                ExecuteContinuousAction(action, prices, currentData.Timestamp);
            }
            else
            {
                ExecuteDiscreteAction(action, prices, currentData.Timestamp);
            }
            
            // Try to move to the next time step
            bool done = !_marketData.MoveNext();
            
            if (!done)
            {
                // Update portfolio with new prices
                var newData = _marketData.GetCurrentData();
                var newPrices = new Dictionary<string, T>();
                
                foreach (var symbol in _symbols)
                {
                    if (newData.TryGetValue(symbol, "Close", out var price))
                    {
                        newPrices[symbol] = price;
                    }
                }
                
                _portfolio.UpdatePrices(newPrices, newData.Timestamp);
            }
            
            // Calculate reward
            T reward = _rewardFunction.CalculateReward(_portfolio, portfolioValueBefore);
            
            // Record performance
            _performanceHistory.Add((currentData.Timestamp, reward, _portfolio.TotalValue));
            
            // Get the new state
            var nextState = GetState();
            
            return (nextState, reward, done);
        }

        /// <summary>
        /// Resets the environment to an initial state and returns that state.
        /// </summary>
        /// <returns>The initial state observation.</returns>
        public override TradingEnvironmentState<T> Reset()
        {
            // Reset market data
            _marketData.Reset(_warmupPeriod > 0);
            
            // Create a new portfolio
            var initialCash = _portfolio.TotalValue; // Use current portfolio value to keep the same scale
            if (_numOps.LessThanOrEquals(initialCash, _numOps.Zero))
            {
                // If portfolio is empty or negative, use a default value
                initialCash = _numOps.FromDouble(100000);
            }
            
            // Clear performance history
            _performanceHistory.Clear();
            
            // Get current prices and update the portfolio
            var currentData = _marketData.GetCurrentData();
            var prices = new Dictionary<string, T>();
            
            foreach (var symbol in _symbols)
            {
                if (currentData.TryGetValue(symbol, "Close", out var price))
                {
                    prices[symbol] = price;
                }
            }
            
            _portfolio.UpdatePrices(prices, currentData.Timestamp);
            
            return GetState();
        }

        /// <summary>
        /// Executes a continuous action (portfolio weights).
        /// </summary>
        /// <param name="action">The action to execute.</param>
        /// <param name="prices">The current prices.</param>
        /// <param name="timestamp">The current timestamp.</param>
        private void ExecuteContinuousAction(object action, Dictionary<string, T> prices, DateTime timestamp)
        {
            // Validate action
            if (action is not Vector<T> weightVector)
            {
                throw new ArgumentException("Continuous action must be a Vector<T>.", nameof(action));
            }
            
            if (weightVector.Length != _symbols.Count)
            {
                throw new ArgumentException($"Action vector length ({weightVector.Length}) does not match number of symbols ({_symbols.Count}).", nameof(action));
            }
            
            // Convert to dictionary
            var targetWeights = new Dictionary<string, T>();
            for (int i = 0; i < _symbols.Count; i++)
            {
                T weight = weightVector[i];
                
                // Clamp weight between 0 and 1
                T minWeight = _allowShort ? _numOps.Negate(_numOps.One) : _numOps.Zero;
                weight = _numOps.LessThan(weight, minWeight) ? 
                    minWeight : (_numOps.GreaterThan(weight, _numOps.One) ? _numOps.One : weight);
                
                targetWeights[_symbols[i]] = weight;
            }
            
            // Rebalance portfolio
            _portfolio.Rebalance(targetWeights, prices, timestamp);
        }

        /// <summary>
        /// Executes a discrete action (buy, sell, hold).
        /// </summary>
        /// <param name="action">The action to execute.</param>
        /// <param name="prices">The current prices.</param>
        /// <param name="timestamp">The current timestamp.</param>
        private void ExecuteDiscreteAction(object action, Dictionary<string, T> prices, DateTime timestamp)
        {
            // Validate action
            int actionId;
            if (action is int intAction)
            {
                actionId = intAction;
            }
            else if (action is T numericAction)
            {
                actionId = _numOps.ToInt32(numericAction);
            }
            else
            {
                throw new ArgumentException("Discrete action must be an int or T.", nameof(action));
            }
            
            if (actionId < 0 || actionId >= ActionSize)
            {
                throw new ArgumentException($"Action ID ({actionId}) out of range (0 to {ActionSize - 1}).", nameof(action));
            }
            
            // Interpret the action
            if (actionId == 0)
            {
                // Hold - do nothing
                return;
            }
            
            // Determine which symbol and whether to buy or sell
            int symbolIndex = (actionId - 1) / 2;
            bool isBuy = (actionId - 1) % 2 == 0;
            
            if (symbolIndex < 0 || symbolIndex >= _symbols.Count)
            {
                throw new InvalidOperationException($"Invalid symbol index: {symbolIndex}");
            }
            
            string symbol = _symbols[symbolIndex];
            
            // Check if we have the price for this symbol
            if (!prices.TryGetValue(symbol, out var price))
            {
                return; // Skip if price is not available
            }
            
            // Calculate quantity
            T quantity;
            if (isBuy)
            {
                // Buy - use 10% of available cash (simplified)
                T cashToUse = _numOps.Multiply(_portfolio.Cash, _numOps.FromDouble(0.1));
                quantity = _portfolio.CalculateMaxBuyQuantity(cashToUse, price);
                
                // Round to whole shares if fractional shares are not allowed
                if (!_allowFractionalShares)
                {
                    quantity = MathHelper.Floor(quantity);
                }
                
                // Skip if quantity is zero
                if (_numOps.LessThanOrEquals(quantity, _numOps.Zero))
                {
                    return;
                }
            }
            else
            {
                // Sell - sell all of the position (simplified)
                var position = _portfolio.GetPosition(symbol);
                if (position == null || _numOps.LessThanOrEquals(position.Quantity, _numOps.Zero))
                {
                    // If short selling is not allowed, skip
                    if (!_allowShort)
                    {
                        return;
                    }
                    
                    // Short sell - use 10% of cash as the value to short
                    T cashToUse = _numOps.Multiply(_portfolio.Cash, _numOps.FromDouble(0.1));
                    quantity = _numOps.Negate(_portfolio.CalculateMaxBuyQuantity(cashToUse, price));
                    
                    // Round to whole shares if fractional shares are not allowed
                    if (!_allowFractionalShares)
                    {
                        quantity = MathHelper.Ceiling(quantity);
                    }
                    
                    // Skip if quantity is zero
                    if (_numOps.GreaterThanOrEquals(quantity, _numOps.Zero))
                    {
                        return;
                    }
                }
                else
                {
                    // Sell all of the position
                    quantity = _numOps.Negate(position.Quantity);
                }
            }
            
            // Execute the trade
            try
            {
                _portfolio.ExecuteTrade(symbol, quantity, price, timestamp);
            }
            catch (InvalidOperationException)
            {
                // Ignore if trade fails (e.g., insufficient cash)
            }
        }

        /// <summary>
        /// Gets the names of all market features in the state.
        /// </summary>
        /// <returns>The list of market feature names.</returns>
        private List<string> GetMarketFeatureNames()
        {
            var names = new List<string>();
            
            foreach (var symbol in _symbols)
            {
                foreach (var feature in _marketFeatures)
                {
                    for (int i = 0; i < _lookbackWindow; i++)
                    {
                        names.Add($"{symbol}_{feature}_{i}");
                    }
                }
            }
            
            return names;
        }

        /// <summary>
        /// Gets the names of all account features in the state.
        /// </summary>
        /// <returns>The list of account feature names.</returns>
        private List<string> GetAccountFeatureNames()
        {
            var names = new List<string>
            {
                "cash_ratio",
                "portfolio_return"
            };
            
            foreach (var symbol in _symbols)
            {
                names.Add($"{symbol}_weight");
                names.Add($"{symbol}_pnl_percent");
            }
            
            return names;
        }

        /// <summary>
        /// Gets information about the environment.
        /// </summary>
        /// <returns>A dictionary containing information about the environment.</returns>
        public override Dictionary<string, object> GetInfo()
        {
            var info = base.GetInfo();
            
            info["symbols"] = _symbols;
            info["features"] = _marketFeatures;
            info["lookback_window"] = _lookbackWindow;
            info["allow_short"] = _allowShort;
            info["allow_fractional_shares"] = _allowFractionalShares;
            info["portfolio_value"] = _portfolio.TotalValue!;
            info["cash"] = _portfolio.Cash!;
            info["total_return"] = _portfolio.ReturnPercentage!;
            info["trading_fee_rate"] = _portfolio.TradingFeeRate!;
            
            return info;
        }

        /// <summary>
        /// Renders the environment (optional, for visualization).
        /// </summary>
        /// <param name="mode">The rendering mode (e.g., "human", "rgb_array").</param>
        /// <returns>Rendering result, if applicable.</returns>
        public override object? Render(string mode = "human")
        {
            // Output current state to console
            if (mode == "human")
            {
                Console.WriteLine($"Time: {_marketData.CurrentTimestamp}, Portfolio Value: {_portfolio.TotalValue}, Return: {_portfolio.ReturnPercentage:P2}");
                
                // Display positions
                Console.WriteLine("Positions:");
                foreach (var position in _portfolio.Positions)
                {
                    if (!position.IsFlat)
                    {
                        Console.WriteLine($"  {position}");
                    }
                }
                
                // Display last few transactions
                Console.WriteLine("Recent Transactions:");
                var transactions = _portfolio.Transactions.ToList();
                var lastTransactions = transactions.Skip(Math.Max(0, transactions.Count - 5));
                foreach (var transaction in lastTransactions)
                {
                    Console.WriteLine($"  {transaction}");
                }
                
                return null;
            }
            
            // Could add support for other rendering modes later
            return null;
        }
    }
}