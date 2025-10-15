using AiDotNet.ReinforcementLearning.Environments.Trading.Portfolio;
using AiDotNet.Interfaces;
using AiDotNet.Helpers;

namespace AiDotNet.ReinforcementLearning.Environments.Trading.Rewards
{
    /// <summary>
    /// A reward function based on the Sharpe ratio (risk-adjusted returns).
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class SharpeRatioRewardFunction<T> : IRewardFunction<T>
    {
        /// <summary>
        /// Gets the numeric operations for type T.
        /// </summary>
        protected INumericOperations<T> NumOps => MathHelper.GetNumericOperations<T>();
        private readonly Queue<T> _returns = default!;
        private readonly int _windowSize;
        private readonly T _riskFreeRate = default!;
        private readonly T _scalingFactor = default!;
        private readonly T _annualizationFactor = default!;
        
        /// <summary>
        /// Initializes a new instance of the <see cref="SharpeRatioRewardFunction{T}"/> class.
        /// </summary>
        /// <param name="windowSize">The window size for calculating the Sharpe ratio.</param>
        /// <param name="riskFreeRate">The risk-free rate expressed as a daily rate.</param>
        /// <param name="scalingFactor">The scaling factor for the reward.</param>
        /// <param name="tradingDaysPerYear">The number of trading days per year for annualization.</param>
        public SharpeRatioRewardFunction(
            int windowSize = 30, 
            double riskFreeRate = 0.0, 
            double scalingFactor = 1.0,
            int tradingDaysPerYear = 252)
        {
            _windowSize = windowSize;
            _riskFreeRate = NumOps.FromDouble(riskFreeRate);
            _scalingFactor = NumOps.FromDouble(scalingFactor);
            _annualizationFactor = NumOps.FromDouble(Math.Sqrt(tradingDaysPerYear));
            _returns = new Queue<T>();
        }
        
        /// <summary>
        /// Calculates the reward based on the Sharpe ratio.
        /// </summary>
        /// <param name="portfolio">The current portfolio state.</param>
        /// <param name="previousPortfolioValue">The previous portfolio value.</param>
        /// <returns>The calculated reward.</returns>
        public T CalculateReward(Portfolio<T> portfolio, T previousPortfolioValue)
        {
            // Calculate the return for this period
            T currentValue = portfolio.TotalValue;
            
            if (NumOps.LessThanOrEquals(previousPortfolioValue, NumOps.Zero))
            {
                return NumOps.Zero;
            }
            
            // Simple return: (current - previous) / previous
            T periodReturn = NumOps.Divide(
                NumOps.Subtract(currentValue, previousPortfolioValue), 
                previousPortfolioValue);
            
            // Add the return to the queue
            _returns.Enqueue(periodReturn);
            
            // Keep only the last windowSize returns
            while (_returns.Count > _windowSize)
            {
                _returns.Dequeue();
            }
            
            // If we don't have enough data yet, return a small reward for surviving
            if (_returns.Count < _windowSize)
            {
                return NumOps.Multiply(periodReturn, _scalingFactor);
            }
            
            // Calculate mean excess return
            T sumReturns = NumOps.Zero;
            foreach (var ret in _returns)
            {
                sumReturns = NumOps.Add(sumReturns, NumOps.Subtract(ret, _riskFreeRate));
            }
            T meanExcessReturn = NumOps.Divide(sumReturns, NumOps.FromDouble(_returns.Count));
            
            // Calculate standard deviation
            T sumSquaredDeviations = NumOps.Zero;
            foreach (var ret in _returns)
            {
                T deviation = NumOps.Subtract(NumOps.Subtract(ret, _riskFreeRate), meanExcessReturn);
                sumSquaredDeviations = NumOps.Add(sumSquaredDeviations, NumOps.Multiply(deviation, deviation));
            }
            
            T stdDev = NumOps.Sqrt(NumOps.Divide(sumSquaredDeviations, NumOps.FromDouble(_returns.Count)));
            
            // Calculate Sharpe ratio
            T sharpeRatio;
            if (NumOps.GreaterThan(stdDev, NumOps.Zero))
            {
                sharpeRatio = NumOps.Multiply(NumOps.Divide(meanExcessReturn, stdDev), _annualizationFactor);
            }
            else
            {
                // If there's no volatility, use a default reward based on the return
                sharpeRatio = NumOps.GreaterThan(meanExcessReturn, NumOps.Zero) ? NumOps.FromDouble(1.0) : NumOps.Zero;
            }
            
            // Scale the reward
            return NumOps.Multiply(sharpeRatio, _scalingFactor);
        }
    }
}