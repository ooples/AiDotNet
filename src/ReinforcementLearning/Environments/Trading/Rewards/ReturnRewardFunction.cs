using AiDotNet.ReinforcementLearning.Environments.Trading.Portfolio;
using AiDotNet.Interfaces;
using AiDotNet.Helpers;

namespace AiDotNet.ReinforcementLearning.Environments.Trading.Rewards
{
    /// <summary>
    /// A reward function based on portfolio returns.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class ReturnRewardFunction<T> : IRewardFunction<T>
    {
        /// <summary>
        /// Gets the numeric operations for type T.
        /// </summary>
        protected INumericOperations<T> NumOps => MathHelper.GetNumericOperations<T>();
        private readonly bool _useLogReturn;
        private readonly T _scalingFactor = default!;
        private readonly bool _punishDrawdown;
        
        /// <summary>
        /// Initializes a new instance of the <see cref="ReturnRewardFunction{T}"/> class.
        /// </summary>
        /// <param name="useLogReturn">Whether to use logarithmic returns instead of simple returns.</param>
        /// <param name="scalingFactor">The scaling factor for the reward.</param>
        /// <param name="punishDrawdown">Whether to additionally punish drawdowns.</param>
        public ReturnRewardFunction(bool useLogReturn = false, double scalingFactor = 1.0, bool punishDrawdown = false)
        {
            _useLogReturn = useLogReturn;
            _scalingFactor = NumOps.FromDouble(scalingFactor);
            _punishDrawdown = punishDrawdown;
        }
        
        /// <summary>
        /// Calculates the reward based on the portfolio returns.
        /// </summary>
        /// <param name="portfolio">The current portfolio state.</param>
        /// <param name="previousPortfolioValue">The previous portfolio value.</param>
        /// <returns>The calculated reward.</returns>
        public T CalculateReward(Portfolio<T> portfolio, T previousPortfolioValue)
        {
            // Calculate return
            T currentValue = portfolio.TotalValue;
            
            if (NumOps.LessThanOrEquals(previousPortfolioValue, NumOps.Zero))
            {
                return NumOps.Zero;
            }
            
            T baseReward;
            
            if (_useLogReturn)
            {
                // Log return: log(current/previous)
                baseReward = NumOps.Log(NumOps.Divide(currentValue, previousPortfolioValue));
            }
            else
            {
                // Simple return: (current - previous) / previous
                baseReward = NumOps.Divide(
                    NumOps.Subtract(currentValue, previousPortfolioValue), 
                    previousPortfolioValue);
            }
            
            // Apply scaling
            baseReward = NumOps.Multiply(baseReward, _scalingFactor);
            
            // Punish drawdowns more if enabled
            if (_punishDrawdown && NumOps.LessThan(baseReward, NumOps.Zero))
            {
                baseReward = NumOps.Multiply(baseReward, NumOps.FromDouble(2.0)); // Double the penalty for losses
            }
            
            return baseReward;
        }
    }
}