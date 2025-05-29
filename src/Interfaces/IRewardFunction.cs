using AiDotNet.ReinforcementLearning.Environments.Trading.Portfolio;

namespace AiDotNet.Interfaces
{
    /// <summary>
    /// Interface for reward functions in trading environments.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public interface IRewardFunction<T>
    {
        /// <summary>
        /// Calculates the reward based on the portfolio state.
        /// </summary>
        /// <param name="portfolio">The current portfolio state.</param>
        /// <param name="previousPortfolioValue">The previous portfolio value.</param>
        /// <returns>The calculated reward.</returns>
        T CalculateReward(Portfolio<T> portfolio, T previousPortfolioValue);
    }
}