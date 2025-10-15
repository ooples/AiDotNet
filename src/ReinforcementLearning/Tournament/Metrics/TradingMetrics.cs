using AiDotNet.ReinforcementLearning.Tournament.Results;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.ReinforcementLearning.Tournament
{
    /// <summary>
    /// Calculates the total return metric for a trading episode.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class TotalReturnMetric<T> : IEvaluationMetric<T>
    {
        /// <summary>
        /// Gets the numeric operations for type T.
        /// </summary>
        protected INumericOperations<T> NumOps => MathHelper.GetNumericOperations<T>();
        
        /// <summary>
        /// Gets the name of the metric.
        /// </summary>
        public string Name => "Total Return";
        
        /// <summary>
        /// Gets the description of the metric.
        /// </summary>
        public string Description => "Total percentage return of the portfolio over the episode.";
        
        /// <summary>
        /// Gets a value indicating whether higher values of this metric are better.
        /// </summary>
        public bool HigherIsBetter => true;
        
        /// <summary>
        /// Calculates the total return for an episode.
        /// </summary>
        /// <param name="episodeResult">The episode result to calculate the metric for.</param>
        /// <returns>The total return.</returns>
        public T Calculate(ModelEpisodeResult<T> episodeResult)
        {
            // Simply return the total reward as it represents the return in a trading environment
            return episodeResult.TotalReward;
        }
    }

    /// <summary>
    /// Calculates the Sharpe ratio metric for a trading episode.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class SharpeRatioMetric<T> : IEvaluationMetric<T>
    {
        private readonly T _riskFreeRate = default!;
        private readonly T _annualizationFactor = default!;
        
        /// <summary>
        /// Gets the numeric operations for type T.
        /// </summary>
        protected INumericOperations<T> NumOps => MathHelper.GetNumericOperations<T>();
        
        /// <summary>
        /// Gets the name of the metric.
        /// </summary>
        public string Name => "Sharpe Ratio";
        
        /// <summary>
        /// Gets the description of the metric.
        /// </summary>
        public string Description => "Risk-adjusted return metric. Higher values indicate better risk-adjusted performance.";
        
        /// <summary>
        /// Gets a value indicating whether higher values of this metric are better.
        /// </summary>
        public bool HigherIsBetter => true;

        /// <summary>
        /// Initializes a new instance of the <see cref="SharpeRatioMetric{T}"/> class.
        /// </summary>
        /// <param name="riskFreeRate">The risk-free rate expressed as a daily rate.</param>
        /// <param name="tradingDaysPerYear">The number of trading days per year for annualization.</param>
        public SharpeRatioMetric(double riskFreeRate = 0.0, int tradingDaysPerYear = 252)
        {
            _riskFreeRate = MathHelper.GetNumericOperations<T>().FromDouble(riskFreeRate);
            _annualizationFactor = MathHelper.GetNumericOperations<T>().FromDouble(Math.Sqrt(tradingDaysPerYear));
        }
        
        /// <summary>
        /// Calculates the Sharpe ratio for an episode.
        /// </summary>
        /// <param name="episodeResult">The episode result to calculate the metric for.</param>
        /// <returns>The Sharpe ratio.</returns>
        public T Calculate(ModelEpisodeResult<T> episodeResult)
        {
            var returns = episodeResult.Rewards;
            
            if (returns.Count == 0)
            {
                return NumOps.Zero;
            }
            
            // Calculate mean excess return
            T sumReturns = NumOps.Zero;
            foreach (var ret in returns)
            {
                sumReturns = NumOps.Add(sumReturns, NumOps.Subtract(ret, _riskFreeRate));
            }
            T meanExcessReturn = NumOps.Divide(sumReturns, NumOps.FromDouble(returns.Count));
            
            // Calculate standard deviation
            T sumSquaredDeviations = NumOps.Zero;
            foreach (var ret in returns)
            {
                T deviation = NumOps.Subtract(NumOps.Subtract(ret, _riskFreeRate), meanExcessReturn);
                sumSquaredDeviations = NumOps.Add(sumSquaredDeviations, NumOps.Multiply(deviation, deviation));
            }
            
            T stdDev = NumOps.Sqrt(NumOps.Divide(sumSquaredDeviations, NumOps.FromDouble(returns.Count)));
            
            // Calculate Sharpe ratio
            if (NumOps.LessThanOrEquals(stdDev, NumOps.Zero))
            {
                return NumOps.GreaterThan(meanExcessReturn, NumOps.Zero) ? 
                    NumOps.FromDouble(10.0) : NumOps.Zero; // Arbitrary high value if no volatility but positive return
            }
            
            return NumOps.Multiply(NumOps.Divide(meanExcessReturn, stdDev), _annualizationFactor);
        }
    }

    /// <summary>
    /// Calculates the maximum drawdown metric for a trading episode.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class MaxDrawdownMetric<T> : IEvaluationMetric<T>
    {
        /// <summary>
        /// Gets the numeric operations for type T.
        /// </summary>
        protected INumericOperations<T> NumOps => MathHelper.GetNumericOperations<T>();
        
        /// <summary>
        /// Gets the name of the metric.
        /// </summary>
        public string Name => "Maximum Drawdown";
        
        /// <summary>
        /// Gets the description of the metric.
        /// </summary>
        public string Description => "Maximum peak-to-trough decline in portfolio value. Lower values indicate less risk.";
        
        /// <summary>
        /// Gets a value indicating whether higher values of this metric are better.
        /// </summary>
        public bool HigherIsBetter => false;
        
        /// <summary>
        /// Calculates the maximum drawdown for an episode.
        /// </summary>
        /// <param name="episodeResult">The episode result to calculate the metric for.</param>
        /// <returns>The maximum drawdown.</returns>
        public T Calculate(ModelEpisodeResult<T> episodeResult)
        {
            var returns = episodeResult.Rewards;
            
            if (returns.Count == 0)
            {
                return NumOps.Zero;
            }
            
            // Convert returns to cumulative returns (starting at 1.0)
            var cumulativeReturns = new List<T> { NumOps.One };
            
            for (int i = 0; i < returns.Count; i++)
            {
                cumulativeReturns.Add(NumOps.Multiply(cumulativeReturns[i], 
                    NumOps.Add(NumOps.One, returns[i])));
            }
            
            // Calculate maximum drawdown
            T maxDrawdown = NumOps.Zero;
            T peak = cumulativeReturns[0];
            
            foreach (var value in cumulativeReturns)
            {
                if (NumOps.GreaterThan(value, peak))
                {
                    peak = value;
                }
                
                T drawdown = NumOps.Divide(NumOps.Subtract(peak, value), peak);
                
                if (NumOps.GreaterThan(drawdown, maxDrawdown))
                {
                    maxDrawdown = drawdown;
                }
            }
            
            return maxDrawdown;
        }
    }

    /// <summary>
    /// Calculates the Calmar ratio metric for a trading episode.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class CalmarRatioMetric<T> : IEvaluationMetric<T>
    {
        private readonly T _annualizationFactor = default!;
        
        /// <summary>
        /// Gets the numeric operations for type T.
        /// </summary>
        protected INumericOperations<T> NumOps => MathHelper.GetNumericOperations<T>();
        
        /// <summary>
        /// Gets the name of the metric.
        /// </summary>
        public string Name => "Calmar Ratio";
        
        /// <summary>
        /// Gets the description of the metric.
        /// </summary>
        public string Description => "Ratio of annualized return to maximum drawdown. Higher values indicate better risk-adjusted performance.";
        
        /// <summary>
        /// Gets a value indicating whether higher values of this metric are better.
        /// </summary>
        public bool HigherIsBetter => true;

        /// <summary>
        /// Initializes a new instance of the <see cref="CalmarRatioMetric{T}"/> class.
        /// </summary>
        /// <param name="tradingDaysPerYear">The number of trading days per year for annualization.</param>
        public CalmarRatioMetric(int tradingDaysPerYear = 252)
        {
            _annualizationFactor = MathHelper.GetNumericOperations<T>().FromDouble(tradingDaysPerYear);
        }
        
        /// <summary>
        /// Calculates the Calmar ratio for an episode.
        /// </summary>
        /// <param name="episodeResult">The episode result to calculate the metric for.</param>
        /// <returns>The Calmar ratio.</returns>
        public T Calculate(ModelEpisodeResult<T> episodeResult)
        {
            var returns = episodeResult.Rewards;
            
            if (returns.Count == 0)
            {
                return NumOps.Zero;
            }
            
            // Calculate average daily return
            T sumReturns = NumOps.Zero;
            foreach (var ret in returns)
            {
                sumReturns = NumOps.Add(sumReturns, ret);
            }
            T averageReturn = NumOps.Divide(sumReturns, NumOps.FromDouble(returns.Count));
            
            // Annualize the return
            T annualizedReturn = NumOps.Subtract(
                NumOps.Multiply(NumOps.Add(NumOps.One, averageReturn), _annualizationFactor), 
                NumOps.One);
            
            // Calculate maximum drawdown
            var maxDrawdownMetric = new MaxDrawdownMetric<T>();
            T maxDrawdown = maxDrawdownMetric.Calculate(episodeResult);
            
            // Calculate Calmar ratio
            if (NumOps.LessThanOrEquals(maxDrawdown, NumOps.Zero))
            {
                return NumOps.GreaterThan(annualizedReturn, NumOps.Zero) ? 
                    NumOps.FromDouble(10.0) : NumOps.Zero; // Arbitrary high value if no drawdown but positive return
            }
            
            return NumOps.Divide(annualizedReturn, maxDrawdown);
        }
    }

    /// <summary>
    /// Calculates the Sortino ratio metric for a trading episode.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class SortinoRatioMetric<T> : IEvaluationMetric<T>
    {
        private readonly T _riskFreeRate = default!;
        private readonly T _annualizationFactor = default!;
        
        /// <summary>
        /// Gets the numeric operations for type T.
        /// </summary>
        protected INumericOperations<T> NumOps => MathHelper.GetNumericOperations<T>();
        
        /// <summary>
        /// Gets the name of the metric.
        /// </summary>
        public string Name => "Sortino Ratio";
        
        /// <summary>
        /// Gets the description of the metric.
        /// </summary>
        public string Description => "Risk-adjusted return metric considering only downside risk. Higher values indicate better risk-adjusted performance.";
        
        /// <summary>
        /// Gets a value indicating whether higher values of this metric are better.
        /// </summary>
        public bool HigherIsBetter => true;

        /// <summary>
        /// Initializes a new instance of the <see cref="SortinoRatioMetric{T}"/> class.
        /// </summary>
        /// <param name="riskFreeRate">The risk-free rate expressed as a daily rate.</param>
        /// <param name="tradingDaysPerYear">The number of trading days per year for annualization.</param>
        public SortinoRatioMetric(double riskFreeRate = 0.0, int tradingDaysPerYear = 252)
        {
            _riskFreeRate = MathHelper.GetNumericOperations<T>().FromDouble(riskFreeRate);
            _annualizationFactor = MathHelper.GetNumericOperations<T>().FromDouble(Math.Sqrt(tradingDaysPerYear));
        }
        
        /// <summary>
        /// Calculates the Sortino ratio for an episode.
        /// </summary>
        /// <param name="episodeResult">The episode result to calculate the metric for.</param>
        /// <returns>The Sortino ratio.</returns>
        public T Calculate(ModelEpisodeResult<T> episodeResult)
        {
            var returns = episodeResult.Rewards;
            
            if (returns.Count == 0)
            {
                return NumOps.Zero;
            }
            
            // Calculate mean excess return
            T sumReturns = NumOps.Zero;
            foreach (var ret in returns)
            {
                sumReturns = NumOps.Add(sumReturns, NumOps.Subtract(ret, _riskFreeRate));
            }
            T meanExcessReturn = NumOps.Divide(sumReturns, NumOps.FromDouble(returns.Count));
            
            // Calculate downside deviation (only negative returns relative to the target rate)
            T sumSquaredDownsideDeviations = NumOps.Zero;
            int downsideCount = 0;
            
            foreach (var ret in returns)
            {
                if (NumOps.LessThan(ret, _riskFreeRate))
                {
                    T deviation = NumOps.Subtract(ret, _riskFreeRate);
                    sumSquaredDownsideDeviations = NumOps.Add(sumSquaredDownsideDeviations, 
                        NumOps.Multiply(deviation, deviation));
                    downsideCount++;
                }
            }
            
            // Calculate downside deviation
            T downsideDeviation;
            if (downsideCount > 0)
            {
                downsideDeviation = NumOps.Sqrt(NumOps.Divide(sumSquaredDownsideDeviations, 
                    NumOps.FromDouble(downsideCount)));
            }
            else
            {
                downsideDeviation = NumOps.Zero;
            }
            
            // Calculate Sortino ratio
            if (NumOps.LessThanOrEquals(downsideDeviation, NumOps.Zero))
            {
                return NumOps.GreaterThan(meanExcessReturn, NumOps.Zero) ? 
                    NumOps.FromDouble(10.0) : NumOps.Zero; // Arbitrary high value if no downside deviation but positive return
            }
            
            return NumOps.Multiply(NumOps.Divide(meanExcessReturn, downsideDeviation), _annualizationFactor);
        }
    }

    /// <summary>
    /// Calculates the win rate metric for a trading episode.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class WinRateMetric<T> : IEvaluationMetric<T>
    {
        /// <summary>
        /// Gets the numeric operations for type T.
        /// </summary>
        protected INumericOperations<T> NumOps => MathHelper.GetNumericOperations<T>();
        
        /// <summary>
        /// Gets the name of the metric.
        /// </summary>
        public string Name => "Win Rate";
        
        /// <summary>
        /// Gets the description of the metric.
        /// </summary>
        public string Description => "Percentage of positive returns in the episode.";
        
        /// <summary>
        /// Gets a value indicating whether higher values of this metric are better.
        /// </summary>
        public bool HigherIsBetter => true;
        
        /// <summary>
        /// Calculates the win rate for an episode.
        /// </summary>
        /// <param name="episodeResult">The episode result to calculate the metric for.</param>
        /// <returns>The win rate.</returns>
        public T Calculate(ModelEpisodeResult<T> episodeResult)
        {
            var returns = episodeResult.Rewards;
            
            if (returns.Count == 0)
            {
                return NumOps.Zero;
            }
            
            // Count positive returns
            int positiveReturns = 0;
            foreach (var ret in returns)
            {
                if (NumOps.GreaterThan(ret, NumOps.Zero))
                {
                    positiveReturns++;
                }
            }
            
            // Calculate win rate
            return NumOps.FromDouble((double)positiveReturns / returns.Count);
        }
    }
}