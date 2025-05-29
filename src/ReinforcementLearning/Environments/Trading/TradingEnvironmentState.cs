using AiDotNet.LinearAlgebra;
using AiDotNet.Interfaces;
using AiDotNet.ReinforcementLearning.Interfaces;
using AiDotNet.Helpers;
using System;
using System.Collections.Generic;

namespace AiDotNet.ReinforcementLearning.Environments.Trading
{
    /// <summary>
    /// Represents the state of a trading environment.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class TradingEnvironmentState<T> : ITensorConvertible<T>
    {
        /// <summary>
        /// Gets the numeric operations for type T.
        /// </summary>
        protected INumericOperations<T> NumOps => MathHelper.GetNumericOperations<T>();
        
        /// <summary>
        /// Gets the market features (price data, technical indicators, etc.).
        /// </summary>
        public T[] MarketFeatures { get; }
        
        /// <summary>
        /// Gets the account features (current positions, portfolio value, etc.).
        /// </summary>
        public T[] AccountFeatures { get; }
        
        /// <summary>
        /// Gets the current timestamp of the environment.
        /// </summary>
        public DateTime Timestamp { get; }
        
        /// <summary>
        /// Gets the feature names in the order they appear in the market features array.
        /// </summary>
        public IReadOnlyList<string> MarketFeatureNames { get; }
        
        /// <summary>
        /// Gets the feature names in the order they appear in the account features array.
        /// </summary>
        public IReadOnlyList<string> AccountFeatureNames { get; }

        /// <summary>
        /// Initializes a new instance of the <see cref="TradingEnvironmentState{T}"/> class.
        /// </summary>
        /// <param name="marketFeatures">The market features array.</param>
        /// <param name="accountFeatures">The account features array.</param>
        /// <param name="timestamp">The current timestamp.</param>
        /// <param name="marketFeatureNames">The names of market features.</param>
        /// <param name="accountFeatureNames">The names of account features.</param>
        public TradingEnvironmentState(
            T[] marketFeatures, 
            T[] accountFeatures, 
            DateTime timestamp,
            IReadOnlyList<string> marketFeatureNames,
            IReadOnlyList<string> accountFeatureNames)
        {
            MarketFeatures = marketFeatures ?? throw new ArgumentNullException(nameof(marketFeatures));
            AccountFeatures = accountFeatures ?? throw new ArgumentNullException(nameof(accountFeatures));
            Timestamp = timestamp;
            MarketFeatureNames = marketFeatureNames ?? throw new ArgumentNullException(nameof(marketFeatureNames));
            AccountFeatureNames = accountFeatureNames ?? throw new ArgumentNullException(nameof(accountFeatureNames));
            
            if (marketFeatures.Length != marketFeatureNames.Count)
            {
                throw new ArgumentException("Market features array length must match feature names count.");
            }
            
            if (accountFeatures.Length != accountFeatureNames.Count)
            {
                throw new ArgumentException("Account features array length must match feature names count.");
            }
        }

        /// <summary>
        /// Gets a specific market feature by name.
        /// </summary>
        /// <param name="name">The name of the feature.</param>
        /// <returns>The feature value.</returns>
        public T GetMarketFeature(string name)
        {
            int index = MarketFeatureNames.ToList().IndexOf(name);
            if (index < 0)
            {
                throw new ArgumentException($"Market feature '{name}' not found.");
            }
            
            return MarketFeatures[index];
        }
        
        /// <summary>
        /// Gets a specific account feature by name.
        /// </summary>
        /// <param name="name">The name of the feature.</param>
        /// <returns>The feature value.</returns>
        public T GetAccountFeature(string name)
        {
            int index = AccountFeatureNames.ToList().IndexOf(name);
            if (index < 0)
            {
                throw new ArgumentException($"Account feature '{name}' not found.");
            }
            
            return AccountFeatures[index];
        }

        /// <summary>
        /// Converts the state to a tensor.
        /// </summary>
        /// <returns>A tensor representation of the state.</returns>
        public Tensor<T> ToTensor()
        {
            // Combine market and account features into a single array
            T[] allFeatures = new T[MarketFeatures.Length + AccountFeatures.Length];
            Array.Copy(MarketFeatures, 0, allFeatures, 0, MarketFeatures.Length);
            Array.Copy(AccountFeatures, 0, allFeatures, MarketFeatures.Length, AccountFeatures.Length);
            
            // Create and return a tensor from the combined array
            return Tensor<T>.FromVector(new Vector<T>(allFeatures));
        }
    }
}