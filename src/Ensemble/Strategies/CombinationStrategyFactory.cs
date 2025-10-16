using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.Ensemble.Strategies;

/// <summary>
/// Factory class for creating combination strategies based on the EnsembleStrategy enum.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This factory class is like a menu that helps you order the right 
/// combination strategy for your ensemble. You tell it what type of strategy you want 
/// (using the EnsembleStrategy enum), and it creates the appropriate strategy object for you.
/// </para>
/// </remarks>
public static class CombinationStrategyFactory
{
    /// <summary>
    /// Creates a combination strategy based on the specified strategy type.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <typeparam name="TInput">The input data type.</typeparam>
    /// <typeparam name="TOutput">The output data type.</typeparam>
    /// <param name="strategyType">The type of strategy to create.</param>
    /// <param name="options">Optional parameters for strategy configuration.</param>
    /// <returns>The created combination strategy.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method is like a vending machine - you select what you want 
    /// (the strategy type), and it gives you the right product (strategy implementation).
    /// Some strategies might need extra configuration, which you can provide in the options.
    /// </remarks>
    public static ICombinationStrategy<T, TInput, TOutput> Create<T, TInput, TOutput>(
        EnsembleStrategy strategyType,
        Dictionary<string, object>? options = null)
        where TOutput : notnull
    {
        options ??= new Dictionary<string, object>();
        
        return strategyType switch
        {
            // Basic Averaging Methods
            EnsembleStrategy.Average => 
                new AveragingStrategy<T, TInput, TOutput>(),
                
            EnsembleStrategy.WeightedAverage => 
                new WeightedAverageStrategy<T, TInput, TOutput>(),
                
            EnsembleStrategy.Median => 
                new MedianStrategy<T, TInput, TOutput>(),
                
            EnsembleStrategy.TrimmedMean => 
                new TrimmedMeanStrategy<T, TInput, TOutput>(
                    GetOptionValue<double>(options, "trimPercentage", 10.0)),
            
            // Voting Methods
            EnsembleStrategy.MajorityVote => 
                new MajorityVoteStrategy<T, TInput, TOutput>(
                    useWeights: false),
                    
            EnsembleStrategy.WeightedVote => 
                new MajorityVoteStrategy<T, TInput, TOutput>(
                    useWeights: true),
                    
            EnsembleStrategy.SoftVote => 
                new VotingStrategy<T, TInput, TOutput>(useSoftVoting: true),
                
            EnsembleStrategy.RankVote => 
                throw new NotImplementedException("RankVote strategy not yet implemented"),
            
            // Advanced Combination Methods
            EnsembleStrategy.Stacking => 
                CreateStackingStrategy<T, TInput, TOutput>(options),
                
            EnsembleStrategy.Blending => 
                new BlendingStrategy<T, TInput, TOutput>(
                    GetOptionValue<double>(options, "validationSplit", 0.2),
                    GetOptionValue<bool>(options, "includeIntercept", true)),
                    
            EnsembleStrategy.MultiLevelStacking => 
                throw new NotImplementedException("MultiLevelStacking strategy not yet implemented"),
            
            // Bayesian Methods
            EnsembleStrategy.BayesianAverage => 
                new BayesianAverageStrategy<T, TInput, TOutput>(
                    GetOptionValue<double>(options, "priorStrength", 1.0)),
                    
            EnsembleStrategy.BayesianCombination => 
                throw new NotImplementedException("BayesianCombination strategy not yet implemented"),
            
            // Dynamic Selection Methods
            EnsembleStrategy.DynamicSelection => 
                new DynamicSelectionStrategy<T, TInput, TOutput>(
                    GetOptionValue<int>(options, "k", 5)),
                    
            EnsembleStrategy.LocalCompetence => 
                throw new NotImplementedException("LocalCompetence strategy not yet implemented"),
                
            EnsembleStrategy.Oracle => 
                throw new NotImplementedException("Oracle strategy not yet implemented"),
            
            // Mixture Methods
            EnsembleStrategy.MixtureOfExperts => 
                throw new NotImplementedException("MixtureOfExperts strategy not yet implemented"),
                
            EnsembleStrategy.HierarchicalMixture => 
                throw new NotImplementedException("HierarchicalMixture strategy not yet implemented"),
                
            EnsembleStrategy.GatedMixture => 
                throw new NotImplementedException("GatedMixture strategy not yet implemented"),
            
            // Statistical Methods
            EnsembleStrategy.MaximumLikelihood => 
                throw new NotImplementedException("MaximumLikelihood strategy not yet implemented"),
                
            EnsembleStrategy.MinimumVariance => 
                new MinimumVarianceStrategy<T, TInput, TOutput>(
                    GetOptionValue<double>(options, "regularization", 0.01)),
                
            EnsembleStrategy.DempsterShafer => 
                throw new NotImplementedException("DempsterShafer strategy not yet implemented"),
            
            // Optimization-based Methods
            EnsembleStrategy.QuadraticProgramming => 
                throw new NotImplementedException("QuadraticProgramming strategy not yet implemented"),
                
            EnsembleStrategy.GeneticOptimization => 
                throw new NotImplementedException("GeneticOptimization strategy not yet implemented"),
                
            EnsembleStrategy.ParticleSwarmOptimization => 
                throw new NotImplementedException("ParticleSwarmOptimization strategy not yet implemented"),
            
            // Boosting Methods
            EnsembleStrategy.AdaBoost => 
                new AdaBoostStrategy<T, TInput, TOutput>(
                    GetOptionValue<double>(options, "learningRate", 1.0),
                    GetOptionValue<int>(options, "maxIterations", 50)),
                
            EnsembleStrategy.GradientBoosting => 
                throw new NotImplementedException("GradientBoosting strategy not yet implemented"),
                
            EnsembleStrategy.XGBoost => 
                throw new NotImplementedException("XGBoost strategy not yet implemented"),
            
            // Other Advanced Methods
            EnsembleStrategy.RandomSubspace => 
                throw new NotImplementedException("RandomSubspace strategy not yet implemented"),
                
            EnsembleStrategy.RotationForest => 
                throw new NotImplementedException("RotationForest strategy not yet implemented"),
                
            EnsembleStrategy.CascadeGeneralization => 
                throw new NotImplementedException("CascadeGeneralization strategy not yet implemented"),
                
            EnsembleStrategy.DynamicWeightedMajority => 
                throw new NotImplementedException("DynamicWeightedMajority strategy not yet implemented"),
                
            EnsembleStrategy.OnlineLearning => 
                throw new NotImplementedException("OnlineLearning strategy not yet implemented"),
                
            EnsembleStrategy.MetaLearning => 
                throw new NotImplementedException("MetaLearning strategy not yet implemented"),
                
            EnsembleStrategy.NeuralCombiner => 
                throw new NotImplementedException("NeuralCombiner strategy not yet implemented"),
                
            EnsembleStrategy.FuzzyIntegral => 
                throw new NotImplementedException("FuzzyIntegral strategy not yet implemented"),
                
            EnsembleStrategy.EvidenceTheory => 
                throw new NotImplementedException("EvidenceTheory strategy not yet implemented"),
                
            EnsembleStrategy.Custom => 
                throw new ArgumentException("Custom strategy requires a user-provided implementation"),
                
            _ => throw new ArgumentException($"Unknown ensemble strategy: {strategyType}")
        };
    }
    
    /// <summary>
    /// Creates a stacking strategy with the specified meta-learner.
    /// </summary>
    private static StackingStrategy<T, TInput, TOutput> CreateStackingStrategy<T, TInput, TOutput>(
        Dictionary<string, object> options)
    {
        var cvFolds = GetOptionValue<int>(options, "cvFolds", 5);
        return new StackingStrategy<T, TInput, TOutput>(cvFolds);
    }
    
    /// <summary>
    /// Gets an option value with a default fallback.
    /// </summary>
    private static TValue GetOptionValue<TValue>(
        Dictionary<string, object> options, 
        string key, 
        TValue defaultValue)
    {
        if (options.TryGetValue(key, out var value) && value is TValue typedValue)
        {
            return typedValue;
        }
        return defaultValue;
    }
    
    /// <summary>
    /// Gets information about which strategies are currently implemented.
    /// </summary>
    /// <returns>A dictionary mapping strategy types to their implementation status.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method tells you which strategies are ready to use 
    /// and which ones are still being developed.
    /// </remarks>
    public static Dictionary<EnsembleStrategy, bool> GetImplementationStatus()
    {
        var status = new Dictionary<EnsembleStrategy, bool>();
        
        foreach (EnsembleStrategy strategy in Enum.GetValues(typeof(EnsembleStrategy)))
        {
            status[strategy] = strategy switch
            {
                EnsembleStrategy.Average => true,
                EnsembleStrategy.WeightedAverage => true,
                EnsembleStrategy.Median => true,
                EnsembleStrategy.TrimmedMean => true,
                EnsembleStrategy.MajorityVote => true,
                EnsembleStrategy.WeightedVote => true,
                EnsembleStrategy.SoftVote => true,
                EnsembleStrategy.Stacking => true,
                EnsembleStrategy.Blending => true,
                EnsembleStrategy.BayesianAverage => true,
                EnsembleStrategy.DynamicSelection => true,
                EnsembleStrategy.MinimumVariance => true,
                EnsembleStrategy.AdaBoost => true,
                _ => false
            };
        }
        
        return status;
    }
}