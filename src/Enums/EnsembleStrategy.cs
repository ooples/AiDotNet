namespace AiDotNet.Enums;

/// <summary>
/// Defines the available strategies for combining predictions in ensemble models.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> These strategies determine how predictions from multiple models 
/// are combined into a single final prediction. Different strategies work better for 
/// different types of problems and data.
/// </para>
/// </remarks>
public enum EnsembleStrategy
{
    // Basic Averaging Methods
    /// <summary>
    /// Simple arithmetic mean of all predictions.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Adds up all predictions and divides by the number of models. 
    /// Works well when all models are similarly accurate.
    /// </remarks>
    Average,
    
    /// <summary>
    /// Weighted average based on model performance.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Like average, but gives more importance to better models. 
    /// A model that's twice as accurate gets twice the influence.
    /// </remarks>
    WeightedAverage,
    
    /// <summary>
    /// Median of predictions (robust to outliers).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Picks the middle value when predictions are sorted. Good when 
    /// some models occasionally make wild predictions.
    /// </remarks>
    Median,
    
    /// <summary>
    /// Trimmed mean (removes extreme predictions).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Removes the highest and lowest predictions before averaging. 
    /// Protects against models that sometimes make extreme errors.
    /// </remarks>
    TrimmedMean,
    
    // Voting Methods
    /// <summary>
    /// Simple voting where each model gets one vote.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Each model gets an equal vote on the final prediction, and the
    /// answer with the most votes wins. Simple and democratic.
    /// </remarks>
    Voting,

    /// <summary>
    /// Each model votes for a class (classification).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Each model votes for what it thinks is the answer, and the
    /// most popular answer wins. Like a democratic election.
    /// </remarks>
    MajorityVote,
    
    /// <summary>
    /// Weighted voting based on model confidence.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Models that are more confident in their predictions get more 
    /// votes. A very confident model might get 3 votes while an uncertain one gets 1.
    /// </remarks>
    WeightedVote,
    
    /// <summary>
    /// Uses prediction probabilities for voting.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Instead of just voting for one answer, models share how confident 
    /// they are about each possible answer, then these are combined.
    /// </remarks>
    SoftVote,
    
    /// <summary>
    /// Rank-based voting.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Models rank their top choices, and the rankings are combined 
    /// to find the overall best choice.
    /// </remarks>
    RankVote,
    
    // Advanced Combination Methods
    /// <summary>
    /// Uses another model to combine predictions.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Trains a separate "boss" model that learns how to best combine 
    /// the predictions from the "worker" models.
    /// </remarks>
    Stacking,
    
    /// <summary>
    /// Linear combination learned from validation data.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Similar to stacking but simpler - finds the best weights to 
    /// multiply each model's prediction before adding them up.
    /// </remarks>
    Blending,
    
    /// <summary>
    /// Multi-level stacking.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Like stacking, but with multiple layers. The first layer combines 
    /// some models, then another layer combines those results.
    /// </remarks>
    MultiLevelStacking,
    
    // Bayesian Methods
    /// <summary>
    /// Bayesian model averaging.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Uses probability theory to combine models based on how likely 
    /// each model is to be correct given the data.
    /// </remarks>
    BayesianAverage,
    
    /// <summary>
    /// Bayesian model combination.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> A more sophisticated Bayesian approach that considers uncertainty 
    /// in both the models and their predictions.
    /// </remarks>
    BayesianCombination,
    
    // Dynamic Selection Methods
    /// <summary>
    /// Selects best model based on input characteristics.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Looks at each new input and picks the model that's likely to 
    /// work best for that specific case.
    /// </remarks>
    DynamicSelection,
    
    /// <summary>
    /// Selects models based on local competence.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Finds which models work best for inputs similar to the current 
    /// one, then uses those models.
    /// </remarks>
    LocalCompetence,
    
    /// <summary>
    /// Oracle-based selection (theoretical best).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> A theoretical strategy that always picks the best model. Used 
    /// for research to see how well other strategies could potentially perform.
    /// </remarks>
    Oracle,
    
    // Mixture Methods
    /// <summary>
    /// Mixture of experts approach.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Different models become "experts" in different areas, and a 
    /// gating network decides which expert(s) to consult for each input.
    /// </remarks>
    MixtureOfExperts,
    
    /// <summary>
    /// Hierarchical mixture of experts.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Organizes experts in a tree structure, with high-level experts 
    /// deciding which low-level experts to use.
    /// </remarks>
    HierarchicalMixture,
    
    /// <summary>
    /// Gated mixture (learnable gates).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Uses neural networks as "gates" that learn when to use each model.
    /// </remarks>
    GatedMixture,
    
    // Statistical Methods
    /// <summary>
    /// Maximum likelihood combination.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Combines models in a way that maximizes the probability of getting 
    /// the correct answers.
    /// </remarks>
    MaximumLikelihood,
    
    /// <summary>
    /// Minimum variance combination.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Combines models to minimize the variation (uncertainty) in the 
    /// final predictions.
    /// </remarks>
    MinimumVariance,
    
    /// <summary>
    /// Dempster-Shafer theory combination.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> A mathematical framework for combining evidence from different 
    /// sources, accounting for uncertainty and conflict.
    /// </remarks>
    DempsterShafer,
    
    // Optimization-based Methods
    /// <summary>
    /// Optimal weights via quadratic programming.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Uses mathematical optimization to find the best possible weights 
    /// for combining models.
    /// </remarks>
    QuadraticProgramming,
    
    /// <summary>
    /// Genetic algorithm for weight optimization.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Uses evolution-inspired algorithms to evolve better and better 
    /// weights over many generations.
    /// </remarks>
    GeneticOptimization,
    
    /// <summary>
    /// Particle swarm optimization for weights.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Inspired by bird flocks, multiple "particles" explore different 
    /// weight combinations to find the best one.
    /// </remarks>
    ParticleSwarmOptimization,
    
    // Boosting Methods
    /// <summary>
    /// Adaptive boosting combination.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Trains models sequentially, with each new model focusing on the 
    /// mistakes made by previous models.
    /// </remarks>
    AdaBoost,
    
    /// <summary>
    /// Gradient boosting combination.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Each new model is trained to correct the errors of all previous 
    /// models combined, using gradient descent.
    /// </remarks>
    GradientBoosting,
    
    /// <summary>
    /// XGBoost-style combination.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> An advanced and efficient version of gradient boosting with 
    /// additional optimizations.
    /// </remarks>
    XGBoost,
    
    // Other Advanced Methods
    /// <summary>
    /// Random subspace method.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Each model is trained on a random subset of features, then 
    /// predictions are combined.
    /// </remarks>
    RandomSubspace,
    
    /// <summary>
    /// Rotation forest combination.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Rotates the feature space for each model to create diversity, 
    /// then combines predictions.
    /// </remarks>
    RotationForest,
    
    /// <summary>
    /// Cascade generalization.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Models are arranged in levels, with each level using predictions 
    /// from the previous level as additional features.
    /// </remarks>
    CascadeGeneralization,
    
    /// <summary>
    /// Dynamic weighted majority.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Continuously adjusts model weights based on their recent performance.
    /// </remarks>
    DynamicWeightedMajority,
    
    /// <summary>
    /// Online learning combination.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Updates the combination strategy continuously as new data arrives.
    /// </remarks>
    OnlineLearning,
    
    /// <summary>
    /// Meta-learning based combination.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Learns from experience across many different tasks how to best 
    /// combine models.
    /// </remarks>
    MetaLearning,
    
    /// <summary>
    /// Neural network combiner.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Uses a neural network to learn complex non-linear combinations 
    /// of model predictions.
    /// </remarks>
    NeuralCombiner,
    
    /// <summary>
    /// Fuzzy integral combination.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Uses fuzzy logic to handle uncertainty and combine predictions 
    /// in a flexible way.
    /// </remarks>
    FuzzyIntegral,
    
    /// <summary>
    /// Evidence theory combination.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Combines predictions while explicitly representing uncertainty 
    /// and conflict between models.
    /// </remarks>
    EvidenceTheory,
    
    /// <summary>
    /// Custom user-defined strategy.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Allows you to implement your own custom combination strategy.
    /// </remarks>
    Custom
}