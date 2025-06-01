namespace AiDotNet.Enums;

/// <summary>
/// Specifies the algorithm to use for meta-learning and few-shot learning.
/// </summary>
public enum MetaLearningAlgorithm
{
    /// <summary>
    /// Model-Agnostic Meta-Learning - learns initialization for fast adaptation.
    /// </summary>
    MAML,
    
    /// <summary>
    /// First-Order MAML - more efficient approximation of MAML.
    /// </summary>
    FOMAML,
    
    /// <summary>
    /// Reptile - simpler alternative to MAML using multiple SGD steps.
    /// </summary>
    Reptile,
    
    /// <summary>
    /// Prototypical Networks - learns a metric space for few-shot classification.
    /// </summary>
    ProtoNet,
    
    /// <summary>
    /// Matching Networks - uses attention and memory for one-shot learning.
    /// </summary>
    MatchingNet,
    
    /// <summary>
    /// Relation Networks - learns to compare query and support examples.
    /// </summary>
    RelationNet,
    
    /// <summary>
    /// ANIL (Almost No Inner Loop) - simplified MAML with frozen feature extractor.
    /// </summary>
    ANIL,
    
    /// <summary>
    /// Meta-SGD - learns both initialization and learning rates.
    /// </summary>
    MetaSGD,
    
    /// <summary>
    /// Probabilistic MAML for uncertainty-aware meta-learning.
    /// </summary>
    ProbabilisticMAML
}