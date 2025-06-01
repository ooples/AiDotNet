namespace AiDotNet.Enums;

/// <summary>
/// Specifies the strategy to use for transfer learning.
/// </summary>
public enum TransferLearningStrategy
{
    /// <summary>
    /// Use the source model as a fixed feature extractor.
    /// Only train the final layers on the new task.
    /// </summary>
    FeatureExtraction,
    
    /// <summary>
    /// Fine-tune all layers of the source model on the new task.
    /// </summary>
    FineTuning,
    
    /// <summary>
    /// Progressively unfreeze layers from top to bottom during training.
    /// </summary>
    ProgressiveUnfreezing,
    
    /// <summary>
    /// Use different learning rates for different layers.
    /// Lower layers use smaller learning rates.
    /// </summary>
    DiscriminativeFineTuning,
    
    /// <summary>
    /// Adapt the model to handle domain shift between source and target.
    /// </summary>
    DomainAdaptation,
    
    /// <summary>
    /// Add task-specific layers while keeping the base model frozen.
    /// </summary>
    TaskSpecificLayers,
    
    /// <summary>
    /// Add adapter modules between layers for efficient fine-tuning.
    /// </summary>
    AdapterBasedTransfer,
    
    /// <summary>
    /// Transfer knowledge using distillation from teacher to student.
    /// </summary>
    KnowledgeDistillation,
    
    /// <summary>
    /// Transfer from multiple source models to one target model.
    /// </summary>
    MultiSourceTransfer,
    
    /// <summary>
    /// Use meta-learning to quickly adapt to new tasks.
    /// </summary>
    MetaLearning,
    
    /// <summary>
    /// Transfer learning with elastic weight consolidation to prevent catastrophic forgetting.
    /// </summary>
    ElasticWeightConsolidation,
    
    /// <summary>
    /// Use continual learning techniques to learn new tasks without forgetting old ones.
    /// </summary>
    ContinualLearning,
    
    /// <summary>
    /// Learn shared representations across multiple related tasks.
    /// </summary>
    MultiTaskLearning
}