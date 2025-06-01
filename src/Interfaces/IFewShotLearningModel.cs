namespace AiDotNet.Interfaces;

/// <summary>
/// Represents a model capable of few-shot learning through transfer learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The type of input data.</typeparam>
/// <typeparam name="TOutput">The type of output data.</typeparam>
/// <remarks>
/// Few-shot learning models can learn new tasks from very few examples
/// by leveraging prior knowledge from related tasks. This is particularly
/// useful when labeled data is scarce or expensive to obtain.
/// </remarks>
public interface IFewShotLearningModel<T, TInput, TOutput> : ITransferLearningModel<T, TInput, TOutput>
{
    /// <summary>
    /// Learns a new task from a few examples.
    /// </summary>
    /// <param name="supportSet">The support set containing few examples of the new task.</param>
    /// <param name="supportLabels">The labels for the support set.</param>
    /// <param name="numShots">Number of examples per class (k in k-shot learning).</param>
    void LearnFromFewExamples(TInput[] supportSet, TOutput[] supportLabels, int numShots);
    
    /// <summary>
    /// Adapts the model using a query set after few-shot learning.
    /// </summary>
    /// <param name="querySet">The query set for evaluation.</param>
    /// <param name="queryLabels">The labels for the query set.</param>
    /// <returns>Accuracy on the query set.</returns>
    T EvaluateOnQuerySet(TInput[] querySet, TOutput[] queryLabels);
    
    /// <summary>
    /// Gets the number of inner loop optimization steps for meta-learning.
    /// </summary>
    int InnerLoopSteps { get; set; }
    
    /// <summary>
    /// Gets or sets the inner loop learning rate for fast adaptation.
    /// </summary>
    T InnerLoopLearningRate { get; set; }
    
    /// <summary>
    /// Performs meta-training on a distribution of tasks.
    /// </summary>
    /// <param name="taskDistribution">A collection of tasks for meta-training.</param>
    void MetaTrain(IEnumerable<FewShotTask<TInput, TOutput>> taskDistribution);
    
    /// <summary>
    /// Creates a task-specific model adapted to the given support set.
    /// </summary>
    /// <param name="supportSet">The support set for the task.</param>
    /// <param name="supportLabels">The labels for the support set.</param>
    /// <returns>A model adapted to the specific task.</returns>
    IFullModel<T, TInput, TOutput> CreateTaskSpecificModel(TInput[] supportSet, TOutput[] supportLabels);
    
    /// <summary>
    /// Gets the meta-learning algorithm being used.
    /// </summary>
    MetaLearningAlgorithm Algorithm { get; }
}

/// <summary>
/// Represents a few-shot learning task.
/// </summary>
public class FewShotTask<TInput, TOutput>
{
    /// <summary>
    /// Gets or sets the support set for the task.
    /// </summary>
    public TInput[] SupportSet { get; set; } = Array.Empty<TInput>();
    
    /// <summary>
    /// Gets or sets the support set labels.
    /// </summary>
    public TOutput[] SupportLabels { get; set; } = Array.Empty<TOutput>();
    
    /// <summary>
    /// Gets or sets the query set for the task.
    /// </summary>
    public TInput[] QuerySet { get; set; } = Array.Empty<TInput>();
    
    /// <summary>
    /// Gets or sets the query set labels.
    /// </summary>
    public TOutput[] QueryLabels { get; set; } = Array.Empty<TOutput>();
    
    /// <summary>
    /// Gets or sets the task identifier.
    /// </summary>
    public string TaskId { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets task-specific metadata.
    /// </summary>
    public Dictionary<string, object> Metadata { get; set; } = new();
}

/// <summary>
/// Specifies the meta-learning algorithm for few-shot learning.
/// </summary>
public enum MetaLearningAlgorithm
{
    /// <summary>
    /// Model-Agnostic Meta-Learning - learns initialization for fast adaptation.
    /// </summary>
    MAML,
    
    /// <summary>
    /// Prototypical Networks - learns a metric space for classification.
    /// </summary>
    ProtoNet,
    
    /// <summary>
    /// Matching Networks - uses attention and memory for one-shot learning.
    /// </summary>
    MatchingNetworks,
    
    /// <summary>
    /// Relation Networks - learns to compare query and support examples.
    /// </summary>
    RelationNetwork,
    
    /// <summary>
    /// Reptile - first-order approximation of MAML.
    /// </summary>
    Reptile,
    
    /// <summary>
    /// ANIL (Almost No Inner Loop) - simplified MAML variant.
    /// </summary>
    ANIL,
    
    /// <summary>
    /// Meta-SGD - learns both initialization and learning rates.
    /// </summary>
    MetaSGD,
    
    /// <summary>
    /// LEO (Latent Embedding Optimization) - uses latent embeddings.
    /// </summary>
    LEO,
    
    /// <summary>
    /// CNP (Conditional Neural Processes) - probabilistic meta-learning.
    /// </summary>
    CNP
}