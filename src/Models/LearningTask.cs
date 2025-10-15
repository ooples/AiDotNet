namespace AiDotNet.Models;

/// <summary>
/// Represents a learning task for meta-learning algorithms.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The type of input data.</typeparam>
/// <typeparam name="TOutput">The type of output data.</typeparam>
public class LearningTask<T, TInput, TOutput>
{
    /// <summary>
    /// Gets or sets the task identifier.
    /// </summary>
    public string TaskId { get; set; } = Guid.NewGuid().ToString();
    
    /// <summary>
    /// Gets or sets the support set (training examples) for the task.
    /// </summary>
    public (TInput Input, TOutput Output)[] SupportSet { get; set; } = new (TInput, TOutput)[0];
    
    /// <summary>
    /// Gets or sets the query set (test examples) for the task.
    /// </summary>
    public (TInput Input, TOutput Output)[] QuerySet { get; set; } = new (TInput, TOutput)[0];
    
    /// <summary>
    /// Gets or sets the task description.
    /// </summary>
    public string Description { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets additional task metadata.
    /// </summary>
    public Dictionary<string, object> Metadata { get; set; } = new Dictionary<string, object>();
}