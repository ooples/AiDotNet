using System.Collections.Generic;

namespace AiDotNet.Models;

/// <summary>
/// Represents statistics for a machine learning model.
/// </summary>
/// <typeparam name="T">The numeric type used for the model.</typeparam>
public class ModelStats<T>
{
    /// <summary>
    /// Gets or sets the number of samples seen by the model.
    /// </summary>
    public long SampleCount { get; set; }
    
    /// <summary>
    /// Gets or sets the current learning rate.
    /// </summary>
    public T LearningRate { get; set; } = default!;
    
    /// <summary>
    /// Gets or sets the training loss.
    /// </summary>
    public T TrainingLoss { get; set; } = default!;
    
    /// <summary>
    /// Gets or sets the validation loss.
    /// </summary>
    public T ValidationLoss { get; set; } = default!;
    
    /// <summary>
    /// Gets or sets additional metrics specific to the model.
    /// </summary>
    public Dictionary<string, T> AdditionalMetrics { get; set; } = new Dictionary<string, T>();
}