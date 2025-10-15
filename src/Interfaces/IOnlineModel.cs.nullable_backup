namespace AiDotNet.Interfaces;

/// <summary>
/// Represents a machine learning model capable of online (incremental) learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The type of input data.</typeparam>
/// <typeparam name="TOutput">The type of output data.</typeparam>
/// <remarks>
/// Online learning models can update their parameters incrementally as new data arrives,
/// without needing to retrain on the entire dataset. This is useful for:
/// - Large datasets that don't fit in memory
/// - Streaming data applications
/// - Adaptive systems that need to respond to changing patterns
/// - Real-time learning scenarios
/// </remarks>
public interface IOnlineModel<T, TInput, TOutput> : IFullModel<T, TInput, TOutput>
{
    /// <summary>
    /// Updates the model with a single data point.
    /// </summary>
    /// <param name="input">The input data point.</param>
    /// <param name="expectedOutput">The expected output for the input.</param>
    void PartialFit(TInput input, TOutput expectedOutput);
    
    /// <summary>
    /// Updates the model with a single data point using a custom learning rate.
    /// </summary>
    /// <param name="input">The input data point.</param>
    /// <param name="expectedOutput">The expected output for the input.</param>
    /// <param name="learningRate">Custom learning rate for this update.</param>
    void PartialFit(TInput input, TOutput expectedOutput, T learningRate);
    
    /// <summary>
    /// Updates the model with a batch of data points.
    /// </summary>
    /// <param name="inputs">The batch of input data.</param>
    /// <param name="expectedOutputs">The expected outputs for the inputs.</param>
    void PartialFitBatch(TInput[] inputs, TOutput[] expectedOutputs);
    
    /// <summary>
    /// Updates the model with a batch of data points using a custom learning rate.
    /// </summary>
    /// <param name="inputs">The batch of input data.</param>
    /// <param name="expectedOutputs">The expected outputs for the inputs.</param>
    /// <param name="learningRate">Custom learning rate for this batch.</param>
    void PartialFitBatch(TInput[] inputs, TOutput[] expectedOutputs, T learningRate);
    
    /// <summary>
    /// Gets the current learning rate of the model.
    /// </summary>
    T LearningRate { get; }
    
    /// <summary>
    /// Gets or sets whether the model should adapt its learning rate automatically.
    /// </summary>
    bool AdaptiveLearningRate { get; set; }
    
    /// <summary>
    /// Gets the number of samples the model has been trained on.
    /// </summary>
    long SamplesSeen { get; }
    
    /// <summary>
    /// Resets the online learning statistics while keeping the learned parameters.
    /// </summary>
    void ResetStatistics();
}