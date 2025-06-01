using System.Collections.Generic;

namespace AiDotNet.Interfaces;

/// <summary>
/// Represents a model capable of transfer learning - adapting knowledge from a source domain to a target domain.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The type of input data.</typeparam>
/// <typeparam name="TOutput">The type of output data.</typeparam>
public interface ITransferLearningModel<T, TInput, TOutput> : IFullModel<T, TInput, TOutput>
{
    /// <summary>
    /// Transfers knowledge from a source model to initialize this model.
    /// </summary>
    /// <param name="sourceModel">The source model to transfer from.</param>
    /// <param name="strategy">The transfer learning strategy to use.</param>
    /// <param name="options">Additional options for the transfer process.</param>
    void TransferFrom(IFullModel<T, TInput, TOutput> sourceModel, 
                     TransferLearningStrategy strategy, 
                     TransferLearningOptions<T>? options = null);
    
    /// <summary>
    /// Transfers knowledge from a source model with different input/output types.
    /// </summary>
    /// <typeparam name="TSourceInput">The input type of the source model.</typeparam>
    /// <typeparam name="TSourceOutput">The output type of the source model.</typeparam>
    void TransferFrom<TSourceInput, TSourceOutput>(
        IFullModel<T, TSourceInput, TSourceOutput> sourceModel,
        IInputAdapter<T, TSourceInput, TInput> inputAdapter,
        IOutputAdapter<T, TSourceOutput, TOutput> outputAdapter,
        TransferLearningStrategy strategy,
        TransferLearningOptions<T>? options = null);
    
    /// <summary>
    /// Fine-tunes the model on new data while preserving transferred knowledge.
    /// </summary>
    /// <param name="inputs">The input data for fine-tuning.</param>
    /// <param name="outputs">The target outputs for fine-tuning.</param>
    /// <param name="options">Fine-tuning specific options.</param>
    void FineTune(TInput[] inputs, TOutput[] outputs, FineTuningOptions<T>? options = null);
    
    /// <summary>
    /// Freezes specified layers to prevent updates during training.
    /// </summary>
    /// <param name="layerIndices">Indices of layers to freeze.</param>
    void FreezeLayers(IEnumerable<int> layerIndices);
    
    /// <summary>
    /// Unfreezes specified layers to allow updates during training.
    /// </summary>
    /// <param name="layerIndices">Indices of layers to unfreeze.</param>
    void UnfreezeLayers(IEnumerable<int> layerIndices);
    
    /// <summary>
    /// Gets the indices of currently frozen layers.
    /// </summary>
    IEnumerable<int> GetFrozenLayers();
    
    /// <summary>
    /// Sets different learning rates for different layers (discriminative fine-tuning).
    /// </summary>
    /// <param name="layerLearningRates">Dictionary mapping layer indices to learning rates.</param>
    void SetLayerLearningRates(IDictionary<int, T> layerLearningRates);
    
    /// <summary>
    /// Gets information about which parts of the model were transferred.
    /// </summary>
    TransferInfo<T> GetTransferInfo();
    
    /// <summary>
    /// Adapts the model to a new domain using domain adaptation techniques.
    /// </summary>
    /// <param name="sourceData">Data from the source domain.</param>
    /// <param name="targetData">Data from the target domain.</param>
    /// <param name="method">The domain adaptation method to use.</param>
    void AdaptDomain(TInput[] sourceData, TInput[] targetData, DomainAdaptationMethod method);
    
    /// <summary>
    /// Gets the transferability score between this model and a target task.
    /// </summary>
    /// <param name="targetData">Sample data from the target task.</param>
    /// <param name="targetLabels">Sample labels from the target task.</param>
    /// <returns>A score indicating how well the model might transfer to the target task.</returns>
    T GetTransferabilityScore(TInput[] targetData, TOutput[] targetLabels);
}

/// <summary>
/// Represents an adapter for converting between different input types during transfer learning.
/// </summary>
public interface IInputAdapter<T, TSourceInput, TTargetInput>
{
    /// <summary>
    /// Adapts input from the source type to the target type.
    /// </summary>
    TTargetInput Adapt(TSourceInput sourceInput);
    
    /// <summary>
    /// Gets the parameters of the adapter that can be learned.
    /// </summary>
    Vector<T> GetParameters();
    
    /// <summary>
    /// Sets the parameters of the adapter.
    /// </summary>
    void SetParameters(Vector<T> parameters);
}

/// <summary>
/// Represents an adapter for converting between different output types during transfer learning.
/// </summary>
public interface IOutputAdapter<T, TSourceOutput, TTargetOutput>
{
    /// <summary>
    /// Adapts output from the source type to the target type.
    /// </summary>
    TTargetOutput Adapt(TSourceOutput sourceOutput);
    
    /// <summary>
    /// Gets the parameters of the adapter that can be learned.
    /// </summary>
    Vector<T> GetParameters();
    
    /// <summary>
    /// Sets the parameters of the adapter.
    /// </summary>
    void SetParameters(Vector<T> parameters);
}