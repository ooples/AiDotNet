using System.Collections.Generic;
using AiDotNet.Enums;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for transfer learning operations.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class TransferLearningOptions<T> : ModelOptions
{
    /// <summary>
    /// Gets or sets the layers to freeze during transfer.
    /// </summary>
    public IList<int> LayersToFreeze { get; set; } = new List<int>();
    
    /// <summary>
    /// Gets or sets the layers to transfer from the source model.
    /// </summary>
    public IList<int> LayersToTransfer { get; set; } = new List<int>();
    
    /// <summary>
    /// Gets or sets whether to reset the final layers after transfer.
    /// </summary>
    public bool ResetFinalLayers { get; set; } = true;
    
    /// <summary>
    /// Gets or sets the learning rate scaling factor for transferred layers.
    /// </summary>
    public T TransferredLayerLearningRateScale { get; set; } = default!;
    
    /// <summary>
    /// Gets or sets whether to use discriminative learning rates.
    /// </summary>
    public bool UseDiscriminativeLearningRates { get; set; } = false;
    
    /// <summary>
    /// Gets or sets the domain adaptation regularization strength.
    /// </summary>
    public T DomainAdaptationStrength { get; set; } = default!;
    
    /// <summary>
    /// Gets or sets additional transfer-specific parameters.
    /// </summary>
    public Dictionary<string, object> AdditionalParameters { get; set; } = new Dictionary<string, object>();
}