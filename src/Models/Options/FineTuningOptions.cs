using AiDotNet.Enums;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options specifically for fine-tuning operations.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class FineTuningOptions<T> : ModelOptions
{
    /// <summary>
    /// Gets or sets the initial learning rate for fine-tuning.
    /// </summary>
    public T InitialLearningRate { get; set; } = default!;
    
    /// <summary>
    /// Gets or sets the number of epochs for fine-tuning.
    /// </summary>
    public int Epochs { get; set; } = 10;
    
    /// <summary>
    /// Gets or sets the batch size for fine-tuning.
    /// </summary>
    public int BatchSize { get; set; } = 32;
    
    /// <summary>
    /// Gets or sets whether to gradually unfreeze layers during training.
    /// </summary>
    public bool UnfreezeGradually { get; set; } = false;
    
    /// <summary>
    /// Gets or sets the number of epochs between unfreezing operations.
    /// </summary>
    public int EpochsPerUnfreeze { get; set; } = 2;
    
    /// <summary>
    /// Gets or sets the learning rate decay factor.
    /// </summary>
    public T LearningRateDecay { get; set; } = default!;
    
    /// <summary>
    /// Gets or sets the warmup epochs for gradual learning rate increase.
    /// </summary>
    public int WarmupEpochs { get; set; } = 0;
    
    /// <summary>
    /// Gets or sets the optimizer type to use for fine-tuning.
    /// </summary>
    public OptimizerType Optimizer { get; set; } = OptimizerType.Adam;
    
    /// <summary>
    /// Gets or sets whether to use mixed precision training.
    /// </summary>
    public bool UseMixedPrecision { get; set; } = false;
    
    /// <summary>
    /// Gets or sets the gradient clipping value (0 = no clipping).
    /// </summary>
    public T GradientClipping { get; set; } = default!;
}