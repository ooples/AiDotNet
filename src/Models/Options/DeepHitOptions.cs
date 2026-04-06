using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for DeepHit survival analysis model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// DeepHit is a deep learning approach to survival analysis that directly learns the
/// distribution of survival times without making assumptions like proportional hazards.
/// It can also handle competing risks (multiple possible failure types).
/// </para>
/// <para>
/// <b>For Beginners:</b> While DeepSurv assumes that factors affect hazard rates
/// proportionally (like "smoking doubles your risk at all times"), DeepHit makes no
/// such assumption - it learns the actual probability of an event happening at each
/// specific time point.
///
/// DeepHit is particularly useful when:
/// - The proportional hazards assumption doesn't hold
/// - You have multiple competing risks (e.g., patient could die from disease OR treatment side effects)
/// - You want to predict exact survival probabilities at specific time horizons
/// - You need to handle complex, non-linear relationships in the data
///
/// For example: "What's the probability this patient survives past 1 year? 2 years? 5 years?"
/// DeepHit directly estimates these probabilities.
/// </para>
/// </remarks>
public class DeepHitOptions<T>
{
    /// <summary>
    /// Gets or sets the number of discrete time bins for survival prediction.
    /// </summary>
    /// <value>Default is 100.</value>
    public int NumTimeBins { get; set; } = 100;

    /// <summary>
    /// Gets or sets the number of hidden layers in the shared sub-network.
    /// </summary>
    /// <value>Default is 2.</value>
    public int NumSharedLayers { get; set; } = 2;

    /// <summary>
    /// Gets or sets the number of hidden layers in each cause-specific sub-network.
    /// </summary>
    /// <value>Default is 2.</value>
    public int NumCauseLayers { get; set; } = 2;

    /// <summary>
    /// Gets or sets the number of neurons in each hidden layer.
    /// </summary>
    /// <value>Default is 64.</value>
    public int HiddenLayerSize { get; set; } = 64;

    /// <summary>
    /// Gets or sets the number of competing risks/causes.
    /// </summary>
    /// <value>Default is 1 (single event type).</value>
    public int NumRisks { get; set; } = 1;

    /// <summary>
    /// Gets or sets the dropout rate for regularization.
    /// </summary>
    /// <value>Default is 0.1.</value>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the learning rate for optimization.
    /// </summary>
    /// <value>Default is 0.001.</value>
    public double LearningRate { get; set; } = 0.001;

    /// <summary>
    /// Gets or sets the number of training epochs.
    /// </summary>
    /// <value>Default is 100.</value>
    public int Epochs { get; set; } = 100;

    /// <summary>
    /// Gets or sets the batch size for training.
    /// </summary>
    /// <value>Default is 64.</value>
    public int BatchSize { get; set; } = 64;

    /// <summary>
    /// Gets or sets the weight for the ranking loss component.
    /// </summary>
    /// <value>Default is 1.0.</value>
    public double RankingWeight { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the sigma parameter for the ranking loss kernel.
    /// </summary>
    /// <value>Default is 0.1.</value>
    public double RankingSigma { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the L2 regularization strength.
    /// </summary>
    /// <value>Default is 0.001.</value>
    public double L2Regularization { get; set; } = 0.001;

    /// <summary>
    /// Gets or sets the activation function for hidden layers.
    /// </summary>
    /// <value>Default is ReLU.</value>
    /// <remarks>
    /// Uses the standard IActivationFunction interface, which provides SIMD-accelerated
    /// Forward/Backward passes via the Engine. Any activation function that implements
    /// the interface can be used — no need to modify DeepHit itself.
    /// </remarks>
    public IActivationFunction<T> Activation { get; set; } = new ReLUActivation<T>();

    /// <summary>
    /// Gets or sets whether to use batch normalization.
    /// </summary>
    /// <value>Default is true.</value>
    public bool UseBatchNormalization { get; set; } = true;

    /// <summary>
    /// Gets or sets the random seed for reproducibility.
    /// </summary>
    public int? Seed { get; set; }

    /// <summary>
    /// Gets or sets the patience for early stopping.
    /// </summary>
    /// <value>Default is 10. Set to null to disable early stopping.</value>
    public int? EarlyStoppingPatience { get; set; } = 10;

    /// <summary>
    /// Gets or sets the time horizons of interest for evaluation.
    /// </summary>
    /// <value>Default is null (all time points).</value>
    public double[]? EvaluationHorizons { get; set; }
}
