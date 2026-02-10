namespace AiDotNet.Training.Configuration;

/// <summary>
/// Root configuration object for a complete training recipe defined in YAML.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A training recipe brings together all the pieces needed to train a model:
/// the model architecture, the dataset, the optimizer, the loss function, and training settings.
/// You can define all of this in a single YAML file and load it with <see cref="AiDotNet.Configuration.YamlConfigLoader"/>.
/// </para>
/// <para>
/// <b>Example YAML:</b>
/// <code>
/// model:
///   name: "ExponentialSmoothing"
///   params:
///     seasonalPeriod: 12
///
/// dataset:
///   name: "sales-data"
///   path: "data/sales.csv"
///   batchSize: 32
///
/// optimizer:
///   name: "Adam"
///   learningRate: 0.001
///
/// lossFunction:
///   name: "MeanSquaredError"
///
/// trainer:
///   epochs: 50
///   enableLogging: true
///   seed: 42
/// </code>
/// </para>
/// </remarks>
public class TrainingRecipeConfig
{
    /// <summary>
    /// Gets or sets the model configuration section.
    /// </summary>
    public ModelConfig? Model { get; set; }

    /// <summary>
    /// Gets or sets the dataset configuration section.
    /// </summary>
    public DatasetConfig? Dataset { get; set; }

    /// <summary>
    /// Gets or sets the optimizer configuration section.
    /// </summary>
    public OptimizerConfig? Optimizer { get; set; }

    /// <summary>
    /// Gets or sets the loss function configuration section.
    /// </summary>
    public LossFunctionConfig? LossFunction { get; set; }

    /// <summary>
    /// Gets or sets the trainer settings section.
    /// </summary>
    public TrainerSettings? Trainer { get; set; }
}
