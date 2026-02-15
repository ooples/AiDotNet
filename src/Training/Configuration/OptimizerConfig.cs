namespace AiDotNet.Training.Configuration;

/// <summary>
/// Configuration for the optimizer section of a training recipe.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The optimizer controls how the model learns from its mistakes.
/// The name should match an <see cref="AiDotNet.Enums.OptimizerType"/> value
/// (e.g., "Adam", "GradientDescent", "Normal").
/// </para>
/// </remarks>
public class OptimizerConfig
{
    /// <summary>
    /// Gets or sets the name of the optimizer type to create.
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the learning rate for the optimizer.
    /// </summary>
    public double LearningRate { get; set; } = 0.001;

    /// <summary>
    /// Gets or sets additional optimizer parameters as key-value pairs.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Some optimizers accept additional parameters beyond the learning rate.
    /// For example, Adam has beta1 and beta2 momentum parameters. This dictionary lets you specify
    /// any extra settings the optimizer needs.
    /// </para>
    /// </remarks>
    public Dictionary<string, object> Params { get; set; } = new();
}
