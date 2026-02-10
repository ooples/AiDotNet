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
}
