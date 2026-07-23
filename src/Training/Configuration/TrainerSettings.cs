namespace AiDotNet.Training.Configuration;

/// <summary>
/// Configuration for the trainer behavior section of a training recipe.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> These settings control how the training loop runs - how many
/// times it goes through the data (epochs), whether to print progress, and an optional
/// random seed for reproducible results.
/// </para>
/// </remarks>
public class TrainerSettings
{
    /// <summary>
    /// Gets or sets the number of training epochs (full passes through the data).
    /// </summary>
    public int Epochs { get; set; } = 10;

    /// <summary>
    /// Gets or sets whether to log training progress (epoch number and loss).
    /// </summary>
    public bool EnableLogging { get; set; } = true;

    /// <summary>
    /// Gets or sets an optional random seed for reproducible training runs.
    /// </summary>
    public int? Seed { get; set; }
}
